"""
Feature engineering pipeline for the Football Predictor backend.

This module turns raw rows from the ``matches`` table into a model-ready feature
matrix: recent form, venue-specific form, Elo ratings, head-to-head records,
league/season encodings and the training targets.

LEAKAGE RULES (the whole point of this file)
--------------------------------------------
Every feature for a match played on date *D* is computed from matches with
``date < D`` only. Concretely:

* Matches are expanded to one row per team per match, accumulated with prefix
  sums, and each window is bounded by the number of matches that took place
  *strictly before* the row's own date. ``groupby().rolling()`` with
  ``shift(1)`` is deliberately avoided: it counts rows, so two matches on the
  same day would see each other.
* A row's own result, goals, shots, corners and cards never enter its own
  features.
* Elo is the most recent rating with ``date < match date``.
* Head-to-head counts only previous meetings of the same pair of teams.

Modelling choices worth knowing about:

* Rolling values are **sums over the window** (wins, points, goals, ...), and a
  team with fewer than ``N`` matches uses whatever history exists
  (``min_periods=1``).
* Missing stats (a season without shots/corners columns) contribute ``0`` to a
  window sum, consistent with the final ``fillna(0)`` on feature columns.
* ``h2h_*`` counts the last five meetings of the two teams at either venue,
  attributed to the team that plays at home today: ``h2h_home_wins_5`` is how
  often today's home team beat today's away team in those meetings, and
  ``h2h_home_gf_5`` the goals it scored in them. See :func:`_h2h_features`.

Output: one row per match, features followed by targets, sorted by date.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import select

from database import EloRating, Match, session_scope

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
# Resolved location of the saved feature matrix and encoders; kept as a named
# path so prediction-time code can locate the same artifacts.
PROCESSED_DIR = DATA_DIR / "processed"

LEAGUE_ENCODER_PATH = DATA_DIR / "league_encoder.joblib"
SEASON_ENCODER_PATH = DATA_DIR / "season_encoder.joblib"

DEFAULT_OUTPUT_PATH = "data/processed/features.csv"

# Rolling windows: the short one for the main feature blocks, the long one for
# the stability features.
WINDOW = 5
LONG_WINDOW = 10

# Stats computed for every team-perspective row, then summed over a window.
STAT_COLUMNS = (
    "wins",
    "draws",
    "losses",
    "points",
    "gf",
    "ga",
    "shots",
    "shots_ot",
    "corners",
)

# Which stats each feature block exposes (``gd`` is derived as gf - ga).
FORM_STATS = ("wins", "draws", "losses", "points", "gf", "ga", "gd", "shots", "shots_ot", "corners")
FORM_STATS_LONG = ("wins", "draws", "losses", "points", "gf", "ga", "gd")
VENUE_STATS = ("wins", "draws", "losses", "points", "gf", "ga", "gd", "shots", "corners")

H2H_STATS = ("home_wins", "draws", "away_wins", "home_gf", "away_gf")

TARGET_COLUMNS = (
    "target_result",
    "target_home_goals",
    "target_away_goals",
    "target_home_corners",
    "target_away_corners",
)

VALID_RESULTS = ("H", "D", "A")


def _feature_columns() -> list[str]:
    """Full feature column list, in the order used by the model and the spec."""
    columns: list[str] = []
    columns += [f"home_form_{stat}_5" for stat in FORM_STATS]
    columns += [f"home_form_{stat}_10" for stat in FORM_STATS_LONG]
    columns += [f"home_home_{stat}_5" for stat in VENUE_STATS]
    columns += [f"away_form_{stat}_5" for stat in FORM_STATS]
    columns += [f"away_form_{stat}_10" for stat in FORM_STATS_LONG]
    columns += [f"away_away_{stat}_5" for stat in VENUE_STATS]
    columns += ["home_elo", "away_elo", "elo_difference"]
    columns += [f"h2h_{stat}_5" for stat in H2H_STATS]
    columns += ["league_encoded", "season_encoded"]
    return columns


FEATURE_COLUMNS = _feature_columns()
OUTPUT_COLUMNS = FEATURE_COLUMNS + list(TARGET_COLUMNS)


# --------------------------------------------------------------------------- #
# Loading from PostgreSQL
# --------------------------------------------------------------------------- #


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """An empty DataFrame carrying the right column names."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def load_matches_from_db() -> pd.DataFrame:
    """
    Load the whole ``matches`` table as a DataFrame.

    Returns an empty DataFrame with the correct column names (and prints a
    warning) when the database is unavailable.
    """
    columns = [column.name for column in Match.__table__.columns]
    try:
        with session_scope() as session:
            rows = session.execute(select(*Match.__table__.columns)).all()
    except Exception as exc:  # noqa: BLE001 - report and continue gracefully
        print(f"[features] WARNING: could not load matches from the database: {exc}")
        return _empty_frame(columns)

    frame = pd.DataFrame(rows, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    print(f"[features] Loaded {len(frame)} matches from the database")
    return frame


def load_elo_from_db() -> pd.DataFrame:
    """
    Load the whole ``elo_ratings`` table as a DataFrame.

    Returns an empty DataFrame with the correct column names (and prints a
    warning) when the database is unavailable.
    """
    columns = [column.name for column in EloRating.__table__.columns]
    try:
        with session_scope() as session:
            rows = session.execute(select(*EloRating.__table__.columns)).all()
    except Exception as exc:  # noqa: BLE001 - report and continue gracefully
        print(f"[features] WARNING: could not load Elo ratings from the database: {exc}")
        return _empty_frame(columns)

    frame = pd.DataFrame(rows, columns=columns)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if "elo" in frame.columns:
        frame["elo"] = pd.to_numeric(frame["elo"], errors="coerce")
    print(f"[features] Loaded {len(frame)} Elo ratings from the database")
    return frame


# --------------------------------------------------------------------------- #
# Elo lookups
# --------------------------------------------------------------------------- #


def _elo_lookup_tables(elo_df: pd.DataFrame | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build ``{team: (sorted_dates, elos)}`` arrays for fast lookups.

    Dates are stored as ``datetime64[ns]`` so they can be searched with
    ``np.searchsorted`` instead of filtering the whole frame per fixture.
    """
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if elo_df is None or len(elo_df) == 0:
        return tables
    if not {"team", "date", "elo"}.issubset(elo_df.columns):
        print("[features] WARNING: Elo data is missing required columns, ignoring it")
        return tables

    frame = elo_df.loc[:, ["team", "date", "elo"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["elo"] = pd.to_numeric(frame["elo"], errors="coerce")
    frame = frame.dropna(subset=["team", "date", "elo"])
    frame = frame.sort_values(["team", "date"], kind="stable")

    for team, group in frame.groupby("team", sort=False):
        tables[str(team)] = (
            group["date"].to_numpy(dtype="datetime64[ns]"),
            group["elo"].to_numpy(dtype="float64"),
        )
    return tables


def _last_elo_before(entry: tuple[np.ndarray, np.ndarray] | None, before_date) -> float | None:
    """Last Elo strictly before ``before_date`` from a ``(dates, elos)`` entry."""
    if entry is None:
        return None
    dates, elos = entry
    if len(dates) == 0:
        return None
    when = pd.Timestamp(before_date).to_datetime64()
    index = int(np.searchsorted(dates, np.datetime64(when, "ns"), side="left")) - 1
    if index < 0:
        return None
    return float(elos[index])


def get_pre_match_elo(elo_df: pd.DataFrame, team: str, before_date: pd.Timestamp) -> float | None:
    """
    Most recent Elo rating for ``team`` with ``date < before_date``.

    Returns ``None`` when the team has no rating before that date -- the case
    prediction time has to cope with for a team ClubElo has never rated.
    """
    if not team or before_date is None or pd.isna(before_date):
        return None
    tables = _elo_lookup_tables(elo_df)
    return _last_elo_before(tables.get(str(team)), before_date)


def _pre_match_elo_series(
    tables: dict[str, tuple[np.ndarray, np.ndarray]],
    teams: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    """Vectorised :func:`get_pre_match_elo` for many fixtures at once."""
    values = np.full(len(teams), np.nan, dtype="float64")
    for position, (team, when) in enumerate(zip(teams, dates)):
        value = _last_elo_before(tables.get(str(team)), when)
        if value is not None:
            values[position] = value
    return values


# --------------------------------------------------------------------------- #
# Rolling windows (leakage-safe)
# --------------------------------------------------------------------------- #


def _window_sums(
    frame: pd.DataFrame,
    group_keys: list[str],
    value_columns: tuple[str, ...] | list[str],
    window: int,
) -> pd.DataFrame:
    """
    Sum ``value_columns`` over each group's previous ``window`` matches.

    The frame needs a ``date`` column and one row per group-match. For every row
    the window covers that group's matches with a date *strictly before* the
    row's own date, so a match can never see itself, nor another match played on
    the same day.

    Implementation: rows are ordered by ``(group, date)`` and prefix sums are
    computed once, so each window is the difference of two prefix sums. That is
    both vectorised and exact about date boundaries.

    Returns a DataFrame indexed like ``frame``; groups without prior matches get
    zeros.
    """
    if len(frame) == 0:
        return pd.DataFrame(index=frame.index, columns=list(value_columns), dtype="float64")

    ordered = frame.sort_values(list(group_keys) + ["date"], kind="stable")
    positions = np.arange(len(ordered), dtype="int64")

    # Position within the group, and within the group's rows for this same date.
    within_group = ordered.groupby(list(group_keys), sort=False).cumcount().to_numpy()
    within_date = (
        ordered.groupby(list(group_keys) + ["date"], sort=False).cumcount().to_numpy()
    )

    group_start = positions - within_group  # global position of the group's first row
    prior_matches = within_group - within_date  # matches strictly before this date

    # Global position of the last row that belongs in the window. When the group
    # has no earlier match this is ``group_start - 1``, so that ``high - low``
    # collapses to zero: both ends are the same prefix value, which also cancels
    # the prefix sums accumulated by *other* groups.
    last_in_window = group_start + prior_matches - 1
    # Global position just before the window starts.
    before_window = group_start + np.maximum(prior_matches - window, 0) - 1

    ordered_result = pd.DataFrame(index=ordered.index)
    for column in value_columns:
        values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype="float64")
        values = np.nan_to_num(values, nan=0.0)
        prefixes = np.concatenate(([0.0], np.cumsum(values)))
        high = prefixes[last_in_window + 1]
        low = prefixes[np.maximum(before_window, -1) + 1]
        ordered_result[column] = high - low

    return ordered_result.sort_index()


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a column, or an all-NaN column when the source lacks it."""
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index)


def _team_rows(work: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    One row per match from one team's perspective.

    ``side`` is ``"home"`` or ``"away"``. The returned frame carries the values
    needed for window sums plus ``row_id`` to map features back to matches.
    """
    if side == "home":
        goals_for, goals_against = "home_goals", "away_goals"
        shots, shots_ot, corners = "home_shots", "home_shots_ot", "home_corners"
        win_result, loss_result = "H", "A"
    else:
        goals_for, goals_against = "away_goals", "home_goals"
        shots, shots_ot, corners = "away_shots", "away_shots_ot", "away_corners"
        win_result, loss_result = "A", "H"

    wins = work["_result"].eq(win_result).astype("float64")
    draws = work["_result"].eq("D").astype("float64")
    losses = work["_result"].eq(loss_result).astype("float64")

    return pd.DataFrame(
        {
            "row_id": work["_row_id"].to_numpy(),
            "is_home": side == "home",
            "team": work[f"{side}_team"].astype(str).to_numpy(),
            "date": work["date"].to_numpy(),
            "wins": wins.to_numpy(),
            "draws": draws.to_numpy(),
            "losses": losses.to_numpy(),
            "points": (wins * 3.0 + draws).to_numpy(),
            "gf": pd.to_numeric(_series(work, goals_for), errors="coerce").to_numpy(dtype="float64"),
            "ga": pd.to_numeric(_series(work, goals_against), errors="coerce").to_numpy(dtype="float64"),
            "shots": pd.to_numeric(_series(work, shots), errors="coerce").to_numpy(dtype="float64"),
            "shots_ot": pd.to_numeric(_series(work, shots_ot), errors="coerce").to_numpy(dtype="float64"),
            "corners": pd.to_numeric(_series(work, corners), errors="coerce").to_numpy(dtype="float64"),
        }
    )


def _with_goal_difference(sums: pd.DataFrame) -> pd.DataFrame:
    """Add the derived ``gd`` column to a block of window sums."""
    sums["gd"] = sums["gf"] - sums["ga"]
    return sums


def _sums_to_features(
    sums: pd.DataFrame,
    row_ids: pd.Series,
    prefix: str,
    suffix: str,
    stats: tuple[str, ...],
    matches: int,
) -> pd.DataFrame:
    """Rename window sums into ``{prefix}_{stat}_{suffix}`` feature columns."""
    block = sums.loc[:, list(stats)].copy()
    block.index = np.asarray(row_ids, dtype="int64")
    block = block.sort_index().reindex(range(matches))
    block.columns = [f"{prefix}_{stat}_{suffix}" for stat in stats]
    return block


# --------------------------------------------------------------------------- #
# Head to head
# --------------------------------------------------------------------------- #


def _h2h_features(work: pd.DataFrame, matches: int) -> pd.DataFrame:
    """
    Head-to-head features over the last ``WINDOW`` meetings of each fixture.

    Meetings are counted at either venue and attributed to the team playing at
    home today. To keep the pair orientation consistent, every meeting is first
    rewritten as ``(first_team, second_team)`` in a canonical (lexicographic)
    order; only the output naming depends on today's venue.
    """
    home = work["home_team"].astype(str).to_numpy()
    away = work["away_team"].astype(str).to_numpy()
    swapped = home > away

    home_goals = pd.to_numeric(_series(work, "home_goals"), errors="coerce").to_numpy(dtype="float64")
    away_goals = pd.to_numeric(_series(work, "away_goals"), errors="coerce").to_numpy(dtype="float64")

    first_goals = np.where(swapped, away_goals, home_goals)
    second_goals = np.where(swapped, home_goals, away_goals)

    first_win = (first_goals > second_goals).astype("float64")
    second_win = (second_goals > first_goals).astype("float64")
    draw = (first_goals == second_goals).astype("float64")
    # A meeting without a recorded score counts as neither a win nor a draw.
    unknown = np.isnan(first_goals) | np.isnan(second_goals)
    first_win[unknown] = 0.0
    second_win[unknown] = 0.0
    draw[unknown] = 0.0

    frame = pd.DataFrame(
        {
            "row_id": work["_row_id"].to_numpy(),
            "first_team": np.where(swapped, away, home),
            "second_team": np.where(swapped, home, away),
            "date": work["date"].to_numpy(),
            "first_win": first_win,
            "second_win": second_win,
            "draw": draw,
            "first_gf": np.nan_to_num(first_goals, nan=0.0),
            "second_gf": np.nan_to_num(second_goals, nan=0.0),
        }
    )

    sums = _window_sums(
        frame,
        ["first_team", "second_team"],
        ("first_win", "second_win", "draw", "first_gf", "second_gf"),
        WINDOW,
    )

    h2h = pd.DataFrame(index=range(matches))
    h2h["h2h_home_wins_5"] = np.where(swapped, sums["second_win"], sums["first_win"])
    h2h["h2h_draws_5"] = sums["draw"].to_numpy()
    h2h["h2h_away_wins_5"] = np.where(swapped, sums["first_win"], sums["second_win"])
    h2h["h2h_home_gf_5"] = np.where(swapped, sums["second_gf"], sums["first_gf"])
    h2h["h2h_away_gf_5"] = np.where(swapped, sums["first_gf"], sums["second_gf"])
    return h2h


# --------------------------------------------------------------------------- #
# Label encoders
# --------------------------------------------------------------------------- #


def fit_label_encoders(df: pd.DataFrame) -> tuple[LabelEncoder, LabelEncoder]:
    """Fit one ``LabelEncoder`` for leagues and one for seasons."""
    league_encoder = LabelEncoder().fit(
        df["league"].fillna("unknown").astype(str).to_numpy()
    )
    season_encoder = LabelEncoder().fit(
        df["season"].fillna("unknown").astype(str).to_numpy()
    )
    return league_encoder, season_encoder


def encode_labels(encoder: LabelEncoder, values) -> np.ndarray:
    """
    Encode labels with a previously fitted encoder.

    Unknown labels become ``-1`` with a warning instead of raising, so a fixture
    from a league the encoders have never seen cannot break prediction time.
    """
    series = pd.Series(values).fillna("unknown").astype(str)
    known = set(encoder.classes_)
    unknown = ~series.isin(known)
    if bool(unknown.any()):
        examples = sorted(series[unknown].unique())[:5]
        print(
            f"[features] WARNING: {int(unknown.sum())} unknown label(s) "
            f"encoded as -1: {examples}"
        )

    codes = np.full(len(series), -1, dtype="int64")
    if (~unknown).any():
        codes[(~unknown).to_numpy()] = encoder.transform(series[~unknown].to_numpy())
    return codes


def save_label_encoders(league_encoder: LabelEncoder, season_encoder: LabelEncoder) -> None:
    """Persist the encoders so predictions reuse the same integer mapping."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(league_encoder, LEAGUE_ENCODER_PATH)
    joblib.dump(season_encoder, SEASON_ENCODER_PATH)
    print(f"[features] Saved league encoder to {LEAGUE_ENCODER_PATH}")
    print(f"[features] Saved season encoder to {SEASON_ENCODER_PATH}")


def load_label_encoders() -> tuple[LabelEncoder | None, LabelEncoder | None]:
    """Load the saved encoders, or ``(None, None)`` when they are missing."""
    try:
        return joblib.load(LEAGUE_ENCODER_PATH), joblib.load(SEASON_ENCODER_PATH)
    except Exception as exc:  # noqa: BLE001 - missing or unreadable files
        print(f"[features] WARNING: could not load label encoders: {exc}")
        return None, None


# --------------------------------------------------------------------------- #
# Core feature builder
# --------------------------------------------------------------------------- #


def build_features(
    df: pd.DataFrame,
    elo_df: pd.DataFrame | None = None,
    league_encoder: LabelEncoder | None = None,
    season_encoder: LabelEncoder | None = None,
) -> pd.DataFrame:
    """
    Build one leakage-free feature row per match.

    ``df`` is the full ``matches`` dataframe (all columns of the matches table).
    ``elo_df`` defaults to :func:`load_elo_from_db`; the encoders default to
    fresh ones fitted on ``df`` (call :func:`fit_label_encoders` when you want to
    control and save them). The encoders actually used are attached to
    ``result.attrs``.

    Returns a DataFrame with ``FEATURE_COLUMNS + TARGET_COLUMNS``, sorted by
    date ascending.
    """
    empty = _empty_frame(OUTPUT_COLUMNS)
    empty.attrs["date_range"] = (None, None)

    if df is None or len(df) == 0:
        print("[features] WARNING: no matches to build features from")
        return empty

    work = df.copy()

    if "date" not in work.columns:
        print("[features] ERROR: matches data has no 'date' column")
        return empty
    required = ["home_team", "away_team", "league", "season"]
    missing_required = [column for column in required if column not in work.columns]
    if missing_required:
        print(f"[features] ERROR: matches data is missing column(s): {missing_required}")
        return empty

    # 1. Dates: coerce, drop unusable rows, sort ascending (never shuffled).
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    dropped_dates = int(work["date"].isna().sum())
    work = work.loc[work["date"].notna()].copy()
    if dropped_dates:
        print(f"[features] Dropped {dropped_dates} row(s) with a null date")

    for column in ("home_team", "away_team"):
        blank = work[column].isna() | work[column].astype(str).str.strip().eq("")
        if bool(blank.any()):
            print(f"[features] Dropped {int(blank.sum())} row(s) with a blank {column}")
            work = work.loc[~blank].copy()

    work = work.sort_values("date", kind="stable").reset_index(drop=True)
    matches = len(work)
    if matches == 0:
        print("[features] WARNING: no usable matches left after cleaning")
        return empty

    work["_row_id"] = np.arange(matches, dtype="int64")
    work["_result"] = _clean_result(work)

    # 2. Team-perspective rows, then leakage-safe rolling windows.
    home_rows = _team_rows(work, "home")
    away_rows = _team_rows(work, "away")
    both_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    home_side = both_rows["is_home"].to_numpy()

    sums_all_5 = _with_goal_difference(_window_sums(both_rows, ["team"], STAT_COLUMNS, WINDOW))
    sums_all_10 = _with_goal_difference(_window_sums(both_rows, ["team"], STAT_COLUMNS, LONG_WINDOW))
    # Venue-restricted frames: only the home rows / only the away rows, so the
    # windows cover the last N matches *at that venue*.
    sums_home_venue_5 = _with_goal_difference(_window_sums(home_rows, ["team"], STAT_COLUMNS, WINDOW))
    sums_away_venue_5 = _with_goal_difference(_window_sums(away_rows, ["team"], STAT_COLUMNS, WINDOW))

    features = pd.DataFrame(index=range(matches))
    features = features.join(
        _sums_to_features(
            sums_all_5[home_side], both_rows.loc[home_side, "row_id"],
            "home_form", "5", FORM_STATS, matches,
        )
    )
    features = features.join(
        _sums_to_features(
            sums_all_10[home_side], both_rows.loc[home_side, "row_id"],
            "home_form", "10", FORM_STATS_LONG, matches,
        )
    )
    features = features.join(
        _sums_to_features(
            sums_home_venue_5, home_rows["row_id"],
            "home_home", "5", VENUE_STATS, matches,
        )
    )
    features = features.join(
        _sums_to_features(
            sums_all_5[~home_side], both_rows.loc[~home_side, "row_id"],
            "away_form", "5", FORM_STATS, matches,
        )
    )
    features = features.join(
        _sums_to_features(
            sums_all_10[~home_side], both_rows.loc[~home_side, "row_id"],
            "away_form", "10", FORM_STATS_LONG, matches,
        )
    )
    features = features.join(
        _sums_to_features(
            sums_away_venue_5, away_rows["row_id"],
            "away_away", "5", VENUE_STATS, matches,
        )
    )

    # 3. Elo: most recent rating strictly before the match date.
    if elo_df is None:
        elo_df = load_elo_from_db()
    elo_tables = _elo_lookup_tables(elo_df)
    if not elo_tables:
        print("[features] WARNING: no Elo ratings available, Elo features default to 0")
    match_dates = work["date"].to_numpy(dtype="datetime64[ns]")
    features["home_elo"] = _pre_match_elo_series(
        elo_tables, work["home_team"].astype(str).to_numpy(), match_dates
    )
    features["away_elo"] = _pre_match_elo_series(
        elo_tables, work["away_team"].astype(str).to_numpy(), match_dates
    )
    features["elo_difference"] = features["home_elo"] - features["away_elo"]

    # 4. Head to head.
    features = features.join(_h2h_features(work, matches))

    # 5. League/season encodings (same mapping as at prediction time).
    if league_encoder is None or season_encoder is None:
        league_encoder, season_encoder = fit_label_encoders(work)
    features["league_encoded"] = encode_labels(league_encoder, work["league"])
    features["season_encoded"] = encode_labels(season_encoder, work["season"])

    # 6. Targets (kept nullable, never filled).
    features["target_result"] = work["_result"].to_numpy()
    features["target_home_goals"] = _to_nullable_int(work.get("home_goals"), work.index)
    features["target_away_goals"] = _to_nullable_int(work.get("away_goals"), work.index)
    features["target_home_corners"] = _to_nullable_int(work.get("home_corners"), work.index)
    features["target_away_corners"] = _to_nullable_int(work.get("away_corners"), work.index)

    # 7. Fill NaN with 0 for feature columns only -- targets stay nullable.
    features = features.loc[:, OUTPUT_COLUMNS]
    features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].astype("float64").fillna(0.0)
    for column in ("league_encoded", "season_encoded"):
        features[column] = features[column].astype("int64")

    features.attrs["date_range"] = (work["date"].min(), work["date"].max())
    features.attrs["rows"] = matches
    features.attrs["league_encoder"] = league_encoder
    features.attrs["season_encoder"] = season_encoder

    print(
        f"[features] Built {features.shape[0]} rows x {features.shape[1]} columns "
        f"({work['date'].min().date()} to {work['date'].max().date()})"
    )
    return features


def _clean_result(work: pd.DataFrame) -> pd.Series:
    """Result labels, derived from the score when the stored value is missing."""
    raw = work.get("result")
    if raw is None:
        raw = pd.Series(np.nan, index=work.index)
    raw = raw.astype("object")
    text = raw.where(raw.isna(), raw.astype(str).str.strip().str.upper())
    result = text.where(text.isin(VALID_RESULTS))

    home_goals = pd.to_numeric(_series(work, "home_goals"), errors="coerce")
    away_goals = pd.to_numeric(_series(work, "away_goals"), errors="coerce")
    derived = pd.Series(np.nan, index=work.index, dtype="object")
    derived[home_goals > away_goals] = "H"
    derived[home_goals < away_goals] = "A"
    derived[(home_goals == away_goals) & home_goals.notna() & away_goals.notna()] = "D"
    return result.fillna(derived)


def _to_nullable_int(series, index) -> pd.Series:
    """Coerce a column to nullable ``Int64`` (targets keep their NA values)."""
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="Int64")
    return pd.to_numeric(series, errors="coerce").astype("Int64")


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


def build_and_save_features(output_path: str = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """
    Load from PostgreSQL, build the feature matrix, save it to CSV and save the
    league/season encoders for prediction time.

    ``output_path`` is resolved relative to the ``backend/`` directory.
    """
    matches = load_matches_from_db()
    if len(matches) == 0:
        print("[features] WARNING: no matches loaded, nothing to save")
        empty = _empty_frame(OUTPUT_COLUMNS)
        empty.attrs["date_range"] = (None, None)
        return empty

    elo_df = load_elo_from_db()
    league_encoder, season_encoder = fit_label_encoders(matches)

    features = build_features(
        matches,
        elo_df=elo_df,
        league_encoder=league_encoder,
        season_encoder=season_encoder,
    )

    path = Path(output_path)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)

    save_label_encoders(league_encoder, season_encoder)

    first_date, last_date = features.attrs.get("date_range", (None, None))
    print("=" * 70)
    print("[features] Saved feature matrix")
    print(f"  Rows         : {len(features)}")
    print(f"  Columns      : {len(features.columns)}")
    if first_date is not None and last_date is not None:
        print(f"  Date range   : {first_date.date()} to {last_date.date()}")
    else:
        print("  Date range   : n/a")
    print(f"  Output       : {path}")
    print("=" * 70)
    return features


if __name__ == "__main__":
    build_and_save_features()
