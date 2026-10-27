"""
Feature engineering pipeline for the Football Predictor.

Model inputs are rolling form, venue form, Elo, H2H and the specialist
rates. No team-season snapshot table is read here: those columns were
always zero for live fixtures and were removed from the model contract.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import inspect, text

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import session_scope

# The model input columns live in ONE module, shared with the trainer and the
# API. The match-history aliases are re-exported here for inference.
try:  # imported as ``features.builder`` (trainer, API, tests)
    from features.feature_columns import (  # noqa: F401 - re-exported
        MATCH_HISTORY_ALIASES,
        MATCH_HISTORY_COLUMNS,
    )
except ImportError:  # pragma: no cover - ``python features/builder.py``
    from feature_columns import (  # type: ignore[no-redef]
        MATCH_HISTORY_ALIASES,
        MATCH_HISTORY_COLUMNS,
    )


#: Suffixes providers append to club names. Elo is a rolling rating: the
#: lookup must find ``Arsenal`` even when the stored row says ``Arsenal FC``.
_ELO_NAME_SUFFIXES = frozenset(
    {"afc", "fc", "cf", "sc", "ac", "sk", "kv", "bk", "if", "cfc"}
)


def normalise_elo_team_name(name: object) -> str:
    """Comparable key for team names in ``elo_ratings`` / ``matches``.

    Lower-cases, strips accents and punctuation, and drops club-type suffixes
    (``FC``, ``AFC``, ...) so a fixture's ``Arsenal`` matches a stored
    ``Arsenal FC``. Read path only: never renames a row in the database.
    """
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name.strip().casefold())
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    words = re.findall(r"[a-z0-9]+", plain)
    # Club-type tokens can be a prefix ("AFC Ajax", "FC Barcelona") or a
    # suffix ("Arsenal FC"); compare on what is left of the actual name.
    stripped = [word for word in words if word not in _ELO_NAME_SUFFIXES]
    return " ".join(stripped or words)


#: Accepted spellings of the three elo_ratings columns we read. The deployed
#: table is introspected at query time so a renamed column no longer turns
#: every lookup into a silent 1500 fallback.
_ELO_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "team":   ("team", "team_name", "club", "name"),
    "rating": ("elo", "elo_rating", "rating", "rating_value"),
    "date":   ("date", "rating_date", "updated_at", "created_at"),
}
_ELO_COLUMN_DEFAULTS = {"team": "team", "rating": "elo", "date": "date"}


def _elo_column_names(session) -> dict[str, str]:
    """Resolve the real ``elo_ratings`` column names (schema-drift safe)."""
    try:
        available = {
            column["name"].casefold(): column["name"]
            for column in inspect(session.bind).get_columns("elo_ratings")
        }
    except Exception:  # noqa: BLE001 - fall back to the documented schema
        return dict(_ELO_COLUMN_DEFAULTS)
    resolved: dict[str, str] = {}
    for role, candidates in _ELO_COLUMN_CANDIDATES.items():
        for candidate in candidates:
            if candidate in available:
                resolved[role] = available[candidate]
                break
        resolved.setdefault(role, _ELO_COLUMN_DEFAULTS[role])
    return resolved


def _quote_identifier(name: str) -> str:
    """Quote a column name discovered from the live schema (SQL-injection safe)."""
    return '"' + name.replace('"', '""') + '"'


def load_latest_elo_ratings() -> dict[str, float]:
    """Return ``{normalised team name: latest rating}`` for every rated team.

    No season, league or match-date filter: Elo is a rolling rating, so the
    most recent stored value is always the correct one. Used to warm the API's
    Elo cache at startup and as the name-insensitive fallback of
    :func:`get_latest_elo_before`. Strictly read-only.
    """
    with session_scope() as session:
        cols = _elo_column_names(session)
        q_team = _quote_identifier(cols["team"])
        q_rating = _quote_identifier(cols["rating"])
        q_date = _quote_identifier(cols["date"])
        rows = session.execute(text(
            f"SELECT latest.{q_team}, latest.{q_rating} "
            f"FROM elo_ratings latest "
            f"JOIN (SELECT {q_team}, MAX({q_date}) AS last_date "
            f"      FROM elo_ratings GROUP BY {q_team}) recent "
            f"  ON latest.{q_team} = recent.{q_team} "
            f" AND latest.{q_date} = recent.last_date"
        )).fetchall()

    ratings: dict[str, float] = {}
    for team, value in rows:
        key = normalise_elo_team_name(team)
        if not key or value is None:
            continue
        try:
            rating = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(rating):
            ratings[key] = rating
    return ratings


def load_matches_from_db() -> pd.DataFrame:
    """Load all rows from the matches table, sorted by date."""
    try:
        with session_scope() as session:
            result = session.execute(
                text("SELECT * FROM matches ORDER BY date ASC")
            )
            rows = result.fetchall()
            cols = list(result.keys())
        df = pd.DataFrame(rows, columns=cols)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"[builder] Loaded {len(df)} matches from DB")
        return df
    except Exception as exc:
        print(f"[builder] WARNING: could not load matches: {exc}")
        return pd.DataFrame()


def load_elo_from_db() -> pd.DataFrame:
    """Load all rows from the elo_ratings table, sorted by date."""
    try:
        with session_scope() as session:
            result = session.execute(
                text("SELECT * FROM elo_ratings ORDER BY date ASC")
            )
            rows = result.fetchall()
            cols = list(result.keys())
        df = pd.DataFrame(rows, columns=cols)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"[builder] Loaded {len(df)} elo records from DB")
        return df
    except Exception as exc:
        print(f"[builder] WARNING: could not load elo ratings: {exc}")
        return pd.DataFrame()


def get_pre_match_elo(
    elo_df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
) -> float | None:
    """Return the most recent Elo rating for a team before a given date."""
    if elo_df.empty:
        return None
    mask = (elo_df["team"] == team) & (elo_df["date"] < before_date)
    filtered = elo_df[mask]
    if filtered.empty:
        return None
    return float(filtered.loc[filtered["date"].idxmax(), "elo"])


def compute_team_rolling_form(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    window: int = 5,
    venue: str = "all",
) -> dict:
    """
    Compute rolling form stats for a team before a given date.

    venue = "all"  -> all matches (home and away combined)
    venue = "home" -> only matches where team was at home
    venue = "away" -> only matches where team was away

    Returns a dict of stat_name -> value.
    All values default to 0 if no history exists.
    """
    if venue == "home":
        mask = (df["home_team"] == team) & (df["date"] < before_date)
        recent = df[mask].tail(window)
        gf       = recent["home_goals"].fillna(0)
        ga       = recent["away_goals"].fillna(0)
        res      = recent["result"].map({"H": "W", "D": "D", "A": "L"})
        shots    = recent["home_shots"].fillna(0)
        shots_ot = recent["home_shots_ot"].fillna(0)
        corners  = recent["home_corners"].fillna(0)

    elif venue == "away":
        mask = (df["away_team"] == team) & (df["date"] < before_date)
        recent = df[mask].tail(window)
        gf       = recent["away_goals"].fillna(0)
        ga       = recent["home_goals"].fillna(0)
        res      = recent["result"].map({"A": "W", "D": "D", "H": "L"})
        shots    = recent["away_shots"].fillna(0)
        shots_ot = recent["away_shots_ot"].fillna(0)
        corners  = recent["away_corners"].fillna(0)

    else:
        home_mask = (df["home_team"] == team) & (df["date"] < before_date)
        away_mask = (df["away_team"] == team) & (df["date"] < before_date)
        home_df = df[home_mask].copy()
        away_df = df[away_mask].copy()

        home_df["_gf"]       = home_df["home_goals"].fillna(0)
        home_df["_ga"]       = home_df["away_goals"].fillna(0)
        home_df["_res"]      = home_df["result"].map(
            {"H": "W", "D": "D", "A": "L"})
        home_df["_shots"]    = home_df["home_shots"].fillna(0)
        home_df["_shots_ot"] = home_df["home_shots_ot"].fillna(0)
        home_df["_corners"]  = home_df["home_corners"].fillna(0)

        away_df["_gf"]       = away_df["away_goals"].fillna(0)
        away_df["_ga"]       = away_df["home_goals"].fillna(0)
        away_df["_res"]      = away_df["result"].map(
            {"A": "W", "D": "D", "H": "L"})
        away_df["_shots"]    = away_df["away_shots"].fillna(0)
        away_df["_shots_ot"] = away_df["away_shots_ot"].fillna(0)
        away_df["_corners"]  = away_df["away_corners"].fillna(0)

        combined = (
            pd.concat([home_df, away_df])
            .sort_values("date")
            .tail(window)
        )
        gf       = combined["_gf"]
        ga       = combined["_ga"]
        res      = combined["_res"]
        shots    = combined["_shots"]
        shots_ot = combined["_shots_ot"]
        corners  = combined["_corners"]

    wins   = int((res == "W").sum())
    draws  = int((res == "D").sum())
    losses = int((res == "L").sum())
    points = wins * 3 + draws
    # Reuse each goal reduction within this call (six calls per match).
    gf_total = gf.sum()
    ga_total = ga.sum()

    return {
        "wins":     wins,
        "draws":    draws,
        "losses":   losses,
        "points":   points,
        "gf":       float(gf_total),
        "ga":       float(ga_total),
        "gd":       float(gf_total - ga_total),
        "shots":    float(shots.sum()),
        "shots_ot": float(shots_ot.sum()),
        "corners":  float(corners.sum()),
    }


def compute_h2h(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date: pd.Timestamp,
    window: int = 5,
) -> dict:
    """
    Head-to-head record between home_team (at home) and
    away_team (away) in the last `window` such meetings
    before before_date.
    """
    mask = (
        (df["home_team"] == home_team)
        & (df["away_team"] == away_team)
        & (df["date"] < before_date)
    )
    recent = df[mask].tail(window)

    if recent.empty:
        return {
            "home_wins": 0, "draws": 0, "away_wins": 0,
            "home_gf": 0.0, "away_gf": 0.0,
        }

    return {
        "home_wins": int((recent["result"] == "H").sum()),
        "draws":     int((recent["result"] == "D").sum()),
        "away_wins": int((recent["result"] == "A").sum()),
        "home_gf":   float(recent["home_goals"].fillna(0).sum()),
        "away_gf":   float(recent["away_goals"].fillna(0).sum()),
    }


def get_latest_elo_before(
    team: str,
    before_date: pd.Timestamp,
    raise_on_error: bool = False,
) -> float | None:
    """Latest Elo for ``team`` from the authoritative ``elo_ratings`` table.

    ``before_date`` is accepted for API/backtest compatibility but is NOT a
    filter: Elo is a rolling rating, so the most recently stored value is
    always the right one. The old exact-name ``team = :team AND date < :date``
    query returned no row whenever the stored spelling differed (``Arsenal``
    vs ``Arsenal FC``) or the newest rating was dated on/after the cutoff —
    which silently produced the hard-coded 1500 fallback and an empty cache.

    ``None`` means "no rating known"; the caller then uses the CSV
    reconstruction and finally the 1500 default. A database error also returns
    None unless ``raise_on_error`` is set, which lets the API tell the two
    apart. This function never writes to the database.
    """
    try:
        with session_scope() as session:
            cols = _elo_column_names(session)
            q_team = _quote_identifier(cols["team"])
            q_rating = _quote_identifier(cols["rating"])
            q_date = _quote_identifier(cols["date"])
            row = session.execute(
                text(
                    f"SELECT {q_rating} FROM elo_ratings "
                    f"WHERE {q_team} = :team AND {q_rating} IS NOT NULL "
                    f"ORDER BY {q_date} DESC LIMIT 1"
                ),
                {"team": team},
            ).fetchone()
        value = row[0] if row else None
        if value is None:
            # Exact-name miss (casing, accents, "FC"/"AFC" suffix): resolve
            # against the latest rating of every stored team instead.
            value = load_latest_elo_ratings().get(normalise_elo_team_name(team))
    except Exception as exc:  # noqa: BLE001 - caller falls back
        if raise_on_error:
            raise
        print(f"[builder] WARNING: could not read Elo for {team}: {exc}")
        return None
    if value is None:
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return rating if np.isfinite(rating) else None


def compute_elo_features(
    elo_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date: pd.Timestamp,
    elo_lookup=None,
) -> dict:
    """
    Returns home_elo, away_elo and elo_difference.

    ``elo_lookup(team, before_date)`` optionally resolves a rating from the
    authoritative source (``elo_ratings``); the reconstructed ``elo_df`` is
    used when it returns None. Falls back to 1500.0 (global Elo average) if a
    team has no rating history at all.
    """
    def resolve(team: str) -> float | None:
        if elo_lookup is not None:
            value = elo_lookup(team, before_date)
            if value is not None:
                return value
        return get_pre_match_elo(elo_df, team, before_date)

    home_elo = resolve(home_team)
    away_elo = resolve(away_team)
    home_elo = home_elo if home_elo is not None else 1500.0
    away_elo = away_elo if away_elo is not None else 1500.0

    return {
        "home_elo":       home_elo,
        "away_elo":       away_elo,
        "elo_difference": home_elo - away_elo,
    }


def compute_btts_rolling(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    window: int = 5,
    venue: str = "home",
) -> dict:
    """Rolling BTTS / clean-sheet / scored rates for a team before a date.

    venue="home" -> matches where team played at home
    venue="away" -> matches where team played away

    Returns fractions over the last ``window`` such matches.
    """
    if venue == "home":
        mask = (df["home_team"] == team) & (df["date"] < before_date)
    else:
        mask = (df["away_team"] == team) & (df["date"] < before_date)

    recent = df[mask].tail(window)
    n = len(recent)
    if n == 0:
        return {"btts_rate": 0.0, "cs_rate": 0.0, "scored_rate": 0.0}

    # Support both home_goals (DB) and home_score (spec) column names
    if "home_goals" in recent.columns:
        hg = recent["home_goals"]
    elif "home_score" in recent.columns:
        hg = recent["home_score"]
    else:
        hg = pd.Series(0, index=recent.index)
    if "away_goals" in recent.columns:
        ag = recent["away_goals"]
    elif "away_score" in recent.columns:
        ag = recent["away_score"]
    else:
        ag = pd.Series(0, index=recent.index)

    hg = pd.to_numeric(hg, errors="coerce").fillna(0)
    ag = pd.to_numeric(ag, errors="coerce").fillna(0)

    if venue == "home":
        btts = int(((hg > 0) & (ag > 0)).sum())
        cs = int((ag == 0).sum())
        scored = int((hg > 0).sum())
    else:
        btts = int(((hg > 0) & (ag > 0)).sum())
        cs = int((hg == 0).sum())
        scored = int((ag > 0).sum())

    return {
        "btts_rate": float(btts) / n,
        "cs_rate": float(cs) / n,
        "scored_rate": float(scored) / n,
    }


def compute_over25_rolling(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    window: int = 5,
    venue: str = "home",
) -> dict:
    """Rolling over-2.5 / high-scoring rates for a team before a date.

    venue="home" -> matches where team played at home
    venue="away" -> matches where team played away

    over25_rate       = fraction of those matches with 3+ total goals
    high_scoring_rate = fraction where this team itself scored 2+ goals

    Returns fractions over the last ``window`` such matches.
    """
    if venue == "home":
        mask = (df["home_team"] == team) & (df["date"] < before_date)
    else:
        mask = (df["away_team"] == team) & (df["date"] < before_date)

    recent = df[mask].tail(window)
    n = len(recent)
    if n == 0:
        return {"over25_rate": 0.0, "high_scoring_rate": 0.0}

    # Support both home_goals (DB) and home_score (spec) column names
    if "home_goals" in recent.columns:
        hg = recent["home_goals"]
    elif "home_score" in recent.columns:
        hg = recent["home_score"]
    else:
        hg = pd.Series(0, index=recent.index)
    if "away_goals" in recent.columns:
        ag = recent["away_goals"]
    elif "away_score" in recent.columns:
        ag = recent["away_score"]
    else:
        ag = pd.Series(0, index=recent.index)

    hg = pd.to_numeric(hg, errors="coerce").fillna(0)
    ag = pd.to_numeric(ag, errors="coerce").fillna(0)

    over25 = int(((hg + ag) >= 3).sum())
    if venue == "home":
        high_scoring = int((hg >= 2).sum())
    else:
        high_scoring = int((ag >= 2).sum())

    return {
        "over25_rate": float(over25) / n,
        "high_scoring_rate": float(high_scoring) / n,
    }


# Corner column aliases. The matches table stores home_corners/away_corners
# (see database.Match); the short variants are accepted so a frame built from
# a differently-named extract still resolves instead of silently zeroing out.
HOME_CORNERS_ALIASES = ("home_corners", "home_crs")
AWAY_CORNERS_ALIASES = ("away_corners", "away_crs")

# A match is "Over 9.5 corners" when it has at least this many corners total.
OVER95_CORNERS_THRESHOLD = 10


def _corner_values(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
) -> pd.Series:
    """Numeric corner column for one side of a slice of the matches table.

    Missing column or unparseable values become 0.0: a match without corners
    must never drop a training row.
    """
    for name in aliases:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _corner_value(row: pd.Series, aliases: tuple[str, ...]) -> float | None:
    """Single match's corner count for one side, or None when unavailable."""
    for name in aliases:
        if name in row.index:
            value = row.get(name)
            if pd.isna(value):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def compute_corners_rolling(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    window: int = 5,
    venue: str = "home",
) -> dict:
    """Rolling corner rates for a team before a date.

    venue="home" -> matches where team played at home
    venue="away" -> matches where team played away

    corners_for_pg     = corners taken per game over the last ``window`` games
    corners_against_pg = corners conceded per game over those games
    over95_rate        = fraction of those matches with >= 10 total corners

    Both rates are per game, not sums, so a team with less than ``window``
    matches of history is still on the same scale as a full-history team.
    Every value is 0.0 when there is no prior match.
    """
    if venue == "home":
        mask = (df["home_team"] == team) & (df["date"] < before_date)
    else:
        mask = (df["away_team"] == team) & (df["date"] < before_date)

    recent = df[mask].tail(window)
    n = len(recent)
    if n == 0:
        return {
            "corners_for_pg": 0.0,
            "corners_against_pg": 0.0,
            "over95_rate": 0.0,
        }

    home_c = _corner_values(recent, HOME_CORNERS_ALIASES)
    away_c = _corner_values(recent, AWAY_CORNERS_ALIASES)

    # Corners conceded is the OTHER side of the same match: at home that is
    # away_corners, away from home it is home_corners.
    if venue == "home":
        for_corners, against_corners = home_c, away_c
    else:
        for_corners, against_corners = away_c, home_c

    over95 = int(((home_c + away_c) >= OVER95_CORNERS_THRESHOLD).sum())

    return {
        "corners_for_pg": float(for_corners.sum()) / n,
        "corners_against_pg": float(against_corners.sum()) / n,
        "over95_rate": float(over95) / n,
    }


def over95_corners_target(row: pd.Series) -> int | None:
    """1 when the match had 10+ corners in total, 0 otherwise, None if a side
    has no recorded corners. Same null handling as target_btts/target_over25.
    """
    home_c = _corner_value(row, HOME_CORNERS_ALIASES)
    away_c = _corner_value(row, AWAY_CORNERS_ALIASES)
    if home_c is None or away_c is None:
        return None
    return 1 if (home_c + away_c) >= OVER95_CORNERS_THRESHOLD else 0


#: Number of match-derived features :func:`compute_match_features` returns
#: (the 60 shared with FEATURE_COLS plus the 16 specialist-classifier rates).
MATCH_FEATURE_COUNT = 76


def compute_match_features(
    df: pd.DataFrame,
    elo_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date: pd.Timestamp,
    elo_lookup=None,
) -> dict[str, float]:
    """Every rolling-form / Elo / H2H feature for one fixture.

    THE definition of those columns. ``build_features`` calls it once per
    training row and ``features/inference.py`` calls it once per live fixture,
    so training and inference share a single implementation and cannot drift.

    Every window and H2H lookup uses matches strictly before ``before_date``.
    Missing history is 0.0 (never NaN), matching what the models trained on.
    """
    hf5  = compute_team_rolling_form(
        df, home_team, before_date, window=5,  venue="all")
    hf10 = compute_team_rolling_form(
        df, home_team, before_date, window=10, venue="all")
    hh5  = compute_team_rolling_form(
        df, home_team, before_date, window=5,  venue="home")
    af5  = compute_team_rolling_form(
        df, away_team, before_date, window=5,  venue="all")
    af10 = compute_team_rolling_form(
        df, away_team, before_date, window=10, venue="all")
    aa5  = compute_team_rolling_form(
        df, away_team, before_date, window=5,  venue="away")
    h2h  = compute_h2h(df, home_team, away_team, before_date)
    elo  = compute_elo_features(
        elo_df, home_team, away_team, before_date, elo_lookup=elo_lookup)
    home_btts = compute_btts_rolling(
        df, home_team, before_date, window=5, venue="home")
    away_btts = compute_btts_rolling(
        df, away_team, before_date, window=5, venue="away")
    home_o25 = compute_over25_rolling(
        df, home_team, before_date, window=5, venue="home")
    away_o25 = compute_over25_rolling(
        df, away_team, before_date, window=5, venue="away")
    home_c5 = compute_corners_rolling(
        df, home_team, before_date, window=5, venue="home")
    away_c5 = compute_corners_rolling(
        df, away_team, before_date, window=5, venue="away")

    feat = {
        # Home form 5
        "home_form_wins_5":      hf5["wins"],
        "home_form_draws_5":     hf5["draws"],
        "home_form_losses_5":    hf5["losses"],
        "home_form_points_5":    hf5["points"],
        "home_form_gf_5":        hf5["gf"],
        "home_form_ga_5":        hf5["ga"],
        "home_form_gd_5":        hf5["gd"],
        "home_form_shots_5":     hf5["shots"],
        "home_form_shots_ot_5":  hf5["shots_ot"],
        "home_form_corners_5":   hf5["corners"],

        # Home form 10
        "home_form_wins_10":     hf10["wins"],
        "home_form_draws_10":    hf10["draws"],
        "home_form_losses_10":   hf10["losses"],
        "home_form_points_10":   hf10["points"],
        "home_form_gf_10":       hf10["gf"],
        "home_form_ga_10":       hf10["ga"],
        "home_form_gd_10":       hf10["gd"],

        # Home at home 5
        "home_home_wins_5":      hh5["wins"],
        "home_home_draws_5":     hh5["draws"],
        "home_home_losses_5":    hh5["losses"],
        "home_home_points_5":    hh5["points"],
        "home_home_gf_5":        hh5["gf"],
        "home_home_ga_5":        hh5["ga"],
        "home_home_gd_5":        hh5["gd"],
        "home_home_shots_5":     hh5["shots"],
        "home_home_corners_5":   hh5["corners"],

        # Away form 5
        "away_form_wins_5":      af5["wins"],
        "away_form_draws_5":     af5["draws"],
        "away_form_losses_5":    af5["losses"],
        "away_form_points_5":    af5["points"],
        "away_form_gf_5":        af5["gf"],
        "away_form_ga_5":        af5["ga"],
        "away_form_gd_5":        af5["gd"],
        "away_form_shots_5":     af5["shots"],
        "away_form_shots_ot_5":  af5["shots_ot"],
        "away_form_corners_5":   af5["corners"],

        # Away form 10
        "away_form_wins_10":     af10["wins"],
        "away_form_draws_10":    af10["draws"],
        "away_form_losses_10":   af10["losses"],
        "away_form_points_10":   af10["points"],
        "away_form_gf_10":       af10["gf"],
        "away_form_ga_10":       af10["ga"],
        "away_form_gd_10":       af10["gd"],

        # Away at away 5
        "away_away_wins_5":      aa5["wins"],
        "away_away_draws_5":     aa5["draws"],
        "away_away_losses_5":    aa5["losses"],
        "away_away_points_5":    aa5["points"],
        "away_away_gf_5":        aa5["gf"],
        "away_away_ga_5":        aa5["ga"],
        "away_away_gd_5":        aa5["gd"],
        "away_away_shots_5":     aa5["shots"],
        "away_away_corners_5":   aa5["corners"],

        # Elo
        "home_elo":              elo["home_elo"],
        "away_elo":              elo["away_elo"],
        "elo_difference":        elo["elo_difference"],

        # BTTS-specific rolling rates
        "home_btts_rate_5":      home_btts["btts_rate"],
        "away_btts_rate_5":      away_btts["btts_rate"],
        "home_cs_rate_5":        home_btts["cs_rate"],
        "away_cs_rate_5":        away_btts["cs_rate"],
        "home_scored_rate_5":    home_btts["scored_rate"],
        "away_scored_rate_5":    away_btts["scored_rate"],

        # Over-2.5-specific rolling rates
        "home_over25_rate_5":        home_o25["over25_rate"],
        "away_over25_rate_5":        away_o25["over25_rate"],
        "home_high_scoring_rate_5":  home_o25["high_scoring_rate"],
        "away_high_scoring_rate_5":  away_o25["high_scoring_rate"],

        # Corner-specific rolling rates (venue-aware), for the Over 9.5
        # corners classifier only — deliberately NOT part of FEATURE_COLS.
        "home_corners_for_pg_5":      home_c5["corners_for_pg"],
        "away_corners_for_pg_5":      away_c5["corners_for_pg"],
        "home_corners_against_pg_5":  home_c5["corners_against_pg"],
        "away_corners_against_pg_5":  away_c5["corners_against_pg"],
        "home_over95_corners_rate_5": home_c5["over95_rate"],
        "away_over95_corners_rate_5": away_c5["over95_rate"],

        # H2H
        "h2h_home_wins_5":       h2h["home_wins"],
        "h2h_draws_5":           h2h["draws"],
        "h2h_away_wins_5":       h2h["away_wins"],
        "h2h_home_gf_5":         h2h["home_gf"],
        "h2h_away_gf_5":         h2h["away_gf"]
    }

    # Model inputs are never NaN: anything the history cannot provide is 0.0.
    # (Targets keep their NaN — the trainer filters on them, never imputes.)
    return {
        name: (0.0 if pd.isna(value) else value) for name, value in feat.items()
    }


def build_feature_row(
    df: pd.DataFrame,
    elo_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date: pd.Timestamp,
    row: pd.Series | None = None,
) -> dict:
    """One complete training row: features + targets + meta + raw history.

    ``row`` is the match itself. Without it only the model inputs are returned,
    which is exactly what inference needs for a fixture not yet played.
    """
    feat = compute_match_features(df, elo_df, home_team, away_team, before_date)
    if row is None:
        return feat

    feat.update({
        # Targets (never filled with 0)
        "target_result":       row.get("result"),
        "target_home_goals":   row.get("home_goals"),
        "target_away_goals":   row.get("away_goals"),
        "target_home_corners": row.get("home_corners"),
        "target_away_corners": row.get("away_corners"),
        "target_btts": (
            None if (pd.isna(row.get("home_goals", row.get("home_score")))
                     or pd.isna(row.get("away_goals", row.get("away_score"))))
            else (1 if (
                (row.get("home_goals", row.get("home_score", 0)) or 0) > 0
                and (row.get("away_goals", row.get("away_score", 0)) or 0) > 0
            ) else 0)
        ),
        "target_over25": (
            None if (pd.isna(row.get("home_goals", row.get("home_score")))
                     or pd.isna(row.get("away_goals", row.get("away_score"))))
            else (1 if (
                (row.get("home_goals", row.get("home_score", 0)) or 0)
                + (row.get("away_goals", row.get("away_score", 0)) or 0) >= 3
            ) else 0)
        ),
        "target_over95_corners": over95_corners_target(row),

        # Meta (kept for chronological splitting later)
        "date":      before_date,
        "league":    row.get("league"),
        "season":    row.get("season"),
        "home_team": home_team,
        "away_team": away_team,

        # Raw match data, carried as meta so the saved features CSV holds the
        # history inference has to replay. Never a model input.
        **{column: row.get(column) for column in MATCH_HISTORY_COLUMNS},
    })
    return feat


def build_features(
    df: pd.DataFrame,
    elo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One feature row per match in ``df``.

    ``df`` is the full matches DataFrame, ``elo_df`` the elo_ratings history.
    All rolling windows and H2H lookups use only matches dated strictly before
    the row's own date (no leakage).

    The raw match columns in :data:`MATCH_HISTORY_COLUMNS` are written with
    every row: the prediction service replays the same rolling windows from
    them (see ``features/inference.py``), so the features CSV doubles as the
    inference-time match history.
    """
    if elo_df is None:
        elo_df = pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with no date or no result — cannot be trained on
    df = df.dropna(subset=["date", "result"])
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"[builder] Processing row {i}/{total}...")
        rows.append(
            build_feature_row(
                df,
                elo_df,
                row["home_team"],
                row["away_team"],
                row["date"],
                row=row,
            )
        )

    out = pd.DataFrame(rows)

    # Fill NaN only in feature columns — never in targets or meta
    feature_cols = [
        c for c in out.columns
        if c.startswith(("home_form_", "away_form_",
                         "home_home_", "away_away_",
                         "elo_", "h2h_",
                         "home_elo", "away_elo"))
    ]
    out[feature_cols] = out[feature_cols].fillna(0)
    btts_rate_cols = [
        "home_btts_rate_5", "away_btts_rate_5",
        "home_cs_rate_5", "away_cs_rate_5",
        "home_scored_rate_5", "away_scored_rate_5",
    ]
    for c in btts_rate_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    over25_rate_cols = [
        "home_over25_rate_5", "away_over25_rate_5",
        "home_high_scoring_rate_5", "away_high_scoring_rate_5",
    ]
    for c in over25_rate_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    corner_rate_cols = [
        "home_corners_for_pg_5", "away_corners_for_pg_5",
        "home_corners_against_pg_5", "away_corners_against_pg_5",
        "home_over95_corners_rate_5", "away_over95_corners_rate_5",
    ]
    for c in corner_rate_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    print(
        f"[builder] Built {len(out)} feature rows, "
        f"{len(feature_cols)} feature columns"
    )
    return out


def build_and_save_features(
    output_path: str = "data/processed/features.csv",
) -> pd.DataFrame:
    """
    Full pipeline: load from DB -> build features ->
    encode league/season -> save CSV and encoders.

    Run this once after data collection completes.
    Encoders are saved so prediction time reuses the
    exact same integer mappings.
    """
    matches_df = load_matches_from_db()
    elo_df     = load_elo_from_db()

    if matches_df.empty:
        print(
            "[builder] ERROR: no match data found. "
            "Run collector.py first."
        )
        return pd.DataFrame()

    features_df = build_features(matches_df, elo_df)

    if features_df.empty:
        print("[builder] ERROR: feature build produced no rows.")
        return pd.DataFrame()

    # Encode league and season with consistent integer mapping
    le_league = LabelEncoder()
    le_season = LabelEncoder()
    features_df["league_encoded"] = le_league.fit_transform(
        features_df["league"].astype(str)
    )
    features_df["season_encoded"] = le_season.fit_transform(
        features_df["season"].astype(str)
    )

    # Save encoders for reuse at prediction time
    enc_dir = Path(__file__).resolve().parent.parent / "data"
    enc_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le_league, enc_dir / "league_encoder.joblib")
    joblib.dump(le_season, enc_dir / "season_encoder.joblib")
    print(f"[builder] Encoders saved to {enc_dir}")

    # Save features CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)

    date_min = features_df["date"].min()
    date_max = features_df["date"].max()
    print(
        f"[builder] Saved {len(features_df)} rows x "
        f"{len(features_df.columns)} cols -> {out_path}"
    )
    print(f"[builder] Date range: {date_min} -> {date_max}")

    return features_df


if __name__ == "__main__":
    # GitHub Actions runs this as `python features/builder.py`. Exiting 0 on an
    # empty build used to let the workflow continue into the trainer (which then
    # found no CSV) and, worse, into the "reload models" step. Fail loudly here.
    built = build_and_save_features()
    if built.empty:
        print("[builder] ERROR: no features were built — exiting 1")
        sys.exit(1)
    print(f"[builder] OK: {len(built)} rows written")
