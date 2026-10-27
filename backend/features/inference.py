"""Inference-time feature assembly, built on the *training* implementation.

``routers/predict.py`` has to score a fixture that has not been played yet, so
it cannot read a pre-computed feature row: it must rebuild the same rolling
windows, venue splits, Elo values and H2H record that ``features/builder.py``
produced for every training row.

This module is the bridge. It

1. reconstructs the match history from the saved features CSV
   (:func:`match_history_from_features`) — the API process has that frame and
   nothing else;
2. reconstructs the Elo history (:func:`elo_history_from_features`) as a
   fallback for when the database cannot answer;
3. calls ``builder.compute_match_features`` — the very function the training
   pipeline calls — with the fixture date as the cutoff.

There is deliberately no second implementation of a rolling window anywhere in
the prediction path. If a formula changes, it changes for both paths at once.

All functions are pure and take their inputs as arguments (no DB, no network),
so ``tests/test_feature_parity.py`` can exercise them with fixtures.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:  # imported as ``features.inference``
    from features import builder
except ImportError:  # pragma: no cover - run from inside backend/features
    import builder  # type: ignore[no-redef]


# --------------------------------------------------------------------------- #
# Dates and seasons
# --------------------------------------------------------------------------- #


def current_season_label(today: date | None = None) -> str:
    """Return the European season label (``"2026-2027"``) for ``today``.

    The season starts in July: a date in July or later belongs to the season
    that started that year, otherwise to the one that started last July. Same
    rule as ``data.collector.current_season`` and ``data.fixtures.derive_season``
    — duplicated here so the prediction service does not pull in the scraping
    dependencies just to name a season.
    """
    current = today or datetime.now(timezone.utc).date()
    start_year = current.year if current.month >= 7 else current.year - 1
    return f"{start_year}-{start_year + 1}"


def inference_cutoff_date(now: Any = None) -> pd.Timestamp:
    """Midnight of the day a live fixture is predicted for (naive, UTC).

    Matches carry date-only precision, and the builder treats the *start* of a
    match day as the cutoff (``date < match_date``), so a fixture today may
    only use matches from earlier days. Using the same normalization at
    inference keeps both paths free of same-day leakage.
    """
    if now is None:
        stamp = pd.Timestamp.now(tz="UTC")
    else:
        stamp = pd.Timestamp(now)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)
    return stamp.normalize()


def _as_naive_date(value: Any) -> pd.Timestamp:
    """Normalize a date/timestamp to a naive midnight ``pd.Timestamp``."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_convert(None)
    return stamp.normalize()


# --------------------------------------------------------------------------- #
# Rebuilding the training inputs from the saved features frame
# --------------------------------------------------------------------------- #


def empty_match_history() -> pd.DataFrame:
    """A zero-row match history with every column the helpers expect.

    An empty ``DataFrame()`` has no columns, so the rolling helpers would raise
    ``KeyError`` instead of returning zeros. A fixture with no history at all
    (a newly promoted club, a first-ever match) must still be scoreable.
    """
    return pd.DataFrame(
        columns=["date", "home_team", "away_team",
                 *builder.MATCH_HISTORY_COLUMNS]
    )


def empty_elo_history() -> pd.DataFrame:
    """A zero-row Elo history; every team then falls back to 1500."""
    return pd.DataFrame(columns=["team", "date", "elo"])


def _first_present(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    """First column of ``names`` present in ``frame`` (else ``None``)."""
    for name in names:
        if name in frame.columns:
            return frame[name]
    return None


def match_history_from_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """The played-match history a features CSV implies.

    Returns the columns ``builder.compute_match_features`` reads
    (``date``/``home_team``/``away_team`` + ``builder.MATCH_HISTORY_COLUMNS``).
    ``build_features`` writes the raw match columns into the CSV for exactly
    this purpose; a CSV written before that change still works because goals,
    result and corners are mapped back from the ``target_*`` columns.

    Shot columns cannot be recovered from an older CSV; they stay NaN, which
    the rolling helpers treat as 0 — the same value a match without recorded
    shots contributes during training.
    """
    if features_df is None or len(features_df) == 0:
        return empty_match_history()

    missing = [c for c in ("date", "home_team", "away_team")
               if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"features frame cannot serve as match history, missing: {missing}"
        )

    history = pd.DataFrame({
        "date": pd.to_datetime(features_df["date"], errors="coerce"),
        "home_team": features_df["home_team"].astype(object),
        "away_team": features_df["away_team"].astype(object),
    })

    for column in builder.MATCH_HISTORY_COLUMNS:
        source = _first_present(
            features_df, builder.MATCH_HISTORY_ALIASES.get(column, (column,))
        )
        if column == "result":
            history[column] = source.astype(object) if source is not None else None
        else:
            history[column] = (
                pd.to_numeric(source, errors="coerce")
                if source is not None
                else np.nan
            )

    history = history.loc[history["date"].notna()]
    history = history.sort_values("date", kind="stable").reset_index(drop=True)
    return history


def elo_history_from_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct an ``elo_ratings``-shaped frame from the features CSV.

    **Fallback only.** Every row stores the home team's and the away team's
    *pre-match* rating (``home_elo`` / ``away_elo``). The table the builder
    reads also holds each team's post-match rating, which is what a later
    fixture needs; the CSV therefore lags by one match per team. The API
    prefers :func:`builder.get_latest_elo_before` (the authoritative table) and
    uses this frame only when the database cannot answer or has no rating.

    Returns ``team``/``date``/``elo`` columns sorted by date.
    """
    empty = empty_elo_history()
    if features_df is None or len(features_df) == 0:
        return empty
    if "date" not in features_df.columns:
        return empty

    frames = []
    for role, team_column, elo_column in (
        ("home", "home_team", "home_elo"),
        ("away", "away_team", "away_elo"),
    ):
        if (team_column not in features_df.columns
                or elo_column not in features_df.columns):
            continue
        frames.append(pd.DataFrame({
            "team": features_df[team_column].astype(object),
            "date": pd.to_datetime(features_df["date"], errors="coerce"),
            "elo": pd.to_numeric(features_df[elo_column], errors="coerce"),
        }))
    if not frames:
        return empty

    elo = pd.concat(frames, ignore_index=True)
    elo = elo.dropna(subset=["team", "date", "elo"])
    elo = elo.sort_values("date", kind="stable")
    return elo.drop_duplicates(
        subset=["team", "date"], keep="last"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Categorical encoding
# --------------------------------------------------------------------------- #


def encode_label(encoder, value: Any, fallback: int = 0) -> int:
    """Encode ``value`` with a fitted ``LabelEncoder``, never raising.

    The encoder object is produced by the builder (``league_encoder.joblib`` /
    ``season_encoder.joblib``) and shipped with the model artifacts, so a known
    label always yields the exact integer the model was trained on. An unseen
    label is reported and mapped to ``fallback`` (0), because predicting with a
    wrong-but-known category is preferable to failing the request.
    """
    if encoder is None:
        return fallback

    text = "" if value is None else str(value)
    try:
        return int(encoder.transform([text])[0])
    except (ValueError, TypeError):
        pass

    # Providers spell labels inconsistently ("TUR-Super Lig" vs "TUR-Süper
    # Lig"); retry case/whitespace-insensitively before giving up.
    classes = list(getattr(encoder, "classes_", []))
    normalised = {str(c).strip().casefold(): c for c in classes}
    match = normalised.get(text.strip().casefold())
    if match is not None:
        try:
            return int(encoder.transform([match])[0])
        except (ValueError, TypeError):
            pass

    print(
        f"[inference] WARNING: unseen label {text!r}; known labels: {classes}. "
        f"Using fallback={fallback}"
    )
    return fallback


# --------------------------------------------------------------------------- #
# One fixture -> one feature row
# --------------------------------------------------------------------------- #


def _numeric(value: Any, default: float = 0.0) -> float:
    """``float(value)`` or ``default`` for None/NaN/junk."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def build_inference_features(
    matches_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date: Any = None,
    league: str | None = None,
    season: str | None = None,
    elo_lookup=None,
) -> dict[str, float]:
    """Every model input for one unplayed fixture, computed like training.

    ``before_date`` defaults to today (see :func:`inference_cutoff_date`): only
    matches from strictly earlier days are used. Pass it explicitly to
    reproduce a historical row (the parity test does).

    ``league``/``season`` are accepted for API compatibility but are not model
    inputs; the categorical encoder is applied by the router.

    ``elo_lookup(team, before_date)`` resolves a rating from the authoritative
    ``elo_ratings`` table. The caller should pass one whenever it can: the
    features CSV only stores pre-match ratings, so the reconstruction alone
    would use each team's rating from one match earlier.
    """
    cutoff = (
        inference_cutoff_date()
        if before_date is None
        else _as_naive_date(before_date)
    )

    # Empty histories must still produce zeros (and the 1500 Elo fallback)
    # rather than raising, so a fixture between teams with no recorded match
    # is scoreable. The builder applies the same defaults.
    if matches_df is None or len(matches_df) == 0:
        matches_df = empty_match_history()
    if elo_df is None or len(elo_df) == 0:
        elo_df = empty_elo_history()

    features = builder.compute_match_features(
        matches_df, elo_df, home_team, away_team, cutoff,
        elo_lookup=elo_lookup,
    )
    return {name: _numeric(value) for name, value in features.items()}
