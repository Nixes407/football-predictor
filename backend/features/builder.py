"""
Feature engineering pipeline for the Football Predictor.

Built in stages. Extended in Phase 3 part 2.
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import session_scope
from database import SessionLocal, FBrefTeamStats


# Key columns that identify one team's season in fbref_team_stats.
FBREF_KEY_COLUMNS = ["team", "league", "season"]

# Stat columns read from fbref_team_stats (keys excluded).
FBREF_STAT_COLUMNS = [
    "games",
    "shots", "shots_on_target", "shot_accuracy",
    "goals_per_shot", "goals_per_sot",
    "yellow_cards", "red_cards", "fouls_committed", "fouls_drawn",
    "offsides", "crosses", "interceptions", "tackles_won",
    "pens_won", "pens_conceded",
    "sota", "goals_against", "saves", "save_pct",
    "clean_sheets", "pk_faced", "pk_saved",
]

# Collection metadata is loaded for the availability cutoff, never as a feature.
FBREF_COLUMNS = FBREF_KEY_COLUMNS + FBREF_STAT_COLUMNS + ["updated_at"]

# Canonical team names keyed by known variants. Team names in the matches
# table (Football-Data.co.uk) differ from those in fbref_team_stats (FBref),
# so both sides are mapped to this canonical form before joining.
TEAM_NAME_MAP: dict[str, str] = {
    # English
    "Man United":          "Manchester Utd",
    "Manchester United":   "Manchester Utd",
    "Man Utd":             "Manchester Utd",
    "Man City":            "Manchester City",
    "Wolves":              "Wolverhampton Wanderers",
    "Spurs":               "Tottenham Hotspur",
    "Tottenham":           "Tottenham Hotspur",
    "Brighton":            "Brighton & Hove Albion",
    "Newcastle":           "Newcastle Utd",
    "Newcastle United":    "Newcastle Utd",
    "West Ham":            "West Ham United",
    "Nott'm Forest":       "Nottingham Forest",
    "Nottm Forest":        "Nottingham Forest",
    "Leicester":           "Leicester City",
    "Sheffield Utd":       "Sheffield United",
    "Leeds":               "Leeds United",
    "Brentford":           "Brentford",
    "Fulham":              "Fulham",
    "Luton":               "Luton Town",
    "Burnley":             "Burnley",
    "Bournemouth":         "Bournemouth",
    "Ipswich":             "Ipswich Town",
    "Southampton":         "Southampton",
    "Everton":             "Everton",
    "Chelsea":             "Chelsea",
    "Arsenal":             "Arsenal",
    "Liverpool":           "Liverpool",
    "Aston Villa":         "Aston Villa",
    "Crystal Palace":      "Crystal Palace",
    # Spanish
    "Almeria":             "Almería",
    "Atletico Madrid":     "Atlético Madrid",
    "Betis":               "Real Betis",
    "Cadiz":               "Cádiz",
    "Espanol":             "Espanyol",
    "Leganes":             "Leganés",
    # German
    "Greuther Furth":      "Greuther Fürth",
    "Koln":                "Köln",
    "FC Koln":             "Köln",
    "Dusseldorf":          "Düsseldorf",
    "Nurnberg":            "Nürnberg",
    "Hannover":            "Hannover 96",
    "Bochum":              "VfL Bochum",
    "M'gladbach":          "Mönchengladbach",
    "Mainz":               "Mainz 05",
    # Italian
    "Inter":               "Internazionale",
    "Inter Milan":         "Internazionale",
    "AC Milan":            "Milan",
    "Hellas Verona":       "Verona",
    # French
    "Paris SG":            "Paris S-G",
    "PSG":                 "Paris S-G",
    "St Etienne":          "Saint-Étienne",
    "Saint-Etienne":       "Saint-Étienne",
    "Marseille":           "Marseille",
}


def normalise_team_name(name: str) -> str:
    """Map known team name variants to a canonical form."""
    if not isinstance(name, str):
        return name
    return TEAM_NAME_MAP.get(name, name)


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


def load_fbref_stats() -> pd.DataFrame:
    """
    Load all rows from the fbref_team_stats table.

    Returns a DataFrame with the columns in FBREF_COLUMNS. If the table is
    empty or cannot be read, a warning is printed and an empty DataFrame with
    those columns is returned.
    """
    empty = pd.DataFrame(columns=FBREF_COLUMNS)
    db = None
    try:
        db = SessionLocal()
        rows = db.query(FBrefTeamStats).all()
        records = [
            {col: getattr(row, col) for col in FBREF_COLUMNS}
            for row in rows
        ]
    except Exception as exc:
        print(f"[builder] WARNING: could not load fbref stats: {exc}")
        return empty
    finally:
        if db is not None:
            db.close()

    if not records:
        print("[builder] WARNING: fbref_team_stats table is empty")
        return empty

    df = pd.DataFrame(records, columns=FBREF_COLUMNS)
    print(f"[builder] Loaded {len(df)} FBref team-season rows from DB")
    return df


def join_fbref_features(
    features_df: pd.DataFrame,
    fbref_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join only FBref values known by each match date.

    ``updated_at`` is the collector's ISO UTC collection timestamp, not a
    model input. Missing/invalid/future timestamps fail closed to zero stats;
    never drop a training row. Matches have date-only precision, so midnight
    UTC is the conservative cutoff (not the end of the match day).

    The source table overwrites one row per team/league/season. It cannot
    reconstruct earlier snapshots: when the stored snapshot is too new, its
    stats must be excluded, not backdated to a guessed season-end date.
    """
    result = features_df.copy()
    all_stat_cols = [
        f"{role}_fbref_{col}"
        for role in ("home", "away")
        for col in FBREF_STAT_COLUMNS
    ]
    # Always emit the trainer's full schema, even with no available FBref data.
    result = result.drop(columns=all_stat_cols, errors="ignore")
    if fbref_df.empty:
        return result.assign(**{col: 0.0 for col in all_stat_cols})

    stats = fbref_df.copy()
    stats["team"] = stats["team"].apply(normalise_team_name)
    # Parse individually: legacy ISO strings may mix offsets and precision.
    # utc=True makes naive legacy values and timezone-aware values comparable.
    timestamps = stats.get(
        "updated_at", pd.Series(index=stats.index, dtype=object)
    )
    stats["updated_at"] = pd.to_datetime(
        timestamps.map(lambda value: pd.to_datetime(value, errors="coerce", utc=True)),
        errors="coerce", utc=True,
    )
    stats = stats.loc[stats["updated_at"].notna()].copy()
    # Canonicalise BEFORE deduplication: provider aliases can collapse to one
    # key. Keep the latest known snapshot, rather than arbitrary input order.
    stats = stats.sort_values("updated_at", kind="stable").drop_duplicates(
        subset=FBREF_KEY_COLUMNS, keep="last"
    )
    stats[FBREF_STAT_COLUMNS] = stats.reindex(columns=FBREF_STAT_COLUMNS).apply(
        pd.to_numeric, errors="coerce"
    )
    match_dates = pd.to_datetime(
        features_df["date"], errors="coerce", utc=True
    ).reset_index(drop=True)

    for role in ("home", "away"):
        team_col = f"{role}_team"
        collected_col = f"_{role}_fbref_updated_at"
        stat_cols = [f"{role}_fbref_{col}" for col in FBREF_STAT_COLUMNS]
        result[team_col] = result[team_col].apply(normalise_team_name)
        team_stats = stats[FBREF_COLUMNS].rename(columns={
            "team": team_col,
            "updated_at": collected_col,
            **{col: f"{role}_fbref_{col}" for col in FBREF_STAT_COLUMNS},
        })
        result = result.merge(
            team_stats, on=[team_col, "league", "season"], how="left",
            validate="many_to_one",
        )
        # A WHERE on the whole left-joined frame would drop historical matches.
        # Mask only this side's stats: updated_at <= match_date is required.
        available = (
            result[collected_col].notna()
            & match_dates.notna()
            & (result[collected_col] <= match_dates)
        )
        result.loc[~available, stat_cols] = 0.0
        result[stat_cols] = result[stat_cols].fillna(0.0)
        result = result.drop(columns=[collected_col])

    if len(result) != len(features_df):
        raise ValueError(f"join changed row count {len(features_df)} -> {len(result)}")
    result.index = features_df.index
    return result


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

    return {
        "wins":     wins,
        "draws":    draws,
        "losses":   losses,
        "points":   points,
        "gf":       float(gf.sum()),
        "ga":       float(ga.sum()),
        "gd":       float(gf.sum() - ga.sum()),
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


def compute_elo_features(
    elo_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date: pd.Timestamp,
) -> dict:
    """
    Returns home_elo, away_elo and elo_difference.
    Falls back to 1500.0 (global Elo average) if a
    team has no rating history.
    """
    home_elo = get_pre_match_elo(elo_df, home_team, before_date)
    away_elo = get_pre_match_elo(elo_df, away_team, before_date)
    home_elo = home_elo if home_elo is not None else 1500.0
    away_elo = away_elo if away_elo is not None else 1500.0

    return {
        "home_elo":       home_elo,
        "away_elo":       away_elo,
        "elo_difference": home_elo - away_elo,
    }


def build_features(
    df: pd.DataFrame,
    elo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build one feature row per match in df.

    df     = full matches DataFrame (all columns from DB).
    elo_df = full elo_ratings DataFrame (optional).

    STRICT RULE: all rolling windows and h2h lookups
    use only matches with date strictly less than the
    current row date. No data leakage of any kind.
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

        date      = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        hf5  = compute_team_rolling_form(
            df, home_team, date, window=5,  venue="all")
        hf10 = compute_team_rolling_form(
            df, home_team, date, window=10, venue="all")
        hh5  = compute_team_rolling_form(
            df, home_team, date, window=5,  venue="home")
        af5  = compute_team_rolling_form(
            df, away_team, date, window=5,  venue="all")
        af10 = compute_team_rolling_form(
            df, away_team, date, window=10, venue="all")
        aa5  = compute_team_rolling_form(
            df, away_team, date, window=5,  venue="away")
        h2h  = compute_h2h(df, home_team, away_team, date)
        elo  = compute_elo_features(
            elo_df, home_team, away_team, date)

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

            # H2H
            "h2h_home_wins_5":       h2h["home_wins"],
            "h2h_draws_5":           h2h["draws"],
            "h2h_away_wins_5":       h2h["away_wins"],
            "h2h_home_gf_5":         h2h["home_gf"],
            "h2h_away_gf_5":         h2h["away_gf"],

            # Targets (never filled with 0)
            "target_result":         row.get("result"),
            "target_home_goals":     row.get("home_goals"),
            "target_away_goals":     row.get("away_goals"),
            "target_home_corners":   row.get("home_corners"),
            "target_away_corners":   row.get("away_corners"),

            # Meta (kept for chronological splitting later)
            "date":      date,
            "league":    row.get("league"),
            "season":    row.get("season"),
            "home_team": home_team,
            "away_team": away_team,
        }

        rows.append(feat)

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

    # Preserve every training row, but only expose a team's FBref snapshot
    # when its collection timestamp is on or before that row's match date.
    fbref_df = load_fbref_stats()
    try:
        features_df = join_fbref_features(features_df, fbref_df)
        print(f"[builder] Added {2 * len(FBREF_STAT_COLUMNS)} date-filtered "
              "FBref features (home + away)")
    except Exception as exc:
        print(f"[builder] WARNING: FBref join failed, using zero FBref "
              f"features: {exc}")
        features_df = join_fbref_features(features_df, pd.DataFrame())

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
    build_and_save_features()
