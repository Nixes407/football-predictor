"""
Feature engineering pipeline for the Football Predictor.

Built in stages. Extended in Phase 3 part 2.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import session_scope


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
