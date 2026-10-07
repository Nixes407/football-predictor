"""
Prediction router for the Football Predictor API.

Loads the trained artifacts (1X2 classifier, Poisson count models and the
label encoders) plus the features CSV at import time, then exposes endpoints
for league/team lookups and full match predictions.

A missing artifact prints a warning and leaves the corresponding global as
``None``; the API keeps booting and ``/predict/match`` answers 503 until the
pipeline has been run.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy.stats import poisson

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Module-level artifact variables (all start as None)
_result_model  = None
_home_goals    = None
_away_goals    = None
_home_corners  = None
_away_corners  = None
_le_league     = None
_le_season     = None
_features_df   = None

FEATURE_COLS = [
    # Home form 5
    "home_form_wins_5", "home_form_draws_5",
    "home_form_losses_5", "home_form_points_5",
    "home_form_gf_5", "home_form_ga_5",
    "home_form_gd_5", "home_form_shots_5",
    "home_form_shots_ot_5", "home_form_corners_5",
    # Home form 10
    "home_form_wins_10", "home_form_draws_10",
    "home_form_losses_10", "home_form_points_10",
    "home_form_gf_10", "home_form_ga_10",
    "home_form_gd_10",
    # Home at home 5
    "home_home_wins_5", "home_home_draws_5",
    "home_home_losses_5", "home_home_points_5",
    "home_home_gf_5", "home_home_ga_5",
    "home_home_gd_5", "home_home_shots_5",
    "home_home_corners_5",
    # Away form 5
    "away_form_wins_5", "away_form_draws_5",
    "away_form_losses_5", "away_form_points_5",
    "away_form_gf_5", "away_form_ga_5",
    "away_form_gd_5", "away_form_shots_5",
    "away_form_shots_ot_5", "away_form_corners_5",
    # Away form 10
    "away_form_wins_10", "away_form_draws_10",
    "away_form_losses_10", "away_form_points_10",
    "away_form_gf_10", "away_form_ga_10",
    "away_form_gd_10",
    # Away at away 5
    "away_away_wins_5", "away_away_draws_5",
    "away_away_losses_5", "away_away_points_5",
    "away_away_gf_5", "away_away_ga_5",
    "away_away_gd_5", "away_away_shots_5",
    "away_away_corners_5",
    # Elo
    "home_elo", "away_elo", "elo_difference",
    # H2H
    "h2h_home_wins_5", "h2h_draws_5",
    "h2h_away_wins_5", "h2h_home_gf_5",
    "h2h_away_gf_5",
    # League and season
    "league_encoded", "season_encoded",
]


def load_artifacts() -> None:
    """
    Load all model .joblib files and the features CSV
    once at module import time. Missing artifacts print
    a warning but never crash the import or the API.
    """
    global _result_model, _home_goals, _away_goals
    global _home_corners, _away_corners
    global _le_league, _le_season, _features_df

    models_dir = BACKEND_DIR / "models"
    data_dir   = BACKEND_DIR / "data"

    artifacts = {
        "_result_model":  models_dir / "result_model_v1.joblib",
        "_home_goals":    models_dir / "home_goals_v1.joblib",
        "_away_goals":    models_dir / "away_goals_v1.joblib",
        "_home_corners":  models_dir / "home_corners_v1.joblib",
        "_away_corners":  models_dir / "away_corners_v1.joblib",
        "_le_league":     data_dir   / "league_encoder.joblib",
        "_le_season":     data_dir   / "season_encoder.joblib",
    }

    for var_name, path in artifacts.items():
        try:
            obj = joblib.load(path)
            globals()[var_name] = obj
            print(f"[predict] Loaded {path.name}")
        except FileNotFoundError:
            print(f"[predict] WARNING: {path.name} not found "
                  f"— run collector, builder and trainer first")
        except Exception as exc:
            print(f"[predict] WARNING: could not load "
                  f"{path.name}: {exc}")

    features_path = BACKEND_DIR / "data" / "processed" / "features.csv"
    try:
        df = pd.read_csv(features_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        _features_df = df
        print(f"[predict] Loaded features.csv "
              f"({len(df)} rows)")
    except FileNotFoundError:
        print("[predict] WARNING: features.csv not found")
    except Exception as exc:
        print(f"[predict] WARNING: could not load "
              f"features.csv: {exc}")


# Call once at import time
load_artifacts()

router = APIRouter(prefix="/predict", tags=["predict"])


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    season: str = "2024-2025"


def _encode_label(encoder, value: str, fallback: int = 0) -> int:
    """
    Encode a string label using a fitted LabelEncoder.
    If the value was not seen during training, returns
    fallback (default 0) and prints a warning.
    Never raises.
    """
    if encoder is None:
        return fallback
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        known = list(encoder.classes_)
        print(f"[predict] WARNING: unseen label '{value}'. "
              f"Known: {known}. Using fallback={fallback}")
        return fallback


def _get_team_features(
    team: str,
    role: str,
    df: pd.DataFrame,
) -> dict[str, float]:
    """
    Extract the most recent feature row for a team.

    role = "home" -> use rows where home_team == team,
                     return all home_* and h2h_* columns
    role = "away" -> use rows where away_team == team,
                     return all away_* columns

    Returns a dict of feature_name -> value.
    Falls back to zeros with a warning if no history.
    """
    if df is None or df.empty:
        print(f"[predict] WARNING: no features data loaded")
        return {}

    if role == "home":
        mask = df["home_team"] == team
    else:
        mask = df["away_team"] == team

    team_rows = df[mask]

    if team_rows.empty:
        print(f"[predict] WARNING: no history for "
              f"{team} as {role} — using zeros")
        return {}

    latest = team_rows.iloc[-1]

    if role == "home":
        prefixes = (
            "home_form_", "home_home_",
            "home_elo", "h2h_",
        )
    else:
        prefixes = (
            "away_form_", "away_away_",
            "away_elo",
        )

    result = {}
    for col in FEATURE_COLS:
        if any(col.startswith(p) for p in prefixes):
            result[col] = float(latest.get(col, 0.0) or 0.0)

    return result


def _build_feature_vector(
    home_team: str,
    away_team: str,
    league_encoded: int,
    season_encoded: int,
    df: pd.DataFrame,
) -> dict[str, float]:
    """
    Build the full 62-feature vector for a prediction request.

    Uses the most recent home stats for home_team and
    the most recent away stats for away_team from the
    features CSV. Elo and H2H come from the home row
    (they are always computed from home perspective).
    """
    home_feats = _get_team_features(home_team, "home", df)
    away_feats = _get_team_features(away_team, "away", df)

    vector: dict[str, float] = {}
    for col in FEATURE_COLS:
        if col == "league_encoded":
            vector[col] = float(league_encoded)
        elif col == "season_encoded":
            vector[col] = float(season_encoded)
        elif col == "elo_difference":
            vector[col] = (
                home_feats.get("home_elo", 0.0)
                - away_feats.get("away_elo", 0.0)
            )
        elif col in home_feats:
            vector[col] = home_feats[col]
        elif col in away_feats:
            vector[col] = away_feats[col]
        else:
            vector[col] = 0.0

    return vector


def _poisson_over(lam: float, threshold: float) -> float:
    """
    Probability that a Poisson(lam) variable exceeds threshold.
    e.g. threshold=2.5 -> P(X > 2) = 1 - poisson.cdf(2, lam)
    """
    k = int(threshold)
    return float(1.0 - poisson.cdf(k, max(lam, 1e-6)))


def _poisson_btts(home_lam: float, away_lam: float) -> float:
    """
    P(home >= 1) * P(away >= 1) assuming independence.
    """
    p_home = float(1.0 - poisson.cdf(0, max(home_lam, 1e-6)))
    p_away = float(1.0 - poisson.cdf(0, max(away_lam, 1e-6)))
    return round(p_home * p_away, 4)


@router.get("/leagues")
def get_leagues() -> dict:
    """Return all known leagues from the league encoder."""
    if _le_league is None:
        return {"leagues": []}
    return {"leagues": sorted(list(_le_league.classes_))}


@router.get("/teams")
def get_teams(league: str = "") -> dict:
    """
    Return all unique team names for a given league
    from the features CSV (home or away side).
    Returns empty list if league not found or no data.
    """
    if _features_df is None or _features_df.empty:
        return {"teams": []}

    df = _features_df
    if league:
        df = df[df["league"] == league]

    if df.empty:
        return {"teams": []}

    home_teams = set(df["home_team"].dropna().unique())
    away_teams = set(df["away_team"].dropna().unique())
    all_teams  = sorted(home_teams | away_teams)
    return {"teams": all_teams}


@router.post("/match")
def predict_match(req: MatchRequest) -> dict[str, Any]:
    """
    Run all prediction models for a given fixture
    and return full market probabilities.
    """
    # Check models are loaded
    models_ready = all([
        _result_model is not None,
        _home_goals   is not None,
        _away_goals   is not None,
        _home_corners is not None,
        _away_corners is not None,
    ])
    if not models_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models not loaded. Run collector.py, "
                "builder.py and trainer.py first."
            ),
        )

    if _features_df is None:
        raise HTTPException(
            status_code=503,
            detail="Features data not loaded.",
        )

    # Encode league and season (unseen -> fallback 0)
    league_enc = _encode_label(_le_league, req.league)
    season_enc = _encode_label(_le_season, req.season)

    # Build feature vector
    vector = _build_feature_vector(
        req.home_team,
        req.away_team,
        league_enc,
        season_enc,
        _features_df,
    )

    # Convert to DataFrame with correct column order
    X = pd.DataFrame([vector])[FEATURE_COLS]

    # 1X2 result probabilities
    result_proba = _result_model.predict_proba(X)[0]
    classes      = list(_result_model.classes_)
    result_map   = dict(zip(classes, result_proba))

    home_win_prob = round(float(result_map.get("H", 0.0)), 4)
    draw_prob     = round(float(result_map.get("D", 0.0)), 4)
    away_win_prob = round(float(result_map.get("A", 0.0)), 4)

    # Expected goals
    exp_home_goals = float(max(_home_goals.predict(X)[0], 0.0))
    exp_away_goals = float(max(_away_goals.predict(X)[0], 0.0))

    # Expected corners
    exp_home_corners = float(max(_home_corners.predict(X)[0], 0.0))
    exp_away_corners = float(max(_away_corners.predict(X)[0], 0.0))
    exp_total_corners = exp_home_corners + exp_away_corners

    # Derived goal markets
    over_1_5 = _poisson_over(
        exp_home_goals + exp_away_goals, 1.5)
    over_2_5 = _poisson_over(
        exp_home_goals + exp_away_goals, 2.5)
    over_3_5 = _poisson_over(
        exp_home_goals + exp_away_goals, 3.5)
    btts = _poisson_btts(exp_home_goals, exp_away_goals)

    # Derived corner markets
    over_corners_8_5  = _poisson_over(exp_total_corners, 8.5)
    over_corners_10_5 = _poisson_over(exp_total_corners, 10.5)

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "league":    req.league,
        "season":    req.season,
        "result": {
            "home_win": home_win_prob,
            "draw":     draw_prob,
            "away_win": away_win_prob,
        },
        "goals": {
            "expected_home": round(exp_home_goals, 3),
            "expected_away": round(exp_away_goals, 3),
            "over_1_5":      round(over_1_5, 4),
            "over_2_5":      round(over_2_5, 4),
            "over_3_5":      round(over_3_5, 4),
            "btts":          round(btts, 4),
        },
        "corners": {
            "expected_home":  round(exp_home_corners, 3),
            "expected_away":  round(exp_away_corners, 3),
            "over_8_5":       round(over_corners_8_5, 4),
            "over_10_5":      round(over_corners_10_5, 4),
        },
        "feature_vector_used": {
            k: round(v, 4) for k, v in vector.items()
        },
    }
