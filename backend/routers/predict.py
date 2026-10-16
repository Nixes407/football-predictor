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

import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy.stats import poisson
from supabase import create_client
import subprocess
import sys
import threading

try:
    from models.formula_engine import (
        predict_match as formula_predict,
        get_engine_status,
    )
    _formula_engine_available = True
    print("[predict] Formula engine loaded")
except Exception as exc:
    formula_predict = None
    get_engine_status = None
    _formula_engine_available = False
    print(f"[predict] WARNING: formula engine "
          f"not available: {exc}")

BACKEND_DIR = Path(__file__).resolve().parent.parent

# The admin endpoints below import the database layer (and the one-off repair
# script) from inside the request handler, i.e. by top-level module name.
# ``database`` and ``scripts.fix_bsa_label`` only resolve while backend/ is on
# sys.path, which is not guaranteed when the app is imported from another
# directory (``uvicorn backend.main:app`` from the repository root, a script
# that mounts the app, ...). ``/admin/fbref-check`` used to insert the path for
# itself inside the handler; do it once, here, for every endpoint.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Module-level artifact variables (all start as None)
_result_model  = None
_home_goals    = None
_away_goals    = None
_home_corners  = None
_away_corners  = None
_le_league     = None
_le_season     = None
_features_df   = None

_pipeline_running = False
_pipeline_status  = "idle"
_pipeline_log     = []
_pipeline_error = ""

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
    # FBref home team stats
    "home_fbref_games",
    "home_fbref_shots", "home_fbref_shots_on_target",
    "home_fbref_shot_accuracy",
    "home_fbref_goals_per_shot", "home_fbref_goals_per_sot",
    "home_fbref_yellow_cards", "home_fbref_red_cards",
    "home_fbref_fouls_committed", "home_fbref_fouls_drawn",
    "home_fbref_offsides", "home_fbref_crosses",
    "home_fbref_interceptions", "home_fbref_tackles_won",
    "home_fbref_pens_won", "home_fbref_pens_conceded",
    "home_fbref_sota", "home_fbref_goals_against",
    "home_fbref_saves", "home_fbref_save_pct",
    "home_fbref_clean_sheets",
    "home_fbref_pk_faced", "home_fbref_pk_saved",
    # FBref away team stats
    "away_fbref_games",
    "away_fbref_shots", "away_fbref_shots_on_target",
    "away_fbref_shot_accuracy",
    "away_fbref_goals_per_shot", "away_fbref_goals_per_sot",
    "away_fbref_yellow_cards", "away_fbref_red_cards",
    "away_fbref_fouls_committed", "away_fbref_fouls_drawn",
    "away_fbref_offsides", "away_fbref_crosses",
    "away_fbref_interceptions", "away_fbref_tackles_won",
    "away_fbref_pens_won", "away_fbref_pens_conceded",
    "away_fbref_sota", "away_fbref_goals_against",
    "away_fbref_saves", "away_fbref_save_pct",
    "away_fbref_clean_sheets",
    "away_fbref_pk_faced", "away_fbref_pk_saved",
]


def download_models_from_supabase() -> bool:
    """
    Pull the five model .joblib files from the "models" bucket
    in Supabase Storage into BACKEND_DIR / "models".

    Credentials come from SUPABASE_URL and SUPABASE_KEY. Returns
    True only when every file downloaded and wrote successfully —
    anything else returns False so the caller can fall back to
    building the models locally. Never raises.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        print("[predict] WARNING: SUPABASE_URL and/or SUPABASE_KEY is not set "
              "— skipping Supabase model download")
        return False

    try:
        client = create_client(url, key)
    except Exception as exc:
        print(f"[predict] WARNING: could not create Supabase client: {exc}")
        return False

    models_dir = BACKEND_DIR / "models"

    # (path inside the "models" bucket, where to save it locally)
    model_files = [
        (
            "result_model_v1.joblib",
            models_dir / "result_model_v1.joblib",
        ),
        (
            "home_goals_v1.joblib",
            models_dir / "home_goals_v1.joblib",
        ),
        (
            "away_goals_v1.joblib",
            models_dir / "away_goals_v1.joblib",
        ),
        (
            "home_corners_v1.joblib",
            models_dir / "home_corners_v1.joblib",
        ),
        (
            "away_corners_v1.joblib",
            models_dir / "away_corners_v1.joblib",
        ),
        (
            "league_encoder.joblib",
            BACKEND_DIR / "data" / "league_encoder.joblib",
        ),
        (
            "season_encoder.joblib",
            BACKEND_DIR / "data" / "season_encoder.joblib",
        ),
        (
            "features.csv",
            BACKEND_DIR / "data" / "processed" / "features.csv",
        ),
    ]

    for file_name, save_path in model_files:
        try:
            data = client.storage.from_("models").download(file_name)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(data)
            print(f"[predict] Downloaded from Supabase -> models/{file_name}")
        except Exception as exc:
            print(f"[predict] WARNING: could not download "
                  f"{file_name}: {exc}")
            return False

    print("Models downloaded from Supabase Storage")
    return True


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


def _get_runtime_feature_cols() -> list[str]:
    """
    Return the feature list from the loaded model.
    Falls back to the hardcoded FEATURE_COLS if
    the model is not loaded or has no feature info.
    This guarantees predict.py always uses exactly
    the features the model was trained on.
    """
    global _result_model

    if _result_model is None:
        return FEATURE_COLS

    try:
        # Pipeline: get the final estimator step
        if hasattr(_result_model, 'steps'):
            final_step = _result_model.steps[-1][1]
        else:
            final_step = _result_model

        if hasattr(final_step, 'feature_names_in_'):
            return list(final_step.feature_names_in_)

        # Try the imputer step which sees raw features
        if hasattr(_result_model, 'steps'):
            imputer = _result_model.steps[0][1]
            if hasattr(imputer, 'feature_names_in_'):
                return list(imputer.feature_names_in_)
    except Exception as exc:
        print(f"[predict] WARNING: could not read feature "
              f"names from model: {exc}")

    return FEATURE_COLS


def auto_start_fast_pipeline() -> None:
    """
    Called at startup. Checks Supabase Storage first: a complete
    set of models there is downloaded and loaded locally, and the
    pipeline is skipped. Otherwise, if models are not loaded,
    automatically starts builder + trainer in a
    background thread. Collector is skipped because
    data is already in the database.
    """
    global _pipeline_running, _pipeline_status, _pipeline_log

    if download_models_from_supabase():
        load_artifacts()
        _pipeline_status = "complete"
        _pipeline_running = False
        print("[pipeline] Models loaded from Supabase Storage "
              "— skipping pipeline")
        return

    if all([
        _result_model  is not None,
        _home_goals    is not None,
        _away_goals    is not None,
        _home_corners  is not None,
        _away_corners  is not None,
    ]):
        print("[pipeline] Models already loaded — skipping auto-start.")
        return

    if _pipeline_running:
        print("[pipeline] Pipeline already running — skipping auto-start.")
        return

    print("[pipeline] Models missing at startup — "
          "auto-starting fast pipeline...")

    def _run():
        global _pipeline_running, _pipeline_status, _pipeline_log
        _pipeline_running = True
        _pipeline_log = ["[AUTO] Starting on startup..."]
        backend_dir = str(BACKEND_DIR)

        steps = [
            ("fbref_collector", ["python", "data/fbref_collector.py"]),
            ("builder",         ["python", "features/builder.py"]),
            ("trainer",         ["python", "models/trainer.py"]),
        ]

        for name, cmd in steps:
            _pipeline_status = f"running_{name}"
            _pipeline_log.append(f"[AUTO] Starting {name}...")
            print(f"[pipeline][AUTO] Starting {name}...")
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=backend_dir,
                    capture_output=True,
                    text=True,
                    timeout=7200,
                )
                if proc.stdout:
                    print(proc.stdout[-3000:])
                if proc.stderr:
                    print(proc.stderr[-1000:])
                if proc.returncode == 0:
                    _pipeline_log.append(f"[AUTO] {name} completed OK")
                    print(f"[pipeline][AUTO] {name} done (rc=0)")
                else:
                    global _pipeline_error
                    _pipeline_error = (
                        proc.stderr[-2000:] if proc.stderr
                        else proc.stdout[-2000:] if proc.stdout
                        else "no output captured"
                    )
                    _pipeline_log.append(
                        f"[AUTO] {name} FAILED rc={proc.returncode}")
                    print(f"[pipeline][AUTO] {name} FAILED")
                    print(_pipeline_error)
                    _pipeline_status = f"failed_{name}"
                    _pipeline_running = False
                    return
            except subprocess.TimeoutExpired:
                _pipeline_log.append(f"[AUTO] {name} timed out")
                _pipeline_status = f"timeout_{name}"
                _pipeline_running = False
                return
            except Exception as exc:
                _pipeline_log.append(f"[AUTO] {name} error: {exc}")
                _pipeline_status = f"error_{name}"
                _pipeline_running = False
                return

        load_artifacts()
        _pipeline_status = "complete"
        _pipeline_log.append("[AUTO] Pipeline complete. Artifacts loaded.")
        print("[pipeline][AUTO] Done. Artifacts reloaded.")
        _pipeline_running = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


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
    feature_cols: list[str] | None = None,
) -> dict[str, float]:
    """
    Extract the most recent feature row for a team.

    role = "home" -> use rows where home_team == team,
                     return all home_* and h2h_* columns
    role = "away" -> use rows where away_team == team,
                     return all away_* columns

    ``feature_cols`` defaults to the hardcoded FEATURE_COLS. Callers that
    know the model's own feature list (``_get_runtime_feature_cols``) pass it
    through, so a model trained on extra columns reads those columns from the
    features CSV instead of scoring them as 0.

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
            "home_fbref_",
        )
    else:
        prefixes = (
            "away_form_", "away_away_",
            "away_elo",
            "away_fbref_",
        )

    result = {}
    for col in (feature_cols if feature_cols is not None else FEATURE_COLS):
        if any(col.startswith(p) for p in prefixes):
            result[col] = float(latest.get(col, 0.0) or 0.0)

    return result


def _build_feature_vector(
    home_team: str,
    away_team: str,
    league_encoded: int,
    season_encoded: int,
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict[str, float]:
    """
    Build the full 62-feature vector for a prediction request.

    Uses the most recent home stats for home_team and
    the most recent away stats for away_team from the
    features CSV. Elo and H2H come from the home row
    (they are always computed from home perspective).
    """
    home_feats = _get_team_features(home_team, "home", df, feature_cols)
    away_feats = _get_team_features(away_team, "away", df, feature_cols)

    cols_to_use = feature_cols if feature_cols is not None else FEATURE_COLS
    vector: dict[str, float] = {}
    for col in cols_to_use:
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
    # Try formula engine first
    if _formula_engine_available and formula_predict:
        try:
            formula_result = formula_predict(
                home_team=req.home_team,
                away_team=req.away_team,
                league=req.league,
                season=req.season,
            )
            if formula_result is not None:
                print(f"[predict] Formula engine: "
                      f"{req.home_team} vs "
                      f"{req.away_team}")
                return formula_result
            else:
                print("[predict] Formula engine "
                      "returned None — using ML")
        except Exception as exc:
            print(f"[predict] Formula engine "
                  f"error: {exc} — using ML")

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
    runtime_cols = _get_runtime_feature_cols()
    vector = _build_feature_vector(
        req.home_team,
        req.away_team,
        league_enc,
        season_enc,
        _features_df,
        runtime_cols,
    )

    # Fill any missing columns with 0
    for col in runtime_cols:
        if col not in vector:
            vector[col] = 0.0

    # Convert to DataFrame with correct column order
    X = pd.DataFrame([vector])[runtime_cols]

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


@router.get("/admin/diagnostics")
def diagnostics() -> dict:
    """
    Report what is actually loaded in memory.
    Shows model feature counts, features.csv columns,
    and whether they match.
    No secret required — read only.
    """
    runtime_cols = _get_runtime_feature_cols()

    features_df_cols = []
    fbref_in_csv = []
    if _features_df is not None:
        features_df_cols = list(_features_df.columns)
        fbref_in_csv = [c for c in features_df_cols
                        if 'fbref' in c]

    fbref_in_model = [c for c in runtime_cols
                      if 'fbref' in c]

    return {
        "hardcoded_feature_cols": len(FEATURE_COLS),
        "model_feature_cols":     len(runtime_cols),
        "features_csv_cols":      len(features_df_cols),
        "fbref_cols_in_model":    len(fbref_in_model),
        "fbref_cols_in_csv":      len(fbref_in_csv),
        "model_fbref_sample":     fbref_in_model[:5],
        "csv_fbref_sample":       fbref_in_csv[:5],
        "cols_match":             (
            set(runtime_cols) == set(FEATURE_COLS)
        ),
        "formula_engine": (
            get_engine_status()
            if _formula_engine_available and get_engine_status
            else {"ready": False, "reason": "not loaded"}
        ),
        "active_engine": (
            "formula"
            if _formula_engine_available and get_engine_status
            and get_engine_status().get("ready")
            else "ml_models"
        ),
    }


@router.post("/admin/run-pipeline")
def run_pipeline(secret: str = "") -> dict:
    """
    Trigger the full data pipeline in a background
    thread so the HTTP request returns immediately.
    Monitor progress via GET /predict/admin/status
    """
    import os
    global _pipeline_running, _pipeline_status, _pipeline_log

    expected = os.getenv("PIPELINE_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing secret key."
        )

    if _pipeline_running:
        return {
            "status": "already_running",
            "message": "Pipeline is already running. "
                       "Check /predict/admin/status for progress."
        }

    def _run():
        global _pipeline_running, _pipeline_status, _pipeline_log, _pipeline_error
        _pipeline_running = True
        _pipeline_log = []
        backend_dir = str(BACKEND_DIR)

        steps = [
            ("collector",       ["python", "data/collector.py"]),
            ("fbref_collector", ["python", "data/fbref_collector.py"]),
            ("builder",         ["python", "features/builder.py"]),
            ("trainer",         ["python", "models/trainer.py"]),
        ]

        for name, cmd in steps:
            _pipeline_status = f"running_{name}"
            _pipeline_log.append(f"Starting {name}...")
            print(f"[pipeline] Starting {name}...")
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=backend_dir,
                    capture_output=True,
                    text=True,
                    timeout=7200,
                )
                if proc.stdout:
                    print(proc.stdout[-3000:])
                if proc.stderr:
                    print(proc.stderr[-1000:])
                if proc.returncode == 0:
                    _pipeline_log.append(f"{name} completed OK")
                    print(f"[pipeline] {name} done (rc=0)")
                else:
                    global _pipeline_error
                    _pipeline_error = (
                        proc.stderr[-2000:] if proc.stderr
                        else proc.stdout[-2000:] if proc.stdout
                        else "no output captured"
                    )
                    _pipeline_log.append(
                        f"{name} FAILED rc={proc.returncode}")
                    print(f"[pipeline] {name} FAILED")
                    print(_pipeline_error)
                    _pipeline_status = f"failed_{name}"
                    _pipeline_running = False
                    return
            except subprocess.TimeoutExpired:
                _pipeline_log.append(f"{name} timed out")
                _pipeline_status = f"timeout_{name}"
                _pipeline_running = False
                return
            except Exception as exc:
                _pipeline_log.append(f"{name} error: {exc}")
                _pipeline_status = f"error_{name}"
                _pipeline_running = False
                return

        load_artifacts()
        _pipeline_status = "complete"
        _pipeline_log.append("Pipeline complete. Artifacts loaded.")
        print("[pipeline] All steps done. Artifacts reloaded.")
        _pipeline_running = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": "Pipeline started in background. "
                   "Poll /predict/admin/status to monitor progress."
    }


@router.post("/admin/run-pipeline-fast")
def run_pipeline_fast(secret: str = "") -> dict:
    """
    Run only builder and trainer — skips collector.
    Use this when data is already in the database
    and you just need to regenerate the model files.
    Takes ~30-40 minutes instead of 2+ hours.
    """
    import os
    global _pipeline_running, _pipeline_status, _pipeline_log

    expected = os.getenv("PIPELINE_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing secret key."
        )

    if _pipeline_running:
        return {
            "status": "already_running",
            "message": "Pipeline is already running. "
                       "Check /predict/admin/status for progress."
        }

    def _run():
        global _pipeline_running, _pipeline_status, _pipeline_log
        _pipeline_running = True
        _pipeline_log = []
        backend_dir = str(BACKEND_DIR)

        steps = [
            ("fbref_collector", ["python", "data/fbref_collector.py"]),
            ("builder",         ["python", "features/builder.py"]),
            ("trainer",         ["python", "models/trainer.py"]),
        ]

        for name, cmd in steps:
            _pipeline_status = f"running_{name}"
            _pipeline_log.append(f"Starting {name}...")
            print(f"[pipeline] Starting {name}...")
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=backend_dir,
                    capture_output=True,
                    text=True,
                    timeout=7200,
                )
                if proc.stdout:
                    print(proc.stdout[-3000:])
                if proc.stderr:
                    print(proc.stderr[-1000:])
                if proc.returncode == 0:
                    _pipeline_log.append(f"{name} completed OK")
                    print(f"[pipeline] {name} done (rc=0)")
                else:
                    global _pipeline_error
                    _pipeline_error = (
                        proc.stderr[-2000:] if proc.stderr
                        else proc.stdout[-2000:] if proc.stdout
                        else "no output captured"
                    )
                    _pipeline_log.append(
                        f"{name} FAILED rc={proc.returncode}")
                    print(f"[pipeline] {name} FAILED")
                    print(_pipeline_error)
                    _pipeline_status = f"failed_{name}"
                    _pipeline_running = False
                    return
            except subprocess.TimeoutExpired:
                _pipeline_log.append(f"{name} timed out")
                _pipeline_status = f"timeout_{name}"
                _pipeline_running = False
                return
            except Exception as exc:
                _pipeline_log.append(f"{name} error: {exc}")
                _pipeline_status = f"error_{name}"
                _pipeline_running = False
                return

        load_artifacts()
        _pipeline_status = "complete"
        _pipeline_log.append("Fast pipeline complete. Artifacts loaded.")
        print("[pipeline] Builder + trainer done. Artifacts reloaded.")
        _pipeline_running = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": "Fast pipeline started (builder + trainer only). "
                   "Poll /predict/admin/status to monitor progress."
    }


@router.get("/admin/status")
def pipeline_status() -> dict:
    """
    Check pipeline progress and loaded artifacts.
    """
    return {
        "pipeline_running": _pipeline_running,
        "pipeline_status":  _pipeline_status,
        "pipeline_log":     _pipeline_log[-10:],
        "pipeline_error":   _pipeline_error,
        "result_model":  _result_model  is not None,
        "home_goals":    _home_goals    is not None,
        "away_goals":    _away_goals    is not None,
        "home_corners":  _home_corners  is not None,
        "away_corners":  _away_corners  is not None,
        "le_league":     _le_league     is not None,
        "le_season":     _le_season     is not None,
        "features_df":   _features_df   is not None,
        "features_rows": len(_features_df) if _features_df is not None else 0,
        "ready": all([
            _result_model is not None,
            _home_goals   is not None,
            _away_goals   is not None,
            _home_corners is not None,
            _away_corners is not None,
        ]),
    }


@router.get("/admin/fbref-check")
def fbref_check() -> dict:
    """
    Query fbref_team_stats directly to see what
    data was actually stored by the FBref collector.
    No secret required.
    """
    try:
        # backend/ is already on sys.path (see BACKEND_DIR at the top), so the
        # imports below resolve no matter which directory the app was started
        # from.
        from sqlalchemy import text as sql_text
        from database import session_scope

        with session_scope() as session:
            total = session.execute(
                sql_text("SELECT COUNT(*) FROM fbref_team_stats")
            ).scalar()

            sample = session.execute(
                sql_text(
                    "SELECT team, league, season "
                    "FROM fbref_team_stats "
                    "LIMIT 10"
                )
            ).fetchall()

            leagues = session.execute(
                sql_text(
                    "SELECT DISTINCT league "
                    "FROM fbref_team_stats"
                )
            ).fetchall()

            seasons = session.execute(
                sql_text(
                    "SELECT DISTINCT season "
                    "FROM fbref_team_stats"
                )
            ).fetchall()

            matches_sample = session.execute(
                sql_text(
                    "SELECT DISTINCT home_team, league, season "
                    "FROM matches "
                    "LIMIT 5"
                )
            ).fetchall()

        return {
            "fbref_total_rows": total,
            "fbref_sample": [
                {"team": r[0], "league": r[1], "season": r[2]}
                for r in sample
            ],
            "fbref_distinct_leagues": [r[0] for r in leagues],
            "fbref_distinct_seasons": [r[0] for r in seasons],
            "matches_sample": [
                {"home_team": r[0], "league": r[1], "season": r[2]}
                for r in matches_sample
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/admin/new-tables-check")
def new_tables_check() -> dict:
    """
    Check new tables and show actual stored values
    to diagnose formula engine data quality.
    """
    try:
        from database import (
            SessionLocal,
            TeamSeasonStats,
            PlayerSeasonStats,
        )
        from sqlalchemy import text as sql_text

        db = SessionLocal()
        try:
            team_count   = db.query(TeamSeasonStats).count()
            player_count = db.query(PlayerSeasonStats).count()

            # Get Arsenal specifically to check values
            arsenal = (
                db.query(TeamSeasonStats)
                .filter(
                    TeamSeasonStats.team.ilike("%Arsenal%")
                ).first()
            )

            chelsea = (
                db.query(TeamSeasonStats)
                .filter(
                    TeamSeasonStats.team.ilike("%Chelsea%")
                ).first()
            )

            def team_snapshot(row):
                if row is None:
                    return None
                return {
                    "team":         row.team,
                    "league":       row.league,
                    "season":       row.season,
                    "sh_xg":        row.sh_xg,
                    "sh_goals":     row.sh_goals,
                    "sh_shots":     row.sh_shots,
                    "sh_sot":       row.sh_sot,
                    "sh_sh_per90":  row.sh_sh_per90,
                    "sh_sot_per90": row.sh_sot_per90,
                    "gk_ga":        row.gk_ga,
                    "gk_ga90":      row.gk_ga90,
                    "gk_cs":        row.gk_cs,
                    "gk_w":         row.gk_w,
                    "gk_d":         row.gk_d,
                    "gk_l":         row.gk_l,
                    "def_press":    row.def_press,
                    "def_press_pct":row.def_press_pct,
                    "pos_poss":     row.pos_poss,
                    "pos_prgc":     row.pos_prgc,
                }

            leagues = db.query(
                TeamSeasonStats.league
            ).distinct().all()

            seasons = db.query(
                TeamSeasonStats.season
            ).distinct().all()

            statsbomb_count = 0
            try:
                result = db.execute(sql_text(
                    "SELECT COUNT(*) FROM "
                    "statsbomb_match_stats"
                ))
                statsbomb_count = result.scalar()
            except Exception:
                statsbomb_count = -1

            from sqlalchemy import nulls_last
            top_scorers = (
                db.query(PlayerSeasonStats)
                .filter(
                    PlayerSeasonStats.league ==
                    "ENG-Premier League",
                    PlayerSeasonStats.goals
                    .isnot(None),
                    PlayerSeasonStats.minutes
                    > 0,
                )
                .order_by(
                    PlayerSeasonStats.goals.desc()
                )
                .limit(5)
                .all()
            )

            # misc_corners coverage. Isolated so a failure here degrades
            # to an error object instead of costing the rest of the
            # snapshot. Note this cannot rescue a database that has not
            # run ensure_team_corners_column yet: every TeamSeasonStats
            # query selects all mapped columns, so the count above fails
            # first and the outer handler answers with its own error.
            try:
                corners_non_null = (
                    db.query(TeamSeasonStats)
                    .filter(
                        TeamSeasonStats.misc_corners
                        .isnot(None)
                    )
                    .count()
                )
                corners_sample = (
                    db.query(
                        TeamSeasonStats.team,
                        TeamSeasonStats.league,
                        TeamSeasonStats.season,
                        TeamSeasonStats.misc_corners,
                    )
                    .filter(
                        TeamSeasonStats.misc_corners
                        .isnot(None)
                    )
                    .order_by(
                        TeamSeasonStats.team,
                        TeamSeasonStats.league,
                        TeamSeasonStats.season,
                    )
                    .limit(3)
                    .all()
                )
                misc_corners_check = {
                    "non_null_count": corners_non_null,
                    "total_rows":     team_count,
                    "sample": [
                        {
                            "team":         r[0],
                            "league":       r[1],
                            "season":       r[2],
                            "misc_corners": r[3],
                        }
                        for r in corners_sample
                    ],
                }
            except Exception as exc:
                misc_corners_check = {"error": str(exc)}

            return {
                "team_season_stats_rows":    team_count,
                "player_season_stats_rows":  player_count,
                "statsbomb_match_stats_rows":statsbomb_count,
                "distinct_leagues": [r[0] for r in leagues],
                "distinct_seasons": [r[0] for r in seasons],
                "arsenal_snapshot": team_snapshot(arsenal),
                "chelsea_snapshot": team_snapshot(chelsea),
                "top_pl_scorers": [
                    {
                        "player":  p.player,
                        "team":    p.team,
                        "goals":   p.goals,
                        "assists": p.assists,
                        "xg":      p.xg,
                        "minutes": p.minutes,
                    }
                    for p in top_scorers
                ],
                "misc_corners_check": misc_corners_check,
            }
        finally:
            db.close()

    except Exception as exc:
        return {"error": str(exc)}


@router.delete("/admin/clear-fixtures-cache")
def clear_fixtures_cache(
    secret: str = "",
    date: str = "",
) -> dict:
    """
    Delete cached fixtures for a specific date
    so they get refetched on the next call.
    Requires the PIPELINE_SECRET.
    """
    import os
    expected = os.getenv("PIPELINE_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing secret."
        )

    from database import SessionLocal
    from sqlalchemy import text as sql_text
    from datetime import date as date_cls

    target = date or date_cls.today().isoformat()

    db = SessionLocal()
    try:
        result = db.execute(
            sql_text(
                "DELETE FROM daily_matches "
                "WHERE match_date = :d"
            ),
            {"d": target},
        )
        db.commit()
        deleted = result.rowcount
        return {
            "deleted": deleted,
            "date": target,
            "message": f"Cleared {deleted} fixtures "
                       f"for {target}. Next call to "
                       f"/fixtures/today will refetch."
        }
    except Exception as exc:
        db.rollback()
        return {"error": str(exc)}
    finally:
        db.close()


@router.get("/admin/fotmob-team-names")
def fotmob_team_names() -> dict:
    """
    Show all team names stored in
    team_season_stats grouped by league.
    Used to debug team name mismatches.
    """
    try:
        from database import (
            SessionLocal, TeamSeasonStats)
        db = SessionLocal()
        try:
            rows = db.query(
                TeamSeasonStats.team,
                TeamSeasonStats.league,
            ).distinct().order_by(
                TeamSeasonStats.league,
                TeamSeasonStats.team,
            ).all()

            result = {}
            for team, league in rows:
                if league not in result:
                    result[league] = []
                result[league].append(team)

            return {
                "total_teams": len(rows),
                "by_league": result,
            }
        finally:
            db.close()
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
# One-off Brazilian Série A label diagnostics / repair
# --------------------------------------------------------------------------- #

# football-data.org serves the Brazilian top flight as competition code "BSA".
# Fixtures fetched before "BSA" was added to data/fixtures.LEAGUE_MAP were
# stored in daily_matches under another competition's label, so both the label
# the rows *should* carry and the stale one they currently carry are needed
# here. The repair itself lives in backend/scripts/fix_bsa_label.py.
BSA_LEAGUE = "BRA-Série A"
STALE_BSA_LEAGUE = "BEL-Jupiler Pro League"

# Rows returned per section. Enough to eyeball a result, small enough that the
# JSON stays readable when a name pattern matches many seasons.
_DIAGNOSTIC_ROW_LIMIT = 200


def _daily_match_rows(rows) -> list[dict]:
    """Serialize daily_matches rows for the diagnostic payload."""
    return [
        {
            "id": row.id,
            "external_id": row.external_id,
            "match_date": row.match_date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "league": row.league,
            "season": row.season,
            "status": row.status,
        }
        for row in rows
    ]


def _bsa_league_filter():
    """SQLAlchemy filter selecting every Brazilian-labelled daily fixture.

    Matches the label the fixtures should carry (``BRA-Série A``), the label
    the pre-fix rows were written with (``BEL-Jupiler Pro League``) and the
    Portuguese spelling (``Brasil``), all case-insensitively so a row stored as
    ``BRA-SERIE A`` still shows up.
    """
    from sqlalchemy import or_

    from database import DailyMatch

    league = DailyMatch.league
    return or_(
        league.ilike("%Jupiler%"),
        league.ilike("%Brasil%"),
        league.ilike(BSA_LEAGUE),
    )


def _team_stats_rows_for(session, pattern: str) -> dict:
    """Rows of team_season_stats / fbref_team_stats whose team matches.

    ``pattern`` is a SQL LIKE pattern (``%deportivo%``). Returns
    ``{"team_name_variants_found": [...], "rows": [...], "counts": {...}}``;
    a table that is missing or unreadable contributes an ``errors`` entry
    instead of failing the whole diagnostic.
    """
    from database import FBrefTeamStats, TeamSeasonStats

    variants: list[str] = []
    rows: list[dict] = []
    errors: list[str] = []

    # Column-limited queries on purpose: selecting whole ORM entities would
    # need every mapped column to exist, and team_season_stats gains columns
    # over time (misc_corners is applied by ensure_team_corners_column).
    try:
        team_rows = (
            session.query(
                TeamSeasonStats.team,
                TeamSeasonStats.league,
                TeamSeasonStats.season,
                TeamSeasonStats.sh_goals,
                TeamSeasonStats.sh_xg,
                TeamSeasonStats.gk_ga,
                TeamSeasonStats.gk_ga90,
            )
            .filter(TeamSeasonStats.team.ilike(pattern))
            .order_by(
                TeamSeasonStats.team,
                TeamSeasonStats.league,
                TeamSeasonStats.season,
            )
            .limit(_DIAGNOSTIC_ROW_LIMIT)
            .all()
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics report, never raise
        team_rows = []
        errors.append(f"team_season_stats: {exc}")

    for team, league, season, goals, xg, ga, ga90 in team_rows:
        if team not in variants:
            variants.append(team)
        rows.append(
            {
                "source": "team_season_stats",
                "team": team,
                "league": league,
                "season": season,
                "goals": goals,
                "xg": xg,
                "goals_against": ga,
                "goals_against_per90": ga90,
            }
        )

    try:
        fbref_rows = (
            session.query(
                FBrefTeamStats.team,
                FBrefTeamStats.league,
                FBrefTeamStats.season,
                FBrefTeamStats.games,
                FBrefTeamStats.shots,
                FBrefTeamStats.goals_per_shot,
            )
            .filter(FBrefTeamStats.team.ilike(pattern))
            .order_by(
                FBrefTeamStats.team,
                FBrefTeamStats.league,
                FBrefTeamStats.season,
            )
            .limit(_DIAGNOSTIC_ROW_LIMIT)
            .all()
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics report, never raise
        fbref_rows = []
        errors.append(f"fbref_team_stats: {exc}")

    for team, league, season, games, shots, goals_per_shot in fbref_rows:
        if team not in variants:
            variants.append(team)
        rows.append(
            {
                "source": "fbref_team_stats",
                "team": team,
                "league": league,
                "season": season,
                "games": games,
                "shots": shots,
                "goals_per_shot": goals_per_shot,
            }
        )

    result = {
        "team_name_variants_found": variants,
        "rows": rows,
        "counts": {
            "team_season_stats": len(
                [r for r in rows if r["source"] == "team_season_stats"]
            ),
            "fbref_team_stats": len(
                [r for r in rows if r["source"] == "fbref_team_stats"]
            ),
        },
    }
    if errors:
        result["errors"] = errors
    return result


@router.get("/admin/la-liga-check")
def la_liga_check() -> dict:
    """
    Temporary diagnostic: what every Brazilian fixture in daily_matches is
    labelled as, plus the Deportivo / Levante rows the La Liga predictor
    would join against. Read-only, no secret required.

    ``rows_needing_bsa_label_fix`` is the number of rows the one-off repair in
    backend/scripts/fix_bsa_label.py would rewrite — 0 means the migration has
    already been applied (or there was nothing to fix).
    """
    try:
        from database import DailyMatch, session_scope

        with session_scope() as session:
            bsa_rows = _daily_match_rows(
                session.query(DailyMatch)
                .filter(_bsa_league_filter())
                .order_by(DailyMatch.match_date.desc(), DailyMatch.id.desc())
                .limit(_DIAGNOSTIC_ROW_LIMIT)
                .all()
            )
            pending_fix = (
                session.query(DailyMatch)
                .filter(DailyMatch.league.ilike("%Jupiler%"))
                .count()
            )
            deportivo = _team_stats_rows_for(session, "%deportivo%")
            levante = _team_stats_rows_for(session, "%levante%")

        return {
            "bsa_stored_rows": bsa_rows,
            "bsa_stored_row_count": len(bsa_rows),
            "expected_bsa_league": BSA_LEAGUE,
            "stale_bsa_league": STALE_BSA_LEAGUE,
            "rows_needing_bsa_label_fix": pending_fix,
            "deportivo_stats": deportivo,
            "levante_stats": levante,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics report, never raise
        print(f"[predict] WARNING: la-liga-check failed: {exc}")
        return {"error": str(exc)}


@router.post("/admin/fix-bsa-label")
def fix_bsa_label(secret: str = "", dry_run: bool = False) -> dict:
    """
    Apply the one-off ``daily_matches`` label repair over HTTP, for a
    deployment that has no shell: sets every Brazilian fixture still stored as
    ``BEL-Jupiler Pro League`` to ``BRA-Série A``.

    Requires PIPELINE_SECRET, like the other write endpoints. Idempotent —
    a second call reports ``"updated": 0``. Pass ``dry_run=true`` to list the
    rows without writing anything.
    """
    import os

    expected = os.getenv("PIPELINE_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing secret.",
        )

    from database import session_scope
    from scripts.fix_bsa_label import apply_bsa_label_fix

    try:
        with session_scope() as session:
            report = apply_bsa_label_fix(session, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 - report, never 500
        print(f"[predict] WARNING: fix-bsa-label failed: {exc}")
        return {"error": str(exc)}

    print(
        f"[predict] fix-bsa-label: updated {report['updated']} row(s) "
        f"(dry_run={report['dry_run']})"
    )
    return report
