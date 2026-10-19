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

import copy
import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.stats import poisson
from supabase import create_client

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

# Shared bucket manifest — the trainer uploads exactly what this file lists.
import model_artifacts as artifacts  # noqa: E402  (needs the sys.path above)

# A reload only downloads pre-built files; if another reload is already in
# flight (e.g. the startup reload racing the GitHub Actions callback) wait this
# long for it before answering 409. Kept well under the endpoint's 60 s budget.
RELOAD_LOCK_TIMEOUT_SECONDS = 25

# Training moved to GitHub Actions (7 GB runners). These endpoints are kept
# only so an old cron/curl target gets an explicit 410 instead of running
# builder.py/trainer.py on a 512 MB Render instance.
_PIPELINE_MOVED_RESPONSE = {
    "error": "training moved to GitHub Actions",
    "status": "gone",
    "detail": (
        "builder.py and trainer.py now run in the daily GitHub Actions "
        "workflow (.github/workflows/daily_fbref_update.yml). This service "
        "only serves predictions. Use POST /predict/admin/reload-models to "
        "re-download the pre-built artifacts from Supabase Storage."
    ),
}

# Module-level artifact variables (all start as None)
_result_model  = None
_xgb_result_model = None
_xgb_label_encoder = None
_home_goals    = None
_away_goals    = None
_home_corners  = None
_away_corners  = None
_btts_model    = None
_over25_model  = None
_over95_corners_model = None
_le_league     = None
_le_season     = None
_features_df   = None

_pipeline_running = False
_pipeline_status  = "idle"
_pipeline_log     = []
_pipeline_error = ""
_last_reload_at = ""
_reload_lock = threading.Lock()

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

BTTS_FEATURE_COLS = [
    "home_btts_rate_5", "away_btts_rate_5",
    "home_cs_rate_5", "away_cs_rate_5",
    "home_scored_rate_5", "away_scored_rate_5",
    "home_form_gf_5", "away_form_gf_5",
    "home_form_ga_5", "away_form_ga_5",
    "home_fbref_goals_against", "away_fbref_goals_against",
    "home_fbref_clean_sheets", "away_fbref_clean_sheets",
    "home_elo", "away_elo",
    "league_encoded",
]

OVER25_FEATURE_COLS = [
    "home_over25_rate_5", "away_over25_rate_5",
    "home_high_scoring_rate_5", "away_high_scoring_rate_5",
    "home_form_gf_5", "away_form_gf_5",
    "home_form_ga_5", "away_form_ga_5",
    "home_form_gf_10", "away_form_gf_10",
    "home_fbref_goals_against", "away_fbref_goals_against",
    "home_fbref_shots_on_target", "away_fbref_shots_on_target",
    "home_elo", "away_elo",
    "elo_difference",
    "league_encoded",
]

# Must stay identical to OVER95_CORNERS_FEATURE_COLS in models/trainer.py —
# the classifier is scored on exactly the columns it was trained on, in order.
OVER95_CORNERS_FEATURE_COLS = [
    "home_corners_for_pg_5", "away_corners_for_pg_5",
    "home_corners_against_pg_5", "away_corners_against_pg_5",
    "home_over95_corners_rate_5", "away_over95_corners_rate_5",
    "home_home_corners_5", "away_away_corners_5",
    "home_form_corners_5", "away_form_corners_5",
    "home_fbref_shots_on_target", "away_fbref_shots_on_target",
    "home_fbref_crosses", "away_fbref_crosses",
    "home_elo", "away_elo",
    "league_encoded",
]

# ── Blend weights: formula_v1 vs ML ──────────────────────────────────────────
BLEND_FORMULA_1X2   = 0.60   # formula weight for home/draw/away win probs
BLEND_ML_1X2        = 0.40   # ML weight for home/draw/away win probs
BLEND_FORMULA_GOALS = 0.70   # formula weight for expected home/away goals
BLEND_ML_GOALS      = 0.30   # ML weight for expected home/away goals
BLEND_FORMULA_CRNRS = 0.70   # formula weight for expected home/away corners
BLEND_ML_CRNRS      = 0.30   # ML weight for expected home/away corners
# Specialized classifiers (BTTS, Over 2.5, Over 9.5) always use 100% classifier
# regardless of formula availability — those overrides live in PRs #72/#73/#74


def _supabase_storage_client():
    """Return ``(client, error_message)`` for the "models" bucket.

    The service role key is preferred (uploads need it; downloads work with
    either the service role or the legacy ``SUPABASE_KEY``). Never raises so
    the API can keep serving the artifacts already on disk.
    """
    url, key = artifacts.supabase_credentials()
    if not url or not key:
        return None, artifacts.missing_credentials_message()
    try:
        return create_client(url, key), ""
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        return None, f"could not create Supabase client: {exc}"


def _download_artifacts_to(staging_dir: Path, client=None) -> list[str]:
    """Download every artifact into ``staging_dir`` and return what arrived.

    Nothing outside ``staging_dir`` is touched, so a failure halfway through
    leaves the artifacts currently served by the API intact. Raises
    ``RuntimeError`` when a required object is missing or empty; optional
    objects are logged and skipped.
    """
    if client is None:
        client, error = _supabase_storage_client()
        if client is None:
            raise RuntimeError(error)

    bucket = client.storage.from_(artifacts.MODEL_BUCKET)
    downloaded: list[str] = []

    for object_name in artifacts.REQUIRED_DOWNLOAD_OBJECTS:
        data = bucket.download(object_name)
        if not data:
            raise RuntimeError(
                f"Supabase Storage returned an empty file for {object_name}"
            )
        target = artifacts.local_path(object_name, staging_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        downloaded.append(object_name)
        print(f"[predict] Downloaded {object_name} "
              f"({len(data) / 1024:.0f} KiB)")

    # Optional market models never fail a reload: a model added by a newer
    # release is not in the bucket until the next training run uploads it, and
    # the router already degrades to the Poisson figure when its global is None.
    for object_name in artifacts.OPTIONAL_DOWNLOAD_OBJECTS:
        try:
            data = bucket.download(object_name)
        except Exception as exc:  # noqa: BLE001 - optional by design
            print(f"[predict] NOTE: optional {object_name} not in Supabase "
                  f"bucket yet ({exc}) — falling back to Poisson")
            continue
        if not data:
            print(f"[predict] NOTE: optional {object_name} is empty in "
                  f"Supabase — falling back to Poisson")
            continue
        target = artifacts.local_path(object_name, staging_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        downloaded.append(object_name)
        print(f"[predict] Downloaded {object_name} "
              f"({len(data) / 1024:.0f} KiB)")

    return downloaded


def _install_artifacts(staging_dir: Path, object_names: list[str]) -> None:
    """Move the staged files into place with an atomic per-file rename."""
    for object_name in object_names:
        source = artifacts.local_path(object_name, staging_dir)
        if not source.exists():
            continue
        destination = artifacts.local_path(object_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)


def download_models_from_supabase() -> bool:
    """Download the bucket into the local artifact directories.

    Thin backwards-compatible wrapper around :func:`_download_artifacts_to`
    plus :func:`_install_artifacts`: returns True when every *required*
    artifact was staged and installed, False otherwise. Never raises.
    """
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=BACKEND_DIR))
    try:
        downloaded = _download_artifacts_to(staging_dir)
        _install_artifacts(staging_dir, downloaded)
        print("[predict] Models downloaded from Supabase Storage")
        return True
    except Exception as exc:  # noqa: BLE001 - legacy contract: never raise
        print(f"[predict] WARNING: model download failed: {exc}")
        return False
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def load_artifacts() -> None:
    """
    Load all model .joblib files and the features CSV
    once at module import time. Missing artifacts print
    a warning but never crash the import or the API.
    """
    global _result_model, _xgb_result_model, _xgb_label_encoder, _home_goals, _away_goals
    global _home_corners, _away_corners, _btts_model, _over25_model
    global _over95_corners_model
    global _le_league, _le_season, _features_df

    models_dir = BACKEND_DIR / "models"
    data_dir   = BACKEND_DIR / "data"

    # NOTE: not named ``artifacts`` — that is the imported manifest module.
    artifact_paths = {
        "_result_model":  models_dir / "result_model_v1.joblib",
        "_xgb_result_model": models_dir / "result_model_xgb_v1.joblib",
        "_xgb_label_encoder": models_dir / "result_model_xgb_v1_label_encoder.joblib",
        "_home_goals":    models_dir / "home_goals_v1.joblib",
        "_away_goals":    models_dir / "away_goals_v1.joblib",
        "_home_corners":  models_dir / "home_corners_v1.joblib",
        "_away_corners":  models_dir / "away_corners_v1.joblib",
        "_btts_model":    models_dir / "btts_model_v1.joblib",
        "_over25_model":  models_dir / "over25_model_v1.joblib",
        "_over95_corners_model": (
            models_dir / "over95_corners_model_v1.joblib"
        ),
        "_le_league":     data_dir   / "league_encoder.joblib",
        "_le_season":     data_dir   / "season_encoder.joblib",
    }

    # Decode everything first, then publish the globals in one dict update.
    # A reload therefore swaps the whole artifact set at once instead of
    # exposing a half-new/half-old set while joblib.load works through the
    # files (and a file that fails to decode keeps its previous object).
    loaded: dict[str, Any] = {}
    for var_name, path in artifact_paths.items():
        try:
            loaded[var_name] = joblib.load(path)
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
        loaded["_features_df"] = df
        print(f"[predict] Loaded features.csv "
              f"({len(df)} rows)")
    except FileNotFoundError:
        print("[predict] WARNING: features.csv not found")
    except Exception as exc:
        print(f"[predict] WARNING: could not load "
              f"features.csv: {exc}")

    globals().update(loaded)


def _get_runtime_feature_cols() -> list[str]:
    """
    Return the feature list from the loaded model.
    Falls back to the hardcoded FEATURE_COLS if
    the model is not loaded or has no feature info.
    This guarantees predict.py always uses exactly
    the features the model was trained on.
    """
    global _result_model, _xgb_result_model

    model = _xgb_result_model if _xgb_result_model is not None else _result_model

    if model is None:
        return FEATURE_COLS

    try:
        # Pipeline: get the final estimator step
        if hasattr(model, 'steps'):
            final_step = model.steps[-1][1]
        else:
            final_step = model

        if hasattr(final_step, 'feature_names_in_'):
            return list(final_step.feature_names_in_)

        # Try the imputer step which sees raw features
        if hasattr(model, 'steps'):
            imputer = model.steps[0][1]
            if hasattr(imputer, 'feature_names_in_'):
                return list(imputer.feature_names_in_)
    except Exception as exc:
        print(f"[predict] WARNING: could not read feature "
              f"names from model: {exc}")

    return FEATURE_COLS


#: Which module global each model artifact populates. Used by
#: /admin/status, /admin/reload-models and the startup reload so that all
#: three report the same set of loaded models.
_ARTIFACT_GLOBALS: dict[str, str] = {
    "result_model_v1.joblib": "_result_model",
    "result_model_xgb_v1.joblib": "_xgb_result_model",
    "result_model_xgb_v1_label_encoder.joblib": "_xgb_label_encoder",
    "home_goals_v1.joblib": "_home_goals",
    "away_goals_v1.joblib": "_away_goals",
    "home_corners_v1.joblib": "_home_corners",
    "away_corners_v1.joblib": "_away_corners",
    "btts_model_v1.joblib": "_btts_model",
    "over25_model_v1.joblib": "_over25_model",
    "over95_corners_model_v1.joblib": "_over95_corners_model",
}


def _loaded_model_names() -> list[str]:
    """Return the model artifacts currently loaded into memory."""
    return [
        object_name
        for object_name, global_name in _ARTIFACT_GLOBALS.items()
        if globals().get(global_name) is not None
    ]


def _models_ready() -> bool:
    """True when the five core markets can be predicted."""
    return all([
        (_result_model is not None or _xgb_result_model is not None),
        _home_goals    is not None,
        _away_goals    is not None,
        _home_corners  is not None,
        _away_corners  is not None,
    ])


class ReloadInProgress(RuntimeError):
    """Raised when a reload already holds the lock."""


def reload_models_from_supabase() -> dict[str, Any]:
    """Download the pre-built artifacts and load them into memory.

    This is the ONLY pipeline action allowed on the Render service. It streams
    files that GitHub Actions already trained — it never runs builder.py,
    trainer.py or a collector, so it stays fast and flat in memory.

    Downloads are staged in a temporary directory inside ``backend/`` and only
    moved into place once every required object has arrived, then the artifacts
    are (re)loaded. A failed reload therefore leaves the previously served
    models untouched.

    Raises :class:`ReloadInProgress` if another reload holds the lock, and
    ``RuntimeError``/``Exception`` if the download or load fails.
    """
    global _pipeline_running, _pipeline_status, _pipeline_log
    global _pipeline_error, _last_reload_at

    if not _reload_lock.acquire(timeout=RELOAD_LOCK_TIMEOUT_SECONDS):
        raise ReloadInProgress(
            "another reload is already in progress; retry shortly"
        )

    started = time.monotonic()
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=BACKEND_DIR))
    try:
        _pipeline_running = True
        _pipeline_status = "reloading_models"
        _pipeline_log = [
            "[reload] downloading pre-built artifacts from Supabase Storage"
        ]
        _pipeline_error = ""
        print("[reload] Downloading pre-built artifacts from Supabase Storage...")

        downloaded = _download_artifacts_to(staging_dir)
        _install_artifacts(staging_dir, downloaded)
        load_artifacts()

        models_loaded = _loaded_model_names()
        # Every *required* object must have decoded in this run — but the
        # globals may still hold yesterday's object if a file was unreadable,
        # so also check the encoders/features explicitly.
        required_globals = [
            _ARTIFACT_GLOBALS[name]
            for name in artifacts.REQUIRED_DOWNLOAD_OBJECTS
            if name in _ARTIFACT_GLOBALS
        ]
        missing = [
            global_name
            for global_name in required_globals + ["_le_league", "_le_season"]
            if globals().get(global_name) is None
        ]
        if missing or _features_df is None:
            raise RuntimeError(
                "reload incomplete: "
                f"{', '.join(missing) or 'features.csv'} missing after "
                f"download (loaded: {models_loaded or 'none'})"
            )

        elapsed = time.monotonic() - started
        _pipeline_status = "complete"
        _last_reload_at = datetime.now(timezone.utc).isoformat()
        _pipeline_log.append(
            f"[reload] complete in {elapsed:.1f}s — "
            f"{len(models_loaded)} models, {len(_features_df)} feature rows"
        )
        print(f"[reload] Complete in {elapsed:.1f}s — "
              f"{len(models_loaded)} models, {len(_features_df)} feature rows")

        return {
            "features_rows": int(len(_features_df)),
            "models_loaded": models_loaded,
            "reload_seconds": round(elapsed, 2),
            "reloaded_at": _last_reload_at,
        }
    except Exception as exc:
        _pipeline_status = "error_reload"
        _pipeline_error = str(exc)
        _pipeline_log.append(f"[reload] FAILED: {exc}")
        print(f"[reload] FAILED: {exc}")
        raise
    finally:
        _pipeline_running = False
        _reload_lock.release()
        shutil.rmtree(staging_dir, ignore_errors=True)


def startup_reload_models() -> None:
    """Kick off a background artifact reload at application startup.

    Runs in a daemon thread so a slow render/network does not delay the port
    bind (Render kills services that take too long to become healthy). It only
    downloads pre-built artifacts — no builder, trainer or collector code is
    ever executed here.

    Set ``SKIP_STARTUP_MODEL_RELOAD=1`` to disable it (useful for tests).
    """
    if os.environ.get("SKIP_STARTUP_MODEL_RELOAD", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        print("[startup] SKIP_STARTUP_MODEL_RELOAD is set — "
              "using artifacts already on disk")
        return

    def _run() -> None:
        try:
            reload_models_from_supabase()
            print("[startup] Models reloaded from Supabase Storage")
        except ReloadInProgress as exc:
            print(f"[startup] NOTE: {exc}")
        except Exception as exc:  # noqa: BLE001 - never crash the app
            print(f"[startup] WARNING: could not reload models from Supabase "
                  f"Storage: {exc}. The API will answer 503 for prediction "
                  f"endpoints until POST /predict/admin/reload-models "
                  f"succeeds.")

    threading.Thread(
        target=_run, name="startup-model-reload", daemon=True
    ).start()


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


def _build_btts_feature_vector(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
) -> dict[str, float] | None:
    """Build the 17-feature vector expected by the BTTS classifier.

    Returns None when the required columns are not present or
    when no history exists for the teams.
    """
    if df is None or df.empty:
        return None
    # Check all 17 columns exist in the CSV
    missing = [c for c in BTTS_FEATURE_COLS if c not in df.columns and c != "league_encoded"]
    # league_encoded is supplied, but also may exist in CSV; ignore it from missing check
    if missing:
        # If any BTTS-specific rolling columns are missing, cannot build
        # but allow partial if only some FBref columns missing – fill with 0
        # We only fail if the 6 new rate cols are missing
        btts_new = ["home_btts_rate_5", "away_btts_rate_5", "home_cs_rate_5", "away_cs_rate_5", "home_scored_rate_5", "away_scored_rate_5"]
        if any(c in missing for c in btts_new):
            print(f"[predict] WARNING: BTTS features missing in CSV: {missing}")
            return None
    # Most recent rows per role
    try:
        home_rows = df[df["home_team"] == home_team]
        away_rows = df[df["away_team"] == away_team]
        if home_rows.empty and away_rows.empty:
            return None
        home_latest = home_rows.iloc[-1] if not home_rows.empty else None
        away_latest = away_rows.iloc[-1] if not away_rows.empty else None
    except Exception as exc:
        print(f"[predict] WARNING: could not get BTTS team rows: {exc}")
        return None

    vector: dict[str, float] = {}
    for col in BTTS_FEATURE_COLS:
        if col == "league_encoded":
            vector[col] = float(league_encoded)
        elif col.startswith("home_"):
            if home_latest is not None and col in home_latest.index:
                try:
                    vector[col] = float(home_latest.get(col, 0.0) or 0.0)
                except Exception:
                    vector[col] = 0.0
            else:
                vector[col] = 0.0
        elif col.startswith("away_"):
            if away_latest is not None and col in away_latest.index:
                try:
                    vector[col] = float(away_latest.get(col, 0.0) or 0.0)
                except Exception:
                    vector[col] = 0.0
            else:
                vector[col] = 0.0
        else:
            vector[col] = 0.0
    return vector


def _get_btts_probability(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
) -> float | None:
    """Return BTTS probability from the dedicated classifier, or None."""
    global _btts_model
    if _btts_model is None:
        return None
    # Quick check: are the 6 new BTTS rate cols present?
    if df is None or df.empty:
        return None
    required_new = ["home_btts_rate_5", "away_btts_rate_5", "home_cs_rate_5", "away_cs_rate_5", "home_scored_rate_5", "away_scored_rate_5"]
    if any(c not in df.columns for c in required_new):
        return None
    vector = _build_btts_feature_vector(home_team, away_team, league_encoded, df)
    if vector is None:
        return None
    # Ensure all expected columns are filled
    for c in BTTS_FEATURE_COLS:
        if c not in vector:
            vector[c] = 0.0
    try:
        X = pd.DataFrame([vector])[BTTS_FEATURE_COLS]
        proba = _btts_model.predict_proba(X)
        # proba shape (1, 2) for binary
        if proba.shape[1] >= 2:
            return float(proba[0][1])
        else:
            return float(proba[0][0])
    except Exception as exc:
        print(f"[predict] WARNING: BTTS inference failed: {exc}")
        return None


def _build_over25_feature_vector(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
) -> dict[str, float] | None:
    """Build the 18-feature vector expected by the Over 2.5 classifier.

    Returns None when the required columns are not present or
    when no history exists for the teams.
    """
    if df is None or df.empty:
        return None
    # Check all 18 columns exist in the CSV
    missing = [c for c in OVER25_FEATURE_COLS if c not in df.columns and c != "league_encoded"]
    # league_encoded is supplied, but also may exist in CSV; ignore it from missing check
    if missing:
        # If any Over-2.5-specific rolling columns are missing, cannot build
        # but allow partial if only some FBref columns missing – fill with 0
        # We only fail if the 4 new rate cols are missing
        over25_new = ["home_over25_rate_5", "away_over25_rate_5",
                      "home_high_scoring_rate_5", "away_high_scoring_rate_5"]
        if any(c in missing for c in over25_new):
            print(f"[predict] WARNING: Over 2.5 features missing in CSV: {missing}")
            return None
    # Most recent rows per role
    try:
        home_rows = df[df["home_team"] == home_team]
        away_rows = df[df["away_team"] == away_team]
        if home_rows.empty and away_rows.empty:
            return None
        home_latest = home_rows.iloc[-1] if not home_rows.empty else None
        away_latest = away_rows.iloc[-1] if not away_rows.empty else None
    except Exception as exc:
        print(f"[predict] WARNING: could not get Over 2.5 team rows: {exc}")
        return None

    vector: dict[str, float] = {}
    for col in OVER25_FEATURE_COLS:
        if col == "league_encoded":
            vector[col] = float(league_encoded)
        elif col == "elo_difference":
            # Derived: always the home perspective Elo gap, never a raw column
            vector[col] = (
                vector.get("home_elo", 0.0) - vector.get("away_elo", 0.0)
            )
        elif col.startswith("home_"):
            if home_latest is not None and col in home_latest.index:
                try:
                    vector[col] = float(home_latest.get(col, 0.0) or 0.0)
                except Exception:
                    vector[col] = 0.0
            else:
                vector[col] = 0.0
        elif col.startswith("away_"):
            if away_latest is not None and col in away_latest.index:
                try:
                    vector[col] = float(away_latest.get(col, 0.0) or 0.0)
                except Exception:
                    vector[col] = 0.0
            else:
                vector[col] = 0.0
        else:
            vector[col] = 0.0
    return vector


def _get_over25_probability(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
) -> float | None:
    """Return Over 2.5 probability from the dedicated classifier, or None."""
    global _over25_model
    if _over25_model is None:
        return None
    # Quick check: are the 4 new Over 2.5 rate cols present?
    if df is None or df.empty:
        return None
    required_new = ["home_over25_rate_5", "away_over25_rate_5",
                    "home_high_scoring_rate_5", "away_high_scoring_rate_5"]
    if any(c not in df.columns for c in required_new):
        return None
    vector = _build_over25_feature_vector(home_team, away_team, league_encoded, df)
    if vector is None:
        return None
    # Ensure all expected columns are filled
    for c in OVER25_FEATURE_COLS:
        if c not in vector:
            vector[c] = 0.0
    try:
        X = pd.DataFrame([vector])[OVER25_FEATURE_COLS]
        proba = _over25_model.predict_proba(X)
        # proba shape (1, 2) for binary
        if proba.shape[1] >= 2:
            return float(proba[0][1])
        else:
            return float(proba[0][0])
    except Exception as exc:
        print(f"[predict] WARNING: Over 2.5 inference failed: {exc}")
        return None


# The 6 corner-specific rates the classifier cannot be scored without: an
# older features.csv without them means the model was trained on different
# inputs, so the caller must keep the Poisson value instead of guessing.
OVER95_CORNERS_REQUIRED_COLS = [
    "home_corners_for_pg_5", "away_corners_for_pg_5",
    "home_corners_against_pg_5", "away_corners_against_pg_5",
    "home_over95_corners_rate_5", "away_over95_corners_rate_5",
]


def _build_over95_corners_feature_vector(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
) -> dict[str, float] | None:
    """Build the 17-feature vector expected by the Over 9.5 corners model.

    Returns None when the corner-specific rolling columns are not present or
    when no history exists for either team.
    """
    if df is None or df.empty:
        return None
    missing = [
        c for c in OVER95_CORNERS_FEATURE_COLS
        if c not in df.columns and c != "league_encoded"
    ]
    # Only the 6 new corner rates are mandatory — FBref and form columns may
    # legitimately be absent from a partial features run, and are then 0.0.
    if any(c in missing for c in OVER95_CORNERS_REQUIRED_COLS):
        print(f"[predict] WARNING: Over 9.5 corners features missing in CSV: "
              f"{missing}")
        return None
    try:
        home_rows = df[df["home_team"] == home_team]
        away_rows = df[df["away_team"] == away_team]
        if home_rows.empty and away_rows.empty:
            return None
        home_latest = home_rows.iloc[-1] if not home_rows.empty else None
        away_latest = away_rows.iloc[-1] if not away_rows.empty else None
    except Exception as exc:
        print(f"[predict] WARNING: could not get Over 9.5 corners team rows: "
              f"{exc}")
        return None

    vector: dict[str, float] = {}
    for col in OVER95_CORNERS_FEATURE_COLS:
        if col == "league_encoded":
            vector[col] = float(league_encoded)
            continue
        if col.startswith("home_") and home_latest is not None \
                and col in home_latest.index:
            source = home_latest
        elif col.startswith("away_") and away_latest is not None \
                and col in away_latest.index:
            source = away_latest
        else:
            vector[col] = 0.0
            continue
        try:
            value = float(source.get(col, 0.0) or 0.0)
        except Exception:
            value = 0.0
        # A NaN here would propagate through the scaler and make the model
        # raise, so an unrecorded corner rate is scored as 0 — the same value
        # builder.py gives a team with no prior matches.
        vector[col] = value if np.isfinite(value) else 0.0
    return vector


def _get_over95_corners_probability(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
) -> float | None:
    """P(total corners >= 10) from the dedicated classifier, or None."""
    global _over95_corners_model
    if _over95_corners_model is None:
        return None
    if df is None or df.empty:
        return None
    if any(c not in df.columns for c in OVER95_CORNERS_REQUIRED_COLS):
        return None
    vector = _build_over95_corners_feature_vector(
        home_team, away_team, league_encoded, df
    )
    if vector is None:
        return None
    # Ensure all expected columns are filled
    for c in OVER95_CORNERS_FEATURE_COLS:
        if c not in vector:
            vector[c] = 0.0
    try:
        X = pd.DataFrame([vector])[OVER95_CORNERS_FEATURE_COLS]
        proba = _over95_corners_model.predict_proba(X)
        # proba shape (1, 2) for binary
        if proba.shape[1] >= 2:
            return float(proba[0][1])
        else:
            return float(proba[0][0])
    except Exception as exc:
        print(f"[predict] WARNING: Over 9.5 corners inference failed: {exc}")
        return None


def _get_xgb_1x2_probabilities(
    feature_vector: Any,
) -> dict[str, float] | None:
    """
    Return {"H": prob, "D": prob, "A": prob} from XGBoost 1X2 model,
    or None if the model or label encoder is not loaded.
    Uses _xgb_label_encoder.classes_ to map predict_proba columns to H/D/A.
    """
    global _xgb_result_model, _xgb_label_encoder

    if _xgb_result_model is None or _xgb_label_encoder is None:
        return None

    try:
        runtime_cols = _get_runtime_feature_cols()

        if isinstance(feature_vector, pd.DataFrame):
            missing = [c for c in runtime_cols if c not in feature_vector.columns]
            if missing:
                df = feature_vector.copy()
                for c in missing:
                    df[c] = 0.0
                X = df[runtime_cols]
            else:
                X = feature_vector[runtime_cols]
        elif isinstance(feature_vector, pd.Series):
            row = feature_vector.to_dict()
            for c in runtime_cols:
                if c not in row:
                    row[c] = 0.0
            X = pd.DataFrame([row])[runtime_cols]
        elif isinstance(feature_vector, dict):
            vec = dict(feature_vector)
            for c in runtime_cols:
                if c not in vec:
                    vec[c] = 0.0
            X = pd.DataFrame([vec])[runtime_cols]
        elif isinstance(feature_vector, (list, np.ndarray)):
            arr = np.asarray(feature_vector, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < len(runtime_cols):
                padded = np.zeros((arr.shape[0], len(runtime_cols)), dtype=float)
                padded[:, :arr.shape[1]] = arr
                X = pd.DataFrame(padded, columns=runtime_cols)
            else:
                X = pd.DataFrame(arr[:, :len(runtime_cols)], columns=runtime_cols)
        else:
            print(f"[predict] WARNING: unexpected feature_vector type for XGBoost: {type(feature_vector)}")
            return None

        proba = _xgb_result_model.predict_proba(X)[0]
        classes = list(_xgb_label_encoder.classes_)
        result_map = dict(zip(classes, proba))

        return {
            "H": round(float(result_map.get("H", 0.0)), 4),
            "D": round(float(result_map.get("D", 0.0)), 4),
            "A": round(float(result_map.get("A", 0.0)), 4),
        }
    except Exception as exc:
        print(f"[predict] WARNING: XGBoost 1X2 inference failed: {exc}")
        return None


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


def _blend_predictions(
    formula_result: dict[str, Any],
    ml_result: dict[str, Any],
    xgb_probs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Blend formula_v1 and ML prediction outputs using weighted averaging
    per market, using formula_result as the base for all other fields,
    and re-derive Poisson-based markets from blended expected goals/corners.
    """
    blended = copy.deepcopy(formula_result)
    blended["result"] = dict(formula_result.get("result") or {})
    blended["goals"] = dict(formula_result.get("goals") or {})
    blended["corners"] = dict(formula_result.get("corners") or {})

    # Determine 1X2 probabilities for ML side: XGBoost or Logistic fallback
    if xgb_probs is not None:
        ml_home = xgb_probs.get("H", 0.0)
        ml_draw = xgb_probs.get("D", 0.0)
        ml_away = xgb_probs.get("A", 0.0)
        xgb_active = True
    else:
        xgb_active = bool(ml_result.get("xgb_1x2_active", False))
        ml_home = ml_result.get("result", {}).get("home_win", 0.0)
        ml_draw = ml_result.get("result", {}).get("draw", 0.0)
        ml_away = ml_result.get("result", {}).get("away_win", 0.0)

    blended["result"]["home_win"] = round(
        formula_result["result"]["home_win"] * BLEND_FORMULA_1X2 +
        ml_home * BLEND_ML_1X2, 4)
    blended["result"]["draw"] = round(
        formula_result["result"]["draw"] * BLEND_FORMULA_1X2 +
        ml_draw * BLEND_ML_1X2, 4)
    blended["result"]["away_win"] = round(
        formula_result["result"]["away_win"] * BLEND_FORMULA_1X2 +
        ml_away * BLEND_ML_1X2, 4)

    blended["goals"]["expected_home"] = round(
        formula_result["goals"]["expected_home"] * BLEND_FORMULA_GOALS +
        ml_result["goals"]["expected_home"] * BLEND_ML_GOALS, 4)
    blended["goals"]["expected_away"] = round(
        formula_result["goals"]["expected_away"] * BLEND_FORMULA_GOALS +
        ml_result["goals"]["expected_away"] * BLEND_ML_GOALS, 4)

    blended["corners"]["expected_home"] = round(
        formula_result["corners"]["expected_home"] * BLEND_FORMULA_CRNRS +
        ml_result["corners"]["expected_home"] * BLEND_ML_CRNRS, 4)
    blended["corners"]["expected_away"] = round(
        formula_result["corners"]["expected_away"] * BLEND_FORMULA_CRNRS +
        ml_result["corners"]["expected_away"] * BLEND_ML_CRNRS, 4)

    # Re-derive Poisson-based markets from blended expected goals/corners
    exp_home_goals = blended["goals"]["expected_home"]
    exp_away_goals = blended["goals"]["expected_away"]
    exp_total_goals = exp_home_goals + exp_away_goals

    blended["goals"]["over_1_5"] = round(_poisson_over(exp_total_goals, 1.5), 4)
    blended["goals"]["over_2_5"] = round(_poisson_over(exp_total_goals, 2.5), 4)
    blended["goals"]["over_3_5"] = round(_poisson_over(exp_total_goals, 3.5), 4)
    blended["goals"]["btts"] = round(_poisson_btts(exp_home_goals, exp_away_goals), 4)

    exp_home_corners = blended["corners"]["expected_home"]
    exp_away_corners = blended["corners"]["expected_away"]
    exp_total_corners = round(exp_home_corners + exp_away_corners, 4)

    blended["corners"]["expected_total"] = exp_total_corners
    blended["corners"]["over_8_5"] = round(_poisson_over(exp_total_corners, 8.5), 4)
    blended["corners"]["over_9_5"] = round(_poisson_over(exp_total_corners, 9.5), 4)
    blended["corners"]["over_10_5"] = round(_poisson_over(exp_total_corners, 10.5), 4)
    blended["corners"]["over_11_5"] = round(_poisson_over(exp_total_corners, 11.5), 4)

    blended["prediction_method"] = "blended"
    blended["xgb_1x2_active"] = xgb_active
    return blended


@router.post("/match")
def predict_match(req: MatchRequest) -> dict[str, Any]:
    """
    Run all prediction models for a given fixture
    and return full market probabilities.
    """
    # 1. Try formula_v1 -> store result as formula_result (or None if unavailable/error)
    formula_result = None
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
            else:
                print("[predict] Formula engine "
                      "returned None")
        except Exception as exc:
            print(f"[predict] Formula engine "
                  f"error: {exc}")
            formula_result = None

    # 2. Try ML -> store result as ml_result (or None if unavailable/error)
    models_ready = all([
        (_result_model is not None or _xgb_result_model is not None),
        _home_goals   is not None,
        _away_goals   is not None,
        _home_corners is not None,
        _away_corners is not None,
    ])

    ml_result = None
    if models_ready and _features_df is not None:
        try:
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

            # 1X2 result probabilities: XGBoost if loaded, else fallback to Logistic regression
            xgb_probs = _get_xgb_1x2_probabilities(X)
            xgb_active = xgb_probs is not None

            if xgb_active:
                home_win_prob = xgb_probs["H"]
                draw_prob     = xgb_probs["D"]
                away_win_prob = xgb_probs["A"]
            elif _result_model is not None:
                result_proba = _result_model.predict_proba(X)[0]
                classes      = list(_result_model.classes_)
                result_map   = dict(zip(classes, result_proba))

                home_win_prob = round(float(result_map.get("H", 0.0)), 4)
                draw_prob     = round(float(result_map.get("D", 0.0)), 4)
                away_win_prob = round(float(result_map.get("A", 0.0)), 4)
            else:
                home_win_prob = draw_prob = away_win_prob = 0.0

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
            over_corners_9_5  = _poisson_over(exp_total_corners, 9.5)
            over_corners_10_5 = _poisson_over(exp_total_corners, 10.5)
            over_corners_11_5 = _poisson_over(exp_total_corners, 11.5)

            ml_result = {
                "home_team": req.home_team,
                "away_team": req.away_team,
                "league":    req.league,
                "season":    req.season,
                "result": {
                    "home_win": home_win_prob,
                    "draw":     draw_prob,
                    "away_win": away_win_prob,
                },
                "xgb_1x2_active": xgb_active,
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
                    # The frontend reads expected_total / over_9_5 / over_11_5 and the
                    # formula engine publishes all three; the ML path used to omit
                    # them, which left those rows blank for every non-formula
                    # prediction (and left no over_9_5 for fixtures.py to store).
                    "expected_total": round(exp_total_corners, 3),
                    "over_8_5":       round(over_corners_8_5, 4),
                    "over_9_5":       round(over_corners_9_5, 4),
                    "over_10_5":      round(over_corners_10_5, 4),
                    "over_11_5":      round(over_corners_11_5, 4),
                },
                "feature_vector_used": {
                    k: round(v, 4) for k, v in vector.items()
                },
            }
        except Exception as exc:
            print(f"[predict] ML prediction error: {exc}")
            ml_result = None

    # 3-6. Combine or select available engine output
    if formula_result is not None and ml_result is not None:
        final_result = _blend_predictions(formula_result, ml_result)
        final_result["prediction_method"] = "blended"
        print(f"[predict] Blended formula_v1 + ML: "
              f"{req.home_team} vs {req.away_team}")
    elif formula_result is not None:
        final_result = formula_result
        final_result["prediction_method"] = "formula_only"
        final_result["xgb_1x2_active"] = False
        print(f"[predict] Using formula_only: "
              f"{req.home_team} vs {req.away_team}")
    elif ml_result is not None:
        final_result = ml_result
        final_result["prediction_method"] = "ml_only"
        final_result["xgb_1x2_active"] = ml_result.get("xgb_1x2_active", False)
        print(f"[predict] Using ml_only: "
              f"{req.home_team} vs {req.away_team}")
    else:
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
        raise HTTPException(
            status_code=503,
            detail="Neither formula engine nor ML models could produce a prediction.",
        )

    # The formula output carries its own engine name, but after blending (or
    # falling back to ML) this response field must identify the method that
    # produced the final prediction.
    final_result["engine"] = {
        "blended": "blended",
        "formula_only": "formula_v1",
        "ml_only": "ml",
    }[final_result["prediction_method"]]

    # Specialized classifiers override their specific fields AFTER blend
    if _btts_model is not None and _features_df is not None:
        try:
            league_enc_for_btts = _encode_label(_le_league, req.league)
            btts_prob = _get_btts_probability(
                req.home_team, req.away_team, league_enc_for_btts, _features_df
            )
            if btts_prob is not None:
                if "goals" in final_result and isinstance(final_result["goals"], dict):
                    final_result["goals"]["btts"] = round(float(btts_prob), 4)
                    print(f"[predict] BTTS overridden by classifier: {btts_prob:.4f}")
        except Exception as exc:
            print(f"[predict] WARNING: BTTS override failed: {exc}")

    if _over25_model is not None and _features_df is not None:
        try:
            league_enc_for_over25 = _encode_label(_le_league, req.league)
            over25_prob = _get_over25_probability(
                req.home_team, req.away_team, league_enc_for_over25, _features_df
            )
            if over25_prob is not None:
                if "goals" in final_result and isinstance(final_result["goals"], dict):
                    final_result["goals"]["over_2_5"] = round(float(over25_prob), 4)
                    print(f"[predict] Over 2.5 overridden by classifier: {over25_prob:.4f}")
        except Exception as exc:
            print(f"[predict] WARNING: Over 2.5 override failed: {exc}")

    if _over95_corners_model is not None and _features_df is not None:
        try:
            league_enc_for_corners = _encode_label(_le_league, req.league)
            corners95_prob = _get_over95_corners_probability(
                req.home_team, req.away_team, league_enc_for_corners, _features_df
            )
            if corners95_prob is not None:
                if "corners" in final_result and isinstance(final_result["corners"], dict):
                    final_result["corners"]["over_9_5"] = round(float(corners95_prob), 4)
                    print(f"[predict] Over 9.5 corners overridden by classifier: {corners95_prob:.4f}")
        except Exception as exc:
            print(f"[predict] WARNING: Over 9.5 corners override failed: {exc}")

    if "xgb_1x2_active" not in final_result:
        final_result["xgb_1x2_active"] = bool(
            ml_result.get("xgb_1x2_active", False) if ml_result else False
        )

    return final_result


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


# --------------------------------------------------------------------------- #
# Training moved to GitHub Actions — heavy endpoints are permanently gone
# --------------------------------------------------------------------------- #
# PRs #78/#80/#81 grew the training set from ~6,776 to 15,000+ rows and the
# builder no longer fits in Render's 512 MB free tier: the service was OOM-killed
# mid-pipeline and restarted by Render. builder.py and trainer.py now run in
# .github/workflows/daily_fbref_update.yml on a 7 GB ubuntu-latest runner.
#
# These two routes are kept (instead of deleted) so any leftover cron job, curl
# script or bookmark gets an explicit 410 Gone instead of a 404, and — most
# importantly — so nothing can ever spawn the heavy pipeline on Render again.
@router.post("/admin/run-pipeline", status_code=410)
def run_pipeline(secret: str = "") -> JSONResponse:  # noqa: ARG001 - legacy arg
    """Gone: the full pipeline now runs in GitHub Actions."""
    return JSONResponse(status_code=410, content=_PIPELINE_MOVED_RESPONSE)


@router.post("/admin/run-pipeline-fast", status_code=410)
def run_pipeline_fast(secret: str = "") -> JSONResponse:  # noqa: ARG001 - legacy arg
    """Gone: FBref + builder + trainer now run in GitHub Actions."""
    return JSONResponse(status_code=410, content=_PIPELINE_MOVED_RESPONSE)


@router.post(
    "/admin/reload-models",
    responses={
        403: {"description": "Invalid or missing PIPELINE_SECRET"},
        409: {"description": "Another reload is already in progress"},
        503: {"description": "Download or load failed — previous models stay live"},
    },
)
def reload_models(secret: str = "") -> dict[str, Any]:
    """Download the pre-built artifacts from Supabase Storage into memory.

    The lightweight half of the daily GitHub Actions handshake: the workflow
    uploads freshly trained models to the ``models`` bucket, then calls this
    endpoint. It only downloads files that already exist — it never runs
    builder, trainer or a collector — so it completes in seconds.

    Requires ``PIPELINE_SECRET`` like the other write endpoints.

    Responses
    ---------
    200 ``{"status": "reloaded", "features_rows": N, "models_loaded": [...]}``
    403 invalid/missing secret
    409 another reload is in progress (retry shortly)
    503 the download or the load failed; previously loaded models stay live
    """
    expected = os.getenv("PIPELINE_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing secret key.",
        )

    try:
        report = reload_models_from_supabase()
    except ReloadInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        raise HTTPException(
            status_code=503,
            detail=f"model reload failed: {exc}",
        ) from exc

    return {"status": "reloaded", **report}


@router.get("/admin/status")
def pipeline_status() -> dict:
    """
    Check which artifacts are loaded and when they were last refreshed.

    ``pipeline_status`` is ``complete`` once a reload has loaded a full set of
    models. It never reports a running builder/trainer: training happens in
    GitHub Actions, and this service only reloads pre-built artifacts.
    """
    return {
        "pipeline_mode":    "github_actions",
        "pipeline_running": _pipeline_running,
        "pipeline_status":  _pipeline_status,
        "pipeline_log":     _pipeline_log[-10:],
        "pipeline_error":   _pipeline_error,
        "last_reload_at":   _last_reload_at,
        "result_model":  _result_model  is not None,
        "xgb_result_model": _xgb_result_model is not None,
        "home_goals":    _home_goals    is not None,
        "away_goals":    _away_goals    is not None,
        "home_corners":  _home_corners  is not None,
        "away_corners":  _away_corners  is not None,
        "btts_model":    _btts_model    is not None,
        "over25_model":  _over25_model  is not None,
        "over95_corners_model": _over95_corners_model is not None,
        "le_league":     _le_league     is not None,
        "le_season":     _le_season     is not None,
        "features_df":   _features_df   is not None,
        "features_rows": len(_features_df) if _features_df is not None else 0,
        "models_loaded": _loaded_model_names(),
        "ready": _models_ready(),
        "blend_weights": {
            "1x2": {"formula": BLEND_FORMULA_1X2, "ml": BLEND_ML_1X2},
            "goals": {"formula": BLEND_FORMULA_GOALS, "ml": BLEND_ML_GOALS},
            "corners": {"formula": BLEND_FORMULA_CRNRS, "ml": BLEND_ML_CRNRS},
            "btts": "classifier_100pct",
            "over25": "classifier_100pct",
            "over95_corners": "classifier_100pct",
        },
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


@router.get("/admin/fix-bsa-label")
def fix_bsa_label(secret: str = "", dry_run: bool = False) -> dict:
    """
    Apply the one-off ``daily_matches`` label repair over HTTP, for a
    deployment that has no shell: sets every Brazilian fixture still stored as
    ``BEL-Jupiler Pro League`` to ``BRA-Série A``.

    Requires PIPELINE_SECRET, like the other write endpoints. Idempotent —
    a second call reports ``"updated": 0``. Pass ``dry_run=true`` to list the
    rows without writing anything.

    GET (not POST) so the repair can be triggered straight from a browser
    address bar on a phone, where issuing a POST is not practical.
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
