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
import logging
import math
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
from pydantic import BaseModel, Field
from scipy.stats import poisson
from supabase import create_client


def compute_first_goal_probs(
    lambda_home: float, lambda_away: float,
) -> dict[str, float]:
    """First-goal probabilities from two independent Poisson processes."""
    if lambda_home <= 0 and lambda_away <= 0:
        return {"home": 0.0, "away": 0.0, "no_goal": 1.0}

    total_rate = lambda_home + lambda_away
    p_no_goal = math.exp(-total_rate)
    p_at_least_one = 1.0 - p_no_goal
    if (
        math.isinf(total_rate)
        and math.isfinite(lambda_home)
        and math.isfinite(lambda_away)
    ):
        # The sum of two finite rates can overflow. Scaling both by the same
        # factor preserves the requested rate ratios without overflowing.
        scale = max(lambda_home, lambda_away)
        home_scaled, away_scaled = lambda_home / scale, lambda_away / scale
        home_share = home_scaled / (home_scaled + away_scaled)
        away_share = away_scaled / (home_scaled + away_scaled)
    else:
        home_share = lambda_home / total_rate
        away_share = lambda_away / total_rate
    probabilities = {
        "home": round(p_at_least_one * home_share, 4),
        "away": round(p_at_least_one * away_share, 4),
        "no_goal": round(p_no_goal, 4),
    }
    if not math.isclose(
        sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=0.001
    ):
        logging.getLogger(__name__).warning(
            "First-goal probabilities do not sum to one: %s", probabilities
        )
    return probabilities


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

# The admin endpoints below import the database layer from inside the request
# handler, i.e. by top-level module name. ``database`` only resolves while
# backend/ is on sys.path, which is not guaranteed when the app is imported
# from another directory (``uvicorn backend.main:app`` from the repository
# root, a script that mounts the app, ...). ``/admin/fbref-check`` used to
# insert the path for itself inside the handler; do it once, here, for every
# endpoint.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Shared bucket manifest — the trainer uploads exactly what this file lists.
import model_artifacts as artifacts  # noqa: E402  (needs the sys.path above)
import fbref_scrape_runner  # noqa: E402  (background StatsBomb->FBref aggregator on Render)

# Feature definitions and assembly. The column lists are the SAME objects the
# trainer fits on (``features/feature_columns.py``), and the feature *values*
# come from the builder's own functions (``features/inference.py``), so the
# training and inference paths cannot drift apart.
from features import builder as feature_builder  # noqa: E402
from features import inference as feature_inference  # noqa: E402
from features.feature_columns import (  # noqa: E402
    BTTS_FEATURE_COLS,
    FEATURE_COLS,
    OVER25_FEATURE_COLS,
    OVER95_CORNERS_FEATURE_COLS,
    OVER95_CORNERS_REQUIRED_COLS,
    validate_feature_column_counts,
)

validate_feature_column_counts()

# A reload only downloads pre-built files; if another reload is already in
# flight (e.g. the startup reload racing the GitHub Actions callback) wait this
# long for it before answering 409. Kept well under the endpoint's 60 s budget.
RELOAD_LOCK_TIMEOUT_SECONDS = 25

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

# The four model input lists are imported above from
# ``features/feature_columns.py`` — the single source of truth shared with
# ``models/trainer.py``. They used to be copy-pasted here, which is exactly how
# the inference path drifted from the training path.


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
    global _history_source, _history, _elo_cache, _elo_lookup_failures
    global _elo_cache_loaded, _elo_latest_cache, _elo_cache_error

    # A reload ships a new features.csv; drop the reconstructed histories and
    # the Elo cache so the next prediction rebuilds them from fresh data.
    _history_source = None
    _history = {}
    _elo_latest_cache = {}
    _elo_cache = {}
    _elo_lookup_failures = 0
    _elo_cache_loaded = False
    _elo_cache_error = None

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
        _refresh_prediction_caches()

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
    # A live fixture belongs to the season it is played in, and the encoder is
    # rebuilt from that season's rows every day. The old fixed "2024-2025"
    # default silently encoded a two-year-old season for any caller that did
    # not send one, feeding the model a value that did not describe the match.
    season: str = Field(default_factory=feature_inference.current_season_label)


def _encode_label(encoder, value: str, fallback: int = 0) -> int:
    """Encode a label with the encoder the builder fitted (never raises).

    Thin wrapper over :func:`features.inference.encode_label` so the trainer's
    encoder and the API cannot apply different rules.
    """
    return feature_inference.encode_label(encoder, value, fallback=fallback)


# --------------------------------------------------------------------------- #
# Inference feature assembly
# --------------------------------------------------------------------------- #
# The features a live fixture is scored on are built by the builder's own
# functions (see features/inference.py): the same rolling windows, venue
# splits, Elo lookup and H2H window as every training row. Reading the
# "latest feature row" of a team out of features.csv — what this module used
# to do — produced stale and opponent-independent values.
#
# No team-season snapshot table is read here. Those columns were zero for
# every live fixture and are no longer part of the model contract; the table
# itself is still maintained by the dormant admin endpoints below.


def _count_fbref_team_stats_rows() -> int:
    """Return exact row count in ``fbref_team_stats``, or 0 when unavailable.

    Only used by the dormant admin scrape-status endpoint below. No prediction
    path reads this table.
    """
    try:
        from database import FBrefTeamStats, SessionLocal

        db = SessionLocal()
        try:
            return int(db.query(FBrefTeamStats).count())
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        return 0


def _refresh_prediction_caches() -> None:
    """Drop cached histories / Elo ratings / formula stats after a reload."""
    global _elo_lookup_failures, _history_source, _history
    global _elo_cache, _elo_cache_loaded, _elo_latest_cache, _elo_cache_error
    _elo_lookup_failures = 0
    _history_source = None
    _history = {}
    _elo_latest_cache = {}
    _elo_cache = {}
    _elo_cache_loaded = False
    _elo_cache_error = None
    try:
        from models import formula_engine as _fe

        if hasattr(_fe, "invalidate_cache"):
            _fe.invalidate_cache()
        elif hasattr(_fe, "_STATS_CACHE"):
            _fe._STATS_CACHE.clear()
    except Exception:  # noqa: BLE001
        pass


# Backwards-compatible callback name for the dormant /admin/run-fbref-scrape
# endpoint. It no longer loads a stats snapshot — only caches are reset.
_refresh_fbref_caches = _refresh_prediction_caches


def _on_fbref_scrape_state(next_state: str, err: str | None) -> None:
    """Update router status tracking as the background FBref scrape progresses."""
    global _pipeline_status, _pipeline_error
    _pipeline_status = next_state
    _pipeline_error = err or ""
    _pipeline_log.append(
        f"[fbref-scrape] {next_state}" + (f" — {err}" if err else "")
    )


_history_source = None
_history: dict[str, pd.DataFrame] = {}


def _inference_history(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Match / Elo history rebuilt from the loaded features CSV.

    ``df`` defaults to the loaded ``_features_df``. Cached until the artifacts
    are reloaded (or a different frame is passed in): rebuilding both frames
    per request would waste time for no benefit.
    """
    global _history_source, _history

    source = _features_df if df is None else df
    if source is None or source.empty:
        return {
            "matches": feature_inference.empty_match_history(),
            "elo": feature_inference.empty_elo_history(),
        }

    if _history_source is source:
        return _history

    _history = {
        "matches": feature_inference.match_history_from_features(source),
        "elo": feature_inference.elo_history_from_features(source),
    }
    _history_source = source
    return _history


#: Columns a specialist classifier cannot be scored without. An older
#: features.csv without them means the model was trained on different inputs,
#: so the caller keeps the Poisson value instead of guessing.
BTTS_REQUIRED_COLS = [
    "home_btts_rate_5", "away_btts_rate_5",
    "home_cs_rate_5", "away_cs_rate_5",
    "home_scored_rate_5", "away_scored_rate_5",
]

OVER25_REQUIRED_COLS = [
    "home_over25_rate_5", "away_over25_rate_5",
    "home_high_scoring_rate_5", "away_high_scoring_rate_5",
]


#: Features derived from shot counts. A features CSV written before the raw
#: match columns were added cannot replay them, so each one falls back to the
#: value stored on the team's latest row instead of silently becoming 0.
_SHOT_DERIVED_FEATURES: tuple[tuple[str, str], ...] = (
    ("home", "home_form_shots_5"),
    ("home", "home_form_shots_ot_5"),
    ("home", "home_home_shots_5"),
    ("away", "away_form_shots_5"),
    ("away", "away_form_shots_ot_5"),
    ("away", "away_away_shots_5"),
)

_SHOT_RAW_COLUMNS = (
    "home_shots", "away_shots", "home_shots_ot", "away_shots_ot",
)


def _latest_stored_value(
    df: pd.DataFrame, team: str, role: str, column: str
) -> float | None:
    """Value of ``column`` on the team's most recent row in that role."""
    team_column = f"{role}_team"
    if df is None or column not in df.columns or team_column not in df.columns:
        return None
    rows = df.loc[df[team_column] == team]
    if rows.empty:
        return None
    value = pd.to_numeric(rows.iloc[-1].get(column), errors="coerce")
    return None if pd.isna(value) else float(value)


def _apply_legacy_shot_fallback(
    features: dict[str, float],
    home_team: str,
    away_team: str,
    df: pd.DataFrame,
) -> dict[str, float]:
    """Keep shot-form features populated when the CSV lacks raw shot counts."""
    if df is None:
        return features
    if all(column in df.columns for column in _SHOT_RAW_COLUMNS):
        return features
    for role, column in _SHOT_DERIVED_FEATURES:
        team = home_team if role == "home" else away_team
        value = _latest_stored_value(df, team, role, column)
        if value is not None:
            features[column] = value
    return features


#: Startup-warmed cache of ``normalised team name -> latest rating``. Elo is a
#: rolling rating, so these entries are date-independent and are the first
#: thing a live lookup consults (diagnostics reports them as cache entries).
_elo_latest_cache: dict[str, float] = {}
_elo_cache_loaded = False
#: Per-team lookup results, keyed by ``(team, cutoff date)``. Kept separate
#: from the warm cache because a historical/backtest lookup is date-sensitive.
_elo_cache: dict[tuple[str, str], float | None] = {}
_ELO_CACHE_LIMIT = 4096
_elo_lookup_failures = 0
_ELO_MAX_FAILURES = 3
#: Last error raised while warming the cache, or None. A warm failure used to
#: be swallowed into a log line only, so a missing/unreachable ``elo_ratings``
#: table still reported ``elo_lookup: "database"`` with 0 entries and nothing
#: in the response explained why. Diagnostics surfaces this string instead.
_elo_cache_error: str | None = None


def warm_elo_cache(force: bool = False) -> int:
    """Load the latest rating of every team into the process-local cache.

    Elo is a rolling rating, so no season/league/match-date filter is applied.
    The query is strictly read-only; a failure is non-fatal because the
    per-team lookup still works. Returns the number of cached entries — the
    number of *teams that actually have a rating*, which is what diagnostics
    reports as ``elo_cache_entries``.

    An exception is still caught (the app must keep serving) but it is now
    recorded in :data:`_elo_cache_error` rather than discarded, so a missing
    table is visible in ``/predict/admin/diagnostics`` instead of looking like
    a healthy database that happens to hold nothing.
    """
    global _elo_latest_cache, _elo_cache_loaded, _elo_lookup_failures
    global _elo_cache_error

    if _elo_cache_loaded and not force:
        return len(_elo_latest_cache)
    try:
        ratings = feature_builder.load_latest_elo_ratings()
    except Exception as exc:  # noqa: BLE001 - never fail startup / diagnostics
        _elo_cache_error = f"{type(exc).__name__}: {exc}"
        print(f"[predict] WARNING: could not warm Elo cache ({exc}); "
              "falling back to per-team lookups")
        _elo_cache_loaded = True
        return len(_elo_latest_cache)
    _elo_latest_cache = dict(ratings)
    _elo_cache_loaded = True
    _elo_lookup_failures = 0
    _elo_cache_error = None
    if not _elo_latest_cache:
        # Not an exception, but equally fatal to the model input: the table is
        # reachable and holds nothing, so every team resolves to the 1500
        # default. Name it so the empty table cannot pass for a healthy one.
        _elo_cache_error = (
            "elo_ratings is reachable but contains no ratings — run "
            "`python data/elo.py` to rebuild them from the matches table"
        )
        print(f"[predict] WARNING: {_elo_cache_error}")
    print(f"[predict] Loaded {len(_elo_latest_cache)} Elo ratings from database")
    return len(_elo_latest_cache)


def _elo_rating_lookup(team: str, before_date: Any) -> float | None:
    """Latest ``elo_ratings`` value for ``team``.

    Elo is a rolling rating: the builder returns the most recent stored value
    rather than filtering by ``before_date`` (kept for API compatibility).
    Kept as its own function so the parity test can substitute the synthetic
    rating history for the database (no DB access in tests). Returns None for
    a genuine miss; ``raise_on_error`` keeps connection errors distinguishable.
    """
    return feature_builder.get_latest_elo_before(
        team, before_date, raise_on_error=True
    )


def _elo_lookup(team: str, before_date: Any) -> float | None:
    """Cached, fail-safe wrapper around :func:`_elo_rating_lookup`.

    A cache hit (populated at startup from the authoritative table) wins. On a
    miss the per-team query resolves name variants and schema drift; only a
    *raised* exception counts as a database failure, and after a few of those
    the lookup stops trying for this process (a reload resets it) so a broken
    connection cannot add latency to every prediction.
    """
    global _elo_lookup_failures

    normalised = feature_builder.normalise_elo_team_name(team)
    if normalised in _elo_latest_cache:
        return _elo_latest_cache[normalised]

    key = (team, str(before_date))
    if key in _elo_cache:
        return _elo_cache[key]
    if _elo_lookup_failures >= _ELO_MAX_FAILURES:
        return None
    try:
        value = _elo_rating_lookup(team, before_date)
    except Exception as exc:  # noqa: BLE001 - fall back to the reconstructed history
        _elo_lookup_failures += 1
        print(f"[predict] WARNING: Elo lookup failed for {team} ({exc}); "
              f"using the ratings stored in features.csv")
        return None
    # A lookup that finds nothing is legitimate (the team has no rating yet);
    # only a raised exception counts as a failure.
    if len(_elo_cache) >= _ELO_CACHE_LIMIT:
        _elo_cache.clear()
    _elo_cache[key] = value
    return value


def _inference_feature_row(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    league: str | None = None,
    season: str | None = None,
    before_date: Any = None,
) -> dict[str, float]:
    """All model inputs for one fixture, exactly as the builder computes them.

    ``df`` is the loaded features CSV; ``before_date`` defaults to today, so
    only matches from earlier days are used (the builder's cutoff rule).
    """
    if df is None:
        return {}
    history = _inference_history(df)
    try:
        features = feature_inference.build_inference_features(
            history["matches"],
            history["elo"],
            home_team,
            away_team,
            before_date=before_date,
            league=league,
            season=season,
            elo_lookup=_elo_lookup,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a warning, not a 500
        print(f"[predict] WARNING: could not build features for "
              f"{home_team} vs {away_team}: {exc}")
        return {}
    return _apply_legacy_shot_fallback(features, home_team, away_team, df)


def _finite(value: Any, default: float = 0.0) -> float:
    """``float(value)`` clamped to a finite number for model input."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _build_feature_vector(
    home_team: str,
    away_team: str,
    league_encoded: int,
    season_encoded: int,
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    league: str | None = None,
    season: str | None = None,
    before_date: Any = None,
) -> dict[str, float]:
    """Build the full feature vector for a prediction request.

    Every value comes from the builder's own feature functions, so a live
    fixture and the equivalent historical training row agree exactly (see
    ``tests/test_feature_parity.py``). ``league``/``season`` are the fixture's
    strings, kept for API parity; ``before_date`` overrides
    today's cutoff (tests and backfills only).
    """
    features = _inference_feature_row(
        df, home_team, away_team, league=league, season=season,
        before_date=before_date,
    )
    features["league_encoded"] = float(league_encoded)
    features["season_encoded"] = float(season_encoded)

    cols_to_use = feature_cols if feature_cols is not None else FEATURE_COLS
    return {
        column: _finite(features.get(column, 0.0))
        for column in cols_to_use
    }


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


def _model_feature_vector(
    model_name: str,
    columns: list[str],
    required: list[str],
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
    league: str | None = None,
    season: str | None = None,
) -> dict[str, float] | None:
    """Feature vector for a specialist classifier, built by the builder.

    ``required`` columns must exist in the CSV and be computable: they are what
    makes the classifier's inputs meaningful. Everything else (form, Elo) may
    legitimately be absent and is then 0.0 — the same value the builder writes
    for a team with no such history.

    Returns None when ``df`` cannot support the model, so the caller keeps the
    Poisson estimate instead of feeding the classifier zeros it never saw.
    """
    if df is None or df.empty:
        return None
    missing = [c for c in required if c not in df.columns
               and c != "league_encoded"]
    if missing:
        print(f"[predict] WARNING: {model_name} features missing in CSV: "
              f"{missing}")
        return None

    features = _inference_feature_row(
        df, home_team, away_team, league=league, season=season
    )
    if not features:
        return None
    features["league_encoded"] = float(league_encoded)
    return {
        column: _finite(features.get(column, 0.0))
        for column in columns
    }


def _build_btts_feature_vector(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
    *,
    league: str | None = None,
    season: str | None = None,
) -> dict[str, float] | None:
    """Build the 13-feature vector expected by the BTTS classifier."""
    return _model_feature_vector(
        "BTTS", BTTS_FEATURE_COLS, BTTS_REQUIRED_COLS,
        home_team, away_team, league_encoded, df,
        league=league, season=season,
    )


def _get_btts_probability(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
    *,
    league: str | None = None,
    season: str | None = None,
) -> float | None:
    """Return BTTS probability from the dedicated classifier, or None."""
    global _btts_model
    if _btts_model is None:
        return None
    vector = _build_btts_feature_vector(
        home_team, away_team, league_encoded, df,
        league=league, season=season,
    )
    if vector is None:
        return None
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
    *,
    league: str | None = None,
    season: str | None = None,
) -> dict[str, float] | None:
    """Build the 14-feature vector expected by the Over 2.5 classifier."""
    return _model_feature_vector(
        "Over 2.5", OVER25_FEATURE_COLS, OVER25_REQUIRED_COLS,
        home_team, away_team, league_encoded, df,
        league=league, season=season,
    )


def _get_over25_probability(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
    *,
    league: str | None = None,
    season: str | None = None,
) -> float | None:
    """Return Over 2.5 probability from the dedicated classifier, or None."""
    global _over25_model
    if _over25_model is None:
        return None
    vector = _build_over25_feature_vector(
        home_team, away_team, league_encoded, df,
        league=league, season=season,
    )
    if vector is None:
        return None
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


def _build_over95_corners_feature_vector(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
    *,
    league: str | None = None,
    season: str | None = None,
) -> dict[str, float] | None:
    """Build the 13-feature vector expected by the Over 9.5 corners model."""
    return _model_feature_vector(
        "Over 9.5 corners", OVER95_CORNERS_FEATURE_COLS,
        OVER95_CORNERS_REQUIRED_COLS,
        home_team, away_team, league_encoded, df,
        league=league, season=season,
    )


def _get_over95_corners_probability(
    home_team: str,
    away_team: str,
    league_encoded: int,
    df: pd.DataFrame,
    *,
    league: str | None = None,
    season: str | None = None,
) -> float | None:
    """P(total corners >= 10) from the dedicated classifier, or None."""
    global _over95_corners_model
    if _over95_corners_model is None:
        return None
    vector = _build_over95_corners_feature_vector(
        home_team, away_team, league_encoded, df,
        league=league, season=season,
    )
    if vector is None:
        return None
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

    merged_fv = dict(ml_result.get("feature_vector_used") or {})
    merged_fv.update(formula_result.get("feature_vector_used") or {})
    blended["feature_vector_used"] = merged_fv

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
                league=req.league,
                season=req.season,
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
                    # Keep these existing markets for API consumers and
                    # fixture storage, even though the UI no longer shows them.
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
                req.home_team, req.away_team, league_enc_for_btts, _features_df,
                league=req.league, season=req.season,
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
                req.home_team, req.away_team, league_enc_for_over25, _features_df,
                league=req.league, season=req.season,
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
                req.home_team, req.away_team, league_enc_for_corners, _features_df,
                league=req.league, season=req.season,
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

    # Formula scorelines use the formula rates, even in a blended response.
    # Those rates are exposed (rounded to three decimals) in the feature
    # vector. ML-only responses have no scorelines, so use their goal rates.
    rate_source = formula_result if formula_result is not None else final_result
    goal_rates = rate_source["goals"]
    formula_features = rate_source.get("feature_vector_used") or {}
    lambda_home = formula_features.get("lambda_home", goal_rates["expected_home"])
    lambda_away = formula_features.get("lambda_away", goal_rates["expected_away"])
    final_result["first_goal"] = compute_first_goal_probs(lambda_home, lambda_away)

    return final_result


@router.get("/admin/diagnostics")
def diagnostics() -> dict:
    """
    Report what is actually loaded in memory.
    Shows model feature counts, features.csv columns,
    and whether they match.
    No secret required — read only.
    """
    # A diagnostics call is the cheapest way to see whether elo_ratings is
    # reachable: refresh the read-only cache so elo_cache_entries reflects the
    # table rather than whatever happened to be queried since the last reload.
    warm_elo_cache(force=True)

    runtime_cols = _get_runtime_feature_cols()

    features_df_cols = []
    legacy_fbref_in_csv = []
    if _features_df is not None:
        features_df_cols = list(_features_df.columns)
        # Legacy CSVs may still carry the removed columns; reporting them here
        # makes a stale artifact obvious. They are never model inputs now.
        legacy_fbref_in_csv = [c for c in features_df_cols
                              if "fbref" in c]

    legacy_fbref_in_model = [c for c in runtime_cols
                             if "fbref" in c]

    # Which assembly path a live request takes, so a deployment can be checked
    # without issuing a prediction: both should report "builder".
    history_columns_present = bool(
        _features_df is not None
        and all(column in _features_df.columns
                for column in feature_builder.MATCH_HISTORY_COLUMNS)
    )

    fbref_rows = _count_fbref_team_stats_rows()

    return {
        "hardcoded_feature_cols": len(FEATURE_COLS),
        "model_feature_cols":     len(runtime_cols),
        "features_csv_cols":      len(features_df_cols),
        "fbref_cols_in_model":    len(legacy_fbref_in_model),
        "fbref_cols_in_csv":      len(legacy_fbref_in_csv),
        "model_fbref_sample":     legacy_fbref_in_model[:5],
        "csv_fbref_sample":       legacy_fbref_in_csv[:5],
        "cols_match":             (
            set(runtime_cols) == set(FEATURE_COLS)
        ),
        # Feature parity (see docs/feature-parity-audit.md and
        # tests/test_feature_parity.py).
        "feature_assembly":            "builder",
        "feature_columns_source":      "features/feature_columns.py",
        "features_csv_has_match_history": history_columns_present,
        # "database" only when elo_ratings was actually read successfully.
        # The warm used to swallow its exception without touching
        # _elo_lookup_failures, so a missing or unreachable table still
        # reported "database" alongside 0 entries and nothing said why.
        "elo_lookup":                  (
            "database"
            if _elo_cache_error is None
            and _elo_lookup_failures < _ELO_MAX_FAILURES
            else "features.csv_fallback"
        ),
        # The number of teams that actually have a rating in elo_ratings.
        # This used to add len(_elo_cache), which counts per-team lookup slots
        # *including misses cached as None* — so two failed Arsenal/Chelsea
        # lookups could report "2 entries" over a completely empty table and
        # hide the very failure it was meant to expose.
        "elo_cache_entries":           len(_elo_latest_cache),
        "elo_lookup_failures":         _elo_lookup_failures,
        # Null when the table read cleanly and holds ratings; otherwise the
        # reason (swallowed exception, or a reachable-but-empty table).
        "elo_cache_error":             _elo_cache_error,
        # The snapshot table is still maintained by the dormant admin
        # endpoints, but it no longer feeds any model input.
        "fbref_source":                "disabled",
        "fbref_team_stats_rows":       fbref_rows,
        "formula_engine": (
            get_engine_status()
            if _formula_engine_available and get_engine_status
            else {"ready": False, "reason": "not loaded"}
        ),
        # PR #75 made the blended engine the only production path: /match
        # returns "engine": "blended" whenever both engines are available.
        # This label used to report "formula" from formula-engine readiness,
        # which contradicted the prediction response.
        "active_engine": "blended",
    }


# --------------------------------------------------------------------------- #
# Dormant FBref snapshot endpoints (kept: the aggregator still runs, but the
# fbref_team_stats table feeds no model feature since the FBref features were
# removed from the ML pipeline)
# --------------------------------------------------------------------------- #
@router.post(
    "/admin/run-fbref-scrape",
    responses={
        403: {"description": "Invalid or missing PIPELINE_SECRET"},
    },
)
def run_fbref_scrape(secret: str = "") -> dict[str, str]:
    """Start ONLY ``data/statsbomb_aggregator.py`` in the background on Render.

    FBref direct scraping is permanently disabled (no Chrome on Render's free
    tier, unsolvable Cloudflare CAPTCHAs on GitHub Actions IPs). The aggregator
    instead folds the match-level rows already collected in
    ``statsbomb_match_stats`` into ``fbref_team_stats`` — same table, same
    column names, no browser. The table is now reference-only and feeds no
    model feature. Never runs ``builder.py`` or ``trainer.py`` (those remain in
    GitHub Actions). Returns ``{"status": "started"}`` immediately.
    """
    expected = os.getenv("PIPELINE_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing secret key.",
        )

    return fbref_scrape_runner.start_fbref_scrape(
        on_status_change=_on_fbref_scrape_state,
        on_complete=_refresh_fbref_caches,
    )


@router.get(
    "/admin/fbref-scrape-status",
    responses={
        403: {"description": "Invalid or missing PIPELINE_SECRET"},
    },
)
def fbref_scrape_status(secret: str = "") -> dict[str, Any]:
    """Poll status of the background StatsBomb aggregation and live DB row count."""
    expected = os.getenv("PIPELINE_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing secret key.",
        )

    return fbref_scrape_runner.get_fbref_scrape_status(
        count_rows_fn=_count_fbref_team_stats_rows,
    )


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
