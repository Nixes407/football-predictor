"""
Single source of truth for the Supabase Storage "models" bucket layout.

The heavy ML pipeline (``features/builder.py`` + ``models/trainer.py``) runs in
GitHub Actions. It uploads every artifact to the ``models`` bucket, and the
Render API — which only serves predictions — downloads them again on startup
and whenever ``POST /predict/admin/reload-models`` is called.

Both sides import this module, so the uploader and the downloader can never
drift apart (the previous duplication of these lists between
``routers/predict.py`` and ``models/trainer.py`` is how a newly added artifact
could silently never reach the API).

Nothing in here touches the network or the database: it is importable without
credentials, which keeps the FastAPI module import cheap.
"""

from __future__ import annotations

import os
from pathlib import Path

#: backend/ — the directory that contains database.py, artifacts, routers/, ...
BACKEND_DIR = Path(__file__).resolve().parent

#: Supabase Storage bucket that holds every pre-built artifact.
MODEL_BUCKET = "models"

# --------------------------------------------------------------------------- #
# Objects written by the training job (GitHub Actions)
# --------------------------------------------------------------------------- #

#: The nine trained models: 1X2 logistic, 1X2 XGBoost, home/away goals,
#: home/away corners, BTTS, Over 2.5 and Over 9.5 corners.
MODEL_OBJECTS: tuple[str, ...] = (
    "result_model_v1.joblib",
    "result_model_xgb_v1.joblib",
    "result_model_xgb_v1_label_encoder.joblib",
    "home_goals_v1.joblib",
    "away_goals_v1.joblib",
    "home_corners_v1.joblib",
    "away_corners_v1.joblib",
    "btts_model_v1.joblib",
    "over25_model_v1.joblib",
    "over95_corners_model_v1.joblib",
)

#: Label encoders written by the builder (they define the league/season integer
#: mapping, so a model trained with one encoder must be served with the same).
ENCODER_OBJECTS: tuple[str, ...] = (
    "league_encoder.joblib",
    "season_encoder.joblib",
)

#: The feature matrix the trainer consumed; the API loads it for team/league
#: lookups and for the rolling-form features.
FEATURES_OBJECT = "features.csv"

#: Everything ``trainer.py --strict-upload`` must publish before the API is
#: allowed to reload. Order matters only for log readability.
UPLOAD_OBJECTS: tuple[str, ...] = (
    MODEL_OBJECTS + ENCODER_OBJECTS + (FEATURES_OBJECT,)
)

# --------------------------------------------------------------------------- #
# What the API needs at boot
# --------------------------------------------------------------------------- #

#: Without these the API cannot serve a single prediction market.
#:
#: The XGBoost 1X2 model and the Over 9.5 corners classifier are *not* listed:
#: they degrade gracefully to the logistic model / Poisson fallback, and a
#: bucket written by an older trainer may not contain them yet. Downloading
#: them is still attempted (see ``OPTIONAL_DOWNLOAD_OBJECTS``).
REQUIRED_DOWNLOAD_OBJECTS: tuple[str, ...] = (
    "result_model_v1.joblib",
    "home_goals_v1.joblib",
    "away_goals_v1.joblib",
    "home_corners_v1.joblib",
    "away_corners_v1.joblib",
    "btts_model_v1.joblib",
    "over25_model_v1.joblib",
) + ENCODER_OBJECTS + (FEATURES_OBJECT,)

#: Best-effort downloads: a missing object is logged and skipped, never fatal.
OPTIONAL_DOWNLOAD_OBJECTS: tuple[str, ...] = tuple(
    name for name in MODEL_OBJECTS if name not in REQUIRED_DOWNLOAD_OBJECTS
)


def local_path(object_name: str, backend_dir: Path | str = BACKEND_DIR) -> Path:
    """Return where ``object_name`` lives on the local filesystem.

    ``backend_dir`` lets the downloader stage files under a temporary root
    before atomically swapping them into place.
    """
    root = Path(backend_dir)
    if object_name == FEATURES_OBJECT:
        return root / "data" / "processed" / object_name
    if object_name in ENCODER_OBJECTS:
        return root / "data" / object_name
    return root / "models" / object_name


def supabase_credentials() -> tuple[str, str]:
    """Return ``(url, key)`` for Supabase Storage.

    ``SUPABASE_SERVICE_ROLE_KEY`` is the preferred variable (uploads need the
    service role). ``SUPABASE_KEY`` is kept as a fallback because that is the
    name the Render service and the older workflow already use, and per the
    handover it already holds the service_role key. Whitespace and a trailing
    slash in the URL are stripped so both ``https://x.supabase.co`` and
    ``https://x.supabase.co/`` work.
    """
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or ""
    ).strip()
    return url, key


def missing_credentials_message() -> str:
    """Human-readable explanation used by both the API and the trainer."""
    return (
        "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or the legacy SUPABASE_KEY) "
        "must be set"
    )
