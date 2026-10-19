"""
Football Match Predictor API — FastAPI application entry point.

This service only *serves* predictions. Data collection and model training run
in the daily GitHub Actions workflow
(`.github/workflows/daily_fbref_update.yml`) on a 7 GB ubuntu-latest runner;
here we just reload the pre-built artifacts from Supabase Storage on startup.

Run locally from the `backend/` directory:

    uvicorn main:app --reload
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import routers.fixtures as fixtures_router_module
import routers.results as results_router_module
from routers.predict import router as predict_router

# Load variables from backend/.env (see .env.example) into the environment.
load_dotenv()

def _allowed_origins() -> list[str]:
    """
    Build the CORS allowlist from the environment.

    Two variables are honoured, so either can be used on its own:

    ``ALLOWED_ORIGINS``  comma-separated list of browser origins. Defaults to
                         the local Next.js dev server.
    ``FRONTEND_URL``     single-origin shortcut (kept for the existing
                         deployment docs). Empty entries are dropped, so an
                         unset value can no longer put ``""`` in the list.

    Trailing slashes are stripped — browsers never send one in ``Origin``, so
    ``https://app.vercel.app/`` would otherwise never match. Order is
    preserved and duplicates removed.
    """
    configured = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    )
    candidates = [*configured.split(","), os.getenv("FRONTEND_URL", "")]
    origins = [
        origin.strip().rstrip("/") for origin in candidates if origin.strip()
    ]
    return list(dict.fromkeys(origins))


ALLOWED_ORIGINS = _allowed_origins()

# Optional regex for hosts that cannot be enumerated, e.g. Vercel preview
# deployments:  ALLOWED_ORIGIN_REGEX=https://.*\.vercel\.app
# Unset by default so the allowlist stays explicit.
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", "").strip() or None

# CORS forbids credentialed requests from a wildcard origin, so a "*" entry
# downgrades the API to anonymous CORS instead of silently rejecting the
# browser's preflight.
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start serving predictions; never train models on this host.

    This service runs on Render's 512 MB free tier. Since PRs #78/#80/#81 the
    training set (15,000+ rows) no longer fits in that budget, so builder.py
    and trainer.py were moved to GitHub Actions (7 GB runners). Startup only
    downloads the pre-built artifacts from Supabase Storage — a lightweight,
    non-blocking reload that cannot OOM the instance.
    """
    print("=" * 60)
    print("Football Predictor API is starting up...")
    print(f"  Environment file loaded : {os.getenv('DATABASE_URL') is not None}")
    print(f"  CORS allowed origins    : {', '.join(ALLOWED_ORIGINS)}")
    if ALLOWED_ORIGIN_REGEX:
        print(f"  CORS origin regex       : {ALLOWED_ORIGIN_REGEX}")
    if not ALLOW_CREDENTIALS:
        print("  CORS credentials        : disabled (wildcard origin)")
    print("  Training pipeline       : GitHub Actions (not this service)")
    print("  Docs available at       : /docs")
    print("=" * 60)

    from database import init_db
    from routers.predict import startup_reload_models

    init_db()
    # Background thread: downloads model artifacts only, then loads them.
    startup_reload_models()

    yield
    print("Football Predictor API is shutting down...")


app = FastAPI(
    title="Football Predictor API",
    description="Backend API for the football match prediction web app.",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_kwargs = {
    "allow_origins": ALLOWED_ORIGINS,
    "allow_credentials": ALLOW_CREDENTIALS,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if ALLOWED_ORIGIN_REGEX:
    _cors_kwargs["allow_origin_regex"] = ALLOWED_ORIGIN_REGEX

app.add_middleware(CORSMiddleware, **_cors_kwargs)


@app.get("/")
def read_root() -> dict:
    """Health/status check used to confirm the API is reachable."""
    return {"status": "Football Predictor API is running"}


# Prediction endpoints live in routers/predict.py and are mounted under /predict.
app.include_router(predict_router)
app.include_router(
    fixtures_router_module.router,
    prefix="/fixtures",
    tags=["fixtures"],
)
app.include_router(
    results_router_module.router,
    prefix="/results",
    tags=["results"],
)


def scheduled_daily_fetch():
    """Fetch fixtures and store their predictions (the API's own job).

    The weekly FBref scrape that used to run here on Mondays was removed when
    fbref_collector.py moved into the daily GitHub Actions workflow: scraping
    on a 512 MB instance competed with the API for memory, duplicated the
    workflow's work, and could get the service OOM-restarted. Data collection
    and training now happen exclusively in
    .github/workflows/daily_fbref_update.yml.
    """
    from data.fixtures import fetch_fixtures
    from routers.fixtures import compute_and_store_predictions
    from database import SessionLocal

    try:
        print("[scheduler] Running daily fixture fetch...")
        fixtures = fetch_fixtures()
        db = SessionLocal()
        try:
            compute_and_store_predictions(fixtures, db)
            print(f"[scheduler] Done. {len(fixtures)} fixtures processed.")
        finally:
            db.close()
    except Exception as e:
        print(f"[scheduler] ERROR in fixture fetch: {e}")


def refresh_todays_fixtures():
    """Refresh fixture predictions every 2 hours."""
    from database import SessionLocal
    from routers.fixtures import (
        _get_or_fetch_matches,
    )
    from datetime import datetime, timezone

    db = SessionLocal()
    try:
        # UTC date, matching /fixtures/today and the UTC scheduler.
        today = datetime.now(timezone.utc).date().isoformat()
        print(f"[scheduler] Refreshing "
              f"fixtures for {today}...")
        _get_or_fetch_matches(today, db)
        print(f"[scheduler] Refresh done")
    except Exception as exc:
        print(f"[scheduler] ERROR: {exc}")
    finally:
        try:
            db.close()
        except Exception:
            pass


scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    scheduled_daily_fetch,
    CronTrigger(hour=0, minute=0),
    id="daily_fixture_fetch",
    replace_existing=True,
)
scheduler.add_job(
    refresh_todays_fixtures,
    trigger=IntervalTrigger(hours=2),
    id="refresh_fixtures",
    replace_existing=True,
)
scheduler.start()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
