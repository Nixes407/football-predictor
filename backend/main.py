"""
Football Match Predictor API — FastAPI application entry point.

Phase 1 scaffold only: the app boots, answers a status check at "/", and mounts
the (currently route-less) prediction router. No prediction logic yet.

Run locally from the `backend/` directory:

    uvicorn main:app --reload
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    """Print a startup confirmation once the app is ready to serve requests."""
    print("=" * 60)
    print("Football Predictor API is starting up...")
    print(f"  Environment file loaded : {os.getenv('DATABASE_URL') is not None}")
    print(f"  CORS allowed origins    : {', '.join(ALLOWED_ORIGINS)}")
    if ALLOWED_ORIGIN_REGEX:
        print(f"  CORS origin regex       : {ALLOWED_ORIGIN_REGEX}")
    if not ALLOW_CREDENTIALS:
        print("  CORS credentials        : disabled (wildcard origin)")
    print("  Docs available at       : /docs")
    print("=" * 60)

    from database import init_db

    init_db()

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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
