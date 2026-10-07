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

# Comma-separated list of browser origins allowed to call this API. The Next.js
# dev server runs on port 3000 by default.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Print a startup confirmation once the app is ready to serve requests."""
    print("=" * 60)
    print("Football Predictor API is starting up...")
    print(f"  Environment file loaded : {os.getenv('DATABASE_URL') is not None}")
    print(f"  CORS allowed origins    : {', '.join(ALLOWED_ORIGINS)}")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict:
    """Health/status check used to confirm the API is reachable."""
    return {"status": "Football Predictor API is running"}


# Prediction endpoints live in routers/predict.py and are mounted under /predict.
app.include_router(predict_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
