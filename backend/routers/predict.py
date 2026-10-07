"""
Prediction router — placeholder.

Endpoints for match predictions, model metadata and team/league lookups will be
added here. Nothing is implemented in Phase 1, but the router is already mounted
by `main.py` under the `/predict` prefix so the route table is ready to grow.
"""

from fastapi import APIRouter

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
)

# Example of what Phase 2 will add:
#
# @router.post("/")
# def create_prediction(request: MatchRequest) -> PredictionResponse:
#     ...
