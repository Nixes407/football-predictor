"""Endpoints for daily football fixtures and their stored predictions."""

import traceback
from datetime import date, datetime, timezone

import pandas as pd
import routers.predict as predict_module
from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from data.fixtures import fetch_fixtures
from database import DailyMatch, SessionLocal
from routers.predict import (
    FEATURE_COLS,
    _away_corners,
    _away_goals,
    _build_feature_vector,
    _encode_label,
    _features_df,
    _home_corners,
    _home_goals,
    _le_league,
    _le_season,
    _result_model,
)


router = APIRouter()


def _current_predictor_value(name: str, imported_value):
    """Use the latest predictor state, including artifacts loaded after import."""
    current_value = getattr(predict_module, name, None)
    return imported_value if current_value is None else current_value


def _compute_prediction_values(
    home_team: str,
    away_team: str,
    league: str,
    season: str,
) -> dict[str, float]:
    """Run the same feature construction and model inference as /predict/match."""
    result_model = _current_predictor_value("_result_model", _result_model)
    home_goals_model = _current_predictor_value("_home_goals", _home_goals)
    away_goals_model = _current_predictor_value("_away_goals", _away_goals)
    home_corners_model = _current_predictor_value("_home_corners", _home_corners)
    away_corners_model = _current_predictor_value("_away_corners", _away_corners)
    league_encoder = _current_predictor_value("_le_league", _le_league)
    season_encoder = _current_predictor_value("_le_season", _le_season)
    features_df = _current_predictor_value("_features_df", _features_df)

    if any(
        model is None
        for model in (
            result_model,
            home_goals_model,
            away_goals_model,
            home_corners_model,
            away_corners_model,
        )
    ):
        raise RuntimeError("Prediction models are not loaded.")
    if features_df is None:
        raise RuntimeError("Features data is not loaded.")

    league_encoded = _encode_label(league_encoder, league)
    season_encoded = _encode_label(season_encoder, season)
    vector = _build_feature_vector(
        home_team,
        away_team,
        league_encoded,
        season_encoded,
        features_df,
    )
    X = pd.DataFrame([vector])[FEATURE_COLS]

    result_probabilities = result_model.predict_proba(X)[0]
    result_map = dict(zip(list(result_model.classes_), result_probabilities))

    return {
        "pred_home_win": round(float(result_map.get("H", 0.0)), 4),
        "pred_draw": round(float(result_map.get("D", 0.0)), 4),
        "pred_away_win": round(float(result_map.get("A", 0.0)), 4),
        "pred_home_goals": round(
            max(float(home_goals_model.predict(X)[0]), 0.0), 3
        ),
        "pred_away_goals": round(
            max(float(away_goals_model.predict(X)[0]), 0.0), 3
        ),
        "pred_home_corners": round(
            max(float(home_corners_model.predict(X)[0]), 0.0), 3
        ),
        "pred_away_corners": round(
            max(float(away_corners_model.predict(X)[0]), 0.0), 3
        ),
    }


def compute_and_store_predictions(fixtures: list, db: Session) -> None:
    """Compute predictions for unseen fixtures and persist every valid fixture."""
    null_predictions = {
        "pred_home_win": None,
        "pred_draw": None,
        "pred_away_win": None,
        "pred_home_goals": None,
        "pred_away_goals": None,
        "pred_home_corners": None,
        "pred_away_corners": None,
    }

    for fixture in fixtures:
        home_team = fixture.get("home_team", "Unknown")
        away_team = fixture.get("away_team", "Unknown")
        try:
            external_id = fixture.get("external_id")
            if external_id is None:
                raise ValueError("fixture has no external_id")
            external_id = str(external_id)

            existing = (
                db.query(DailyMatch)
                .filter(DailyMatch.external_id == external_id)
                .first()
            )
            if existing is not None:
                continue
        except Exception as exc:
            db.rollback()
            print(
                f"[fixtures] WARNING: could not check {home_team} vs "
                f"{away_team}: {exc}"
            )
            traceback.print_exc()
            continue

        predictions = null_predictions.copy()
        try:
            predictions = _compute_prediction_values(
                home_team=home_team,
                away_team=away_team,
                league=fixture["league"],
                season=fixture["season"],
            )
        except Exception as exc:
            print(
                f"[fixtures] WARNING: prediction failed for {home_team} vs "
                f"{away_team}: {exc}"
            )
            traceback.print_exc()

        try:
            row = DailyMatch(
                external_id=external_id,
                match_date=fixture.get("match_date"),
                kickoff_time=fixture.get("kickoff_time"),
                home_team=home_team,
                away_team=away_team,
                league=fixture.get("league"),
                season=fixture.get("season"),
                status=fixture.get("status"),
                **predictions,
                prediction_computed_at=datetime.now(timezone.utc).isoformat(),
            )
            db.add(row)
            db.commit()
        except Exception as exc:
            db.rollback()
            print(
                f"[fixtures] WARNING: could not store {home_team} vs "
                f"{away_team}: {exc}"
            )
            traceback.print_exc()


def _row_to_dict(row: DailyMatch) -> dict:
    """Serialize a DailyMatch row using the fixtures API response shape."""
    return {
        "id": row.id,
        "external_id": row.external_id,
        "match_date": row.match_date,
        "kickoff_time": row.kickoff_time,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "league": row.league,
        "season": row.season,
        "status": row.status,
        "pred_home_win": row.pred_home_win,
        "pred_draw": row.pred_draw,
        "pred_away_win": row.pred_away_win,
        "pred_home_goals": row.pred_home_goals,
        "pred_away_goals": row.pred_away_goals,
        "pred_home_corners": row.pred_home_corners,
        "pred_away_corners": row.pred_away_corners,
        "prediction_computed_at": row.prediction_computed_at,
    }


def _get_or_fetch_matches(target_date: str, db: Session) -> list[dict]:
    """Return cached matches for a date, fetching and storing them if needed."""
    rows = (
        db.query(DailyMatch)
        .filter(DailyMatch.match_date == target_date)
        .all()
    )
    if not rows:
        fixtures = fetch_fixtures(target_date)
        compute_and_store_predictions(fixtures, db)
        rows = (
            db.query(DailyMatch)
            .filter(DailyMatch.match_date == target_date)
            .all()
        )
    return [_row_to_dict(row) for row in rows]


def _close_session(db) -> None:
    if db is not None:
        try:
            db.close()
        except Exception as exc:
            print(f"[fixtures] WARNING: could not close DB session: {exc}")


@router.get("/today")
def get_today_fixtures() -> list[dict]:
    """Return today's UTC fixtures, fetching them if they are not cached."""
    db = None
    try:
        target_date = datetime.now(timezone.utc).date().isoformat()
        db = SessionLocal()
        return _get_or_fetch_matches(target_date, db)
    except Exception as exc:
        print(f"[fixtures] WARNING: could not load today's fixtures: {exc}")
        traceback.print_exc()
        return []
    finally:
        _close_session(db)


@router.get("/date")
def get_fixtures_for_date(
    target_date: str = Query(..., alias="date", description="Date in YYYY-MM-DD format"),
) -> list[dict]:
    """Return fixtures for the requested date, fetching them if not cached."""
    db = None
    try:
        parsed_date = date.fromisoformat(target_date)
        if parsed_date.isoformat() != target_date:
            raise ValueError("date must be in YYYY-MM-DD format")
        db = SessionLocal()
        return _get_or_fetch_matches(target_date, db)
    except Exception as exc:
        print(f"[fixtures] WARNING: could not load fixtures for {target_date}: {exc}")
        traceback.print_exc()
        return []
    finally:
        _close_session(db)
