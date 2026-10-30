"""First-goal math and additive contracts for both prediction APIs."""

from __future__ import annotations

import copy
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import DailyMatch  # noqa: E402
from routers import fixtures, predict  # noqa: E402


def test_first_goal_normal():
    probabilities = predict.compute_first_goal_probs(2.0, 1.0)
    assert probabilities == {"home": 0.6335, "away": 0.3167, "no_goal": 0.0498}
    assert probabilities["home"] > probabilities["away"]
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.001)


def test_first_goal_equal():
    probabilities = predict.compute_first_goal_probs(1.5, 1.5)
    assert probabilities["home"] == pytest.approx(probabilities["away"])
    assert probabilities["no_goal"] == round(math.exp(-3.0), 4)


def test_first_goal_zero():
    assert predict.compute_first_goal_probs(0.0, 0.0) == {
        "home": 0.0, "away": 0.0, "no_goal": 1.0,
    }


@pytest.mark.parametrize(
    ("home_rate", "away_rate"),
    [(0, 0), (0, 3), (3, 0), (0.0001, 0.0002), (0.3, 0.8),
     (1.5, 1.5), (2, 1), (8.0, 0.001), (100, 100), (1e308, 1e308)],
)
def test_first_goal_sums_to_one(home_rate, away_rate):
    probabilities = predict.compute_first_goal_probs(home_rate, away_rate)
    assert abs(sum(probabilities.values()) - 1.0) <= 0.001
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    assert probabilities["no_goal"] == round(math.exp(-(home_rate + away_rate)), 4)


def test_first_goal_sums_across_goal_rate_grid():
    for home_step in range(21):
        for away_step in range(21):
            probabilities = predict.compute_first_goal_probs(home_step / 10, away_step / 10)
            assert abs(sum(probabilities.values()) - 1.0) <= 0.001
            assert all(0.0 <= value <= 1.0 for value in probabilities.values())


def test_first_goal_warns_if_probabilities_do_not_sum_to_one(monkeypatch, caplog):
    # Simulate broken rounding to exercise the defensive production warning.
    monkeypatch.setattr(predict, "round", lambda value, digits: 0.4, raising=False)
    with caplog.at_level(logging.WARNING, logger=predict.__name__):
        predict.compute_first_goal_probs(2.0, 1.0)
    assert "do not sum to one" in caplog.text


@pytest.fixture
def formula_response():
    return {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "league": "ENG-Premier League",
        "season": "2026-2027",
        "result": {"home_win": 0.6, "draw": 0.25, "away_win": 0.15},
        "goals": {
            "expected_home": 2.0, "expected_away": 1.0,
            "over_1_5": 0.8009, "over_2_5": 0.5768,
            "over_3_5": 0.3528, "btts": 0.5466,
        },
        "corners": {
            "expected_home": 6.0, "expected_away": 4.0,
            "expected_total": 10.0, "over_8_5": 0.6672,
            "over_9_5": 0.5421, "over_10_5": 0.417,
            "over_11_5": 0.3032,
        },
        "top_scorelines": [
            {"score": "2-1", "probability": 0.0996},
            {"score": "0-0", "probability": 0.0498},
        ],
        "ratings": {"home": {"attack_rel": 1.2}, "away": {"attack_rel": 0.9}},
        "feature_vector_used": {"lambda_home": 2.0, "lambda_away": 1.0},
        "engine": "formula_v1",
    }


@pytest.fixture
def client(monkeypatch):
    # Keep these tests independent of whichever model artifacts are installed.
    for name in (
        "_result_model", "_xgb_result_model", "_xgb_label_encoder",
        "_home_goals", "_away_goals", "_home_corners", "_away_corners",
        "_btts_model", "_over25_model", "_over95_corners_model", "_features_df",
    ):
        monkeypatch.setattr(predict, name, None)
    monkeypatch.setattr(predict, "_formula_engine_available", False)
    app = FastAPI()
    app.include_router(predict.router)
    return TestClient(app)


def _post_arsenal_chelsea(client):
    return client.post("/predict/match", json={
        "home_team": "Arsenal", "away_team": "Chelsea",
        "league": "ENG-Premier League", "season": "2026-2027",
    })


def _install_ml_models(monkeypatch):
    class ResultModel:
        classes_ = ["H", "D", "A"]

        def predict_proba(self, _frame):
            return np.array([[0.3, 0.3, 0.4]])

    class ConstantModel:
        def __init__(self, value):
            self.value = value

        def predict(self, _frame):
            return np.array([self.value])

    monkeypatch.setattr(predict, "_result_model", ResultModel())
    monkeypatch.setattr(predict, "_home_goals", ConstantModel(1.0))
    monkeypatch.setattr(predict, "_away_goals", ConstantModel(3.0))
    monkeypatch.setattr(predict, "_home_corners", ConstantModel(5.0))
    monkeypatch.setattr(predict, "_away_corners", ConstantModel(4.0))
    monkeypatch.setattr(predict, "_features_df", pd.DataFrame({"home_team": ["Arsenal"]}))
    monkeypatch.setattr(predict, "_get_runtime_feature_cols", lambda: ["dummy"])
    monkeypatch.setattr(predict, "_build_feature_vector", lambda *args, **kwargs: {"dummy": 1.0})


def test_formula_response_adds_first_goal_without_changing_existing_markets(
    client, monkeypatch, formula_response,
):
    monkeypatch.setattr(predict, "_formula_engine_available", True)
    monkeypatch.setattr(predict, "formula_predict", lambda **kwargs: copy.deepcopy(formula_response))

    response = _post_arsenal_chelsea(client)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction_method"] == "formula_only"
    assert result["first_goal"] == {"home": 0.6335, "away": 0.3167, "no_goal": 0.0498}
    assert result["first_goal"]["no_goal"] == result["top_scorelines"][1]["probability"]
    for field in ("result", "goals", "corners", "top_scorelines", "ratings", "feature_vector_used"):
        assert result[field] == formula_response[field]


def test_brazilian_league_round_trips_through_teams_and_prediction(
    client, monkeypatch, formula_response,
):
    league = "BRA-Série A"
    monkeypatch.setattr(predict, "_features_df", pd.DataFrame({
        "league": [league], "home_team": ["Flamengo"], "away_team": ["Palmeiras"],
    }))
    monkeypatch.setattr(predict, "_formula_engine_available", True)
    calls = []

    def formula(**kwargs):
        calls.append(kwargs)
        return {**copy.deepcopy(formula_response), **kwargs}

    monkeypatch.setattr(predict, "formula_predict", formula)
    teams = client.get("/predict/teams", params={"league": league})
    assert teams.status_code == 200
    assert teams.json()["teams"] == ["Flamengo", "Palmeiras"]
    response = client.post("/predict/match", json={
        "home_team": "Flamengo", "away_team": "Palmeiras",
        "league": league, "season": "2026",
    })
    assert response.status_code == 200, response.text
    assert calls[0]["league"] == league
    assert response.json()["league"] == league
    assert response.json()["first_goal"] == predict.compute_first_goal_probs(2.0, 1.0)


def test_ml_only_response_includes_first_goal_and_keeps_corners(client, monkeypatch):
    _install_ml_models(monkeypatch)

    response = _post_arsenal_chelsea(client)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction_method"] == "ml_only"
    assert result["first_goal"] == predict.compute_first_goal_probs(1.0, 3.0)
    assert result["result"] == {"home_win": 0.3, "draw": 0.3, "away_win": 0.4}
    assert result["goals"]["expected_home"] == 1.0
    assert result["goals"]["expected_away"] == 3.0
    assert result["corners"]["expected_total"] == 9.0
    assert "over_9_5" in result["corners"]
    assert result["feature_vector_used"] == {"dummy": 1.0}


def test_ml_only_zero_rates_return_certain_no_goal(client, monkeypatch):
    _install_ml_models(monkeypatch)

    class ZeroModel:
        def predict(self, _frame):
            return np.array([0.0])

    monkeypatch.setattr(predict, "_home_goals", ZeroModel())
    monkeypatch.setattr(predict, "_away_goals", ZeroModel())
    response = _post_arsenal_chelsea(client)
    assert response.status_code == 200, response.text
    assert response.json()["first_goal"] == {"home": 0.0, "away": 0.0, "no_goal": 1.0}


def test_blended_first_goal_uses_scoreline_rates_not_blended_rates(
    client, monkeypatch, formula_response,
):
    _install_ml_models(monkeypatch)
    monkeypatch.setattr(predict, "_formula_engine_available", True)
    monkeypatch.setattr(predict, "formula_predict", lambda **kwargs: copy.deepcopy(formula_response))
    # Market classifiers must continue to override their existing fields.
    monkeypatch.setattr(predict, "_btts_model", object())
    monkeypatch.setattr(predict, "_over25_model", object())
    monkeypatch.setattr(predict, "_over95_corners_model", object())
    monkeypatch.setattr(predict, "_get_btts_probability", lambda *args, **kwargs: 0.61)
    monkeypatch.setattr(predict, "_get_over25_probability", lambda *args, **kwargs: 0.62)
    monkeypatch.setattr(predict, "_get_over95_corners_probability", lambda *args, **kwargs: 0.63)

    response = _post_arsenal_chelsea(client)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction_method"] == "blended"
    assert result["goals"]["expected_home"] == 1.7
    assert result["goals"]["expected_away"] == 1.6
    assert result["first_goal"] == predict.compute_first_goal_probs(2.0, 1.0)
    assert result["first_goal"] != predict.compute_first_goal_probs(1.7, 1.6)
    assert result["top_scorelines"] == formula_response["top_scorelines"]
    assert result["ratings"] == formula_response["ratings"]
    assert result["feature_vector_used"]["lambda_home"] == 2.0
    assert result["feature_vector_used"]["lambda_away"] == 1.0
    assert result["goals"]["btts"] == 0.61
    assert result["goals"]["over_2_5"] == 0.62
    assert result["corners"]["over_9_5"] == 0.63


def test_fixture_first_goal_is_derived_without_writing_stored_values():
    row = DailyMatch(
        external_id="fixture-1", home_team="Arsenal", away_team="Chelsea",
        league="ENG-Premier League", pred_home_win=0.5, pred_draw=0.3,
        pred_away_win=0.2, pred_home_goals=2.0, pred_away_goals=1.0,
        pred_home_corners=6.0, pred_away_corners=4.0,
    )
    before = {name: getattr(row, name, None) for name in fixtures.PREDICTION_FIELDS}
    data = fixtures._row_to_dict(row)
    assert data["first_goal"] == predict.compute_first_goal_probs(2.0, 1.0)
    assert data["has_prediction"] is True
    assert data["pred_home_corners"] == 6.0
    assert data["pred_away_corners"] == 4.0
    assert {name: getattr(row, name, None) for name in fixtures.PREDICTION_FIELDS} == before


def test_today_fixture_endpoint_exposes_first_goal(monkeypatch):
    row = DailyMatch(
        external_id="fixture-3", home_team="Flamengo", away_team="Palmeiras",
        league="BRA-Série A", pred_home_win=0.5, pred_draw=0.3,
        pred_away_win=0.2, pred_home_goals=1.5, pred_away_goals=1.0,
    )

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(fixtures, "SessionLocal", Session)
    monkeypatch.setattr(
        fixtures, "_get_or_fetch_matches", lambda target_date, db: [fixtures._row_to_dict(row)]
    )
    app = FastAPI()
    app.include_router(fixtures.router, prefix="/fixtures")
    response = TestClient(app).get("/fixtures/today")
    assert response.status_code == 200
    assert response.json()[0]["first_goal"] == predict.compute_first_goal_probs(1.5, 1.0)
    assert response.json()[0]["league"] == "BRA-Série A"


@pytest.mark.parametrize("invalid_rate", [None, -1.0, float("nan"), float("inf")])
def test_fixture_without_valid_goal_rates_does_not_invent_first_goal(invalid_rate):
    row = DailyMatch(external_id="fixture-2", pred_home_goals=invalid_rate, pred_away_goals=1.0)
    data = fixtures._row_to_dict(row)
    assert "first_goal" not in data
    assert data["has_prediction"] is False
