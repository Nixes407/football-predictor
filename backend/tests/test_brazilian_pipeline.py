"""Brazilian match ingestion, season, training, and serving regressions.

All provider responses and databases are local test doubles; no credentials or
network are needed. In particular, provider coverage is not a live API test.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import collector, elo, fixtures as fixture_feed  # noqa: E402
from data.league_config import (  # noqa: E402
    BRA_LEAGUE,
    CALENDAR_YEAR_LEAGUES,
    MIN_ROWS_PER_LEAGUE,
)
from database import DailyMatch, EloRating, Match  # noqa: E402
from features import builder  # noqa: E402
from features.feature_columns import FEATURE_COLS  # noqa: E402
from models import trainer  # noqa: E402
from routers import fixtures, predict, results  # noqa: E402

EUROPEAN_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "ENG-Championship",
    "POR-Primeira Liga",
    "NED-Eredivisie",
    "TUR-Süper Lig",
    "NOR-Eliteserien",
    "AUT-Bundesliga",
]


def test_bra_in_trainer_leagues():
    assert trainer.TRAINING_LEAGUES == [*EUROPEAN_LEAGUES, BRA_LEAGUE]
    assert trainer.canonical_league("bra-serie a") == BRA_LEAGUE


def test_bra_in_collector_leagues():
    assert collector.TRAINING_LEAGUES == [*EUROPEAN_LEAGUES, BRA_LEAGUE]
    assert fixture_feed.LEAGUE_MAP["BSA"] == BRA_LEAGUE


def test_daily_workflow_passes_api_key_and_trains_default_leagues():
    workflow_path = (
        Path(__file__).resolve().parents[2] / ".github/workflows/daily_fbref_update.yml"
    )
    steps = yaml.safe_load(workflow_path.read_text())["jobs"]["collect-and-retrain"]["steps"]
    match_step = next(step for step in steps if step.get("id") == "matches")
    trainer_step = next(step for step in steps if "Train + upload models" in step.get("name", ""))
    assert match_step["env"]["FOOTBALL_DATA_API_KEY"] == "${{ secrets.FOOTBALL_DATA_API_KEY }}"
    assert match_step["continue-on-error"] is True
    assert "python data/collector.py" in match_step["run"]
    assert "::warning::FOOTBALL_DATA_API_KEY" in match_step["run"]
    assert "python models/trainer.py" in trainer_step["run"]
    assert "--leagues" not in trainer_step["run"]


def test_calendar_year_seasons():
    assert CALENDAR_YEAR_LEAGUES == {BRA_LEAGUE}
    assert collector.CALENDAR_YEAR_LEAGUES is CALENDAR_YEAR_LEAGUES
    assert builder.CALENDAR_YEAR_LEAGUES is CALENDAR_YEAR_LEAGUES
    assert fixture_feed.CALENDAR_YEAR_LEAGUES is CALENDAR_YEAR_LEAGUES
    assert collector.get_seasons_for_league(BRA_LEAGUE, date(2026, 10, 30)) == [
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "2026",
    ]
    assert collector.get_seasons_for_league(BRA_LEAGUE, date(2027, 1, 1))[-1] == "2027"


def test_european_seasons_unchanged():
    for league in EUROPEAN_LEAGUES:
        assert collector.get_seasons_for_league(
            league, date(2026, 10, 30)
        ) == collector.season_range(today=date(2026, 10, 30))
        assert "2024-2025" in collector.get_seasons_for_league(league, date(2026, 10, 30))
    # Norway has calendar-year FotMob labels but its match collector's existing
    # cross-year DB labels must not be changed by this Brazil-only override.
    assert collector.get_seasons_for_league("NOR-Eliteserien", date(2026, 6, 1))[-1] == "2025-2026"


def test_season_string_bra():
    assert builder.get_season_string(BRA_LEAGUE, "2026-2027") == "2026"
    assert builder.get_season_string(BRA_LEAGUE, "2026") == "2026"
    assert builder.get_season_string(BRA_LEAGUE, "2026/2027") == "2026"


def test_season_string_european():
    for league in EUROPEAN_LEAGUES:
        assert builder.get_season_string(league, "2026-2027") == "2026-2027"


def sample_match(**overrides):
    match = {
        "id": 123,
        "utcDate": "2026-05-18T22:30:00Z",
        "status": "FINISHED",
        "season": {"startDate": "2026-04-01"},
        "homeTeam": {"name": "CR Flamengo"},
        "awayTeam": {"name": "SE Palmeiras"},
        "score": {"fullTime": {"home": 2, "away": 1}},
    }
    match.update(overrides)
    return match


class FakeResponse:
    def __init__(self, payload=None, status=200, headers=None):
        self.payload = payload
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_bsa_finished_matches_are_requested_and_normalized(monkeypatch):
    payload = {
        "matches": [
            sample_match(),
            sample_match(id=124, status="SCHEDULED"),
            sample_match(id=125, score={"fullTime": {"home": None, "away": 1}}),
            sample_match(id=126, season={"startDate": "2025-04-01"}),
            sample_match(id=127, homeTeam="malformed"),
            sample_match(id=128, score={"fullTime": {"home": -1, "away": 0}}),
        ]
    }
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(payload)

    monkeypatch.setattr(collector, "_football_data_get", fake_get)
    frame = collector._load_brazilian_matches("2026", "test-key")
    rows = collector._build_match_rows(frame, BRA_LEAGUE, "2026")
    assert calls[0][0].endswith("/competitions/BSA/matches")
    assert calls[0][1]["params"] == {"season": 2026, "status": "FINISHED"}
    assert calls[0][1]["headers"] == {"X-Auth-Token": "test-key"}
    assert len(rows) == 1
    row = rows[0]
    assert (row["league"], row["season"], row["date"]) == (BRA_LEAGUE, "2026", date(2026, 5, 18))
    assert (row["home_team"], row["away_team"]) == ("CR Flamengo", "SE Palmeiras")
    assert (row["home_goals"], row["away_goals"], row["result"]) == (2, 1, "H")
    for name in ("home_shots", "away_shots", "home_corners", "away_corners", "odds_home"):
        assert row[name] is None


def test_bsa_valid_empty_season_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(
        collector, "_football_data_get", lambda *a, **k: FakeResponse({"matches": []})
    )
    frame = collector._load_brazilian_matches("2026", "test-key")
    assert frame is not None and frame.empty
    assert collector._build_match_rows(frame, BRA_LEAGUE, "2026") == []


def test_bsa_http_and_malformed_payload_fail_independently(monkeypatch):
    for response in (FakeResponse({}, 403), FakeResponse({"matches": {}}, 200)):
        monkeypatch.setattr(collector, "_football_data_get", lambda *a, **k: response)
        assert collector._load_brazilian_matches("2021", "test-key") is None


def test_bsa_rate_limit_retry_is_bounded(monkeypatch):
    responses = iter(
        [FakeResponse(status=429, headers={"Retry-After": "3"}), FakeResponse({"matches": []})]
    )
    delays = []
    monkeypatch.setattr(collector.requests, "get", lambda *a, **kw: next(responses))
    monkeypatch.setattr(collector.time, "sleep", delays.append)
    monkeypatch.setattr(collector.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(collector, "_last_football_data_request", None)
    response = collector._football_data_get("https://example.invalid", params={}, headers={})
    assert response.status_code == 200
    assert 3.0 in delays


def test_collector_still_imports_when_requests_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    spec = importlib.util.spec_from_file_location("collector_no_requests", collector.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "requests" in module._MISSING_DEPENDENCIES
    assert not module._check_dependencies()


def test_collection_without_key_skips_brazil_but_keeps_original_leagues(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.setattr(collector, "LEAGUES", ["ENG-Premier League", BRA_LEAGUE])
    monkeypatch.setattr(collector, "_check_dependencies", lambda: True)
    monkeypatch.setattr(collector, "_register_extra_leagues_with_soccerdata", lambda: None)
    monkeypatch.setattr(
        collector,
        "get_seasons_for_league",
        lambda league: ["2026"] if league == BRA_LEAGUE else ["2026-2027"],
    )
    calls = []
    monkeypatch.setattr(
        collector,
        "_scrape_matches",
        lambda league, season: calls.append((league, season)) or pd.DataFrame(),
    )
    monkeypatch.setattr(collector, "_build_match_rows", lambda *args: [])
    monkeypatch.setattr(collector, "_upsert_matches", lambda rows: (0, 0))
    monkeypatch.setattr(
        collector,
        "_load_brazilian_matches",
        lambda *args: pytest.fail("should skip without API key"),
    )
    counts = collector.collect_match_history()
    assert calls == [("ENG-Premier League", "2026-2027")]
    assert counts["failed"] == 1


def test_current_bsa_season_is_collected_before_denied_history(monkeypatch):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    monkeypatch.setattr(collector, "LEAGUES", [BRA_LEAGUE])
    monkeypatch.setattr(collector, "_check_dependencies", lambda: True)
    monkeypatch.setattr(collector, "_register_extra_leagues_with_soccerdata", lambda: None)
    monkeypatch.setattr(
        collector, "get_seasons_for_league", lambda league: ["2024", "2025", "2026"]
    )
    calls = []

    def load(season, _key):
        calls.append(season)
        if season != "2026":
            return None  # e.g. account does not allow this historical season
        return pd.DataFrame(
            [{"date": "2026-05-18", "home_team": "A", "away_team": "B", "FTHG": 1, "FTAG": 0}]
        )

    written = []
    monkeypatch.setattr(collector, "_load_brazilian_matches", load)
    monkeypatch.setattr(
        collector, "_upsert_matches", lambda rows: (written.extend(rows) or len(rows), 0)
    )
    counts = collector.collect_match_history()
    assert calls == ["2026", "2025", "2024"]
    assert counts == {"inserted": 1, "skipped": 0, "failed": 2}
    assert written[0]["season"] == "2026"


@contextmanager
def sqlite_scope_factory(engine):
    session = sessionmaker(bind=engine)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_bsa_match_upsert_never_overwrites_existing_values(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Match.__table__.create(engine)
    monkeypatch.setattr(collector, "session_scope", lambda: sqlite_scope_factory(engine))
    monkeypatch.setattr(collector, "pg_insert", sqlite_insert)
    frame = pd.DataFrame(
        [
            {
                "date": "2026-05-18",
                "home_team": "CR Flamengo",
                "away_team": "SE Palmeiras",
                "FTHG": 2,
                "FTAG": 1,
                "HS": 17,
            }
        ]
    )
    original = collector._build_match_rows(frame, BRA_LEAGUE, "2026")
    assert collector._upsert_matches(original) == (1, 0)
    replacement = [{**original[0], "home_goals": 9, "home_shots": None}]
    assert collector._upsert_matches(replacement) == (0, 1)
    with sqlite_scope_factory(engine) as session:
        saved = session.query(Match).one()
        assert saved.home_goals == 2
        assert saved.home_shots == 17
    engine.dispose()


def test_collected_bsa_rows_build_features_and_fit_real_result_model(monkeypatch, tmp_path):
    scores = [(2, 1), (1, 1), (0, 2)]
    matches = [
        sample_match(
            id=1000 + i,
            utcDate=f"2026-05-{i + 1:02d}T18:00:00Z",
            score={"fullTime": {"home": scores[i % 3][0], "away": scores[i % 3][1]}},
        )
        for i in range(20)
    ]
    monkeypatch.setattr(
        collector, "_football_data_get", lambda *a, **k: FakeResponse({"matches": matches})
    )
    engine = create_engine("sqlite:///:memory:")
    Match.__table__.create(engine)
    scope = lambda: sqlite_scope_factory(engine)
    monkeypatch.setattr(collector, "session_scope", scope)
    monkeypatch.setattr(collector, "pg_insert", sqlite_insert)
    monkeypatch.setattr(builder, "session_scope", scope)

    frame = collector._load_brazilian_matches("2026", "test-key")
    rows = collector._build_match_rows(frame, BRA_LEAGUE, "2026")
    assert collector._upsert_matches(rows) == (20, 0)
    features = builder.build_features(builder.load_matches_from_db())
    features["league_encoded"] = 0
    features["season_encoded"] = 0
    assert features["target_home_corners"].isna().all()
    assert set(FEATURE_COLS).issubset(features.columns)
    train, val, test = trainer.chronological_split(trainer.filter_training_leagues(features))
    metrics = trainer.train_result_model(train, val, test, str(tmp_path))
    model = joblib.load(metrics["model_path"])
    assert model.brazil_training_rows_ == len(train) + len(val) >= MIN_ROWS_PER_LEAGUE
    assert predict._brazil_ml_ready(BRA_LEAGUE, "CR Flamengo", "SE Palmeiras", model, features)
    assert model.predict_proba(test[FEATURE_COLS]).shape == (len(test), 3)
    engine.dispose()


def test_brazil_elo_conflicts_preserve_non_null_values(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    EloRating.__table__.create(engine)
    monkeypatch.setattr(elo, "engine", engine)
    monkeypatch.setattr(elo, "session_scope", lambda: sqlite_scope_factory(engine))
    day = date(2026, 5, 18)
    with sqlite_scope_factory(engine) as session:
        session.add_all(
            [
                EloRating(team="CR Flamengo", date=day, league=BRA_LEAGUE, elo=1712),
                EloRating(team="Arsenal", date=day, league="ENG-Premier League", elo=1600),
                EloRating(team="Grêmio", date=day, league=None, elo=1630),
            ]
        )
    elo.upsert_elo_rows(
        [
            {"team": "CR Flamengo", "date": day, "league": BRA_LEAGUE, "elo": 1500},
            {"team": "Arsenal", "date": day, "league": "ENG-Premier League", "elo": 1650},
            {"team": "Grêmio", "date": day, "league": BRA_LEAGUE, "elo": 1500},
            {"team": "SE Palmeiras", "date": day, "league": BRA_LEAGUE, "elo": 1510},
        ]
    )
    with sqlite_scope_factory(engine) as session:
        saved = {row.team: row for row in session.query(EloRating).all()}
        assert saved["CR Flamengo"].elo == 1712
        assert saved["CR Flamengo"].league == BRA_LEAGUE
        assert saved["Grêmio"].elo == 1630  # preserve rating, fill only missing league
        assert saved["Grêmio"].league == BRA_LEAGUE
        assert saved["SE Palmeiras"].elo == 1510
        assert saved["Arsenal"].elo == 1650  # existing European behavior
    engine.dispose()


def test_postgres_elo_upsert_compiles_with_brazil_preservation(monkeypatch):
    from sqlalchemy.dialects.postgresql import dialect

    statements = []

    @contextmanager
    def fake_scope():
        yield SimpleNamespace(execute=statements.append)

    monkeypatch.setattr(elo, "engine", SimpleNamespace(dialect=SimpleNamespace(name="postgresql")))
    monkeypatch.setattr(elo, "session_scope", fake_scope)
    elo.upsert_elo_rows(
        [
            {"team": "CR Flamengo", "date": date(2026, 5, 18), "league": BRA_LEAGUE, "elo": 1500},
        ]
    )
    sql = str(statements[0].compile(dialect=dialect()))
    assert "ON CONFLICT (team, date) DO UPDATE" in sql
    assert "CASE WHEN" in sql and "coalesce(elo_ratings.elo, excluded.elo)" in sql
    assert "coalesce(elo_ratings.league, excluded.league)" in sql


def test_builder_normalizes_only_brazilian_season_in_memory():
    frame = pd.DataFrame(
        [
            {"date": "2026-06-01", "home_team": "A", "away_team": "B", "FTHG": 1, "FTAG": 0},
        ]
    )
    bra = collector._build_match_rows(frame, BRA_LEAGUE, "2026-2027")[0]
    eng = collector._build_match_rows(frame, "ENG-Premier League", "2026-2027")[0]
    matches = pd.DataFrame([bra, eng])
    built = builder.build_features(matches, pd.DataFrame())
    assert list(built["season"]) == ["2026", "2026-2027"]
    assert list(matches["season"]) == ["2026-2027", "2026-2027"]
    assert len(FEATURE_COLS) == 62


def test_fixture_and_result_seasons_use_api_start_year(monkeypatch):
    assert fixture_feed.derive_season("2026-02-15T00:00:00Z", BRA_LEAGUE) == "2026"
    assert fixture_feed.derive_season("2027-01-15T00:00:00Z", BRA_LEAGUE, "2026-04-01") == "2026"
    assert fixture_feed.derive_season("2026-02-15T00:00:00Z", "ENG-Premier League") == "2025-2026"
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    monkeypatch.setattr(fixture_feed, "LEAGUE_MAP", {"BSA": BRA_LEAGUE})
    monkeypatch.setattr(fixture_feed, "_collect_mls_fixtures", lambda day: [])
    match = sample_match(utcDate="2027-01-15T00:00:00Z")
    monkeypatch.setattr(
        fixture_feed.requests, "get", lambda *a, **k: FakeResponse({"matches": [match]})
    )
    assert fixture_feed.fetch_fixtures("2027-01-15")[0]["season"] == "2026"
    monkeypatch.setattr(results, "LEAGUE_MAP", {"BSA": BRA_LEAGUE})
    monkeypatch.setattr(results, "LEAGUE_CODE_MAP", {BRA_LEAGUE: "BSA"})
    assert results.fetch_results("2027-01-15")[0]["season"] == "2026"


def test_stored_legacy_bsa_fixture_season_is_resolved_without_db_writes():
    row = DailyMatch(
        external_id="123",
        league=BRA_LEAGUE,
        season="2025-2026",
        kickoff_time="2026-04-20T18:00:00Z",
        match_date="2026-04-20",
    )
    assert fixtures._stored_fixture_season(row) == "2026"
    assert row.season == "2025-2026"
    # When available, the provider's season is authoritative for a delayed
    # January match from the prior season.
    row.kickoff_time = "2027-01-05T18:00:00Z"
    assert fixtures._stored_fixture_season(row, {"season": "2026"}) == "2026"
    row.league = "ENG-Premier League"
    assert fixtures._stored_fixture_season(row) == "2025-2026"


def training_frame(europe=30, brazil=0, brazil_first=False):
    dates = pd.date_range("2023-01-01", periods=europe + brazil, freq="D")
    leagues = ["ENG-Premier League"] * europe + [BRA_LEAGUE] * brazil
    if brazil_first:
        leagues = [BRA_LEAGUE] * brazil + ["ENG-Premier League"] * europe
    frame = pd.DataFrame({column: np.arange(len(leagues), dtype=float) for column in FEATURE_COLS})
    frame["date"] = dates
    frame["league"] = leagues
    frame["target_result"] = (["H", "D", "A"] * (len(leagues) // 3 + 1))[: len(leagues)]
    return frame


def stub_model_training(monkeypatch, captured):
    def result_model(train, val, test, model_dir):
        captured.append((train.copy(), val.copy(), test.copy()))
        return {}

    monkeypatch.setattr(trainer, "train_result_model", result_model)
    for name in (
        "train_xgb_result_model",
        "train_count_model",
        "train_btts_model",
        "train_over25_model",
        "train_over95_corners_model",
    ):
        monkeypatch.setattr(trainer, name, lambda *args, **kwargs: {})


def test_trainer_min_rows_guard(monkeypatch, tmp_path):
    captured = []
    stub_model_training(monkeypatch, captured)
    monkeypatch.setattr(trainer, "load_features", lambda path: training_frame(europe=30, brazil=0))
    summary = trainer.train_all_models(model_dir=str(tmp_path), upload=False)
    assert summary["_dataset"]["features_rows"] == 30
    assert summary["_dataset"]["brazil_trainval_rows"] == 0
    assert set(captured[0][0]["league"]) == {"ENG-Premier League"}
    assert MIN_ROWS_PER_LEAGUE == 10


def test_brazil_only_with_too_few_rows_exits_without_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer, "load_features", lambda path: training_frame(europe=0, brazil=9))
    assert trainer.train_all_models(model_dir=str(tmp_path), upload=False) == {}


def test_thin_brazil_does_not_filter_thin_existing_leagues():
    df = pd.DataFrame({"league": ["ENG-Premier League"] + [BRA_LEAGUE] * 9})
    kept = trainer.filter_training_leagues(df)
    assert kept["league"].tolist() == ["ENG-Premier League"]
    df = pd.concat([df, pd.DataFrame({"league": [BRA_LEAGUE]})], ignore_index=True)
    assert len(trainer.filter_training_leagues(df)) == 11


def test_bsa_only_in_held_out_split_is_skipped_without_leakage(monkeypatch, tmp_path):
    captured = []
    stub_model_training(monkeypatch, captured)
    monkeypatch.setattr(
        trainer, "load_features", lambda path: training_frame(europe=40, brazil=12)
    )
    summary = trainer.train_all_models(model_dir=str(tmp_path), upload=False)
    assert summary["_dataset"]["features_rows"] == 40
    assert summary["_dataset"]["brazil_trainval_rows"] == 0
    assert all(set(part["league"]) <= {"ENG-Premier League"} for part in captured[0])
    assert [len(part) for part in captured[0]] == [26, 6, 8]


def test_sufficient_brazil_rows_enter_chronological_model_fit(monkeypatch, tmp_path):
    captured = []
    stub_model_training(monkeypatch, captured)
    monkeypatch.setattr(
        trainer,
        "load_features",
        lambda path: training_frame(europe=40, brazil=20, brazil_first=True),
    )
    summary = trainer.train_all_models(model_dir=str(tmp_path), upload=False)
    assert summary["_dataset"]["features_rows"] == 60
    assert summary["_dataset"]["brazil_trainval_rows"] == 20
    assert BRA_LEAGUE in set(captured[0][0]["league"])


def test_result_artifact_records_actual_brazil_fit_rows(tmp_path):
    df = training_frame(europe=20, brazil=20, brazil_first=True)
    metrics = trainer.train_result_model(df.iloc[:25], df.iloc[25:32], df.iloc[32:], str(tmp_path))
    model = joblib.load(metrics["model_path"])
    assert model.brazil_training_rows_ == 20


def test_brazil_ml_readiness_needs_training_and_known_teams():
    history = pd.DataFrame(
        {
            "league": [BRA_LEAGUE],
            "home_team": ["CR Flamengo"],
            "away_team": ["SE Palmeiras"],
        }
    )
    assert not predict._brazil_ml_ready(
        BRA_LEAGUE, "CR Flamengo", "SE Palmeiras", object(), history
    )
    assert not predict._brazil_ml_ready(
        BRA_LEAGUE,
        "CR Flamengo",
        "SE Palmeiras",
        SimpleNamespace(brazil_training_rows_=9),
        history,
    )
    ready_model = SimpleNamespace(brazil_training_rows_=10)
    assert predict._brazil_ml_ready(
        BRA_LEAGUE, "CR Flamengo", "SE Palmeiras", ready_model, history
    )
    assert not predict._brazil_ml_ready(BRA_LEAGUE, "CR Flamengo", "Unknown", ready_model, history)
    assert predict._brazil_ml_ready("ENG-Premier League", "Unknown", "Other", None, None)


def test_untrained_brazil_match_returns_503_instead_of_an_extrapolation(monkeypatch):
    monkeypatch.setattr(predict, "_formula_engine_available", False)
    monkeypatch.setattr(predict, "_result_model", object())
    for name in ("_home_goals", "_away_goals", "_home_corners", "_away_corners"):
        monkeypatch.setattr(predict, name, object())
    monkeypatch.setattr(predict, "_features_df", training_frame())
    app = FastAPI()
    app.include_router(predict.router)
    response = TestClient(app).post(
        "/predict/match",
        json={
            "home_team": "CR Flamengo",
            "away_team": "SE Palmeiras",
            "league": BRA_LEAGUE,
        },
    )
    assert response.status_code == 503
    assert "collect enough finished matches" in response.json()["detail"]


def test_brazil_omitted_season_defaults_to_calendar_year_in_february(monkeypatch):
    class February(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 1, tzinfo=tz or timezone.utc)

    calls = []
    monkeypatch.setattr(predict, "datetime", February)
    monkeypatch.setattr(predict, "_formula_engine_available", True)
    monkeypatch.setattr(predict, "formula_predict", lambda **kw: calls.append(kw) or None)
    monkeypatch.setattr(predict, "_result_model", None)
    monkeypatch.setattr(predict, "_features_df", None)
    app = FastAPI()
    app.include_router(predict.router)
    response = TestClient(app).post(
        "/predict/match",
        json={
            "home_team": "CR Flamengo",
            "away_team": "SE Palmeiras",
            "league": BRA_LEAGUE,
        },
    )
    assert response.status_code == 503
    assert calls[0]["season"] == "2026"


def test_trained_brazil_match_uses_canonical_encoded_season(monkeypatch):
    class ResultModel:
        brazil_training_rows_ = 20
        classes_ = ["H", "D", "A"]

        def predict_proba(self, _frame):
            return np.array([[0.5, 0.3, 0.2]])

    class CountModel:
        def predict(self, _frame):
            return np.array([1.2])

    history = pd.DataFrame(
        {
            "league": [BRA_LEAGUE],
            "home_team": ["CR Flamengo"],
            "away_team": ["SE Palmeiras"],
        }
    )
    league_encoder = LabelEncoder().fit([BRA_LEAGUE, "ENG-Premier League"])
    season_encoder = LabelEncoder().fit(["2026", "2026-2027"])
    monkeypatch.setattr(predict, "_formula_engine_available", False)
    monkeypatch.setattr(predict, "_result_model", ResultModel())
    monkeypatch.setattr(predict, "_xgb_result_model", None)
    monkeypatch.setattr(predict, "_features_df", history)
    monkeypatch.setattr(predict, "_le_league", league_encoder)
    monkeypatch.setattr(predict, "_le_season", season_encoder)
    for name in ("_home_goals", "_away_goals", "_home_corners", "_away_corners"):
        monkeypatch.setattr(predict, name, CountModel())
    for name in ("_btts_model", "_over25_model", "_over95_corners_model"):
        monkeypatch.setattr(predict, name, None)
    monkeypatch.setattr(predict, "_get_runtime_feature_cols", lambda: ["dummy"])
    calls = []

    def vector(home, away, league_encoded, season_encoded, *args, **kwargs):
        calls.append((league_encoded, season_encoded, kwargs["season"]))
        return {"dummy": 1.0}

    monkeypatch.setattr(predict, "_build_feature_vector", vector)
    app = FastAPI()
    app.include_router(predict.router)
    response = TestClient(app).post(
        "/predict/match",
        json={
            "home_team": "CR Flamengo",
            "away_team": "SE Palmeiras",
            "league": BRA_LEAGUE,
            "season": "2026-2027",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["prediction_method"] == "ml_only"
    assert response.json()["season"] == "2026"
    assert response.json()["first_goal"] == predict.compute_first_goal_probs(1.2, 1.2)
    assert calls == [
        (
            int(league_encoder.transform([BRA_LEAGUE])[0]),
            int(season_encoder.transform(["2026"])[0]),
            "2026",
        )
    ]


def test_brazil_fixture_uses_provider_team_names_and_normalized_season(monkeypatch):
    history = pd.DataFrame(
        {"league": [BRA_LEAGUE], "home_team": ["CR Flamengo"], "away_team": ["SE Palmeiras"]}
    )
    model = SimpleNamespace(brazil_training_rows_=10)
    monkeypatch.setattr(predict, "_features_df", history)
    monkeypatch.setattr(predict, "_result_model", model)
    monkeypatch.setattr(
        fixtures, "_load_fotmob_names", lambda: {BRA_LEAGUE: ["Flamengo", "Palmeiras"]}
    )
    monkeypatch.setattr(fixtures, "_ml_models_available", lambda: True)
    monkeypatch.setattr(fixtures, "_predict_with_formula", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        fixtures,
        "_compute_prediction_values",
        lambda *a: calls.append(a) or {"pred_home_win": 0.5},
    )
    assert fixtures._predict_fixture_values(
        "CR Flamengo", "SE Palmeiras", BRA_LEAGUE, "2026-2027"
    ) == {"pred_home_win": 0.5}
    assert calls == [("CR Flamengo", "SE Palmeiras", BRA_LEAGUE, "2026")]


def test_brazil_fixture_does_not_use_untrained_model(monkeypatch):
    history = pd.DataFrame(
        {"league": [BRA_LEAGUE], "home_team": ["CR Flamengo"], "away_team": ["SE Palmeiras"]}
    )
    monkeypatch.setattr(predict, "_features_df", history)
    monkeypatch.setattr(predict, "_result_model", object())
    monkeypatch.setattr(fixtures, "_load_fotmob_names", lambda: {})
    monkeypatch.setattr(fixtures, "_ml_models_available", lambda: True)
    monkeypatch.setattr(fixtures, "_predict_with_formula", lambda *a, **k: None)
    monkeypatch.setattr(
        fixtures,
        "_compute_prediction_values",
        lambda *a: pytest.fail("untrained BSA model must not run"),
    )
    assert (
        fixtures._predict_fixture_values("CR Flamengo", "SE Palmeiras", BRA_LEAGUE, "2026") == {}
    )
