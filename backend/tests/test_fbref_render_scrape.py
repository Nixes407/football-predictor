"""End-to-end tests for Render FBref scraping, status polling, workflow, and live inference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "daily_fbref_update.yml"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import database  # noqa: E402
import fbref_scrape_runner  # noqa: E402
from data import fbref_collector  # noqa: E402
from features import builder, inference  # noqa: E402
from features.feature_columns import FBREF_STAT_COLUMNS, FEATURE_COLS  # noqa: E402
from models import formula_engine  # noqa: E402
from routers import predict as predict_router  # noqa: E402


def _make_multiindex_stat_frames(teams_by_league_season: dict[tuple[str, str], list[str]]):
    """Build realistic two-level MultiIndex DataFrames matching ``soccerdata.FBref`` output."""
    idx_tuples = []
    shoot_rows = []
    misc_rows = []
    keep_rows = []
    for (league, short_season), teams in teams_by_league_season.items():
        for i, team in enumerate(teams):
            idx_tuples.append((league, short_season, team))
            shoot_rows.append([10.0, 145.0 + i * 5, 52.0 + i * 2, 35.9, 0.14, 0.38])
            misc_rows.append([10.0, 18.0 + i, 1.0, 112.0, 108.0, 19.0, 164.0, 88.0, 96.0, 3.0, 2.0])
            keep_rows.append([10.0, 34.0 + i, 9.0 + i, 25.0, 73.5, 5.0, 2.0, 1.0])

    index = pd.MultiIndex.from_tuples(idx_tuples, names=["league", "season", "team"])
    shooting_df = pd.DataFrame(
        shoot_rows,
        index=index,
        columns=pd.MultiIndex.from_tuples(
            [
                ("Playing Time", "90s"),
                ("Standard", "Sh"),
                ("Standard", "SoT"),
                ("Standard", "SoT%"),
                ("Standard", "G/Sh"),
                ("Standard", "G/SoT"),
            ]
        ),
    )
    misc_df = pd.DataFrame(
        misc_rows,
        index=index,
        columns=pd.MultiIndex.from_tuples(
            [
                ("Playing Time", "90s"),
                ("Performance", "CrdY"),
                ("Performance", "CrdR"),
                ("Performance", "Fls"),
                ("Performance", "Fld"),
                ("Performance", "Off"),
                ("Performance", "Crs"),
                ("Performance", "Int"),
                ("Performance", "TklW"),
                ("Performance", "PKwon"),
                ("Performance", "PKcon"),
            ]
        ),
    )
    keeper_df = pd.DataFrame(
        keep_rows,
        index=index,
        columns=pd.MultiIndex.from_tuples(
            [
                ("Playing Time", "MP"),
                ("Performance", "SoTA"),
                ("Performance", "GA"),
                ("Performance", "Saves"),
                ("Performance", "Save%"),
                ("Performance", "CS"),
                ("Penalty Kicks", "PKatt"),
                ("Penalty Kicks", "PKsv"),
            ]
        ),
    )
    return shooting_df, misc_df, keeper_df


class FBrefRenderEndpointsTests(unittest.TestCase):
    """Contract tests for POST /predict/admin/run-fbref-scrape & GET /predict/admin/fbref-scrape-status."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_file = Path(self._tmp.name) / "test.db"
        self.engine = create_engine(f"sqlite:///{db_file}")
        self.Session = sessionmaker(bind=self.engine)
        database.Base.metadata.create_all(bind=self.engine)

        self._patches = [
            patch.object(database, "engine", self.engine),
            patch.object(database, "SessionLocal", self.Session),
            patch.dict(os.environ, {"PIPELINE_SECRET": "top-secret"}),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

        fbref_scrape_runner.reset_runner_state()
        self.addCleanup(fbref_scrape_runner.reset_runner_state)

        app = FastAPI()
        app.include_router(predict_router.router)
        self.client = TestClient(app)

    def test_auth_rejects_missing_or_wrong_secret_on_both_endpoints(self):
        self.assertEqual(
            self.client.post("/predict/admin/run-fbref-scrape").status_code, 403
        )
        self.assertEqual(
            self.client.post("/predict/admin/run-fbref-scrape?secret=wrong").status_code,
            403,
        )
        self.assertEqual(
            self.client.get("/predict/admin/fbref-scrape-status").status_code, 403
        )
        self.assertEqual(
            self.client.get("/predict/admin/fbref-scrape-status?secret=wrong").status_code,
            403,
        )

    def test_idle_status_returns_expected_contract_and_db_row_count(self):
        with self.Session() as db:
            db.add(
                database.FBrefTeamStats(
                    team="Arsenal",
                    league="ENG-Premier League",
                    season="2026-2027",
                    shots=150.0,
                    clean_sheets=5.0,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            db.commit()

        res = self.client.get("/predict/admin/fbref-scrape-status?secret=top-secret")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "idle")
        self.assertIsNone(body["last_run_at"])
        self.assertIsNone(body["last_result"])
        self.assertEqual(body["fbref_team_stats_rows"], 1)

    def test_non_blocking_scrape_runs_only_fbref_collector_and_updates_status(self):
        gate = threading.Event()
        spawned_cmds: list[list[str]] = []

        class FakeProc:
            def __init__(self, cmd, **_kwargs):
                spawned_cmds.append(list(cmd))
                self.stdout = iter(["[fbref] Starting FBref Big Five stats scrape...\n"])

            def wait(self):
                gate.wait(timeout=5)
                return 0

        with patch.object(fbref_scrape_runner.subprocess, "Popen", side_effect=FakeProc):
            start_res = self.client.post(
                "/predict/admin/run-fbref-scrape?secret=top-secret"
            )
            self.assertEqual(start_res.status_code, 200)
            self.assertEqual(start_res.json(), {"status": "started"})
            self.assertEqual(predict_router._pipeline_status, "fbref_scrape_running")

            running_res = self.client.get(
                "/predict/admin/fbref-scrape-status?secret=top-secret"
            )
            self.assertEqual(running_res.status_code, 200)
            self.assertEqual(running_res.json()["status"], "running")
            self.assertIsNotNone(running_res.json()["last_run_at"])

            # Insert 2 rows before finishing so completion picks up the DB count
            with self.Session() as db:
                db.add_all(
                    [
                        database.FBrefTeamStats(
                            team="Arsenal",
                            league="ENG-Premier League",
                            season="2026-2027",
                            shots=155.0,
                            clean_sheets=6.0,
                            updated_at=datetime.now(timezone.utc).isoformat(),
                        ),
                        database.FBrefTeamStats(
                            team="Chelsea",
                            league="ENG-Premier League",
                            season="2026-2027",
                            shots=142.0,
                            clean_sheets=4.0,
                            updated_at=datetime.now(timezone.utc).isoformat(),
                        ),
                    ]
                )
                db.commit()

            gate.set()

            # Wait briefly for daemon worker to finalize state
            done_body = None
            for _ in range(50):
                done_body = self.client.get(
                    "/predict/admin/fbref-scrape-status?secret=top-secret"
                ).json()
                if done_body["status"] != "running":
                    break
                threading.Event().wait(0.02)

        self.assertEqual(len(spawned_cmds), 1)
        self.assertTrue(
            spawned_cmds[0][-1].endswith("data/fbref_collector.py"),
            spawned_cmds,
        )
        self.assertNotIn("builder.py", " ".join(spawned_cmds[0]))
        self.assertNotIn("trainer.py", " ".join(spawned_cmds[0]))
        assert done_body is not None
        self.assertEqual(done_body["status"], "complete")
        self.assertEqual(done_body["last_result"], "fbref_collector completed OK")
        self.assertEqual(done_body["fbref_team_stats_rows"], 2)
        self.assertEqual(predict_router._pipeline_status, "fbref_scrape_complete")

    def test_failed_subprocess_sets_failed_status_and_error_message(self):
        class FailingProc:
            def __init__(self, *_args, **_kwargs):
                self.stdout = iter(
                    ["[fbref] ERROR: WebDriver crashed unexpectedly\n"]
                )

            def wait(self):
                return 1

        with patch.object(fbref_scrape_runner.subprocess, "Popen", side_effect=FailingProc):
            res = self.client.post("/predict/admin/run-fbref-scrape?secret=top-secret")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json(), {"status": "started"})

            status_body = None
            for _ in range(50):
                status_body = self.client.get(
                    "/predict/admin/fbref-scrape-status?secret=top-secret"
                ).json()
                if status_body["status"] != "running":
                    break
                threading.Event().wait(0.02)

        assert status_body is not None
        self.assertEqual(status_body["status"], "failed")
        self.assertIn("WebDriver crashed", status_body["last_result"])
        self.assertEqual(predict_router._pipeline_status, "fbref_scrape_failed")


class FBrefCollectorSchemaAndResilienceTests(unittest.TestCase):
    """Verify schema migration, Big-Five filter, short-season MultiIndex handling, and partial failures."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_file = Path(self._tmp.name) / "fbref.db"
        self.engine = create_engine(f"sqlite:///{db_file}")
        self.Session = sessionmaker(bind=self.engine)
        database.Base.metadata.create_all(bind=self.engine)

        self._patches = [
            patch.object(database, "engine", self.engine),
            patch.object(database, "SessionLocal", self.Session),
            patch.object(fbref_collector, "SessionLocal", self.Session),
            patch.object(builder, "SessionLocal", self.Session),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def test_big_five_leagues_exact_constraint(self):
        self.assertEqual(
            fbref_collector.LEAGUES,
            [
                "ENG-Premier League",
                "ESP-La Liga",
                "GER-Bundesliga",
                "ITA-Serie A",
                "FRA-Ligue 1",
            ],
        )

    def test_additive_column_migration_never_drops_existing_columns(self):
        legacy_file = Path(self._tmp.name) / "legacy.db"
        legacy_engine = create_engine(f"sqlite:///{legacy_file}")
        with legacy_engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE TABLE fbref_team_stats ("
                "id INTEGER PRIMARY KEY, "
                "team VARCHAR, league VARCHAR, season VARCHAR, "
                "shots FLOAT, custom_keep_col VARCHAR)"
            )
            conn.exec_driver_sql(
                "INSERT INTO fbref_team_stats (team, league, season, shots, custom_keep_col) "
                "VALUES ('Arsenal', 'ENG-Premier League', '2025-2026', 512.0, 'preserved')"
            )

        self.assertTrue(database.ensure_fbref_team_stats_columns(legacy_engine))
        cols = {c["name"] for c in inspect(legacy_engine).get_columns("fbref_team_stats")}
        for model_col in database.FBrefTeamStats.__table__.columns:
            self.assertIn(model_col.name, cols)
        self.assertIn("custom_keep_col", cols)

        with legacy_engine.connect() as conn:
            row = conn.execute(
                text("SELECT shots, custom_keep_col FROM fbref_team_stats WHERE team = 'Arsenal'")
            ).fetchone()
        self.assertEqual(row, (512.0, "preserved"))

    def test_soccerdata_4digit_season_index_writes_canonical_seasons_and_all_23_columns(self):
        shooting_df, misc_df, keeper_df = _make_multiindex_stat_frames(
            {
                ("ENG-Premier League", "2425"): ["Arsenal", "Chelsea"],
                ("ENG-Premier League", "2627"): ["Arsenal", "Chelsea"],
                ("ESP-La Liga", "2627"): ["Barcelona", "Real Madrid"],
            }
        )
        fake_driver = MagicMock()
        fake_fbref = MagicMock()
        fake_fbref._driver = fake_driver
        fake_fbref.read_team_season_stats.side_effect = lambda stat_type: {
            "shooting": shooting_df,
            "misc": misc_df,
            "keeper": keeper_df,
        }[stat_type]

        with patch.object(fbref_collector.sd, "FBref", return_value=fake_fbref):
            first_run = fbref_collector.scrape_fbref_stats(
                seasons=["2024-2025", "2026-2027"]
            )
            second_run = fbref_collector.scrape_fbref_stats(
                seasons=["2024-2025", "2026-2027"]
            )

        self.assertEqual(first_run["inserted"], 6)
        self.assertEqual(first_run["updated"], 0)
        self.assertEqual(second_run["inserted"], 0)
        self.assertEqual(second_run["updated"], 6)
        self.assertTrue(fake_driver.quit.called)

        with self.Session() as db:
            arsenal = (
                db.query(database.FBrefTeamStats)
                .filter_by(
                    team="Arsenal",
                    league="ENG-Premier League",
                    season="2026-2027",
                )
                .one()
            )
            for stat_col in FBREF_STAT_COLUMNS:
                val = getattr(arsenal, stat_col)
                self.assertIsNotNone(val, f"{stat_col} was None")
                self.assertGreater(float(val), 0.0, f"{stat_col} was not > 0")

    def test_one_failing_stat_table_still_commits_remaining_tables(self):
        _, misc_df, keeper_df = _make_multiindex_stat_frames(
            {("ENG-Premier League", "2627"): ["Arsenal", "Chelsea"]}
        )
        fake_fbref = MagicMock()
        fake_fbref.read_team_season_stats.side_effect = [
            RuntimeError("shooting table transient 503"),
            misc_df,
            keeper_df,
        ]

        with patch.object(fbref_collector.sd, "FBref", return_value=fake_fbref):
            summary = fbref_collector.scrape_fbref_stats(
                leagues=["ENG-Premier League"],
                seasons=["2026-2027"],
            )

        self.assertEqual(summary["inserted"], 2)
        with self.Session() as db:
            arsenal = db.query(database.FBrefTeamStats).filter_by(team="Arsenal").one()
            self.assertEqual(arsenal.clean_sheets, 5.0)
            self.assertEqual(arsenal.yellow_cards, 18.0)
            self.assertEqual(arsenal.games, 10.0)


class EndToEndArsenalVsChelseaPredictionTests(unittest.TestCase):
    """End-to-end verification of the exact post-merge acceptance checks."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_file = Path(self._tmp.name) / "live.db"
        self.engine = create_engine(f"sqlite:///{db_file}")
        self.Session = sessionmaker(bind=self.engine)
        database.Base.metadata.create_all(bind=self.engine)

        self._patches = [
            patch.object(database, "engine", self.engine),
            patch.object(database, "SessionLocal", self.Session),
            patch.object(fbref_collector, "SessionLocal", self.Session),
            patch.object(builder, "SessionLocal", self.Session),
            patch.object(formula_engine, "SessionLocal", self.Session),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

        formula_engine.invalidate_cache()

    def test_same_day_scrape_populates_diagnostics_and_arsenal_vs_chelsea_features(self):
        pl_clubs = [
            "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Tottenham",
            "Newcastle Utd", "Aston Villa", "Brighton", "Fulham", "Brentford",
            "Bournemouth", "Crystal Palace", "Everton", "West Ham", "Wolves",
            "Nott'ham Forest", "Leicester City", "Ipswich Town", "Southampton", "Leeds United",
        ]
        shooting_df, misc_df, keeper_df = _make_multiindex_stat_frames(
            {
                ("ENG-Premier League", "2425"): pl_clubs,
                ("ENG-Premier League", "2526"): pl_clubs,
                ("ENG-Premier League", "2627"): pl_clubs,
            }
        )
        fake_fbref = MagicMock()
        fake_fbref.read_team_season_stats.side_effect = lambda stat_type: {
            "shooting": shooting_df,
            "misc": misc_df,
            "keeper": keeper_df,
        }[stat_type]

        # 1. Run collector right now (same UTC calendar day as prediction!)
        with patch.object(fbref_collector.sd, "FBref", return_value=fake_fbref):
            fbref_collector.scrape_fbref_stats(
                leagues=["ENG-Premier League"],
                seasons=["2024-2025", "2025-2026", "2026-2027"],
            )

        # 2. Seed FotMob-shaped team_season_stats rows (even with sh_shots=None on
        # 2024-2025 and sh_shots populated on 2026-2027) so both paths are tested.
        with self.Session() as db:
            for idx, club in enumerate(pl_clubs):
                db.add(
                    database.TeamSeasonStats(
                        team=club,
                        league="ENG-Premier League",
                        season="2026-2027",
                        sh_xg=16.5 - idx * 0.3,
                        sh_shots=150.0 - idx * 3.0,
                        sh_sot=54.0 - idx,
                        sh_goals=18.0 - idx * 0.4,
                        pos_poss=58.0 - idx * 0.6,
                        gk_saves=24.0 + idx,
                        gk_cs=5.0,
                        gk_ga=8.0 + idx * 0.5,
                        gk_w=6.0,
                        gk_d=2.0,
                        gk_l=2.0,
                        misc_corners=62.0 - idx,
                        misc_crs=160.0 - idx * 2,
                    )
                )
            db.commit()

        # 3. Verify builder reads the freshly scraped fbref_team_stats rows
        loaded_fbref = builder.load_fbref_stats()
        self.assertEqual(len(loaded_fbref), 60)

        # 4. Install lightweight trained ML models + features history into predict_router
        n_rows = 12
        history_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-08-15", periods=n_rows, freq="3D"),
                "home_team": ["Arsenal", "Chelsea"] * (n_rows // 2),
                "away_team": ["Chelsea", "Arsenal"] * (n_rows // 2),
                "league": ["ENG-Premier League"] * n_rows,
                "season": ["2026-2027"] * n_rows,
                "home_goals": [2, 1] * (n_rows // 2),
                "away_goals": [1, 1] * (n_rows // 2),
                "result": ["H", "D"] * (n_rows // 2),
                "home_corners": [7, 5] * (n_rows // 2),
                "away_corners": [4, 5] * (n_rows // 2),
                "home_shots": [15, 12] * (n_rows // 2),
                "away_shots": [10, 11] * (n_rows // 2),
                "home_shots_ot": [6, 4] * (n_rows // 2),
                "away_shots_ot": [3, 4] * (n_rows // 2),
            }
        )
        X_dummy = pd.DataFrame(
            np.ones((6, len(FEATURE_COLS))), columns=FEATURE_COLS
        )
        clf = Pipeline([("clf", DummyClassifier(strategy="uniform", random_state=0))]).fit(
            X_dummy, ["A", "D", "H", "A", "D", "H"]
        )
        reg = Pipeline([("reg", DummyRegressor(strategy="constant", constant=1.5))]).fit(
            X_dummy, [1.5] * 6
        )
        le_lg = LabelEncoder().fit(["ENG-Premier League"])
        le_sn = LabelEncoder().fit(["2024-2025", "2025-2026", "2026-2027"])

        saved_globals = {
            k: getattr(predict_router, k)
            for k in (
                "_result_model",
                "_xgb_result_model",
                "_home_goals",
                "_away_goals",
                "_home_corners",
                "_away_corners",
                "_le_league",
                "_le_season",
                "_features_df",
                "_fbref_df",
                "_fbref_load_attempted",
            )
        }
        self.addCleanup(lambda: predict_router.__dict__.update(saved_globals))

        predict_router._result_model = clf
        predict_router._xgb_result_model = None
        predict_router._home_goals = reg
        predict_router._away_goals = reg
        predict_router._home_corners = reg
        predict_router._away_corners = reg
        predict_router._le_league = le_lg
        predict_router._le_season = le_sn
        predict_router._features_df = history_df
        predict_router._refresh_fbref_caches()

        app = FastAPI()
        app.include_router(predict_router.router)
        client = TestClient(app)

        # Check diagnostics contract
        diag = client.get("/predict/admin/diagnostics").json()
        self.assertEqual(diag["fbref_source"], "database")
        self.assertEqual(diag["fbref_team_stats_rows"], 60)

        # Check Arsenal vs Chelsea prediction across default (2026-2027) AND 2024-2025
        for req_payload in (
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "league": "ENG-Premier League",
            },
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "league": "ENG-Premier League",
                "season": "2024-2025",
            },
        ):
            with self.subTest(payload=req_payload):
                pred_res = client.post("/predict/match", json=req_payload)
                self.assertEqual(pred_res.status_code, 200, pred_res.text)
                pred = pred_res.json()
                fv = pred["feature_vector_used"]
                self.assertEqual(pred["prediction_method"], "blended")
                self.assertGreater(fv["home_fbref_shots"], 0.0)
                self.assertGreater(fv["home_fbref_clean_sheets"], 0.0)
                self.assertGreater(fv["home_shots_per_game"], 0.0)


class WorkflowContractAndShellTests(unittest.TestCase):
    """Validate .github/workflows/daily_fbref_update.yml structure and execute its polling bash loop."""

    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        self.steps = self.workflow["jobs"]["collect-and-retrain"]["steps"]

    def test_workflow_never_runs_fbref_collector_on_github_actions_and_preserves_order(self):
        step_names = [s.get("name", "") for s in self.steps]
        for step in self.steps:
            run_script = step.get("run", "")
            self.assertNotIn(
                "fbref_collector.py",
                run_script,
                f"GitHub Actions step {step.get('name')!r} must NOT execute fbref_collector.py",
            )

        def _idx(prefix: str) -> int:
            for idx, name in enumerate(step_names):
                if prefix in name:
                    return idx
            self.fail(f"Step matching {prefix!r} missing from {step_names}")

        self.assertLess(_idx("[1/5] Run StatsBomb"), _idx("[2/5] Run FotMob"))
        self.assertLess(_idx("[2/5] Run FotMob"), _idx("[2/5] Run Understat"))
        self.assertLess(_idx("[2/5] Run Understat"), _idx("[3/5] Run Match Collector"))
        self.assertLess(
            _idx("[3/5] Run Match Collector"),
            _idx("[4/5] Trigger FBref scrape on Render"),
        )
        self.assertLess(
            _idx("[4/5] Trigger FBref scrape on Render"),
            _idx("[4/5] Wait for FBref scrape on Render"),
        )
        self.assertLess(
            _idx("[4/5] Wait for FBref scrape on Render"),
            _idx("[5/5] Build features (builder.py)"),
        )
        self.assertLess(
            _idx("[5/5] Build features (builder.py)"),
            _idx("[5/5] Train + upload models (trainer.py)"),
        )
        self.assertLess(
            _idx("[5/5] Train + upload models (trainer.py)"),
            _idx("[6/6] Notify Render to reload models"),
        )

    def _run_wait_step_against_responses(self, responses: list[dict]) -> tuple[int, str, int]:
        calls = {"count": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                idx = min(calls["count"], len(responses) - 1)
                calls["count"] += 1
                payload = json.dumps(responses[idx]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            wait_step = next(
                s for s in self.steps if "Wait for FBref scrape on Render" in s.get("name", "")
            )
            env = dict(os.environ)
            env["RENDER_BACKEND"] = f"http://127.0.0.1:{port}"
            env["PIPELINE_SECRET"] = "top-secret"
            env["FBREF_POLL_INTERVAL_SECONDS"] = "0"
            proc = subprocess.run(
                ["bash", "-c", wait_step["run"]],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            return proc.returncode, proc.stdout + proc.stderr, calls["count"]
        finally:
            server.shutdown()
            server.server_close()

    def test_wait_step_exits_0_immediately_when_complete(self):
        rc, out, polls = self._run_wait_step_against_responses(
            [
                {
                    "status": "running",
                    "last_run_at": "2026-10-20T21:00:00+00:00",
                    "last_result": "fbref_collector running",
                    "fbref_team_stats_rows": 120,
                },
                {
                    "status": "complete",
                    "last_run_at": "2026-10-20T21:03:00+00:00",
                    "last_result": "fbref_collector completed OK",
                    "fbref_team_stats_rows": 580,
                },
            ]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(polls, 2)
        self.assertIn("fbref_team_stats_rows=580", out)

    def test_wait_step_exits_0_with_warning_when_failed(self):
        rc, out, polls = self._run_wait_step_against_responses(
            [
                {
                    "status": "failed",
                    "last_run_at": "2026-10-20T21:01:00+00:00",
                    "last_result": "CAPTCHA detected",
                    "fbref_team_stats_rows": 420,
                }
            ]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(polls, 1)
        self.assertIn("::warning::", out)

    def test_wait_step_exits_0_after_24_polls_timeout(self):
        rc, out, polls = self._run_wait_step_against_responses(
            [
                {
                    "status": "running",
                    "last_run_at": "2026-10-20T21:00:00+00:00",
                    "last_result": "fbref_collector running",
                    "fbref_team_stats_rows": 200,
                }
            ]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(polls, 24)
        self.assertIn("::warning::FBref scrape did not finish within 12 minutes", out)


if __name__ == "__main__":
    unittest.main()
