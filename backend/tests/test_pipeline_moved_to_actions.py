"""Guardrails for the "training moved to GitHub Actions" architecture.

The Render free tier (512 MB) OOM-killed the service while builder.py ran on
the expanded ~15,000-row dataset. These tests pin the new contract:

* ``/admin/run-pipeline`` and ``/admin/run-pipeline-fast`` answer 410 Gone and
  no longer import ``subprocess`` — Render can never spawn builder/trainer.
* startup only reloads pre-built artifacts from Supabase Storage.
* ``/admin/reload-models`` downloads the bucket into memory, reports
  ``features_rows`` + ``models_loaded``, and leaves the previously served
  models untouched when the download fails.
"""

from __future__ import annotations

import ast
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _source(name: str) -> str:
    return (BACKEND_DIR / name).read_text()


def _functions(source: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source)
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


class HeavyEndpointsAreGoneTests(unittest.TestCase):
    def test_predict_router_cannot_spawn_subprocesses(self):
        tree = ast.parse(_source("routers/predict.py"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("subprocess", imported)

    def test_run_pipeline_functions_only_return_the_gone_response(self):
        functions = _functions(_source("routers/predict.py"))
        for name in ("run_pipeline", "run_pipeline_fast"):
            with self.subTest(function=name):
                fn = functions[name]
                # decorator: @router.post("/admin/...", status_code=410)
                decorator = fn.decorator_list[0]
                status_codes = {
                    kw.value.value
                    for kw in decorator.keywords
                    if kw.arg == "status_code"
                }
                self.assertEqual(status_codes, {410})

                calls = [
                    node.func.id
                    for node in ast.walk(fn)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                ]
                self.assertEqual(calls, ["JSONResponse"])

    def test_old_auto_start_is_gone_and_startup_only_reloads(self):
        predict_source = _source("routers/predict.py")
        functions = _functions(predict_source)
        self.assertNotIn("auto_start_fast_pipeline", functions)
        self.assertIn("startup_reload_models", functions)

        main_source = _source("main.py")
        self.assertNotIn("auto_start_fast_pipeline", main_source)
        self.assertIn("startup_reload_models", main_source)
        # The Monday FBref scrape that used to run on Render is gone too.
        self.assertNotIn("scrape_fbref_stats", main_source)


class ReloadEndpointTests(unittest.TestCase):
    """Exercise the real router with a fake Supabase Storage client."""

    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from routers import predict

        cls.predict = predict
        cls.app = FastAPI()
        cls.app.include_router(predict.router)
        cls.client = TestClient(cls.app)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmp.name)
        self._env = patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-key",
                "PIPELINE_SECRET": "test-secret",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._tmp.cleanup)

        real_local_path = self.predict.artifacts.local_path

        def local_path(object_name, backend_dir=None):
            return real_local_path(object_name, backend_dir or self.tmp_root)

        patches = [
            patch.object(self.predict, "BACKEND_DIR", self.tmp_root),
            patch.object(self.predict.artifacts, "local_path", local_path),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    # ------------------------------------------------------------------ #
    # Fixtures
    # ------------------------------------------------------------------ #
    def _files(self, include_all=True, features_rows=3):
        payload = {}
        model_names = list(self.predict.artifacts.MODEL_OBJECTS)
        if not include_all:
            model_names = model_names[:1]
        for name in model_names:
            buffer = io.BytesIO()
            joblib.dump({"artifact": name}, buffer)
            payload[name] = buffer.getvalue()
        for name in self.predict.artifacts.ENCODER_OBJECTS:
            buffer = io.BytesIO()
            joblib.dump(["ENG-Premier League"], buffer)
            payload[name] = buffer.getvalue()
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=features_rows),
                "home_team": ["A"] * features_rows,
                "away_team": ["B"] * features_rows,
                "league": ["ENG-Premier League"] * features_rows,
            }
        )
        payload[self.predict.artifacts.FEATURES_OBJECT] = (
            frame.to_csv(index=False).encode()
        )
        return payload

    def _fake_client(self, files, failing=None):
        class FakeBucket:
            def download(self, name):
                if failing and name == failing:
                    raise RuntimeError(f"network error for {name}")
                if name not in files:
                    raise RuntimeError(f"object not found: {name}")
                return files[name]

        class FakeStorage:
            def from_(self, _bucket):
                return FakeBucket()

        class FakeClient:
            storage = FakeStorage()

        return FakeClient()

    # ------------------------------------------------------------------ #
    # Tests
    # ------------------------------------------------------------------ #
    def test_heavy_endpoints_return_410(self):
        for path in ("/predict/admin/run-pipeline",
                     "/predict/admin/run-pipeline-fast"):
            with self.subTest(path=path):
                response = self.client.post(path)
                self.assertEqual(response.status_code, 410)
                self.assertEqual(
                    response.json()["error"],
                    "training moved to GitHub Actions",
                )

    def test_reload_requires_pipeline_secret(self):
        response = self.client.post("/predict/admin/reload-models")
        self.assertEqual(response.status_code, 403)

        response = self.client.post(
            "/predict/admin/reload-models?secret=wrong"
        )
        self.assertEqual(response.status_code, 403)

    def test_reload_downloads_models_and_features(self):
        files = self._files()
        with patch.object(
            self.predict, "create_client",
            return_value=self._fake_client(files),
        ):
            response = self.client.post(
                "/predict/admin/reload-models?secret=test-secret"
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "reloaded")
        self.assertEqual(body["features_rows"], 3)
        self.assertEqual(
            sorted(body["models_loaded"]),
            sorted(self.predict.artifacts.MODEL_OBJECTS),
        )
        self.assertEqual(self.predict._pipeline_status, "complete")
        self.assertFalse(self.predict._pipeline_running)
        self.assertIsNotNone(self.predict._features_df)
        self.assertIsNotNone(self.predict._result_model)
        # Artifacts really landed in the (temporary) backend directory.
        self.assertTrue(
            (self.tmp_root / "models" / "result_model_v1.joblib").exists()
        )
        self.assertTrue(
            (self.tmp_root / "data" / "processed" / "features.csv").exists()
        )
        # No staging directory is left behind.
        self.assertEqual(
            [p.name for p in self.tmp_root.iterdir()
             if p.name.startswith(".staging-")],
            [],
        )

    def test_missing_optional_models_still_reload(self):
        files = self._files()
        for name in self.predict.artifacts.OPTIONAL_DOWNLOAD_OBJECTS:
            files.pop(name, None)

        with patch.object(
            self.predict, "create_client",
            return_value=self._fake_client(files),
        ):
            response = self.client.post(
                "/predict/admin/reload-models?secret=test-secret"
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "reloaded")
        for name in self.predict.artifacts.OPTIONAL_DOWNLOAD_OBJECTS:
            self.assertNotIn(name, body["models_loaded"])
        # The core set is still served.
        self.assertIsNotNone(self.predict._result_model)
        self.assertIsNotNone(self.predict._home_goals)

    def test_failed_reload_keeps_previous_models_and_reports_503(self):
        sentinel = object()
        self.predict._result_model = sentinel
        self.predict._pipeline_status = "complete"

        files = self._files()
        client = self._fake_client(files, failing="home_goals_v1.joblib")
        with patch.object(self.predict, "create_client", return_value=client):
            response = self.client.post(
                "/predict/admin/reload-models?secret=test-secret"
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.predict._pipeline_status, "error_reload")
        self.assertFalse(self.predict._pipeline_running)
        # The old model is still the one being served.
        self.assertIs(self.predict._result_model, sentinel)
        # A partial download must not have replaced anything on disk.
        self.assertFalse(
            (self.tmp_root / "models" / "result_model_v1.joblib").exists()
        )

    def test_reload_reports_missing_credentials(self):
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": "",
             "SUPABASE_KEY": ""},
        ):
            response = self.client.post(
                "/predict/admin/reload-models?secret=test-secret"
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("SUPABASE_URL", response.json()["detail"])

    def test_startup_hook_never_trains(self):
        calls = []

        def fake_reload():
            calls.append("reload")
            return {}

        captured = {}

        class FakeThread:
            def __init__(self, target, name=None, daemon=None):
                captured["name"] = name
                captured["daemon"] = daemon
                self._target = target

            def start(self):
                self._target()

        with patch.object(
            self.predict, "reload_models_from_supabase", fake_reload
        ), patch.object(self.predict.threading, "Thread", FakeThread), \
                patch.dict(os.environ, {"SKIP_STARTUP_MODEL_RELOAD": ""}):
            self.predict.startup_reload_models()

        self.assertEqual(calls, ["reload"])
        self.assertEqual(captured["name"], "startup-model-reload")
        self.assertTrue(captured["daemon"])


class TrainerUploadTests(unittest.TestCase):
    """``--strict-upload`` must fail the CI job instead of publishing a
    mixed set of old and new artifacts."""

    def setUp(self):
        from models import trainer

        self.trainer = trainer
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_file = Path(self._tmp.name) / "result_model_v1.joblib"
        self.model_file.write_bytes(b"model")

    def _client(self, fail=False):
        uploaded = []

        class FakeBucket:
            def upload(self, path=None, file=None, file_options=None):
                if fail:
                    raise RuntimeError("storage unavailable")
                uploaded.append(path)

        class FakeStorage:
            def from_(self, _bucket):
                return FakeBucket()

        class FakeClient:
            storage = FakeStorage()

        return FakeClient(), uploaded

    def test_strict_upload_fails_on_missing_artifacts(self):
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x.supabase.co",
             "SUPABASE_SERVICE_ROLE_KEY": "key"},
        ), patch.object(
            self.trainer, "_artifacts_to_upload",
            return_value=[(Path(self._tmp.name) / "nope.joblib",
                           "nope.joblib")],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.trainer.upload_models_to_supabase(
                    self._tmp.name, strict=True
                )
        self.assertIn("nope.joblib", str(ctx.exception))

    def test_strict_upload_fails_when_an_upload_fails(self):
        client, _ = self._client(fail=True)
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x.supabase.co",
             "SUPABASE_SERVICE_ROLE_KEY": "key"},
        ), patch.object(
            self.trainer, "_artifacts_to_upload",
            return_value=[(self.model_file, "result_model_v1.joblib")],
        ), patch.object(self.trainer, "create_client", return_value=client):
            with self.assertRaises(RuntimeError):
                self.trainer.upload_models_to_supabase(
                    self._tmp.name, strict=True
                )

    def test_non_strict_upload_is_best_effort(self):
        client, uploaded = self._client()
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x.supabase.co",
             "SUPABASE_SERVICE_ROLE_KEY": "key"},
        ), patch.object(
            self.trainer, "_artifacts_to_upload",
            return_value=[(self.model_file, "result_model_v1.joblib")],
        ), patch.object(self.trainer, "create_client", return_value=client):
            report = self.trainer.upload_models_to_supabase(self._tmp.name)
        self.assertEqual(report["uploaded"], ["result_model_v1.joblib"])
        self.assertEqual(uploaded, ["result_model_v1.joblib"])

    def test_cli_exits_1_when_training_produced_nothing(self):
        with patch.object(self.trainer, "train_all_models", return_value={}):
            self.assertEqual(
                self.trainer.main(["--no-upload"]), 1
            )

    def test_cli_exits_1_when_a_model_file_is_missing(self):
        results = {"result": {"accuracy": 0.5}}
        with patch.object(
            self.trainer, "train_all_models", return_value=results
        ):
            self.assertEqual(
                self.trainer.main(
                    ["--no-upload", "--model-dir", self._tmp.name]
                ),
                1,
            )


class ArtifactManifestTests(unittest.TestCase):
    def test_required_and_optional_are_disjoint_and_cover_the_bucket(self):
        from model_artifacts import (
            MODEL_OBJECTS,
            OPTIONAL_DOWNLOAD_OBJECTS,
            REQUIRED_DOWNLOAD_OBJECTS,
            UPLOAD_OBJECTS,
        )

        self.assertEqual(len(MODEL_OBJECTS), 10)  # 9 models + XGBoost encoder
        self.assertFalse(
            set(REQUIRED_DOWNLOAD_OBJECTS) & set(OPTIONAL_DOWNLOAD_OBJECTS)
        )
        self.assertEqual(
            set(UPLOAD_OBJECTS),
            set(MODEL_OBJECTS) | {"league_encoder.joblib",
                                  "season_encoder.joblib", "features.csv"},
        )
        self.assertIn("features.csv", REQUIRED_DOWNLOAD_OBJECTS)
        self.assertIn("result_model_v1.joblib", REQUIRED_DOWNLOAD_OBJECTS)

    def test_local_paths_match_the_trainer_layout(self):
        from model_artifacts import local_path

        self.assertEqual(
            local_path("result_model_v1.joblib").parent.name, "models"
        )
        self.assertEqual(
            local_path("league_encoder.joblib"),
            BACKEND_DIR / "data" / "league_encoder.joblib",
        )
        self.assertEqual(
            local_path("features.csv"),
            BACKEND_DIR / "data" / "processed" / "features.csv",
        )

    def test_service_role_key_wins_over_legacy_key(self):
        from model_artifacts import supabase_credentials

        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x.supabase.co/",
             "SUPABASE_SERVICE_ROLE_KEY": "service",
             "SUPABASE_KEY": "legacy"},
        ):
            self.assertEqual(
                supabase_credentials(),
                ("https://x.supabase.co", "service"),
            )

        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x.supabase.co",
             "SUPABASE_SERVICE_ROLE_KEY": "",
             "SUPABASE_KEY": "legacy"},
        ):
            self.assertEqual(
                supabase_credentials(), ("https://x.supabase.co", "legacy")
            )


if __name__ == "__main__":
    unittest.main()
