"""Exercise pipeline runners without importing API startup/model side effects."""
import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class PipelineTimeoutTests(unittest.TestCase):
    def runner(self, name, timeout=False):
        path = Path(__file__).resolve().parents[1] / "routers" / "predict.py"
        tree = ast.parse(path.read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {
                    "PIPELINE_STEP_TIMEOUT_SECONDS", "BUILDER_TIMEOUT_SECONDS"
                } for t in node.targets
            ):
                nodes.append(node)
            if isinstance(node, ast.FunctionDef) and node.name == name:
                node.decorator_list = []
                nodes.append(node)
        calls = []

        def run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if timeout and cmd[-1] == "features/builder.py":
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        namespace = {
            "os": os, "BACKEND_DIR": path.parents[1],
            "subprocess": SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
            "threading": SimpleNamespace(Thread=lambda target, **kw: SimpleNamespace(start=target)),
            "_pipeline_running": False, "_pipeline_status": "idle", "_pipeline_log": [],
            "_pipeline_error": "", "load_artifacts": Mock(),
            "download_models_from_supabase": lambda: False,
            **{key: None for key in ("_result_model", "_xgb_result_model", "_home_goals",
                                     "_away_goals", "_home_corners", "_away_corners")},
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        with patch.dict(os.environ, {"PIPELINE_SECRET": "test-secret"}):
            if name == "auto_start_fast_pipeline":
                namespace[name]()
            else:
                namespace[name](secret="test-secret")
        return namespace, calls

    def test_builder_gets_six_hours_in_all_runners(self):
        for name in ("run_pipeline", "run_pipeline_fast", "auto_start_fast_pipeline"):
            with self.subTest(runner=name):
                ns, calls = self.runner(name)
                self.assertTrue(any(cmd[-1] == "features/builder.py" for cmd, _ in calls))
                for cmd, kwargs in calls:
                    expected = 21600 if cmd[-1] == "features/builder.py" else 7200
                    self.assertEqual(kwargs["timeout"], expected)
                self.assertEqual(ns["_pipeline_status"], "complete")
                self.assertFalse(ns["_pipeline_running"])
                ns["load_artifacts"].assert_called_once()

    def test_builder_timeout_stops_before_trainer_and_clears_running(self):
        for name in ("run_pipeline", "run_pipeline_fast", "auto_start_fast_pipeline"):
            with self.subTest(runner=name):
                ns, calls = self.runner(name, timeout=True)
                self.assertEqual(ns["_pipeline_status"], "timeout_builder")
                self.assertFalse(ns["_pipeline_running"])
                self.assertFalse(any(cmd[-1] == "models/trainer.py" for cmd, _ in calls))
                ns["load_artifacts"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
