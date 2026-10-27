"""Regression tests: the builder emits no team-season snapshot columns.

The 46 ``home_fbref_*`` / ``away_fbref_*`` columns were always zero for live
fixtures and were removed from the ML pipeline. Neither the training path nor
the single-match path may query that table, zero-fill those keys or emit them
into the saved features CSV.

No live database or network access is used.

Run from the repository root:
    python -m unittest discover -s backend/tests -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from features import builder  # noqa: E402
from features.feature_columns import FEATURE_COLS  # noqa: E402


def synthetic_matches() -> pd.DataFrame:
    """Three played matches with every column the rolling helpers read."""
    return pd.DataFrame({
        "date": pd.to_datetime(
            ["2024-01-05", "2024-01-12", "2024-01-19"]
        ),
        "home_team": ["Arsenal", "Chelsea", "Arsenal"],
        "away_team": ["Chelsea", "Arsenal", "Liverpool"],
        "league": ["ENG-Premier League"] * 3,
        "season": ["2023-2024"] * 3,
        "home_goals": [2, 1, 0],
        "away_goals": [1, 1, 2],
        "result": ["H", "D", "A"],
        "home_shots": [14, 9, 7],
        "away_shots": [8, 11, 16],
        "home_shots_ot": [6, 4, 3],
        "away_shots_ot": [3, 5, 8],
        "home_corners": [7, 4, 2],
        "away_corners": [3, 5, 9],
    })


def snapshot_keys(frame: pd.DataFrame | dict) -> list[str]:
    keys = frame.columns if isinstance(frame, pd.DataFrame) else frame.keys()
    return [
        str(key) for key in keys
        if str(key).startswith(("home_fbref_", "away_fbref_"))
    ]


class BuilderHasNoSnapshotFeaturesTests(unittest.TestCase):
    def test_builder_no_fbref_keys(self):
        """Required regression: mocked-DB feature row carries no fbref keys."""
        matches = synthetic_matches()
        with patch.object(builder, "load_matches_from_db", return_value=matches), \
                patch.object(builder, "load_elo_from_db", return_value=pd.DataFrame()), \
                patch.object(builder.joblib, "dump"):
            row = builder.compute_match_features(
                matches, pd.DataFrame(), "Arsenal", "Chelsea",
                pd.Timestamp("2024-01-20"),
            )

        self.assertTrue(row)  # form / Elo / H2H / specialist rates still there
        self.assertEqual(snapshot_keys(row), [])
        self.assertIn("home_elo", row)
        self.assertIn("home_form_gf_5", row)

    def test_build_and_save_features_never_reads_the_snapshot_table(self):
        """Training path: no snapshot loader, no join, full 62-column schema."""
        matches = synthetic_matches()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder, "load_matches_from_db", return_value=matches), \
                patch.object(builder, "load_elo_from_db", return_value=pd.DataFrame()), \
                patch.object(builder.joblib, "dump"):
            path = str(Path(directory) / "features.csv")
            built = builder.build_and_save_features(path)
            saved = pd.read_csv(path)

        self.assertEqual(len(built), len(matches))
        self.assertEqual(snapshot_keys(built), [])
        self.assertEqual(snapshot_keys(saved), [])
        # The trainer's contract is still fully present in the saved CSV.
        self.assertEqual(len(FEATURE_COLS), 62)
        self.assertTrue(set(FEATURE_COLS).issubset(saved.columns))
        self.assertFalse(any("updated_at" in str(c) for c in saved.columns))

    def test_snapshot_helpers_are_gone_from_the_builder(self):
        for name in ("load_fbref_stats", "join_fbref_features",
                     "normalise_season_label", "FBREF_STAT_COLUMNS",
                     "FBREF_KEY_COLUMNS", "FBREF_COLUMNS", "SessionLocal"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(builder, name))
        source = (BACKEND_DIR / "features" / "builder.py").read_text()
        self.assertNotIn("fbref", source.lower())

    def test_feature_columns_module_keeps_only_the_deprecated_reference(self):
        source = (BACKEND_DIR / "features" / "feature_columns.py").read_text()
        lower = source.lower()
        self.assertIn("# Deprecated — FBref features removed from ML pipeline. "
                      "Kept for reference only.", source)
        self.assertNotIn("home_fbref_", lower)
        self.assertNotIn("away_fbref_", lower)
        self.assertIn(
            'assert len(FEATURE_COLS) == 62, '
            'f"Expected 62 FEATURE_COLS, got {len(FEATURE_COLS)}"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
