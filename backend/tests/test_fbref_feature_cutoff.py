"""Regression tests for collection-time-safe FBref training features.

Run from the repository root:
    python -m unittest discover -s backend/tests -v

No live database or FBref requests are used.
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import builder


class FBrefFeatureCutoffTests(unittest.TestCase):
    def matches(self, dates=("2024-01-15",)):
        return pd.DataFrame({
            "date": list(dates),
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league": "ENG-Premier League",
            "season": "2023-2024",
            "home_form_gf_5": 7.0,
            "home_elo": 1600.0,
            "h2h_home_wins_5": 2,
            "target_result": "H",
            "target_home_goals": 2,
        })

    def snapshot(self, team="Arsenal", updated_at="2024-01-01T00:00:00+00:00",
                 value=12.0, **overrides):
        return {
            "team": team,
            "league": "ENG-Premier League",
            "season": "2023-2024",
            "updated_at": updated_at,
            **{col: value for col in builder.FBREF_STAT_COLUMNS},
            **overrides,
        }

    def assert_stats(self, result, role, expected, row=0):
        for col in builder.FBREF_STAT_COLUMNS:
            self.assertEqual(result.iloc[row][f"{role}_fbref_{col}"], expected)

    def test_january_match_cannot_see_march_stats(self):
        matches = self.matches()
        stats = pd.DataFrame([
            self.snapshot(updated_at="2024-03-01T00:00:00Z"),
            self.snapshot("Chelsea", "2024-03-01T00:00:00Z"),
        ])
        result = builder.join_fbref_features(matches, stats)
        self.assert_stats(result, "home", 0.0)
        self.assert_stats(result, "away", 0.0)
        pd.testing.assert_frame_equal(result[matches.columns], matches)

    def test_cutoff_applies_independently_to_both_teams_and_each_match(self):
        matches = self.matches(("2024-01-15", "2024-04-15"))
        stats = pd.DataFrame([
            self.snapshot(), self.snapshot("Chelsea", "2024-03-01T00:00:00Z", 20),
        ])
        result = builder.join_fbref_features(matches, stats)
        self.assert_stats(result, "home", 12.0, row=0)
        self.assert_stats(result, "away", 0.0, row=0)
        self.assert_stats(result, "home", 12.0, row=1)
        self.assert_stats(result, "away", 20.0, row=1)

    def test_equal_timestamp_allowed_but_later_on_match_day_is_not(self):
        stats = pd.DataFrame([
            self.snapshot(updated_at="2024-01-15T00:00:00Z"),
            self.snapshot("Chelsea", "2024-01-15T12:00:00Z"),
        ])
        result = builder.join_fbref_features(self.matches(), stats)
        self.assert_stats(result, "home", 12.0)
        self.assert_stats(result, "away", 0.0)

    def test_missing_or_invalid_collection_timestamp_fails_closed(self):
        for timestamp in (None, "", "not-a-date", "2024-99-99"):
            with self.subTest(timestamp=timestamp):
                stats = pd.DataFrame([self.snapshot(updated_at=timestamp)])
                result = builder.join_fbref_features(self.matches(), stats)
                self.assert_stats(result, "home", 0.0)
                self.assert_stats(result, "away", 0.0)

    def test_absent_timestamp_column_fails_closed(self):
        stats = pd.DataFrame([self.snapshot()]).drop(columns="updated_at")
        result = builder.join_fbref_features(self.matches(), stats)
        self.assert_stats(result, "home", 0.0)

    def test_timezone_offsets_and_naive_iso_values_are_compared_in_utc(self):
        stats = pd.DataFrame([
            self.snapshot(updated_at="2024-01-15T01:00:00+01:00"),
            self.snapshot("Chelsea", "2024-01-15T00:00:01"),
        ])
        result = builder.join_fbref_features(self.matches(), stats)
        self.assert_stats(result, "home", 12.0)
        self.assert_stats(result, "away", 0.0)

    def test_invalid_match_date_cannot_receive_stats(self):
        result = builder.join_fbref_features(
            self.matches((None, "bad-date")), pd.DataFrame([self.snapshot()])
        )
        self.assert_stats(result, "home", 0.0, row=0)
        self.assert_stats(result, "home", 0.0, row=1)

    def test_empty_fbref_data_still_emits_all_46_stat_features(self):
        matches = self.matches()
        result = builder.join_fbref_features(matches, pd.DataFrame())
        self.assertEqual(len(result.columns) - len(matches.columns), 46)
        self.assert_stats(result, "home", 0.0)
        self.assert_stats(result, "away", 0.0)
        pd.testing.assert_frame_equal(result[matches.columns], matches)

    def test_aliases_are_normalised_before_dedup_and_latest_timestamp_wins(self):
        matches = self.matches()
        matches["home_team"] = "Man United"
        # Latest first: selection must not depend on input order.
        stats = pd.DataFrame([
            self.snapshot("Manchester United", "2024-01-10T00:00:00Z", 30),
            self.snapshot("Man United", "2024-01-01T00:00:00Z", 10),
        ])
        result = builder.join_fbref_features(matches, stats)
        self.assertEqual(len(result), len(matches))
        self.assertEqual(result.iloc[0]["home_team"], "Manchester Utd")
        self.assert_stats(result, "home", 30.0)

    def test_other_league_or_season_does_not_match(self):
        stats = pd.DataFrame([
            self.snapshot(league="OTHER"),
            self.snapshot("Chelsea", season="2022-2023"),
        ])
        result = builder.join_fbref_features(self.matches(), stats)
        self.assert_stats(result, "home", 0.0)
        self.assert_stats(result, "away", 0.0)

    def test_missing_and_nonnumeric_stats_are_zero_not_metadata_features(self):
        stats = pd.DataFrame([self.snapshot(shots=None, saves="unavailable")])
        stats = stats.drop(columns="games")
        result = builder.join_fbref_features(self.matches(), stats)
        self.assertEqual(result.iloc[0]["home_fbref_shots"], 0.0)
        self.assertEqual(result.iloc[0]["home_fbref_saves"], 0.0)
        self.assertEqual(result.iloc[0]["home_fbref_games"], 0.0)
        self.assertFalse(any("updated_at" in col for col in result.columns))

    def test_input_order_index_and_frames_are_preserved(self):
        matches = self.matches(("2024-04-15", "2024-01-15"))
        matches.index = [7, 3]
        stats = pd.DataFrame([self.snapshot(updated_at="2024-03-01T00:00:00Z")])
        original_matches, original_stats = matches.copy(), stats.copy()
        result = builder.join_fbref_features(matches, stats)
        pd.testing.assert_frame_equal(result[matches.columns], matches)
        pd.testing.assert_frame_equal(matches, original_matches)
        pd.testing.assert_frame_equal(stats, original_stats)
        self.assert_stats(result, "home", 12.0, row=0)
        self.assert_stats(result, "home", 0.0, row=1)

    def test_rejoining_does_not_retain_previously_leaked_features(self):
        matches = self.matches().assign(home_fbref_shots=999.0)
        stats = pd.DataFrame([self.snapshot(updated_at="2024-03-01T00:00:00Z")])
        result = builder.join_fbref_features(matches, stats)
        self.assert_stats(result, "home", 0.0)
        self.assertFalse(any(col.endswith(("_x", "_y")) for col in result.columns))

    def test_join_error_falls_back_to_full_zero_feature_schema(self):
        matches = self.matches()
        malformed = pd.DataFrame({"unexpected": ["bad snapshot"]})
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder, "load_matches_from_db", return_value=matches), \
                patch.object(builder, "build_features", return_value=matches), \
                patch.object(builder, "load_elo_from_db", return_value=pd.DataFrame()), \
                patch.object(builder, "load_fbref_stats", return_value=malformed), \
                patch.object(builder.joblib, "dump"):
            result = builder.build_and_save_features(str(Path(directory) / "features.csv"))
        self.assertEqual(len(result), len(matches))
        self.assert_stats(result, "home", 0.0)
        self.assert_stats(result, "away", 0.0)

    def test_loader_includes_collection_metadata(self):
        record = MagicMock()
        for column, value in self.snapshot().items():
            setattr(record, column, value)
        db = MagicMock()
        db.query.return_value.all.return_value = [record]
        with patch.object(builder, "SessionLocal", return_value=db):
            loaded = builder.load_fbref_stats()
        self.assertEqual(loaded.iloc[0]["updated_at"], record.updated_at)
        db.close.assert_called_once()

    def test_saved_csv_is_safe_and_has_the_trainers_full_feature_schema(self):
        import ast

        match = self.matches().assign(
            result="H", home_goals=2, away_goals=1,
            home_shots=10, away_shots=8, home_shots_ot=4, away_shots_ot=3,
            home_corners=5, away_corners=4,
        )
        stats = pd.DataFrame([self.snapshot(updated_at="2024-03-01T00:00:00Z")])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder, "load_matches_from_db", return_value=match), \
                patch.object(builder, "load_elo_from_db", return_value=pd.DataFrame()), \
                patch.object(builder, "load_fbref_stats", return_value=stats), \
                patch.object(builder.joblib, "dump"):
            path = str(Path(directory) / "features.csv")
            result = builder.build_and_save_features(path)
            saved = pd.read_csv(path)
        # Read the existing trainer declaration without importing it or training.
        trainer = ast.parse((Path(builder.__file__).parents[1] / "models/trainer.py").read_text())
        feature_cols = next(
            ast.literal_eval(node.value) for node in trainer.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "FEATURE_COLS" for t in node.targets)
        )
        self.assertEqual(len(feature_cols), 108)
        self.assertTrue(set(feature_cols).issubset(saved.columns))
        self.assert_stats(result, "home", 0.0)
        self.assertFalse(any("updated_at" in col for col in saved.columns))


if __name__ == "__main__":
    unittest.main()
