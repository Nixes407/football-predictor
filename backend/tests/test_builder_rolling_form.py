"""Guard rolling-form totals and strict historical cutoff after reduction reuse."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.builder import compute_team_rolling_form


class RollingFormTests(unittest.TestCase):
    def matches(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "home_team": ["A", "B", "A"], "away_team": ["B", "A", "B"],
            "home_goals": [2.0, None, 99.0], "away_goals": [1.0, 3.0, 99.0],
            "result": ["H", "A", "D"],
            "home_shots": [5, 6, 99], "away_shots": [3, 4, 99],
            "home_shots_ot": [3, 2, 99], "away_shots_ot": [1, 2, 99],
            "home_corners": [4, 2, 99], "away_corners": [2, 3, 99],
        })

    def test_totals_all_venues_and_windows(self):
        for venue, window, gf, ga, wins in [
            ("all", 5, 5.0, 1.0, 2), ("home", 5, 2.0, 1.0, 1),
            ("away", 5, 3.0, 0.0, 1), ("all", 1, 3.0, 0.0, 1),
            ("all", 10, 5.0, 1.0, 2),
        ]:
            with self.subTest(venue=venue, window=window):
                result = compute_team_rolling_form(
                    self.matches(), "A", pd.Timestamp("2024-01-03"), window, venue
                )
                self.assertEqual(result["gf"], gf)
                self.assertEqual(result["ga"], ga)
                self.assertEqual(result["gd"], gf - ga)
                self.assertEqual(result["wins"], wins)
                self.assertEqual(result["points"], wins * 3)

    def test_empty_history_is_zero(self):
        for venue in ("all", "home", "away"):
            result = compute_team_rolling_form(
                self.matches(), "A", pd.Timestamp("2024-01-01"), venue=venue
            )
            self.assertTrue(all(value == 0 for value in result.values()))


if __name__ == "__main__":
    unittest.main()
