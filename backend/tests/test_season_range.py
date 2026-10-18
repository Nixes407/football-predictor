"""Tests for the dynamic training season range.

The collector used to hard-code 2021-2022 .. 2024-2025, so retraining in a
later season silently kept the training set stale. SEASONS is now derived
from today's date. Run from the repository root:
    python -m unittest discover -s backend/tests -v
"""

from datetime import date
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import collector


class CurrentSeasonTests(unittest.TestCase):
    def test_october_is_second_half_of_year_starting_season(self):
        # The case that motivated the fix: October 2026 -> 2026-2027 in
        # progress, 2025-2026 finished.
        self.assertEqual(collector.current_season(date(2026, 10, 18)),
                         "2026-2027")

    def test_july_boundary(self):
        self.assertEqual(collector.current_season(date(2026, 6, 30)),
                         "2025-2026")
        self.assertEqual(collector.current_season(date(2026, 7, 1)),
                         "2026-2027")

    def test_winter_and_spring_belong_to_previous_start(self):
        self.assertEqual(collector.current_season(date(2026, 1, 5)),
                         "2025-2026")
        self.assertEqual(collector.current_season(date(2026, 5, 31)),
                         "2025-2026")


class SeasonRangeTests(unittest.TestCase):
    def test_full_range_through_in_progress_season(self):
        self.assertEqual(
            collector.season_range(today=date(2026, 10, 18)),
            ["2021-2022", "2022-2023", "2023-2024", "2024-2025",
             "2025-2026", "2026-2027"],
        )

    def test_before_july_stops_at_previous_start(self):
        self.assertEqual(
            collector.season_range(today=date(2026, 6, 30)),
            ["2021-2022", "2022-2023", "2023-2024", "2024-2025",
             "2025-2026"],
        )

    def test_custom_first_year(self):
        self.assertEqual(
            collector.season_range(2024, today=date(2024, 8, 1)),
            ["2024-2025"],
        )

    def test_first_year_after_current_start_is_empty(self):
        # May 2024 still belongs to the season started in 2023: a window
        # that begins in 2024 correctly yields nothing rather than reaching
        # into the future.
        self.assertEqual(
            collector.season_range(2024, today=date(2024, 5, 1)),
            [],
        )

    def test_module_seasons_end_at_current_season(self):
        self.assertEqual(collector.SEASONS[-1], collector.current_season())
        self.assertEqual(collector.SEASONS[0], "2021-2022")
        self.assertTrue(all(a < b for a, b in zip(collector.SEASONS,
                                                  collector.SEASONS[1:])))

    def test_recent_finished_and_in_progress_seasons_present(self):
        # On or after July 2025 both 2025-2026 and (from July 2026)
        # 2026-2027 must be collected — the original bug window.
        seasons = collector.season_range(today=date(2026, 10, 1))
        self.assertIn("2025-2026", seasons)
        self.assertIn("2026-2027", seasons)


class CrossModuleConsistencyTests(unittest.TestCase):
    def test_fbref_collector_follows_match_collector(self):
        try:
            from data import fbref_collector
        except Exception:  # pragma: no cover - env without soccerdata
            self.skipTest("fbref_collector dependencies missing")
        self.assertEqual(list(fbref_collector.SEASONS),
                         list(collector.SEASONS))

    def test_analyze_script_defaults_follow_match_collector(self):
        try:
            from scripts import analyze_training_leagues as az
        except Exception:  # pragma: no cover - env without sqlalchemy etc.
            self.skipTest("analyze script dependencies missing")
        self.assertEqual(list(az.DEFAULT_SEASONS), list(collector.SEASONS))


if __name__ == "__main__":
    unittest.main()
