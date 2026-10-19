"""Tests for the incremental match-collection behaviour of data/collector.py.

The daily GitHub Actions run must not re-download league-seasons that are
already stored, while still fetching the in-progress season and anything
missing. No network access is used — every source is patched.

Run from the repository root:
    python -m unittest discover -s backend/tests -v
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import collector


def fake_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-08-16")],
            "home_team": ["Man City"],
            "away_team": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"],
        }
    )


class RequiredMatchesTests(unittest.TestCase):
    def test_reference_counts_cover_all_training_leagues(self):
        # Every league must have a reference size, otherwise the loose floor is
        # used and small competitions would look complete too early.
        for league in collector.LEAGUES:
            self.assertIn(league, collector.EXPECTED_MATCHES_PER_SEASON)
            self.assertGreater(collector.required_matches(league), 0)

    def test_threshold_is_a_fraction_of_the_reference(self):
        self.assertEqual(
            collector.required_matches("ENG-Premier League"),
            int(380 * collector.COMPLETE_SEASON_RATIO),
        )

    def test_unknown_league_falls_back_to_the_floor(self):
        self.assertEqual(
            collector.required_matches("XXX-Unknown"),
            collector.MIN_COMPLETE_SEASON_ROWS,
        )


class LeagueSeasonIsStoredTests(unittest.TestCase):
    def test_stored_finished_season_is_skipped(self):
        counts = {("ENG-Premier League", "2021-2022"): 380}
        self.assertTrue(
            collector.league_season_is_stored(
                "ENG-Premier League", "2021-2022", counts, current="2026-2027"
            )
        )

    def test_partial_season_is_fetched_again(self):
        counts = {("ENG-Premier League", "2021-2022"): 12}
        self.assertFalse(
            collector.league_season_is_stored(
                "ENG-Premier League", "2021-2022", counts, current="2026-2027"
            )
        )

    def test_missing_season_is_fetched(self):
        self.assertFalse(
            collector.league_season_is_stored(
                "NOR-Eliteserien", "2022-2023", {}, current="2026-2027"
            )
        )

    def test_current_season_is_always_fetched(self):
        counts = {("ENG-Premier League", "2026-2027"): 380}
        self.assertFalse(
            collector.league_season_is_stored(
                "ENG-Premier League", "2026-2027", counts, current="2026-2027"
            )
        )


class CollectMatchHistoryTests(unittest.TestCase):
    """The loop must fetch only the current season when the rest is stored."""

    def setUp(self):
        self.all_seasons = list(collector.SEASONS)
        self.current = self.all_seasons[-1]
        self.counts = {
            (league, season): collector.required_matches(league)
            for league in collector.LEAGUES
            for season in self.all_seasons[:-1]
        }

    def _run(self, counts=None, force=False, only_missing=True):
        fetched = []

        def fake_scrape(league, season):
            fetched.append(("soccerdata", league, season))
            return fake_frame()

        def fake_extra(league, config, season):
            fetched.append(("extra-csv", league, season))
            return fake_frame()

        with patch.object(collector, "_register_extra_leagues_with_soccerdata"), \
                patch.object(collector, "current_season", return_value=self.current), \
                patch.object(collector, "stored_match_counts",
                             return_value=self.counts if counts is None else counts), \
                patch.object(collector, "_scrape_matches", side_effect=fake_scrape), \
                patch.object(collector, "_load_extra_matches", side_effect=fake_extra), \
                patch.object(collector, "_upsert_matches", return_value=(1, 0)):
            totals = collector.collect_match_history(
                force=force, only_missing=only_missing
            )
        return totals, fetched

    def test_only_the_current_season_is_downloaded(self):
        totals, fetched = self._run()
        seasons = {season for _, _, season in fetched}
        self.assertEqual(seasons, {self.current})
        self.assertEqual(len(fetched), len(collector.LEAGUES))
        self.assertEqual(
            totals["skipped_seasons"],
            len(collector.LEAGUES) * (len(self.all_seasons) - 1),
        )
        self.assertEqual(totals["inserted"], len(collector.LEAGUES))

    def test_extra_csv_leagues_are_skipped_the_same_way(self):
        _, fetched = self._run()
        # NOR/AUT (all-time CSV downloader) are fetched once — the in-progress
        # season only — not once per season.
        for league in collector.EXTRA_CSV_LEAGUES:
            self.assertEqual(
                [season for kind, name, season in fetched
                 if kind == "extra-csv" and name == league],
                [self.current],
            )

    def test_missing_season_is_fetched(self):
        counts = dict(self.counts)
        del counts[("ENG-Championship", "2022-2023")]
        totals, fetched = self._run(counts=counts)
        self.assertIn(
            ("soccerdata", "ENG-Championship", "2022-2023"), fetched
        )
        self.assertEqual(totals["skipped_seasons"],
                         len(collector.LEAGUES) * (len(self.all_seasons) - 1) - 1)

    def test_force_ignores_stored_counts(self):
        totals, fetched = self._run(force=True)
        self.assertEqual(len(fetched), len(collector.LEAGUES) * len(self.all_seasons))
        self.assertEqual(totals["skipped_seasons"], 0)

    def test_unreadable_database_falls_back_to_fetching_everything(self):
        with patch.object(collector, "session_scope", side_effect=OSError("db down")):
            self.assertEqual(collector.stored_match_counts(), {})

    def test_upsert_failure_is_counted_and_does_not_stop_the_run(self):
        with patch.object(collector, "_register_extra_leagues_with_soccerdata"), \
                patch.object(collector, "current_season", return_value=self.current), \
                patch.object(collector, "stored_match_counts", return_value=self.counts), \
                patch.object(collector, "_scrape_matches", return_value=fake_frame()), \
                patch.object(collector, "_load_extra_matches", return_value=fake_frame()), \
                patch.object(collector, "_upsert_matches",
                             side_effect=RuntimeError("write failed")):
            totals = collector.collect_match_history()
        self.assertEqual(totals["failed"], len(collector.LEAGUES))
        self.assertEqual(totals["inserted"], 0)


class SoccerdataCacheInvalidationTests(unittest.TestCase):
    """Registering extra leagues must work even after soccerdata cached its map."""

    def test_registration_after_a_cached_league_map(self):
        try:
            from soccerdata import MatchHistory
            from soccerdata import _config as sd_config
        except Exception:  # pragma: no cover - env without soccerdata
            self.skipTest("soccerdata not installed")

        saved = dict(sd_config.LEAGUE_DICT)
        try:
            for name in collector.EXTRA_MATCH_HISTORY_CODES:
                sd_config.LEAGUE_DICT.pop(name, None)
            # Populate the cache the way any earlier soccerdata call would.
            MatchHistory.available_leagues()
            if hasattr(MatchHistory, "_all_leagues_dict"):
                del MatchHistory._all_leagues_dict
            MatchHistory.available_leagues()  # cache is now definitely warm
            self.assertNotIn("ENG-Championship", MatchHistory.available_leagues())

            collector._register_extra_leagues_with_soccerdata()

            for name, code in collector.EXTRA_MATCH_HISTORY_CODES.items():
                self.assertEqual(MatchHistory._all_leagues()[name], code)
        finally:
            sd_config.LEAGUE_DICT.clear()
            sd_config.LEAGUE_DICT.update(saved)
            if hasattr(MatchHistory, "_all_leagues_dict"):
                del MatchHistory._all_leagues_dict


if __name__ == "__main__":
    unittest.main()
