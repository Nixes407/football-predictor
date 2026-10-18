"""Tests for the extra-league ingestion added to data/collector.py.

Covers the football-data.co.uk "extra leagues" (NOR-Eliteserien,
AUT-Bundesliga) CSV normalisation and the soccerdata league registration
that fixes the silent "Invalid league" ingestion failure for the six new
training leagues. No network access is used — downloads are patched.

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


def fake_extra_csv() -> str:
    # One all-time CSV in the football-data.co.uk /new/ format: dd/mm/yyyy
    # dates, goals + odds only, plus a July match that belongs to the
    # FOLLOWING season for non calendar-year leagues.
    return """Country,League,Season,Date,Time,Home,Away,HG,AG,Res
Norway,Eliteserien,2021,07/08/2021,17:00,Molde,Rosenborg,2,1,H
Norway,Eliteserien,2021,01/05/2021,18:00,Brann,Viking,1,1,D
Norway,Eliteserien,2022,07/08/2022,17:00,Molde,Brann,3,0,H
"""


def fake_extra_csv_no_season() -> str:
    return """Country,League,Date,Home,Away,HG,AG,Res
Austria,Bundesliga,01/08/2021,Rapid Wien,Admira,2,0,H
Austria,Bundesliga,15/02/2021,Rapid Wien,Admira,1,1,D
"""


class FakeResponse:
    def __init__(self, text: str):
        self.content = text.encode("utf-8")

    def raise_for_status(self):
        return None


class SeasonStartYearTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(collector._season_start_year("2021-2022"), 2021)
        self.assertEqual(collector._season_start_year("2021/22"), 2021)
        self.assertEqual(collector._season_start_year("2024"), 2024)
        self.assertIsNone(collector._season_start_year("nope"))


class ExtraLeagueLoadTests(unittest.TestCase):
    def setUp(self):
        collector._EXTRA_CSV_CACHE.clear()

    def _load(self, csv_text, config, season, name="NOR-Eliteserien"):
        with patch.object(
            collector.requests, "get",
            return_value=FakeResponse(csv_text),
        ):
            frame = collector._load_extra_matches(name, config, season)
        return frame

    def test_season_filter_from_season_column(self):
        config = collector.EXTRA_CSV_LEAGUES["NOR-Eliteserien"]
        frame = self._load(fake_extra_csv(), config, "2021-2022")
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame), 2)  # both 2021 rows, not the 2022 one
        row = frame.iloc[0]
        self.assertEqual(row["home_team"], "Molde")
        self.assertEqual(row["FTHG"], 2)
        self.assertEqual(row["date"], pd.Timestamp("2021-08-07"))

    def test_rows_flow_into_build_match_rows(self):
        config = collector.EXTRA_CSV_LEAGUES["NOR-Eliteserien"]
        frame = self._load(fake_extra_csv(), config, "2021-2022")
        rows = collector._build_match_rows(frame, "NOR-Eliteserien",
                                           "2021-2022")
        self.assertEqual(len(rows), 2)
        first = rows[0]
        self.assertEqual(first["result"], "H")
        self.assertEqual(first["home_goals"], 2)
        self.assertEqual(first["away_goals"], 1)
        self.assertEqual(first["league"], "NOR-Eliteserien")
        # shots/corners are absent in extra-league files -> None
        self.assertIsNone(first["home_shots"])
        self.assertIsNone(first["home_corners"])

    def test_missing_season_column_falls_back_to_dates(self):
        config = {
            "code": "AUT",
            "country": "Austria",
            "calendar_year": False,
        }
        frame = self._load(fake_extra_csv_no_season(), config, "2021-2022",
                           name="AUT-Bundesliga")
        # 01/08/2021 -> season 2021-22; 15/02/2021 -> still 2020-21 season.
        self.assertEqual(len(frame), 1)
        rows = collector._build_match_rows(frame, "AUT-Bundesliga",
                                           "2021-2022")
        self.assertEqual(rows[0]["home_team"], "Rapid Wien")

    def test_country_guard_rejects_wrong_file(self):
        config = collector.EXTRA_CSV_LEAGUES["NOR-Eliteserien"]
        wrong = fake_extra_csv().replace("Norway", "Sweden")
        with patch.object(collector.requests, "get",
                          return_value=FakeResponse(wrong)):
            frame = collector._load_extra_matches(
                "NOR-Eliteserien", config, "2021-2022")
        self.assertIsNone(frame)
        self.assertIsNone(collector._EXTRA_CSV_CACHE["NOR"])

    def test_download_failure_is_cached_as_none(self):
        config = collector.EXTRA_CSV_LEAGUES["NOR-Eliteserien"]
        with patch.object(collector.requests, "get",
                          side_effect=OSError("blocked network")):
            frame = collector._load_extra_matches(
                "NOR-Eliteserien", config, "2021-2022")
        self.assertIsNone(frame)


class SoccerdataRegistrationTests(unittest.TestCase):
    def test_registration_adds_missing_match_history_codes(self):
        try:
            from soccerdata import _config as sd_config
        except Exception:  # pragma: no cover - env without soccerdata
            self.skipTest("soccerdata not installed")
        saved = dict(sd_config.LEAGUE_DICT)
        try:
            for name in collector.EXTRA_MATCH_HISTORY_CODES:
                sd_config.LEAGUE_DICT.pop(name, None)
            collector._register_extra_leagues_with_soccerdata()
            for name, code in collector.EXTRA_MATCH_HISTORY_CODES.items():
                self.assertEqual(
                    sd_config.LEAGUE_DICT[name]["MatchHistory"], code)
            # MatchHistory must now accept the league name: it appears in the
            # classmethod league list without raising "Invalid league".
            from soccerdata import MatchHistory
            self.assertIn("ENG-Championship", MatchHistory.available_leagues())
            self.assertEqual(
                MatchHistory._all_leagues()["ENG-Championship"], "E1")
        except Exception as exc:  # pragma: no cover
            if "soccerdata" in str(exc):
                self.skipTest(f"soccerdata unavailable: {exc}")
            raise
        finally:
            sd_config.LEAGUE_DICT.clear()
            sd_config.LEAGUE_DICT.update(saved)

    def test_leagues_list_matches_trainer(self):
        try:
            from models import trainer
        except Exception:  # pragma: no cover - optional heavy deps
            self.skipTest("trainer deps not installed")
        self.assertEqual(list(collector.LEAGUES), list(trainer.TRAINING_LEAGUES))


if __name__ == "__main__":
    unittest.main()
