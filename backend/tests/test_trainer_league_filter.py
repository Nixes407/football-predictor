"""Tests for the expanded training-league filtering in models/trainer.py.

Run from the repository root:
    python -m unittest discover -s backend/tests -v

No database or network access is required.
"""

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import trainer


class TrainingLeagueListTests(unittest.TestCase):
    def test_list_is_the_documented_eleven(self):
        self.assertEqual(trainer.TRAINING_LEAGUES, [
            "ENG-Premier League", "ESP-La Liga", "GER-Bundesliga",
            "ITA-Serie A", "FRA-Ligue 1", "ENG-Championship",
            "POR-Primeira Liga", "NED-Eredivisie", "TUR-Süper Lig",
            "NOR-Eliteserien", "AUT-Bundesliga",
        ])

    def test_big_five_regressions_stay_absent(self):
        # BSA (Brazil) and the Jupiler Pro League must not sneak in.
        joined = " ".join(trainer.TRAINING_LEAGUES).casefold()
        self.assertNotIn("brasileir", joined)
        self.assertNotIn("série a", joined.replace("ita-série a", ""))
        self.assertNotIn("jupiler", joined)
        self.assertNotIn("BEL-", joined)


class CanonicalLeagueNameTests(unittest.TestCase):
    def test_exact_names_round_trip(self):
        for league in trainer.TRAINING_LEAGUES:
            self.assertEqual(trainer.canonical_league(league), league)

    def test_case_and_accent_variants(self):
        self.assertEqual(
            trainer.canonical_league("tur-super lig"), "TUR-Süper Lig")
        self.assertEqual(
            trainer.canonical_league("  TUR-SÜPER  LIG "), "TUR-Süper Lig")
        self.assertEqual(
            trainer.canonical_league("Fra-Ligue 1"), "FRA-Ligue 1")

    def test_sponsor_and_renamed_league_aliases(self):
        self.assertEqual(
            trainer.canonical_league("NOR-OBOS-ligaen"), "NOR-Eliteserien")
        self.assertEqual(
            trainer.canonical_league("POR-Liga Portugal"), "POR-Primeira Liga")
        self.assertEqual(
            trainer.canonical_league("AUT-Admiral Bundesliga"),
            "AUT-Bundesliga")
        self.assertEqual(
            trainer.canonical_league("ENG-League Championship"),
            "ENG-Championship")

    def test_aut_bundesliga_never_matches_ger_bundesliga(self):
        # Both contain "Bundesliga": matching must be prefix-aware.
        self.assertEqual(trainer.canonical_league("AUT-Bundesliga"),
                         "AUT-Bundesliga")
        self.assertEqual(trainer.canonical_league("GER-Bundesliga"),
                         "GER-Bundesliga")

    def test_unrelated_leagues_return_none(self):
        for name in ["BEL-Jupiler Pro League", "BRA-Série A", "USA-MLS",
                     "UEFA-Champions League", "", None, float("nan")]:
            self.assertIsNone(trainer.canonical_league(name))


class FilterTrainingLeaguesTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "league": [
                "ENG-Premier League", "ENG-Premier League",
                "ENG-Championship",
                "TUR-Super Lig",          # accent-variant alias
                "NOR-Eliteserien",
                "AUT-Bundesliga",
                "BEL-Jupiler Pro League", # excluded by constraint
                "USA-MLS",                # not a training league
            ],
            "v": range(8),
        })

    def test_keeps_wanted_drops_other_leagues(self):
        out = trainer.filter_training_leagues(self.frame())
        self.assertEqual(len(out), 6)
        self.assertEqual(
            sorted(map(str, out["league"])),
            ["AUT-Bundesliga", "ENG-Championship", "ENG-Premier League",
             "ENG-Premier League", "NOR-Eliteserien", "TUR-Super Lig"],
        )

    def test_explicit_leagues_argument(self):
        out = trainer.filter_training_leagues(
            self.frame(), leagues=["ENG-Championship", "TUR-Süper Lig"])
        self.assertEqual(len(out), 2)

    def test_no_league_column_is_passthrough(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        self.assertIs(trainer.filter_training_leagues(df), df)

    def test_empty_frame(self):
        self.assertTrue(trainer.filter_training_leagues(
            pd.DataFrame()).empty)


if __name__ == "__main__":
    unittest.main()
