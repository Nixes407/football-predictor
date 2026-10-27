"""Tests for data/statsbomb_aggregator.py (StatsBomb -> reference stats table).

Run from the repository root:
    python -m unittest discover -s backend/tests -v

No live database, network or scraping is used — everything runs on a
throwaway SQLite database.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import database  # noqa: E402
from data import statsbomb_aggregator as agg  # noqa: E402
from features import builder, inference  # noqa: E402


class LabelNormalisationTests(unittest.TestCase):
    """competition / season labels come in several shapes; all must normalise."""

    def test_league_accepts_clean_and_full_label_big_five(self):
        self.assertEqual(agg.normalise_league("ENG-Premier League"), "ENG-Premier League")
        self.assertEqual(agg.normalise_league("ESP-La Liga"), "ESP-La Liga")
        self.assertEqual(
            agg.normalise_league("ENG-Premier League 2015-2016"), "ENG-Premier League"
        )
        self.assertEqual(
            agg.normalise_league("ESP-La Liga 2021-2022"), "ESP-La Liga"
        )
        self.assertEqual(
            agg.normalise_league("FRA-Ligue 1 2019-2020"), "FRA-Ligue 1"
        )

    def test_league_rejects_everything_outside_big_five(self):
        rejected = [
            "",
            None,
            "UEFA Champions League 2021-2022",
            "FIFA World Cup 2022",
            "Women's Super League 2021-2022",
            "Big 5 European Leagues Combined",
            "ENG-Championship",
            "AUT-Bundesliga",
            "TUR-Süper Lig",
            "BEL-Jupiler Pro League",
            "USA-MLS",
            "BRA-Série A",
        ]
        for label in rejected:
            self.assertIsNone(agg.normalise_league(label), label)

    def test_season_normalises_range_code_and_date_fallback(self):
        self.assertEqual(agg.normalise_season("2024-2025"), "2024-2025")
        self.assertEqual(agg.normalise_season("ESP-La Liga 2021-2022"), "2021-2022")
        self.assertEqual(agg.normalise_season("2425"), "2024-2025")
        # Date fallback: seasons start in July.
        self.assertEqual(agg.normalise_season(None, "2025-09-01"), "2025-2026")
        self.assertEqual(agg.normalise_season("junk", "2024-03-05"), "2023-2024")
        self.assertIsNone(agg.normalise_season("junk", None))

    def test_seasons_follow_match_collector(self):
        from data import collector

        self.assertEqual(list(agg.SEASONS), list(collector.SEASONS))


class AggregatorDBTests(unittest.TestCase):
    """Aggregate + upsert against a mock statsbomb_match_stats table."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_file = Path(self._tmp.name) / "agg.db"
        self.engine = create_engine(f"sqlite:///{db_file}")
        self.Session = sessionmaker(bind=self.engine)
        database.Base.metadata.create_all(bind=self.engine)

        self._patches = [
            patch.object(database, "engine", self.engine),
            patch.object(database, "SessionLocal", self.Session),
            patch.object(agg, "SessionLocal", self.Session),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _add_match(self, db, **overrides):
        row = {
            "match_id": 1,
            "competition": "ENG-Premier League",
            "season": "2025-2026",
            "date": "2025-09-01",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_goals": 2,
            "away_goals": 1,
            "home_shots": 15,
            "away_shots": 10,
            "home_shots_ot": 6,
            "away_shots_ot": 3,
            "home_gk_saves": 2,
            "away_gk_saves": 4,
        }
        row.update(overrides)
        db.add(database.StatsBombMatchStats(**row))
        db.commit()

    def _get(self, db, team, league="ENG-Premier League", season="2025-2026"):
        return (
            db.query(database.FBrefTeamStats)
            .filter_by(team=team, league=league, season=season)
            .one()
        )

    def test_totals_rates_and_unmapped_zeroes_are_written(self):
        with self.Session() as db:
            self._add_match(
                db,
                match_id=1,
                home_team="Arsenal",
                away_team="Chelsea",
                home_goals=2,
                away_goals=1,
                home_shots=15,
                away_shots=10,
                home_shots_ot=6,
                away_shots_ot=3,
                home_gk_saves=2,
                away_gk_saves=4,
            )
            self._add_match(
                db,
                match_id=2,
                date="2025-09-15",
                home_team="Chelsea",
                away_team="Arsenal",
                home_goals=0,
                away_goals=0,
                home_shots=8,
                away_shots=12,
                home_shots_ot=2,
                away_shots_ot=5,
                home_gk_saves=5,
                away_gk_saves=2,
            )

        summary = agg.run_aggregation()

        self.assertEqual(summary["source_rows"], 2)
        self.assertEqual(summary["inserted"], 2)
        self.assertEqual(summary["updated"], 0)
        self.assertEqual(summary["failed_groups"], 0)
        self.assertEqual(
            summary["per_league_season"]["ENG-Premier League 2025-2026"]["inserted"], 2
        )

        with self.Session() as db:
            arsenal = self._get(db, "Arsenal")
            # totals
            self.assertEqual(arsenal.games, 2)
            self.assertEqual(arsenal.shots, 27.0)  # 15 + 12
            self.assertEqual(arsenal.shots_on_target, 11.0)  # 6 + 5
            self.assertEqual(arsenal.saves, 4.0)  # 2 + 2
            self.assertEqual(arsenal.goals_against, 1.0)  # 1 + 0
            self.assertEqual(arsenal.clean_sheets, 1.0)  # conceded 0 in match 2
            self.assertEqual(arsenal.sota, 5.0)  # faced 3 + 2 on target
            # rates (FBref conventions)
            self.assertAlmostEqual(arsenal.shot_accuracy, 11 / 27 * 100, places=4)
            self.assertAlmostEqual(arsenal.goals_per_shot, 2 / 27, places=4)
            self.assertAlmostEqual(arsenal.goals_per_sot, 2 / 11, places=4)
            self.assertAlmostEqual(arsenal.save_pct, 80.0, places=4)  # 4 / 5
            self.assertIsNotNone(arsenal.updated_at)

            chelsea = self._get(db, "Chelsea")
            self.assertEqual(chelsea.games, 2)
            self.assertEqual(chelsea.shots, 18.0)  # 10 + 8
            self.assertEqual(chelsea.shots_on_target, 5.0)  # 3 + 2
            self.assertEqual(chelsea.saves, 9.0)  # 4 + 5
            self.assertEqual(chelsea.goals_against, 2.0)  # 2 + 0
            self.assertEqual(chelsea.clean_sheets, 1.0)  # 0-0 in match 2
            self.assertEqual(chelsea.sota, 11.0)  # faced 6 + 5
            self.assertAlmostEqual(chelsea.save_pct, 9 / 11 * 100, places=4)

            # No StatsBomb equivalent in today's schema -> 0.0 (never None)
            for field in (
                "yellow_cards",
                "red_cards",
                "fouls_committed",
                "fouls_drawn",
                "offsides",
                "crosses",
                "interceptions",
                "tackles_won",
                "pens_won",
                "pens_conceded",
                "pk_faced",
                "pk_saved",
            ):
                self.assertEqual(getattr(arsenal, field), 0.0, field)
                self.assertEqual(getattr(chelsea, field), 0.0, field)

    def test_second_run_updates_rows_instead_of_inserting(self):
        with self.Session() as db:
            self._add_match(db, match_id=1)

        first = agg.run_aggregation()
        second = agg.run_aggregation()

        self.assertEqual(first["inserted"], 2)
        self.assertEqual(first["updated"], 0)
        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["updated"], 2)
        with self.Session() as db:
            self.assertEqual(db.query(database.FBrefTeamStats).count(), 2)

    def test_ignores_out_of_scope_competitions_and_seasons(self):
        with self.Session() as db:
            # In scope
            self._add_match(db, match_id=1)
            # Wrong competition
            self._add_match(
                db,
                match_id=2,
                competition="FIFA World Cup 2022",
                season="2022-2023",
                home_team="Brazil",
                away_team="Argentina",
            )
            self._add_match(
                db,
                match_id=3,
                competition="AUT-Bundesliga",
                home_team="Salzburg",
                away_team="Sturm Graz",
            )
            self._add_match(
                db,
                match_id=4,
                competition="ENG-Championship",
                home_team="Leeds",
                away_team="Burnley",
            )
            # Big Five but season outside collector.season_range()
            self._add_match(
                db,
                match_id=5,
                season="2018-2019",
                home_team="Man City",
                away_team="Liverpool",
            )

        summary = agg.run_aggregation()

        self.assertEqual(summary["inserted"], 2)  # Arsenal + Chelsea only
        with self.Session() as db:
            self.assertEqual(db.query(database.FBrefTeamStats).count(), 2)
            self.assertIsNotNone(self._get(db, "Arsenal"))
            self.assertIsNotNone(self._get(db, "Chelsea"))

    def test_statsbomb_full_labels_normalise_to_canonical_keys(self):
        with self.Session() as db:
            self._add_match(
                db,
                match_id=1,
                competition="ESP-La Liga 2021-2022",
                season="ESP-La Liga 2021-2022",
                home_team="Real Madrid",
                away_team="Barcelona",
            )

        summary = agg.run_aggregation()

        self.assertEqual(summary["inserted"], 2)
        with self.Session() as db:
            real = self._get(db, "Real Madrid", league="ESP-La Liga", season="2021-2022")
            self.assertEqual(real.shots, 15.0)
            self.assertEqual(real.games, 1)

    def test_understat_style_rows_with_null_stats_still_count(self):
        """Rows with goals/xG only (Understat's writer) must still yield games,
        goals and clean sheets; stat sums fall back to 0."""
        with self.Session() as db:
            self._add_match(
                db,
                match_id=1,
                home_team="Everton",
                away_team="Fulham",
                home_goals=1,
                away_goals=0,
                home_shots=None,
                away_shots=None,
                home_shots_ot=None,
                away_shots_ot=None,
                home_gk_saves=None,
                away_gk_saves=None,
            )

        agg.run_aggregation()

        with self.Session() as db:
            everton = self._get(db, "Everton")
            self.assertEqual(everton.games, 1)
            self.assertEqual(everton.goals_against, 0.0)
            self.assertEqual(everton.clean_sheets, 1.0)
            self.assertEqual(everton.shots, 0.0)
            self.assertEqual(everton.shots_on_target, 0.0)
            fulham = self._get(db, "Fulham")
            self.assertEqual(fulham.goals_against, 1.0)
            self.assertEqual(fulham.clean_sheets, 0.0)

    def test_extended_source_columns_are_picked_up_dynamically(self):
        """If the live table carries fouls/cards/crosses columns, the mapping
        uses them without any code change."""
        with self.engine.begin() as conn:
            for col in (
                "home_yellow_cards",
                "away_yellow_cards",
                "home_fouls",
                "away_fouls",
                "home_crosses",
                "away_crosses",
            ):
                conn.exec_driver_sql(
                    f"ALTER TABLE statsbomb_match_stats ADD COLUMN {col} INTEGER"
                )
            conn.exec_driver_sql(
                "INSERT INTO statsbomb_match_stats (match_id, competition, season, "
                "date, home_team, away_team, home_goals, away_goals, home_shots, "
                "away_shots, home_shots_ot, away_shots_ot, home_gk_saves, "
                "away_gk_saves, home_yellow_cards, away_yellow_cards, home_fouls, "
                "away_fouls, home_crosses, away_crosses) "
                "VALUES (1, 'ENG-Premier League', '2025-2026', '2025-09-01', "
                "'Arsenal', 'Chelsea', 2, 1, 15, 10, 6, 3, 2, 4, 2, 3, 11, 9, 20, 12)"
            )

        summary = agg.run_aggregation()
        self.assertEqual(summary["inserted"], 2)
        self.assertIn("yellow_cards", summary["resolved_fields"])
        self.assertIn("fouls_committed", summary["resolved_fields"])
        self.assertIn("crosses", summary["resolved_fields"])

        with self.Session() as db:
            arsenal = self._get(db, "Arsenal")
            self.assertEqual(arsenal.yellow_cards, 2.0)
            self.assertEqual(arsenal.fouls_committed, 11.0)
            self.assertEqual(arsenal.crosses, 20.0)
            chelsea = self._get(db, "Chelsea")
            self.assertEqual(chelsea.yellow_cards, 3.0)
            self.assertEqual(chelsea.crosses, 12.0)

    def test_unmapped_fields_are_preserved_on_update(self):
        """A legacy FBref row keeps values the aggregator cannot recompute."""
        with self.Session() as db:
            db.add(
                database.FBrefTeamStats(
                    team="Arsenal",
                    league="ENG-Premier League",
                    season="2025-2026",
                    yellow_cards=55.0,
                    pk_saved=3.0,
                    shots=1.0,
                )
            )
            db.commit()
            self._add_match(db, match_id=1)

        summary = agg.run_aggregation()

        self.assertEqual(summary["inserted"], 1)  # Chelsea only
        self.assertEqual(summary["updated"], 1)  # Arsenal merged
        with self.Session() as db:
            arsenal = self._get(db, "Arsenal")
            # Mapped fields are the aggregator's source of truth...
            self.assertEqual(arsenal.shots, 15.0)
            # ...unmapped ones are never clobbered.
            self.assertEqual(arsenal.yellow_cards, 55.0)
            self.assertEqual(arsenal.pk_saved, 3.0)

    def test_missing_source_table_is_not_an_error(self):
        with self.engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE statsbomb_match_stats")

        summary = agg.run_aggregation()

        self.assertEqual(summary["source_rows"], 0)
        self.assertEqual(summary["inserted"], 0)
        self.assertEqual(summary["failed_groups"], 0)

    def test_dry_run_writes_nothing(self):
        with self.Session() as db:
            self._add_match(db, match_id=1)

        summary = agg.run_aggregation(dry_run=True)

        self.assertEqual(summary["inserted"], 2)  # "would insert"
        with self.Session() as db:
            self.assertEqual(db.query(database.FBrefTeamStats).count(), 0)

    def test_main_always_returns_zero_even_when_the_db_is_down(self):
        with patch.object(agg, "SessionLocal", side_effect=RuntimeError("db down")):
            self.assertEqual(agg.main([]), 0)
        self.assertEqual(agg.main(["--dry-run"]), 0)


class AggregatorToFeaturePipelineTests(unittest.TestCase):
    """Aggregated rows are reference-only: inference never emits their columns."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_file = Path(self._tmp.name) / "pipe.db"
        self.engine = create_engine(f"sqlite:///{db_file}")
        self.Session = sessionmaker(bind=self.engine)
        database.Base.metadata.create_all(bind=self.engine)

        self._patches = [
            patch.object(database, "engine", self.engine),
            patch.object(database, "SessionLocal", self.Session),
            patch.object(agg, "SessionLocal", self.Session),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def test_inference_features_have_no_fbref_keys(self):
        with self.Session() as db:
            for i, (home, away) in enumerate(
                [("Arsenal", "Chelsea"), ("Chelsea", "Arsenal")], start=1
            ):
                db.add(
                    database.StatsBombMatchStats(
                        match_id=i,
                        competition="ENG-Premier League",
                        season="2026-2027",
                        date="2026-09-01",
                        home_team=home,
                        away_team=away,
                        home_goals=2,
                        away_goals=1,
                        home_shots=15,
                        away_shots=10,
                        home_shots_ot=6,
                        away_shots_ot=3,
                        home_gk_saves=2,
                        away_gk_saves=4,
                    )
                )
            db.commit()

        summary = agg.run_aggregation()
        self.assertEqual(summary["inserted"], 2)

        # The dormant aggregator still writes its reference rows...
        with self.Session() as db:
            rows = db.query(database.FBrefTeamStats).all()
            self.assertEqual(len(rows), 2)
            arsenal = next(row for row in rows if row.team == "Arsenal")
            self.assertEqual(arsenal.shots, 25.0)
            self.assertEqual(arsenal.clean_sheets, 0.0)  # both seeded matches conceded
            self.assertEqual(arsenal.saves, 6.0)

        # ...but the builder / inference path neither reads nor emits them.
        features = inference.build_inference_features(
            inference.empty_match_history(),
            inference.empty_elo_history(),
            "Arsenal",
            "Chelsea",
            league="ENG-Premier League",
            season="2026-2027",
        )
        self.assertFalse(any(
            key.startswith(("home_fbref_", "away_fbref_"))
            for key in features
        ))
        from features.feature_columns import FEATURE_COLS
        self.assertTrue(
            (set(FEATURE_COLS) - {"league_encoded", "season_encoded"})
            <= set(features)
        )
        self.assertFalse(hasattr(builder, "load_fbref_stats"))
        self.assertFalse(hasattr(builder, "join_fbref_features"))


if __name__ == "__main__":
    unittest.main()
