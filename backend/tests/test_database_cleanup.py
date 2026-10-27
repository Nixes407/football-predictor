"""Regression tests for the BEL-Jupiler Pro League purge and write guard.

The Belgian competition was never supported: a bad BSA -> BEL fixture label
wrote ``team_season_stats`` rows under that league. Startup deletes them once
and every collector write goes through a guarded upsert, so they cannot come
back. All database work here uses throwaway SQLite databases.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import database  # noqa: E402


class BelJupilerGuardTests(unittest.TestCase):
    def test_bel_jupiler_blocked(self):
        """Required regression: a blocked league never touches the DB cursor."""
        cursor = MagicMock()
        db = MagicMock()
        db.cursor.return_value = cursor

        written = database.upsert_team_season_stats(
            db, "Anderlecht", "BEL-Jupiler Pro League", "2026-2027",
            {"sh_xg": 12.5},
        )

        self.assertFalse(written)
        cursor.execute.assert_not_called()
        db.cursor.assert_not_called()
        db.query.assert_not_called()
        db.execute.assert_not_called()
        db.add.assert_not_called()

    def test_blocked_label_variants(self):
        for label in (
            "BEL-Jupiler Pro League",
            "  bel-jupiler pro league  ",
            "Jupiler Pro League",
            "BEL-Jupiler",
        ):
            with self.subTest(label=label):
                self.assertTrue(database.is_blocked_league(label))
        for label in ("ENG-Premier League", "BRA-Série A", "USA-MLS", None, 7):
            with self.subTest(label=label):
                self.assertFalse(database.is_blocked_league(label))


class BelJupilerPurgeTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        database.Base.metadata.create_all(
            self.engine, tables=[database.TeamSeasonStats.__table__]
        )
        self.Session = sessionmaker(bind=self.engine)
        self.addCleanup(self.engine.dispose)

    def _seed(self):
        with self.Session() as db:
            db.add_all([
                database.TeamSeasonStats(
                    team="Anderlecht", league="BEL-Jupiler Pro League",
                    season="2025-2026", sh_xg=1.0,
                ),
                database.TeamSeasonStats(
                    team="Club Brugge", league="BEL-Jupiler Pro League",
                    season="2026-2027", sh_xg=2.0,
                ),
                database.TeamSeasonStats(
                    team="Arsenal", league="ENG-Premier League",
                    season="2026-2027", sh_xg=20.0,
                ),
            ])
            db.commit()

    def _rows(self):
        with self.Session() as db:
            return [
                (row.team, row.league, row.sh_xg)
                for row in db.query(database.TeamSeasonStats).all()
            ]

    def test_purge_deletes_only_blocked_rows_and_is_idempotent(self):
        self._seed()
        self.assertEqual(database.purge_bel_jupiler_rows(bind=self.engine), 2)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "Arsenal")
        self.assertEqual(rows[0][2], 20.0)  # non-null values never overwritten
        # After the first clean run the startup hook is a no-op.
        self.assertEqual(database.purge_bel_jupiler_rows(bind=self.engine), 0)

    def test_guarded_upsert_writes_valid_league_and_skips_none(self):
        with self.Session() as db:
            self.assertTrue(database.upsert_team_season_stats(
                db, "Arsenal", "ENG-Premier League", "2026-2027",
                {"sh_xg": 21.0, "sh_shots": None},
            ))
            db.commit()
            self.assertFalse(database.upsert_team_season_stats(
                db, "Club Brugge", "BEL-Jupiler Pro League", "2026-2027",
                {"sh_xg": 9.0},
            ))
            # A partial feed must not clear an existing non-null value.
            database.upsert_team_season_stats(
                db, "Arsenal", "ENG-Premier League", "2026-2027",
                {"sh_xg": None},
            )
            db.commit()

        self.assertEqual(self._rows(), [("Arsenal", "ENG-Premier League", 21.0)])


class BelJupilerIsNotAValidLeagueTests(unittest.TestCase):
    def test_no_collector_or_fixture_map_exposes_the_league(self):
        from data import api_football_collector, fotmob_collector
        from routers import fixtures

        self.assertNotIn("BEL-Jupiler Pro League", api_football_collector.LEAGUE_MAP)
        self.assertNotIn("BEL-Jupiler Pro League", api_football_collector.SEASON_MAP)
        self.assertNotIn("BEL-Jupiler Pro League", fotmob_collector.FOTMOB_LEAGUES)
        self.assertNotIn("BEL-Jupiler Pro League", fotmob_collector.LEAGUES)
        self.assertNotIn(
            "BEL-Jupiler Pro League",
            set(fixtures.UCL_TEAM_DOMESTIC_LEAGUE.values()),
        )

    def test_startup_hook_runs_the_purge(self):
        source = (BACKEND_DIR / "main.py").read_text()
        self.assertIn("purge_bel_jupiler_rows()", source)
        self.assertIn("init_db()", source)


if __name__ == "__main__":
    unittest.main()
