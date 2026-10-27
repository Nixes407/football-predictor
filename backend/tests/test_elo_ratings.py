"""Regression tests for the Elo write path, the cache and the diagnostics.

The bug these pin down: ``elo_ratings`` had exactly one writer
(``data.collector.collect_elo_ratings``, a clubelo.com scrape) and every
per-team failure was swallowed into a ``skipped`` counter, so a run that stored
**zero** rows returned normally and exited 0 — and the
``[3/5] Run Match Collector`` step was ``continue-on-error: true`` on top of
that. The table stayed empty, the warmed cache held 0 entries, every lookup
returned ``None`` and ``compute_elo_features`` applied its hard-coded 1500
default: Arsenal vs Chelsea reported ``home_elo=1500, away_elo=1500,
elo_difference=0`` while diagnostics still claimed ``elo_lookup: "database"``.

Coverage here:

* the standard Elo recursion (K=32, start 1500) is exact and zero-sum;
* ``data/elo.py`` really writes rows from a ``matches`` table, idempotently,
  and never writes None/NaN/0 in place of a rating;
* the cache load returns one entry per rated team and diagnostics reports that
  count — not the number of lookup slots, which used to include misses;
* a missing/empty table can no longer masquerade as a healthy database;
* the workflow runs the updater after the match collector, before the builder,
  and is allowed to fail the run.

Only a throwaway in-memory SQLite database is used; nothing touches a
production database and no network call is made.

Run from ``backend/``::

    python -m unittest tests.test_elo_ratings -v
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "daily_fbref_update.yml"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import yaml  # noqa: E402
from sqlalchemy import create_engine, func, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import database  # noqa: E402
from data import collector, elo  # noqa: E402
from features import builder  # noqa: E402
from routers import predict  # noqa: E402


def _match(home, away, hg, ag, played, league="ENG-Premier League", result=None,
           match_id=None):
    """One ``matches``-shaped dict.

    Pass ``hg=None, ag=None`` (and no ``result``) for a fixture whose outcome
    is not known yet — the writer must skip those rather than rate them.
    """
    if result is None and hg is not None and ag is not None:
        result = "H" if hg > ag else "D" if hg == ag else "A"
    return {
        "match_id": match_id or f"{home}-{away}-{played}",
        "league": league,
        "season": "2024-2025",
        "date": played,
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "result": result,
    }


class EloFormulaTests(unittest.TestCase):
    """The recursion must be the textbook one, exactly."""

    def test_expected_score_is_half_for_equal_ratings(self):
        self.assertAlmostEqual(elo.expected_score(1500.0, 1500.0), 0.5, places=12)

    def test_expected_score_matches_the_documented_formula(self):
        for own, opponent in ((1500, 1500), (1900, 1500), (1500, 1900),
                              (1602.5, 1487.25), (2100, 1200)):
            with self.subTest(own=own, opponent=opponent):
                reference = 1.0 / (1.0 + 10.0 ** ((opponent - own) / 400.0))
                self.assertAlmostEqual(
                    elo.expected_score(own, opponent), reference, places=12
                )

    def test_expected_score_for_a_400_point_gap(self):
        # 1 / (1 + 10^-1) = 0.909090...
        self.assertAlmostEqual(
            elo.expected_score(1900.0, 1500.0), 1.0 / 1.1, places=12
        )

    def test_expected_scores_of_both_sides_sum_to_one(self):
        self.assertAlmostEqual(
            elo.expected_score(1700.0, 1450.0)
            + elo.expected_score(1450.0, 1700.0),
            1.0, places=12,
        )

    def test_expected_score_never_raises_on_degenerate_input(self):
        for own, opponent in ((None, 1500), (1500, None), (float("nan"), 1500),
                              (float("inf"), 1500), ("x", "y")):
            with self.subTest(own=own, opponent=opponent):
                value = elo.expected_score(own, opponent)
                self.assertTrue(0.0 <= value <= 1.0)

    def test_expected_score_survives_an_absurd_rating_gap(self):
        # CPython raises OverflowError for 10.0 ** 400, so an unclamped gap
        # used to blow up mid-replay. It must saturate instead of raising.
        low = elo.expected_score(-1e9, 1e9)
        high = elo.expected_score(1e9, -1e9)
        self.assertTrue(0.0 <= low <= 1.0)
        self.assertTrue(0.0 <= high <= 1.0)
        self.assertAlmostEqual(low, 0.0, places=12)
        self.assertAlmostEqual(high, 1.0, places=12)
        # A merely extreme but finite gap saturates the same way.
        self.assertAlmostEqual(elo.expected_score(0.0, 1e6), 0.0, places=12)
        self.assertAlmostEqual(elo.expected_score(1e6, 0.0), 1.0, places=12)
        # A non-finite rating is treated as "unknown" rather than propagated.
        self.assertAlmostEqual(
            elo.expected_score(float("inf"), 1500.0), 0.5, places=12
        )

    def test_actual_score_reads_the_result_code(self):
        self.assertEqual(elo.actual_score("H"), 1.0)
        self.assertEqual(elo.actual_score("D"), 0.5)
        self.assertEqual(elo.actual_score("A"), 0.0)

    def test_actual_score_is_case_and_whitespace_insensitive(self):
        self.assertEqual(elo.actual_score(" h "), 1.0)
        self.assertEqual(elo.actual_score("d"), 0.5)

    def test_actual_score_derives_from_goals_when_the_code_is_missing(self):
        # The football-data.co.uk /new/ leagues store goals but no result code.
        self.assertEqual(elo.actual_score(None, 3, 1), 1.0)
        self.assertEqual(elo.actual_score(None, 2, 2), 0.5)
        self.assertEqual(elo.actual_score(None, 0, 1), 0.0)
        self.assertEqual(elo.actual_score("", None, None), None)

    def test_actual_score_is_none_when_the_outcome_is_unknown(self):
        self.assertIsNone(elo.actual_score(None, None, None))
        self.assertIsNone(elo.actual_score("?", None, None))
        self.assertIsNone(elo.actual_score(None, 2, None))


class EloReplayTests(unittest.TestCase):
    """``compute_elo_ratings`` is a pure function of the match list."""

    def test_first_result_moves_both_sides_by_exactly_half_k(self):
        rows = elo.compute_elo_ratings(
            [_match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10))]
        )
        by_team = {row["team"]: row["elo"] for row in rows}
        # Both start at 1500 -> expected 0.5 -> K * (1 - 0.5) = +16 / -16.
        self.assertAlmostEqual(by_team["Arsenal"], 1516.0, places=6)
        self.assertAlmostEqual(by_team["Chelsea"], 1484.0, places=6)

    def test_second_result_uses_the_updated_ratings(self):
        rows = elo.compute_elo_ratings([
            _match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)),
            _match("Chelsea", "Arsenal", 2, 0, date(2024, 8, 17)),
        ])
        # Recomputed here from the documented formula, not by calling
        # expected_score, so the test is an independent check.
        home_before, away_before = 1484.0, 1516.0
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_before - home_before) / 400.0))
        want_chelsea = home_before + 32.0 * (1.0 - expected_home)
        want_arsenal = away_before + 32.0 * (0.0 - (1.0 - expected_home))
        latest = {row["team"]: row["elo"] for row in rows[-2:]}
        self.assertAlmostEqual(latest["Chelsea"], want_chelsea, delta=1e-4)
        self.assertAlmostEqual(latest["Arsenal"], want_arsenal, delta=1e-4)

    def test_every_match_is_zero_sum(self):
        matches = [
            _match("A", "B", 3, 0, date(2024, 8, 1)),
            _match("B", "C", 1, 1, date(2024, 8, 8)),
            _match("C", "A", 0, 2, date(2024, 8, 15)),
            _match("A", "C", 4, 1, date(2024, 8, 22)),
        ]
        rows = elo.compute_elo_ratings(matches)
        totals: dict[str, float] = {}
        for row in rows:
            totals[row["team"]] = row["elo"]
        played = {m["home_team"] for m in matches} | {m["away_team"] for m in matches}
        # Points are only ever exchanged, never created or destroyed, so the
        # system's total stays at 1500 per team no matter the results.
        self.assertAlmostEqual(
            sum(totals.values()), 1500.0 * len(played), places=6
        )

    def test_a_draw_lifts_the_underdog_and_cuts_the_favourite(self):
        rows = elo.compute_elo_ratings([
            _match("Strong", "Weak", 3, 0, date(2024, 8, 1)),
            _match("Weak", "Strong", 1, 1, date(2024, 8, 8)),
        ])
        final = {row["team"]: row["elo"] for row in rows[-2:]}
        before = elo.final_ratings([
            _match("Strong", "Weak", 3, 0, date(2024, 8, 1))
        ])
        self.assertGreater(final["Weak"], before["Weak"])
        self.assertLess(final["Strong"], before["Strong"])

    def test_a_new_team_starts_at_1500(self):
        rows = elo.compute_elo_ratings(
            [_match("Debutant", "Veteran", 1, 1, date(2024, 8, 1))]
        )
        by_team = {row["team"]: row["elo"] for row in rows}
        # A draw between two 1500s leaves both exactly where they started.
        self.assertAlmostEqual(by_team["Debutant"], 1500.0, places=6)
        self.assertAlmostEqual(by_team["Veteran"], 1500.0, places=6)
        self.assertEqual(elo.DEFAULT_RATING, 1500.0)
        self.assertEqual(elo.K_FACTOR, 32.0)

    def test_matches_without_a_determinable_result_are_skipped_not_zeroed(self):
        rows = elo.compute_elo_ratings([
            _match("A", "B", None, None, date(2024, 8, 1), result=None),
            _match("A", "B", 2, 1, date(2024, 8, 8)),
        ])
        self.assertEqual(len(rows), 2)  # only the second match produced rows
        for row in rows:
            self.assertIsNotNone(row["elo"])
            self.assertNotEqual(row["elo"], 0.0)
            self.assertTrue(row["elo"] > 0.0)

    def test_unusable_rows_are_skipped(self):
        rows = elo.compute_elo_ratings([
            {"date": None, "home_team": "A", "away_team": "B",
             "home_goals": 1, "away_goals": 0, "result": "H"},
            {"date": date(2024, 8, 1), "home_team": "", "away_team": "B",
             "home_goals": 1, "away_goals": 0, "result": "H"},
            {"date": date(2024, 8, 1), "home_team": "A", "away_team": None,
             "home_goals": 1, "away_goals": 0, "result": "H"},
            _match("A", "A", 2, 0, date(2024, 8, 2)),  # a team cannot play itself
        ])
        self.assertEqual(rows, [])

    def test_rows_are_one_per_team_per_match_date(self):
        rows = elo.compute_elo_ratings([
            _match("A", "B", 2, 0, date(2024, 8, 1)),
        ])
        keys = [(row["team"], row["date"]) for row in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual({row["date"] for row in rows}, {date(2024, 8, 1)})

    def test_two_fixtures_on_one_date_collapse_to_the_last_rating(self):
        rows = elo.compute_elo_ratings([
            _match("A", "B", 2, 0, date(2024, 8, 1)),
            _match("A", "C", 2, 0, date(2024, 8, 1)),
        ])
        per_team = {}
        for row in rows:
            per_team.setdefault(row["team"], []).append(row["elo"])
        self.assertEqual(len(per_team["A"]), 1)  # not two rows on one date
        # The surviving value is the post-second-match rating.
        self.assertGreater(per_team["A"][0], 1516.0)

    def test_replay_is_deterministic(self):
        matches = [
            _match("A", "B", 2, 1, date(2024, 8, 1)),
            _match("B", "C", 0, 0, date(2024, 8, 8)),
            _match("C", "A", 1, 3, date(2024, 8, 15)),
        ]
        self.assertEqual(
            elo.compute_elo_ratings(matches), elo.compute_elo_ratings(matches)
        )

    def test_league_is_carried_onto_the_row(self):
        rows = elo.compute_elo_ratings(
            [_match("A", "B", 1, 0, date(2024, 8, 1), league="ITA-Serie A")]
        )
        self.assertEqual({row["league"] for row in rows}, {"ITA-Serie A"})

    def test_replay_counts_rated_and_skipped_matches_exactly(self):
        matches = [
            _match("A", "B", 2, 1, date(2024, 8, 1)),
            _match("A", "B", 1, 1, date(2024, 8, 1)),   # same teams, same date
            _match("C", "D", None, None, date(2024, 8, 8), result=None),
        ]
        rows, rated, skipped = elo._replay(matches)
        self.assertEqual(rated, 2)
        self.assertEqual(skipped, 1)
        # Two rated matches but a colliding (team, date) key -> fewer than 4
        # rows. Deriving `rated` from len(rows)//2 would have misreported this.
        self.assertLess(len(rows), 2 * rated)


class EloDatabaseTestCase(unittest.TestCase):
    """Base class: a throwaway in-memory database with the real schema."""

    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        database.Base.metadata.create_all(
            self.engine,
            tables=[database.Match.__table__, database.EloRating.__table__],
        )
        self.Session = sessionmaker(bind=self.engine)
        for patcher in (
            mock.patch.object(database, "engine", self.engine),
            mock.patch.object(database, "SessionLocal", self.Session),
            mock.patch.object(elo, "engine", self.engine),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.engine.dispose)

    def seed_matches(self, *matches):
        with self.Session() as db:
            for row in matches:
                payload = {k: v for k, v in row.items()}
                db.add(database.Match(**payload))
            db.commit()

    def seed_ratings(self, *rows):
        with self.Session() as db:
            for team, day, rating in rows:
                db.add(database.EloRating(
                    team=team, date=date.fromisoformat(day), elo=rating
                ))
            db.commit()

    def elo_rows(self):
        with self.Session() as db:
            return db.execute(
                select(database.EloRating).order_by(
                    database.EloRating.date, database.EloRating.team
                )
            ).scalars().all()

    def count(self, model):
        with self.Session() as db:
            return db.execute(select(func.count()).select_from(model)).scalar_one()


class EloWriterTests(EloDatabaseTestCase):
    """``data/elo.py`` must actually put ratings in the table."""

    def test_update_writes_one_row_per_team_per_match(self):
        self.seed_matches(
            _match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)),
            _match("Chelsea", "Arsenal", 1, 1, date(2024, 8, 17)),
            _match("Liverpool", "Everton", 3, 1, date(2024, 8, 18)),
        )
        totals = elo.update_elo_from_matches(verbose=False)

        self.assertEqual(totals["matches"], 3)
        self.assertEqual(totals["rated_matches"], 3)
        self.assertEqual(totals["skipped_matches"], 0)
        self.assertEqual(totals["teams"], 4)
        self.assertEqual(totals["records"], 6)
        self.assertEqual(self.count(database.EloRating), 6)

    def test_written_ratings_are_the_formula_output(self):
        self.seed_matches(
            _match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10))
        )
        elo.update_elo_from_matches(verbose=False)
        stored = {row.team: row.elo for row in self.elo_rows()}
        self.assertAlmostEqual(stored["Arsenal"], 1516.0, places=6)
        self.assertAlmostEqual(stored["Chelsea"], 1484.0, places=6)

    def test_update_is_idempotent(self):
        self.seed_matches(
            _match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)),
            _match("Chelsea", "Arsenal", 0, 1, date(2024, 8, 17)),
        )
        first = elo.update_elo_from_matches(verbose=False)
        snapshot = [(r.team, r.date, r.elo) for r in self.elo_rows()]
        second = elo.update_elo_from_matches(verbose=False)
        again = [(r.team, r.date, r.elo) for r in self.elo_rows()]

        self.assertEqual(first["records"], second["records"])
        self.assertEqual(snapshot, again)
        self.assertEqual(self.count(database.EloRating), first["records"])

    def test_rerun_after_new_matches_extends_history_without_losing_rows(self):
        self.seed_matches(_match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)))
        elo.update_elo_from_matches(verbose=False)
        before = self.count(database.EloRating)

        with self.Session() as db:
            db.add(database.Match(**_match(
                "Chelsea", "Arsenal", 1, 1, date(2024, 8, 17),
                match_id="second",
            )))
            db.commit()
        elo.update_elo_from_matches(verbose=False)

        self.assertEqual(before, 2)
        self.assertEqual(self.count(database.EloRating), 4)

    def test_an_empty_matches_table_reports_zeros_without_crashing(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            totals = elo.update_elo_from_matches()
        self.assertEqual(totals["records"], 0)
        self.assertEqual(totals["matches"], 0)
        self.assertEqual(self.count(database.EloRating), 0)
        self.assertIn("matches table is empty", buffer.getvalue())

    def test_matches_with_no_result_write_nothing(self):
        self.seed_matches(
            _match("Arsenal", "Chelsea", None, None, date(2024, 8, 10),
                   result=None)
        )
        totals = elo.update_elo_from_matches(verbose=False)
        self.assertEqual(totals["records"], 0)
        self.assertEqual(self.count(database.EloRating), 0)

    def test_upsert_never_writes_a_non_finite_or_null_rating(self):
        written = elo.upsert_elo_rows([
            {"team": "Good", "date": date(2024, 8, 1), "elo": 1520.0},
            {"team": "Nan", "date": date(2024, 8, 1), "elo": float("nan")},
            {"team": "Inf", "date": date(2024, 8, 1), "elo": float("inf")},
            {"team": "None", "date": date(2024, 8, 1), "elo": None},
            {"team": "Text", "date": date(2024, 8, 1), "elo": "abc"},
            {"team": "NoDate", "date": None, "elo": 1500.0},
            {"team": "", "date": date(2024, 8, 1), "elo": 1500.0},
        ])
        self.assertEqual(written, 1)
        stored = self.elo_rows()
        self.assertEqual([row.team for row in stored], ["Good"])

    def test_upsert_refreshes_an_existing_row_instead_of_duplicating(self):
        self.seed_ratings(("Arsenal", "2024-08-10", 1500.0))
        written = elo.upsert_elo_rows([
            {"team": "Arsenal", "date": date(2024, 8, 10), "elo": 1602.5,
             "league": "ENG-Premier League"},
        ])
        self.assertEqual(written, 1)
        rows = self.elo_rows()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].elo, 1602.5, places=6)
        self.assertEqual(rows[0].league, "ENG-Premier League")

    def test_upsert_deduplicates_a_repeated_key_before_inserting(self):
        written = elo.upsert_elo_rows([
            {"team": "Arsenal", "date": date(2024, 8, 10), "elo": 1510.0},
            {"team": "Arsenal", "date": date(2024, 8, 10), "elo": 1520.0},
        ])
        self.assertEqual(written, 1)
        rows = self.elo_rows()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].elo, 1520.0, places=6)  # last wins

    def test_summary_tells_a_missing_table_from_an_empty_one(self):
        empty = elo.elo_summary()
        self.assertTrue(empty["table_exists"])
        self.assertEqual(empty["rows"], 0)
        self.assertEqual(empty["teams"], 0)
        self.assertIsNone(empty["error"])

        database.EloRating.__table__.drop(self.engine)
        missing = elo.elo_summary()
        self.assertFalse(missing["table_exists"])
        self.assertIn("does not exist", missing["error"])
        self.assertFalse(elo.elo_table_exists())

    def test_summary_reports_the_rating_spread(self):
        self.seed_ratings(
            ("Arsenal", "2024-08-10", 1602.5),
            ("Arsenal", "2024-08-17", 1615.0),
            ("Chelsea", "2024-08-17", 1480.0),
        )
        summary = elo.elo_summary()
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["teams"], 2)
        self.assertAlmostEqual(summary["min"], 1480.0, places=6)
        self.assertAlmostEqual(summary["max"], 1615.0, places=6)
        self.assertEqual(summary["latest_date"], "2024-08-17")

    def test_rating_for_returns_the_latest_row_only(self):
        self.seed_ratings(
            ("Arsenal", "2024-08-10", 1500.0),
            ("Arsenal", "2026-10-25", 1676.31),
        )
        self.assertAlmostEqual(elo.rating_for("Arsenal"), 1676.31, places=6)
        self.assertIsNone(elo.rating_for("Unknown Team XYZ"))

    def test_cli_exits_1_when_nothing_was_written(self):
        """The silent-success failure mode must be impossible to repeat."""
        buffer = io.StringIO()
        with redirect_stdout(buffer), mock.patch(
            "database.init_db", return_value=True
        ):
            code = elo.main([])
        self.assertEqual(code, 1)
        self.assertIn("FAILED", buffer.getvalue())

    def test_cli_exits_0_and_reports_the_table_when_ratings_exist(self):
        self.seed_matches(
            _match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)),
            _match("Liverpool", "Everton", 1, 1, date(2024, 8, 11)),
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer), mock.patch(
            "database.init_db", return_value=True
        ):
            code = elo.main([])
        output = buffer.getvalue()
        self.assertEqual(code, 0, output)
        self.assertIn("[elo] OK: 4 teams rated", output)
        self.assertIn("rows=4", output)

    def test_cli_exits_1_without_a_database_engine(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer), mock.patch.object(elo, "engine", None):
            code = elo.main([])
        self.assertEqual(code, 1)
        self.assertIn("DATABASE_URL", buffer.getvalue())

    def test_cli_help_does_not_touch_the_database(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer), mock.patch.object(elo, "engine", None):
            code = elo.main(["--help"])
        self.assertEqual(code, 0)


class EloCollectorWiringTests(EloDatabaseTestCase):
    """The collector must route through the computed writer, not only ClubElo."""

    def test_clubelo_upsert_delegates_to_the_shared_writer(self):
        rows = [{"team": "Arsenal", "date": date(2024, 8, 10), "elo": 1602.5,
                 "league": "ENG-Premier League"}]
        self.assertEqual(collector._upsert_elo(rows), 1)
        stored = self.elo_rows()
        self.assertEqual(len(stored), 1)
        self.assertAlmostEqual(stored[0].elo, 1602.5, places=6)

    def test_clubelo_upsert_of_nothing_writes_nothing(self):
        self.assertEqual(collector._upsert_elo([]), 0)
        self.assertEqual(self.count(database.EloRating), 0)

    def test_run_collection_computes_elo_from_the_matches_it_collected(self):
        self.seed_matches(
            _match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)),
            _match("Liverpool", "Everton", 1, 1, date(2024, 8, 11)),
        )
        with mock.patch.object(collector, "_check_dependencies",
                               return_value=True), \
                mock.patch.object(collector, "collect_match_history",
                                  return_value={}) as history, \
                mock.patch.object(collector, "collect_elo_ratings",
                                  return_value={}) as clubelo, \
                mock.patch.dict("os.environ", {}, clear=False), \
                redirect_stdout(io.StringIO()):
            import os

            os.environ.pop("COLLECT_CLUBELO", None)
            collector.run_collection()

        history.assert_called_once()
        # ClubElo is opt-in now; the computed updater is the default writer.
        clubelo.assert_not_called()
        self.assertEqual(self.count(database.EloRating), 4)

    def test_run_collection_survives_a_failing_elo_update(self):
        self.seed_matches(_match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)))
        with mock.patch.object(collector, "_check_dependencies",
                               return_value=True), \
                mock.patch.object(collector, "collect_match_history",
                                  return_value={}), \
                mock.patch("data.elo.update_elo_from_matches",
                           side_effect=RuntimeError("db down")), \
                mock.patch.dict("os.environ", {}, clear=False), \
                redirect_stdout(io.StringIO()):
            collector.run_collection()  # must not raise

    def test_clubelo_is_opt_in_through_the_environment(self):
        for value, want in (("", False), ("0", False), ("no", False),
                            ("1", True), ("true", True), ("YES", True),
                            ("on", True)):
            with self.subTest(value=value), mock.patch.dict(
                "os.environ", {"COLLECT_CLUBELO": value}
            ):
                self.assertEqual(collector._clubelo_enabled(), want)

    def test_summary_shouts_when_matches_exist_but_no_ratings_do(self):
        """The state that shipped the bug must not print as a clean summary."""
        self.seed_matches(_match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            collector._print_summary()
        output = buffer.getvalue()
        self.assertIn("elo_ratings is EMPTY", output)
        self.assertIn("1500 default", output)

    def test_summary_is_quiet_when_ratings_are_present(self):
        self.seed_matches(_match("Arsenal", "Chelsea", 2, 0, date(2024, 8, 10)))
        elo.update_elo_from_matches(verbose=False)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            collector._print_summary()
        output = buffer.getvalue()
        self.assertNotIn("elo_ratings is EMPTY", output)
        self.assertIn("Teams with an elo rating: 2", output)
        self.assertIn("Elo rating range", output)


class EloCacheAndDiagnosticsTests(EloDatabaseTestCase):
    """Steps 6/7: the cache count and the lookup the API actually performs."""

    _GLOBALS = (
        "_elo_latest_cache", "_elo_cache", "_elo_cache_loaded",
        "_elo_lookup_failures", "_elo_cache_error",
    )

    def setUp(self):
        super().setUp()
        self._saved = {name: getattr(predict, name) for name in self._GLOBALS}
        self.addCleanup(self._restore)
        for patcher in (
            mock.patch.object(builder, "session_scope", database.session_scope),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.reset()

    def _restore(self):
        for name, value in self._saved.items():
            setattr(predict, name, value)

    def reset(self):
        predict._elo_latest_cache = {}
        predict._elo_cache = {}
        predict._elo_cache_loaded = False
        predict._elo_lookup_failures = 0
        predict._elo_cache_error = None

    def diagnostics(self):
        with mock.patch.object(predict, "get_engine_status",
                               return_value={"ready": True}), \
                mock.patch.object(predict, "_count_fbref_team_stats_rows",
                                  return_value=0):
            return predict.diagnostics()

    def test_elo_cache_loads(self):
        """Required: 3 rated rows in the DB -> elo_cache_entries == 3."""
        self.seed_ratings(
            ("Arsenal FC", "2026-10-26", 2012.0),
            ("Chelsea AFC", "2026-10-25", 1888.0),
            ("Liverpool", "2026-10-24", 1975.0),
        )
        ratings = builder.load_latest_elo_ratings()
        self.assertEqual(len(ratings), 3)
        self.assertEqual(predict.warm_elo_cache(force=True), 3)
        self.assertEqual(self.diagnostics()["elo_cache_entries"], 3)

    def test_cache_keeps_only_the_latest_row_per_team(self):
        self.seed_ratings(
            ("Arsenal", "2024-08-10", 1500.0),
            ("Arsenal", "2026-10-26", 1676.31),
        )
        self.assertEqual(predict.warm_elo_cache(force=True), 1)
        self.assertAlmostEqual(
            builder.load_latest_elo_ratings()["arsenal"], 1676.31, places=6
        )

    def test_elo_lookup_exact_match(self):
        """Required: lookup("Arsenal") returns the rating, not 1500."""
        self.seed_ratings(("Arsenal", "2026-10-26", 1676.31))
        predict.warm_elo_cache(force=True)
        value = predict._elo_lookup("Arsenal", pd.Timestamp("2026-10-27"))
        self.assertAlmostEqual(value, 1676.31, places=6)
        self.assertNotEqual(value, 1500.0)

    def test_elo_lookup_name_normalisation(self):
        """Required: a suffix/case/accent variant resolves to the same rating."""
        self.seed_ratings(
            ("Arsenal FC", "2026-10-26", 1676.31),
            ("Atlético Madrid", "2026-10-26", 1580.0),
        )
        predict.warm_elo_cache(force=True)
        cutoff = pd.Timestamp("2026-10-27")
        for asked in ("Arsenal", "arsenal", "  Arsenal FC ", "ARSENAL"):
            with self.subTest(asked=asked):
                self.assertAlmostEqual(
                    predict._elo_lookup(asked, cutoff), 1676.31, places=6
                )
        self.assertAlmostEqual(
            predict._elo_lookup("Atletico Madrid", cutoff), 1580.0, places=6
        )

    def test_elo_fallback(self):
        """Required: an unknown team still gets the documented 1500 default."""
        self.seed_ratings(("Arsenal", "2026-10-26", 1676.31))
        predict.warm_elo_cache(force=True)
        cutoff = pd.Timestamp("2026-10-27")

        self.assertIsNone(predict._elo_lookup("Unknown Team XYZ", cutoff))
        self.assertIsNone(
            builder.get_latest_elo_before("Unknown Team XYZ", cutoff)
        )
        features = builder.compute_elo_features(
            pd.DataFrame(), "Unknown Team XYZ", "Also Unknown", cutoff,
            elo_lookup=predict._elo_lookup,
        )
        self.assertEqual(features, {
            "home_elo": 1500.0, "away_elo": 1500.0, "elo_difference": 0.0,
        })

    def test_arsenal_vs_chelsea_is_no_longer_the_1500_pair(self):
        self.seed_matches(
            _match("Arsenal", "Chelsea", 2, 0, date(2026, 8, 10)),
            _match("Chelsea", "Arsenal", 1, 1, date(2026, 8, 17)),
            _match("Arsenal", "Liverpool", 3, 1, date(2026, 8, 24)),
            _match("Chelsea", "Everton", 0, 2, date(2026, 8, 25)),
        )
        elo.update_elo_from_matches(verbose=False)
        predict.warm_elo_cache(force=True)

        features = builder.compute_elo_features(
            pd.DataFrame(), "Arsenal", "Chelsea", pd.Timestamp("2026-10-27"),
            elo_lookup=predict._elo_lookup,
        )
        self.assertNotEqual(features["home_elo"], 1500.0)
        self.assertNotEqual(features["away_elo"], 1500.0)
        self.assertNotEqual(features["elo_difference"], 0.0)
        self.assertGreater(features["home_elo"], features["away_elo"])

    def test_cache_entries_count_teams_not_miss_slots(self):
        """A miss used to be cached as None and counted as an "entry"."""
        self.reset()
        self.seed_ratings(("Arsenal", "2026-10-26", 1676.31))
        predict.warm_elo_cache(force=True)
        # Two unknown teams populate _elo_cache with None values.
        self.assertIsNone(
            predict._elo_lookup("Unknown One", pd.Timestamp("2026-10-27"))
        )
        self.assertIsNone(
            predict._elo_lookup("Unknown Two", pd.Timestamp("2026-10-27"))
        )
        self.assertEqual(len(predict._elo_cache), 2)
        self.assertEqual(self.diagnostics()["elo_cache_entries"], 1)

    def test_an_empty_table_does_not_report_a_healthy_database(self):
        self.reset()
        self.assertEqual(predict.warm_elo_cache(force=True), 0)
        payload = self.diagnostics()
        self.assertEqual(payload["elo_cache_entries"], 0)
        self.assertEqual(payload["elo_lookup"], "features.csv_fallback")
        self.assertIsNotNone(payload["elo_cache_error"])
        self.assertIn("elo_ratings", payload["elo_cache_error"])

    def test_a_missing_table_reports_the_swallowed_error(self):
        self.reset()
        database.EloRating.__table__.drop(self.engine)
        payload = self.diagnostics()
        self.assertEqual(payload["elo_cache_entries"], 0)
        self.assertEqual(payload["elo_lookup"], "features.csv_fallback")
        self.assertIsNotNone(payload["elo_cache_error"])

    def test_a_populated_table_reports_database_lookup(self):
        self.reset()
        self.seed_ratings(
            ("Arsenal", "2026-10-26", 1676.31),
            ("Chelsea", "2026-10-25", 1528.70),
        )
        payload = self.diagnostics()
        self.assertEqual(payload["elo_lookup"], "database")
        self.assertEqual(payload["elo_cache_entries"], 2)
        self.assertIsNone(payload["elo_cache_error"])
        self.assertEqual(payload["elo_lookup_failures"], 0)

    def test_repeated_lookup_failures_still_flip_the_label(self):
        self.reset()
        self.seed_ratings(("Arsenal", "2026-10-26", 1676.31))
        with mock.patch.object(predict, "_elo_rating_lookup",
                               side_effect=RuntimeError("connection reset")):
            for _ in range(predict._ELO_MAX_FAILURES):
                self.assertIsNone(
                    predict._elo_lookup("Chelsea", pd.Timestamp("2026-10-27"))
                )
        self.assertEqual(
            predict._elo_lookup_failures, predict._ELO_MAX_FAILURES
        )
        payload = self.diagnostics()
        # warm_elo_cache(force=True) succeeds and resets the failure counter,
        # so the label recovers as soon as the table is readable again.
        self.assertEqual(payload["elo_lookup"], "database")
        self.assertEqual(payload["elo_lookup_failures"], 0)

    def test_warm_cache_is_cached_until_forced(self):
        self.reset()
        self.seed_ratings(("Arsenal", "2026-10-26", 1676.31))
        self.assertEqual(predict.warm_elo_cache(), 1)
        self.seed_ratings(("Chelsea", "2026-10-25", 1528.70))
        self.assertEqual(predict.warm_elo_cache(), 1)          # not refreshed
        self.assertEqual(predict.warm_elo_cache(force=True), 2)

    def test_refresh_prediction_caches_clears_the_error_too(self):
        self.reset()
        predict._elo_cache_error = "something broke"
        predict._elo_latest_cache = {"arsenal": 1676.31}
        predict._refresh_prediction_caches()
        self.assertIsNone(predict._elo_cache_error)
        self.assertEqual(predict._elo_latest_cache, {})
        self.assertFalse(predict._elo_cache_loaded)


class EloWorkflowContractTests(unittest.TestCase):
    """The updater must run in the pipeline, after matches, before building."""

    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        self.steps = self.workflow["jobs"]["collect-and-retrain"]["steps"]
        self.names = [step.get("name", "") for step in self.steps]

    def _idx(self, prefix):
        for index, name in enumerate(self.names):
            if prefix in name:
                return index
        self.fail(f"step matching {prefix!r} missing from {self.names}")

    def test_the_elo_step_exists_and_runs_the_updater(self):
        index = self._idx("[4/5] Update Elo ratings from match results")
        step = self.steps[index]
        self.assertEqual(step.get("working-directory"), "backend")
        self.assertIn("python data/elo.py", step["run"])

    def test_the_elo_step_runs_after_the_match_collector(self):
        self.assertLess(
            self._idx("[3/5] Run Match Collector"),
            self._idx("[4/5] Update Elo ratings from match results"),
        )

    def test_the_elo_step_runs_before_the_builder(self):
        """builder.py reads elo_ratings, so the ratings must exist first."""
        self.assertLess(
            self._idx("[4/5] Update Elo ratings from match results"),
            self._idx("[5/5] Build features (builder.py)"),
        )

    def test_the_elo_step_is_allowed_to_fail_the_run(self):
        """continue-on-error is what let an empty table ship as a green run."""
        step = self.steps[
            self._idx("[4/5] Update Elo ratings from match results")
        ]
        self.assertNotIn("continue-on-error", step)
        self.assertIn("::error::", step["run"])
        self.assertIn("exit $RC", step["run"])

    def test_the_match_collector_step_no_longer_claims_to_collect_elo(self):
        step = self.steps[self._idx("[3/5] Run Match Collector")]
        self.assertNotIn("collect_elo_ratings", step["run"])


class EloScriptTests(unittest.TestCase):
    """``scripts/test_elo_lookup.py`` must be importable and inert."""

    def test_module_has_no_side_effects_on_import(self):
        """It lives under scripts/, but pytest could still import it."""
        sys.path.insert(0, str(BACKEND_DIR))
        import importlib

        module = importlib.import_module("scripts.test_elo_lookup")
        self.assertEqual(module.PROBE_TEAMS, ("Arsenal", "Chelsea"))
        self.assertEqual(module.DEFAULT_RATING, 1500.0)
        self.assertTrue(callable(module.run))
        self.assertTrue(callable(module.main))

    def test_it_reports_a_missing_database_instead_of_crashing(self):
        from scripts import test_elo_lookup

        with mock.patch.object(database, "engine", None):
            report = test_elo_lookup.run(update=False)
        self.assertIs(report["database_url_configured"], False)
        self.assertFalse(report["pass"])
        self.assertIn("DATABASE_URL", report["reason"])

    def test_it_fails_on_an_empty_table_and_passes_on_a_rated_one(self):
        from scripts import test_elo_lookup

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        database.Base.metadata.create_all(
            engine,
            tables=[database.Match.__table__, database.EloRating.__table__],
        )
        Session = sessionmaker(bind=engine)
        self.addCleanup(engine.dispose)

        with mock.patch.object(database, "engine", engine), \
                mock.patch.object(database, "SessionLocal", Session), \
                mock.patch.object(elo, "engine", engine), \
                mock.patch.object(predict, "_elo_latest_cache", {}), \
                mock.patch.object(predict, "_elo_cache", {}), \
                mock.patch.object(predict, "_elo_cache_loaded", False), \
                mock.patch.object(predict, "_elo_lookup_failures", 0), \
                mock.patch.object(predict, "_elo_cache_error", None):

            empty = test_elo_lookup.run(update=False)
            self.assertFalse(empty["pass"])
            self.assertIn("EMPTY", empty["reason"])

            with Session() as db:
                for row in (
                    _match("Arsenal", "Chelsea", 2, 0, date(2026, 8, 10)),
                    _match("Chelsea", "Arsenal", 1, 1, date(2026, 8, 17)),
                    _match("Arsenal", "Liverpool", 3, 1, date(2026, 8, 24)),
                    _match("Chelsea", "Everton", 0, 2, date(2026, 8, 25)),
                ):
                    db.add(database.Match(**row))
                db.commit()

            report = test_elo_lookup.run(update=True)

        self.assertTrue(report["pass"], report["reason"])
        self.assertTrue(report["table_exists"])
        self.assertEqual(report["table_teams"], 4)
        self.assertEqual(report["elo_lookup"], "database")
        self.assertIsNone(report["elo_cache_error"])
        self.assertEqual(report["elo_cache_entries"], 4)
        self.assertNotEqual(report["lookups"]["Arsenal"], 1500.0)
        self.assertNotEqual(report["lookups"]["Chelsea"], 1500.0)
        self.assertNotEqual(report["fixture"]["elo_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
