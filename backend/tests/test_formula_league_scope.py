"""
League-scope tests for the formula engine's league stats.

``league_teams`` / ``mean_attack`` / ``mean_defence`` / ``goals_per_game`` /
``corner_scale`` are all produced by ``formula_engine.compute_league_baselines``
and by the league averages it is built on. They describe **one competition in
one season**: the Premier League's 20 clubs of 2026-2027, not every club that
has rows in ``team_season_stats``.

These tests pin that scope down with a synthetic two-league table
(20 clubs in League A, 18 in League B — 38 in total, so a missing league
filter is caught by a single assertion) over two seasons.

Run from the repository root:

    python -m unittest discover -s backend/tests -v

No database server and no network access are required: the ORM models are
created in an in-memory SQLite database and the engine's session factory is
pointed at it for the duration of each test.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database
from models import formula_engine as fe


LEAGUE_A = "League A"          # 20 clubs
LEAGUE_B = "League B"          # 18 clubs
SEASON = "2026-2027"
PREVIOUS_SEASON = "2025-2026"

TEAMS_A = 20
TEAMS_B = 18


def team_row(team, league, season, xg=1.30, **overrides):
    """One plausible team_season_stats row (per-game scale)."""
    values = dict(
        team=team,
        league=league,
        season=season,
        # Shooting — sh_* / *_per90 pairs that agree make the row read as a
        # per-game row (see games_played), so nothing is divided again.
        sh_goals=xg,
        sh_shots=13.0,
        sh_sot=4.5,
        sh_xg=xg,
        sh_npxg=xg,
        sh_sh_per90=13.0,
        sh_sot_per90=4.5,
        # Defence / goalkeeping
        gk_ga=xg,
        gk_ga90=xg,
        gk_saves=3.0,
        def_press=150.0,
        def_tkl_int=20.0,
        # Possession / progression
        pos_poss=50.0,
        pos_touch_att3rd=200.0,
        pos_prgc=60.0,
        ps_prgp=40.0,
        gsc_gca90=5.5,
        misc_crs=20.0,
    )
    values.update(overrides)
    return database.TeamSeasonStats(**values)


class LeagueScopeTestCase(unittest.TestCase):
    """In-memory team_season_stats, wired into the formula engine."""

    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        database.Base.metadata.create_all(
            self.engine,
            tables=[
                database.TeamSeasonStats.__table__,
                database.PlayerSeasonStats.__table__,
            ],
        )
        self.Session = sessionmaker(bind=self.engine)
        self._original_session_local = fe.SessionLocal
        fe.SessionLocal = self.Session
        fe.clear_cache()

    def tearDown(self):
        fe.SessionLocal = self._original_session_local
        fe.clear_cache()
        self.engine.dispose()

    # -- helpers ---------------------------------------------------------- #

    def add_rows(self, *rows):
        session = self.Session()
        try:
            session.add_all(rows)
            session.commit()
        finally:
            session.close()

    def add_leagues(
        self,
        seasons=(SEASON,),
        teams_a=TEAMS_A,
        teams_b=TEAMS_B,
        xg_a=1.30,
        xg_b=1.30,
    ):
        """Populate both leagues for every season given."""
        rows = []
        for season in seasons:
            xg = xg_a if season == SEASON else xg_a
            rows += [
                team_row(f"Team A{i:02d}", LEAGUE_A, season, xg=xg)
                for i in range(teams_a)
            ]
            rows += [
                team_row(f"Team B{i:02d}", LEAGUE_B, season, xg=xg_b)
                for i in range(teams_b)
            ]
        self.add_rows(*rows)


class LeagueTeamsCountTests(LeagueScopeTestCase):
    """The bug: league_teams counted every club in the table, not the league."""

    def test_league_teams_is_the_requested_league_only(self):
        self.add_leagues(seasons=(SEASON,))
        baselines = fe.compute_league_baselines(LEAGUE_A, SEASON)

        # 20, not 38 (League A + League B) and not 40 (both seasons).
        self.assertEqual(baselines["teams"], 20)
        self.assertEqual(baselines["season"], SEASON)

    def test_the_other_league_gets_its_own_count(self):
        self.add_leagues(seasons=(SEASON,))
        baselines = fe.compute_league_baselines(LEAGUE_B, SEASON)
        self.assertEqual(baselines["teams"], 18)

    def test_league_teams_is_scoped_to_one_season(self):
        # Same clubs in both campaigns: a season filter that is dropped
        # would still report 20 here, so the averages are checked too.
        self.add_leagues(seasons=(SEASON, PREVIOUS_SEASON))
        baselines = fe.compute_league_baselines(LEAGUE_A, SEASON)

        self.assertEqual(baselines["teams"], 20)
        self.assertEqual(baselines["season"], SEASON)

    def test_a_missing_season_never_aggregates_every_season(self):
        self.add_leagues(seasons=(SEASON, PREVIOUS_SEASON))

        # Callers that do not know the season (scripts, the CLI entry point)
        # must get one campaign, not the union of all of them.
        baselines = fe.compute_league_baselines(LEAGUE_A, None)

        self.assertEqual(baselines["teams"], 20)
        self.assertEqual(baselines["season"], SEASON)

    def test_unknown_season_falls_back_instead_of_using_all_seasons(self):
        self.add_leagues(seasons=(SEASON, PREVIOUS_SEASON))
        baselines = fe.compute_league_baselines(LEAGUE_A, "1999-2000")

        # No rows for that campaign: neutral baselines, never a mix of the
        # two seasons that do exist.
        self.assertEqual(baselines["teams"], 0)
        self.assertEqual(baselines["mean_attack"], 1.0)


class LeagueAveragesScopeTests(LeagueScopeTestCase):
    """goals_per_game / league_avg_xg / league_avg_sot are per league+season."""

    def test_averages_ignore_the_other_league(self):
        # League B scores twice as much: if its rows leaked into League A's
        # averages, League A's goals_per_game would drift upwards.
        self.add_leagues(seasons=(SEASON,), xg_a=1.00, xg_b=2.00)

        averages = fe.load_league_averages(LEAGUE_A, SEASON)

        self.assertEqual(averages["teams"], 20)
        self.assertAlmostEqual(averages["avg_xg"], 1.00, places=6)
        self.assertAlmostEqual(averages["avg_sot"], 4.5, places=6)

    def test_averages_ignore_the_other_season(self):
        rows = [
            team_row(f"Team A{i:02d}", LEAGUE_A, SEASON, xg=1.00)
            for i in range(TEAMS_A)
        ]
        rows += [
            team_row(f"Team A{i:02d}", LEAGUE_A, PREVIOUS_SEASON, xg=3.00)
            for i in range(TEAMS_A)
        ]
        self.add_rows(*rows)

        current = fe.load_league_averages(LEAGUE_A, SEASON)
        previous = fe.load_league_averages(LEAGUE_A, PREVIOUS_SEASON)

        # An all-seasons average would be 2.00 for both.
        self.assertAlmostEqual(current["avg_xg"], 1.00, places=6)
        self.assertAlmostEqual(previous["avg_xg"], 3.00, places=6)
        self.assertEqual(current["teams"], 20)

    def test_baselines_goals_per_game_follows_the_requested_season(self):
        rows = [
            team_row(f"Team A{i:02d}", LEAGUE_A, SEASON, xg=1.00)
            for i in range(TEAMS_A)
        ]
        rows += [
            team_row(f"Team A{i:02d}", LEAGUE_A, PREVIOUS_SEASON, xg=3.00)
            for i in range(TEAMS_A)
        ]
        self.add_rows(*rows)

        fe.clear_cache()
        current = fe.compute_league_baselines(LEAGUE_A, SEASON)
        fe.clear_cache()
        previous = fe.compute_league_baselines(LEAGUE_A, PREVIOUS_SEASON)

        self.assertLess(current["goals_per_game"], previous["goals_per_game"])


class TeamRowHygieneTests(LeagueScopeTestCase):
    """Rows that are not a club's season must not be counted as teams."""

    def test_empty_shell_rows_are_not_counted(self):
        self.add_leagues(seasons=(SEASON,))
        # A standings-only upsert: no measured stat at all.
        self.add_rows(
            database.TeamSeasonStats(
                team="Ghost FC", league=LEAGUE_A, season=SEASON),
            database.TeamSeasonStats(
                team="1234", league=LEAGUE_A, season=SEASON, sh_xg=1.0),
            database.TeamSeasonStats(
                team="   ", league=LEAGUE_A, season=SEASON, sh_xg=1.0),
        )

        baselines = fe.compute_league_baselines(LEAGUE_A, SEASON)
        self.assertEqual(baselines["teams"], 20)

    def test_provider_name_variants_collapse_to_one_team(self):
        self.add_leagues(seasons=(SEASON,), teams_a=0, teams_b=0)
        self.add_rows(
            team_row("Manchester City", LEAGUE_A, SEASON),
            team_row("Man City", LEAGUE_A, SEASON),      # FotMob spelling
            team_row("Manchester Utd", LEAGUE_A, SEASON),
            team_row("Man Utd", LEAGUE_A, SEASON),       # FotMob spelling
        )

        baselines = fe.compute_league_baselines(LEAGUE_A, SEASON)

        # Two clubs, four stored rows — and not one: without the aliases the
        # suffix strip collapses both Manchester clubs onto "man".
        self.assertEqual(baselines["teams"], 2)

    def test_rows_are_never_taken_from_another_league_or_season(self):
        self.add_leagues(seasons=(SEASON,), teams_a=0, teams_b=0)
        # Look-alike labels that must not match LEAGUE_A / SEASON.
        self.add_rows(
            team_row("Team A99", "League A ", SEASON),
            team_row("Team A98", LEAGUE_A, "2026/2027"),
            team_row("Team A97", "League B", SEASON),
        )

        baselines = fe.compute_league_baselines(LEAGUE_A, SEASON)
        self.assertEqual(baselines["teams"], 0)

    def test_leaked_fotmob_player_rows_are_not_counted_as_teams(self):
        """
        Reproduces the production bug where FotMob's legacy teamOrPlayer=team
        fallback inserted 330 player leaderboard rows into ENG-Premier League
        2026-2027 (350 total rows) and 466 player rows into ENG-Championship
        2026-2027 (490 total rows).
        """
        epl = "ENG-Premier League"
        epl_clubs = [
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
            "Leicester City", "Liverpool", "Manchester City", "Manchester United",
            "Newcastle United", "Nottingham Forest", "Southampton",
            "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers",
        ]
        self.assertEqual(len(epl_clubs), 20)

        # 20 real clubs in 2026-2027 + 20 in 2025-2026
        club_rows = [
            team_row(club, epl, SEASON, xg=1.45)
            for club in epl_clubs
        ] + [
            team_row(club, epl, PREVIOUS_SEASON, xg=2.40)
            for club in epl_clubs
        ]
        self.add_rows(*club_rows)

        clean_avgs = fe.load_league_averages(epl, SEASON)
        clean_base = fe.compute_league_baselines(epl, SEASON)
        fe.clear_cache()

        # Add 330 leaked player rows (350 total in ENG-Premier League 2026-2027)
        # carrying the exact columns FotMob's player leaderboards populate
        # (sh_shots, misc_crs, sh_xg, sh_sot, sh_goals, gk_saves, misc_corners,
        # pos_touch_attpen), with no standings or team-identity columns.
        player_rows = [
            database.TeamSeasonStats(
                team=f"Premier League Player {i:03d}",
                league=epl,
                season=SEASON,
                sh_shots=42.0,
                sh_sot=18.0,
                sh_xg=6.5,
                sh_goals=7.0,
                misc_crs=25.0,
                misc_corners=12.0,
                pos_touch_attpen=55.0,
                gk_saves=10.0,
            )
            for i in range(330)
        ]
        self.add_rows(*player_rows)

        # Active prediction path: predict_match -> _predict_match ->
        # load_league_averages & compute_league_baselines -> _league_season_rows
        pred = fe.predict_match("Arsenal", "Chelsea", epl, SEASON)
        self.assertIsNotNone(pred)
        fv = pred["feature_vector_used"]
        self.assertEqual(fv["league_teams"], 20)
        self.assertEqual(pred["ratings"]["league"]["teams"], 20)
        self.assertAlmostEqual(fv["league_avg_xg"], clean_avgs["avg_xg"], places=6)
        self.assertAlmostEqual(fv["league_avg_sot"], clean_avgs["avg_sot"], places=6)
        self.assertAlmostEqual(
            pred["ratings"]["league"]["goals_per_game"],
            clean_base["goals_per_game"],
            places=6,
        )
        self.assertAlmostEqual(
            pred["ratings"]["league"]["mean_attack"],
            clean_base["mean_attack"],
            places=6,
        )
        self.assertAlmostEqual(
            pred["ratings"]["league"]["mean_defence"],
            clean_base["mean_defence"],
            places=6,
        )
        self.assertAlmostEqual(
            pred["ratings"]["league"]["corner_scale"],
            clean_base["corner_scale"],
            places=6,
        )

    def test_championship_active_predict_match_counts_24_teams_not_490(self):
        """
        Active code path test for ENG-Championship 2026-2027 with 24 clubs
        (including Leeds United, Sheffield United, Sheffield Wednesday) plus
        466 leaked FotMob player rows (490 rows total in team_season_stats).
        """
        champ = "ENG-Championship"
        champ_clubs = [
            "Leeds United", "Sheffield United", "Sheffield Wednesday",
            "Burnley", "Sunderland", "Middlesbrough", "West Bromwich Albion",
            "Blackburn Rovers", "Norwich City", "Coventry City",
            "Bristol City", "Watford", "Swansea City", "Millwall",
            "Preston North End", "Queens Park Rangers", "Stoke City",
            "Hull City", "Cardiff City", "Oxford United",
            "Portsmouth", "Derby County", "Luton Town", "Plymouth Argyle",
        ]
        self.assertEqual(len(champ_clubs), 24)

        rows = [
            team_row(club, champ, SEASON, xg=1.25)
            for club in champ_clubs
        ] + [
            team_row(club, champ, PREVIOUS_SEASON, xg=1.95)
            for club in champ_clubs
        ] + [
            database.TeamSeasonStats(
                team=f"Championship Player {i:03d}",
                league=champ,
                season=SEASON,
                sh_shots=35.0,
                misc_crs=19.0,
                sh_xg=4.2,
                sh_sot=12.0,
                sh_goals=4.0,
            )
            for i in range(466)
        ]
        self.add_rows(*rows)

        # Call predict_match with shorthand names "Leeds" and "Sheffield"
        pred = fe.predict_match("Leeds", "Sheffield", champ, SEASON)
        self.assertIsNotNone(pred)
        self.assertEqual(pred["feature_vector_used"]["league_teams"], 24)
        self.assertEqual(pred["ratings"]["league"]["teams"], 24)

        # Verify startup DB cleanup removes the 466 leaked player rows while
        # keeping all 48 real club rows (24 per season across 2 seasons).
        deleted = database.cleanup_non_team_rows(self.engine)
        self.assertEqual(deleted, 466)
        session = self.Session()
        try:
            remaining = session.query(database.TeamSeasonStats).count()
            self.assertEqual(remaining, 48)
        finally:
            session.close()

    def test_fotmob_collector_rejects_player_leaderboard_rows(self):
        from data import fotmob_collector as fc

        player_row = {
            "id": 737066,
            "teamId": 8456,
            "name": "Erling Haaland",
            "teamName": "Manchester City",
            "statValue": {"value": 85},
        }
        team_row_payload = {
            "id": 9825,
            "teamId": 9825,
            "name": "Arsenal",
            "teamName": "Arsenal",
            "statValue": {"value": 58.4},
        }
        self.assertTrue(fc._is_player_leaderboard_row(player_row))
        self.assertIsNone(fc._row_team(player_row))
        self.assertFalse(fc._is_player_leaderboard_row(team_row_payload))
        self.assertEqual(fc._row_team(team_row_payload), "Arsenal")

        merged = fc._merge_teams(
            {"Arsenal": {"expectedGoals": 40.0}, "Erling Haaland": {"shotsTotal": 85.0}},
            {"Arsenal": {"matchesPlayed": 20}, "Chelsea": {"matchesPlayed": 20}},
        )
        self.assertEqual(set(merged.keys()), {"Arsenal", "Chelsea"})


if __name__ == "__main__":
    unittest.main()
