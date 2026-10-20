"""Regression tests: training features and inference features must agree.

The trainer fits on rows produced by ``features/builder.py``. The API scores a
live fixture by rebuilding those same features in ``routers/predict.py`` (via
``features/inference.py``). A difference between the two code paths is
invisible at runtime — the model still answers, just from wrong numbers — so
this module pins them together:

* a synthetic match history is turned into a features frame by the *builder*;
* the same frame, cut off at a fixture date, is turned into a feature vector by
  *predict*;
* every one of the 108 / 17 / 18 / 17 model inputs is asserted equal.

No database and no network are touched: the builder's feature functions are
pure, the frame the API would download is passed in directly, and the FBref
snapshot is injected as a DataFrame (which is exactly what
``load_artifacts`` caches in production).

Run from ``backend/``::

    python -m unittest tests.test_feature_parity -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import builder  # noqa: E402
from features.feature_columns import (  # noqa: E402
    BTTS_FEATURE_COLS,
    EXPECTED_FEATURE_COUNTS,
    FBREF_STAT_COLUMNS,
    FEATURE_COLS,
    OVER25_FEATURE_COLS,
    OVER95_CORNERS_FEATURE_COLS,
    validate_feature_column_counts,
)
from features import inference  # noqa: E402
import routers.predict as predict  # noqa: E402

TOLERANCE = 1e-6

TEAMS = ["Alpha", "Beta", "Gamma", "Delta"]
LEAGUES = ["ENG-Premier League", "ESP-La Liga"]
SEASONS = ["2023-2024", "2024-2025"]
PAIRINGS = [
    ("Alpha", "Beta"), ("Gamma", "Delta"),
    ("Alpha", "Gamma"), ("Beta", "Delta"),
    ("Delta", "Alpha"), ("Beta", "Gamma"),
    ("Alpha", "Delta"), ("Gamma", "Beta"),
]


# --------------------------------------------------------------------------- #
# Synthetic inputs (deterministic, no DB)
# --------------------------------------------------------------------------- #


def synthetic_matches(rows: int = 48) -> pd.DataFrame:
    """A small, realistic match history with every column the builder reads."""
    rng = np.random.default_rng(20261020)
    start = pd.Timestamp("2023-08-01")
    records = []
    for index in range(rows):
        home, away = PAIRINGS[index % len(PAIRINGS)]
        home_goals = int(rng.integers(0, 4))
        away_goals = int(rng.integers(0, 4))
        if home_goals > away_goals:
            result = "H"
        elif away_goals > home_goals:
            result = "A"
        else:
            result = "D"
        records.append({
            "date": start + pd.Timedelta(days=7 * index),
            "league": LEAGUES[index % len(LEAGUES)],
            "season": SEASONS[0] if index < rows // 2 else SEASONS[-1],
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
            "home_shots": int(rng.integers(3, 20)),
            "away_shots": int(rng.integers(3, 20)),
            "home_shots_ot": int(rng.integers(1, 9)),
            "away_shots_ot": int(rng.integers(1, 9)),
            "home_corners": int(rng.integers(0, 12)),
            "away_corners": int(rng.integers(0, 12)),
        })
    return pd.DataFrame(records)


def synthetic_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """A rating per team per match date, starting at its first recorded match.

    A ClubElo rating only moves when the team plays, so the extra row Alpha
    gets a few days after each match must not change what
    ``get_pre_match_elo`` returns — the reconstruction has to survive a rating
    row on a day without a match.

    Ratings start at each team's first match because that is exactly what the
    features CSV can carry: it only knows a team's rating on days it played.
    (A club whose real rating history predates its first row in the CSV is the
    one case the reconstruction cannot see; see docs/feature-parity-audit.md.)
    """
    ratings = {"Alpha": 1700.0, "Beta": 1600.0, "Gamma": 1500.0, "Delta": 1450.0}
    records = []
    seen: set[str] = set()
    for date in sorted(set(matches["date"])):
        seen |= set(matches.loc[matches["date"] == date, "home_team"])
        seen |= set(matches.loc[matches["date"] == date, "away_team"])
        for team in TEAMS:
            if team in seen:
                records.append({"team": team, "date": date, "elo": ratings[team]})
        if "Alpha" in seen:
            records.append({
                "team": "Alpha",
                "date": date + pd.Timedelta(days=3),
                "elo": ratings["Alpha"],
            })
    return pd.DataFrame(records)


def synthetic_fbref() -> pd.DataFrame:
    """A team-season snapshot available before every synthetic match."""
    records = []
    for team in TEAMS:
        for league in LEAGUES:
            for season in SEASONS:
                records.append({
                    "team": team,
                    "league": league,
                    "season": season,
                    "updated_at": "2023-07-01T00:00:00+00:00",
                    **{
                        stat: float(10 + position)
                        for position, stat in enumerate(FBREF_STAT_COLUMNS)
                    },
                })
    return pd.DataFrame(records)


def training_frame():
    """``(features frame, (league_encoder, season_encoder), fbref rows, elo)``.

    Everything ``build_and_save_features`` would produce, without the database.
    """
    matches = synthetic_matches()
    elo = synthetic_elo(matches)
    fbref = synthetic_fbref()

    features = builder.build_features(matches, elo)

    leagues = LabelEncoder()
    seasons = LabelEncoder()
    features["league_encoded"] = leagues.fit_transform(features["league"].astype(str))
    features["season_encoded"] = seasons.fit_transform(features["season"].astype(str))

    saved = builder.join_fbref_features(features, fbref)
    return saved, (leagues, seasons), fbref, elo


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class FeatureColumnContractTests(unittest.TestCase):
    """The published column lists and their lengths cannot drift."""

    def test_counts_are_unchanged(self):
        self.assertEqual(len(FEATURE_COLS), 108)
        self.assertEqual(len(BTTS_FEATURE_COLS), 17)
        self.assertEqual(len(OVER25_FEATURE_COLS), 18)
        self.assertEqual(len(OVER95_CORNERS_FEATURE_COLS), 17)
        validate_feature_column_counts()

    def test_expected_counts_map_is_the_contract(self):
        self.assertEqual(
            EXPECTED_FEATURE_COUNTS,
            {
                "FEATURE_COLS": 108,
                "BTTS_FEATURE_COLS": 17,
                "OVER25_FEATURE_COLS": 18,
                "OVER95_CORNERS_FEATURE_COLS": 17,
            },
        )

    def test_specialist_lists_subset_of_main_list(self):
        # Every specialist input except the 16 rolling rates it adds on top
        # must be scoreable from the main feature row.
        main = set(FEATURE_COLS)
        extra = {
            "home_btts_rate_5", "away_btts_rate_5",
            "home_cs_rate_5", "away_cs_rate_5",
            "home_scored_rate_5", "away_scored_rate_5",
            "home_over25_rate_5", "away_over25_rate_5",
            "home_high_scoring_rate_5", "away_high_scoring_rate_5",
            "home_corners_for_pg_5", "away_corners_for_pg_5",
            "home_corners_against_pg_5", "away_corners_against_pg_5",
            "home_over95_corners_rate_5", "away_over95_corners_rate_5",
        }
        for columns in (
            BTTS_FEATURE_COLS, OVER25_FEATURE_COLS, OVER95_CORNERS_FEATURE_COLS
        ):
            self.assertTrue(set(columns) <= main | extra, columns)

    def test_predict_uses_the_shared_lists(self):
        self.assertIs(predict.FEATURE_COLS, FEATURE_COLS)
        self.assertIs(predict.BTTS_FEATURE_COLS, BTTS_FEATURE_COLS)
        self.assertIs(predict.OVER25_FEATURE_COLS, OVER25_FEATURE_COLS)
        self.assertIs(
            predict.OVER95_CORNERS_FEATURE_COLS, OVER95_CORNERS_FEATURE_COLS
        )

    def test_features_csv_carries_the_match_history(self):
        # Inference replays the rolling windows from these columns; dropping
        # them from the builder output silently degrades the shot features.
        features, _, _, _ = training_frame()
        for column in builder.MATCH_HISTORY_COLUMNS:
            self.assertIn(column, features.columns)
        self.assertEqual(
            set(builder.MATCH_HISTORY_COLUMNS) - {"result"},
            {
                "home_goals", "away_goals", "home_shots", "away_shots",
                "home_shots_ot", "away_shots_ot", "home_corners",
                "away_corners",
            },
        )


class InferenceParityTests(unittest.TestCase):
    """Builder row for a match == predict vector for the same match."""

    @classmethod
    def setUpClass(cls):
        (
            cls.features,
            (cls.league_encoder, cls.season_encoder),
            cls.fbref,
            cls.elo,
        ) = training_frame()

    def setUp(self):
        # The API normally fills these in load_artifacts(); inject the fixture
        # snapshot instead so no database is contacted.
        self._saved = {
            name: getattr(predict, name)
            for name in (
                "_features_df", "_fbref_df", "_fbref_load_attempted",
                "_history_source", "_history_fbref_source", "_history",
            )
        }
        predict._fbref_df = self.fbref
        predict._fbref_load_attempted = True
        predict._elo_cache = {}
        predict._elo_lookup_failures = 0

        # In production this reads elo_ratings — the same table the builder
        # read. Substituting the fixture history keeps the test DB-free while
        # exercising the identical code path.
        patcher = mock.patch.object(
            predict,
            "_elo_rating_lookup",
            side_effect=lambda team, before: builder.get_pre_match_elo(
                self.elo, team, pd.Timestamp(before)
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(predict, name, value)

    # -- helpers ---------------------------------------------------------- #

    def fixture_csv(self, before_date: pd.Timestamp) -> pd.DataFrame:
        """The features CSV as the API would have it before ``before_date``."""
        frame = self.features.loc[self.features["date"] < before_date]
        predict._features_df = frame.sort_values("date").reset_index(drop=True)
        predict._history_source = None
        predict._history_fbref_source = None
        predict._history = {}
        return predict._features_df

    def assert_vector_matches_row(self, row: pd.Series, columns, vector):
        mismatches = []
        for column in columns:
            expected = float(row[column])
            actual = float(vector[column]) if column in vector else float("nan")
            if not np.isfinite(expected) or not np.isfinite(actual):
                mismatches.append((column, expected, actual))
            elif abs(expected - actual) > TOLERANCE:
                mismatches.append((column, expected, actual))
        self.assertEqual(
            mismatches, [],
            f"{len(mismatches)} feature(s) disagree between builder and "
            f"predict: {mismatches[:10]}",
        )

    # -- main model ------------------------------------------------------- #

    def test_main_feature_vector_matches_builder_for_last_match(self):
        target = self.features.iloc[-1]
        cutoff = target["date"]
        csv = self.fixture_csv(cutoff)

        vector = predict._build_feature_vector(
            target["home_team"],
            target["away_team"],
            int(target["league_encoded"]),
            int(target["season_encoded"]),
            csv,
            FEATURE_COLS,
            league=target["league"],
            season=target["season"],
            before_date=cutoff,
        )

        self.assertEqual(set(vector), set(FEATURE_COLS))
        self.assert_vector_matches_row(target, FEATURE_COLS, vector)

    def test_main_feature_vector_matches_builder_for_mid_history_match(self):
        target = self.features.iloc[30]
        cutoff = target["date"]
        csv = self.fixture_csv(cutoff)

        vector = predict._build_feature_vector(
            target["home_team"],
            target["away_team"],
            int(target["league_encoded"]),
            int(target["season_encoded"]),
            csv,
            FEATURE_COLS,
            league=target["league"],
            season=target["season"],
            before_date=cutoff,
        )

        self.assert_vector_matches_row(target, FEATURE_COLS, vector)

    def test_every_feature_row_agrees(self):
        """No cherry-picking: every synthetic match must reproduce exactly."""
        for position in range(len(self.features)):
            target = self.features.iloc[position]
            cutoff = target["date"]
            csv = self.fixture_csv(cutoff)
            vector = predict._build_feature_vector(
                target["home_team"],
                target["away_team"],
                int(target["league_encoded"]),
                int(target["season_encoded"]),
                csv,
                FEATURE_COLS,
                league=target["league"],
                season=target["season"],
                before_date=cutoff,
            )
            self.assert_vector_matches_row(target, FEATURE_COLS, vector)

    # -- specialist classifiers ------------------------------------------- #

    def test_specialist_vectors_match_builder(self):
        target = self.features.iloc[-1]
        cutoff = target["date"]
        csv = self.fixture_csv(cutoff)

        cases = (
            (BTTS_FEATURE_COLS, predict._build_btts_feature_vector),
            (OVER25_FEATURE_COLS, predict._build_over25_feature_vector),
            (OVER95_CORNERS_FEATURE_COLS,
             predict._build_over95_corners_feature_vector),
        )
        for columns, build in cases:
            with self.subTest(model=build.__name__):
                vector = build(
                    target["home_team"],
                    target["away_team"],
                    int(target["league_encoded"]),
                    csv,
                    league=target["league"],
                    season=target["season"],
                )
                self.assertIsNotNone(vector)
                self.assertEqual(list(vector), list(columns))
                self.assert_vector_matches_row(target, columns, vector)

    # -- encoding --------------------------------------------------------- #

    def test_label_encoding_matches_the_builder(self):
        for column, encoder, value in (
            ("league_encoded", self.league_encoder, "ENG-Premier League"),
            ("league_encoded", self.league_encoder, "ESP-La Liga"),
            ("season_encoded", self.season_encoder, SEASONS[0]),
            ("season_encoded", self.season_encoder, SEASONS[-1]),
        ):
            with self.subTest(column=column, value=value):
                expected = int(encoder.transform([value])[0])
                self.assertEqual(predict._encode_label(encoder, value), expected)

    def test_encoding_is_case_and_space_insensitive(self):
        expected = int(self.league_encoder.transform(["ENG-Premier League"])[0])
        self.assertEqual(
            predict._encode_label(self.league_encoder, " eng-premier league "),
            expected,
        )

    def test_unseen_label_uses_the_documented_fallback(self):
        self.assertEqual(
            predict._encode_label(self.league_encoder, "XYZ-No Such League"), 0
        )


class InferenceRobustnessTests(unittest.TestCase):
    """Inference degrades predictably, and never returns NaN."""

    @classmethod
    def setUpClass(cls):
        (cls.features, _, cls.fbref, cls.elo) = training_frame()

    def setUp(self):
        self._saved = {
            name: getattr(predict, name)
            for name in (
                "_features_df", "_fbref_df", "_fbref_load_attempted",
                "_history_source", "_history_fbref_source", "_history",
                "_elo_cache", "_elo_lookup_failures",
            )
        }
        predict._fbref_load_attempted = True
        predict._fbref_df = None
        predict._elo_cache = {}
        predict._elo_lookup_failures = 0

        patcher = mock.patch.object(
            predict,
            "_elo_rating_lookup",
            side_effect=lambda team, before: builder.get_pre_match_elo(
                self.elo, team, pd.Timestamp(before)
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(predict, name, value)

    def set_csv(self, frame: pd.DataFrame) -> pd.DataFrame:
        predict._features_df = (
            frame if frame is None or frame.empty
            else frame.sort_values("date").reset_index(drop=True)
        )
        predict._history_source = None
        predict._history_fbref_source = None
        predict._history = {}
        return predict._features_df

    def test_legacy_csv_without_raw_match_columns_still_scores(self):
        # A features.csv published before the raw match columns existed: goals,
        # result and corners are recovered from the target columns, and the
        # shot features fall back to the stored values of the team's last row.
        target = self.features.iloc[-1]
        cutoff = target["date"]
        legacy = self.features.loc[self.features["date"] < cutoff].drop(
            columns=["home_shots", "away_shots",
                     "home_shots_ot", "away_shots_ot"]
        )
        csv = self.set_csv(legacy)

        vector = predict._build_feature_vector(
            target["home_team"], target["away_team"],
            int(target["league_encoded"]), int(target["season_encoded"]),
            csv, FEATURE_COLS,
            league=target["league"], season=target["season"],
            before_date=cutoff,
        )

        self.assertEqual(set(vector), set(FEATURE_COLS))
        self.assertTrue(all(np.isfinite(value) for value in vector.values()))

        # Everything the CSV can still describe must match the builder exactly.
        shot_columns = {
            "home_form_shots_5", "home_form_shots_ot_5", "home_home_shots_5",
            "away_form_shots_5", "away_form_shots_ot_5", "away_away_shots_5",
        }
        for column in set(FEATURE_COLS) - shot_columns:
            self.assertAlmostEqual(
                float(target[column]), vector[column], delta=TOLERANCE,
                msg=f"{column} should still match without raw shot counts",
            )

    def test_fbref_falls_back_to_the_csv_snapshot(self):
        target = self.features.iloc[-1]
        cutoff = target["date"]
        csv = self.set_csv(self.features.loc[self.features["date"] < cutoff])

        vector = predict._build_feature_vector(
            target["home_team"], target["away_team"],
            int(target["league_encoded"]), int(target["season_encoded"]),
            csv, FEATURE_COLS,
            league=target["league"], season=target["season"],
            before_date=cutoff,
        )
        # The synthetic snapshot predates every match, so the CSV-carried
        # values and the (unavailable) database values are the same.
        for column in ("home_fbref_games", "away_fbref_crosses"):
            self.assertAlmostEqual(
                float(target[column]), vector[column], delta=TOLERANCE
            )

    def test_empty_frame_still_returns_the_full_schema(self):
        # Degenerate case (no features CSV at all): the vector keeps the
        # model's schema and every value is finite, so scoring cannot raise on
        # shape. A team with no history is the documented 1500 Elo fallback,
        # exactly as the builder scores a team with no rating.
        empty = pd.DataFrame()
        vector = predict._build_feature_vector(
            "Alpha", "Beta", 0, 0, empty, FEATURE_COLS
        )
        self.assertEqual(list(vector), list(FEATURE_COLS))
        self.assertTrue(all(np.isfinite(value) for value in vector.values()))
        self.assertEqual(vector["home_form_gf_5"], 0.0)
        self.assertEqual(vector["home_home_corners_5"], 0.0)
        self.assertEqual(vector["h2h_home_wins_5"], 0.0)
        self.assertIsNone(predict._build_btts_feature_vector(
            "Alpha", "Beta", 0, empty
        ))

    def test_teams_with_no_history_share_the_1500_elo_fallback(self):
        # Both paths must agree on the fallback for a team with no rating yet.
        features = self.features
        first = features.iloc[0]
        vector = predict._build_feature_vector(
            first["home_team"], first["away_team"],
            int(first["league_encoded"]), int(first["season_encoded"]),
            self.set_csv(pd.DataFrame()), FEATURE_COLS,
            league=first["league"], season=first["season"],
            before_date=first["date"],
        )
        self.assertEqual(float(first["home_elo"]), vector["home_elo"])
        self.assertEqual(float(first["away_elo"]), vector["away_elo"])

    def test_database_elo_wins_over_the_csv_reconstruction(self):
        # The rating table is authoritative: the CSV only knows each team's
        # pre-match rating, so a database answer must be used as-is.
        target = self.features.iloc[-1]
        cutoff = target["date"]
        csv = self.set_csv(self.features.loc[self.features["date"] < cutoff])

        with mock.patch.object(
            predict, "_elo_rating_lookup", return_value=1999.0
        ):
            predict._elo_cache = {}
            vector = predict._build_feature_vector(
                target["home_team"], target["away_team"],
                int(target["league_encoded"]), int(target["season_encoded"]),
                csv, FEATURE_COLS,
                league=target["league"], season=target["season"],
                before_date=cutoff,
            )

        self.assertEqual(vector["home_elo"], 1999.0)
        self.assertEqual(vector["away_elo"], 1999.0)
        self.assertEqual(vector["elo_difference"], 0.0)

    def test_history_reconstruction_is_sorted_and_numeric(self):
        history = inference.match_history_from_features(self.features)
        self.assertIn("date", history.columns)
        self.assertTrue(history["date"].is_monotonic_increasing)
        self.assertTrue(
            pd.api.types.is_numeric_dtype(history["home_goals"])
        )
        elo = inference.elo_history_from_features(self.features)
        self.assertEqual(set(elo.columns), {"team", "date", "elo"})
        self.assertFalse(elo["elo"].isna().any())

    def test_cutoff_only_uses_earlier_days(self):
        # Same-day matches are excluded, exactly like the builder's cutoff.
        target = self.features.iloc[-1]
        history = inference.match_history_from_features(
            self.features.loc[self.features["date"] <= target["date"]]
        )
        cutoff = inference._as_naive_date(target["date"])
        same_day = history.loc[history["date"] >= cutoff]
        self.assertFalse(same_day.empty)  # the row is there...
        features = inference.build_inference_features(
            history,
            inference.elo_history_from_features(self.features),
            self.fbref,
            target["home_team"],
            target["away_team"],
            before_date=cutoff,
            league=target["league"],
            season=target["season"],
        )
        # ...but it must not contribute to the rolling windows.
        self.assertAlmostEqual(
            features["home_form_wins_5"],
            float(target["home_form_wins_5"]),
            delta=TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
