"""
Single source of truth for the model input columns.

The training pipeline (``features/builder.py`` -> ``models/trainer.py``) and
the inference path (``routers/predict.py`` -> ``features/inference.py``) must
feed the models *exactly* the same columns, in the same order. These lists used
to be copy-pasted into ``models/trainer.py`` and ``routers/predict.py``; a typo
or an added column in only one copy produced a model that scored plausible
numbers from wrong or zeroed inputs without raising anything.

Import them from here. Never redefine them elsewhere.

Counts are part of the contract: 108 / 17 / 18 / 17. ``tests/test_feature_parity.py``
asserts both the counts and that builder-side rows and predict-side vectors
agree column by column.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# FBref team-season stats
# --------------------------------------------------------------------------- #

#: Stat columns copied from ``fbref_team_stats`` into the feature frame, once
#: per side (``home_fbref_*`` / ``away_fbref_*``). This order is also the order
#: the columns appear in :data:`FEATURE_COLS`.
FBREF_STAT_COLUMNS = [
    "games", "shots",
    "shots_on_target", "shot_accuracy",
    "goals_per_shot", "goals_per_sot",
    "yellow_cards", "red_cards",
    "fouls_committed", "fouls_drawn",
    "offsides", "crosses",
    "interceptions", "tackles_won",
    "pens_won", "pens_conceded",
    "sota", "goals_against",
    "saves", "save_pct",
    "clean_sheets", "pk_faced",
    "pk_saved",
]


#: The 46 FBref features, in ``FEATURE_COLS`` order (home block, then away).
FBREF_FEATURE_COLS: list[str] = [
    f"{role}_fbref_{stat}"
    for role in ("home", "away")
    for stat in FBREF_STAT_COLUMNS
]

# --------------------------------------------------------------------------- #
# Match history carried by the features CSV
# --------------------------------------------------------------------------- #

#: Raw match columns the rolling-window helpers read. ``build_features`` writes
#: them into every training row (as meta, never as features) so a saved
#: features CSV can reconstruct the match history; ``features/inference.py``
#: replays the same helpers from them for a live fixture.
MATCH_HISTORY_COLUMNS: list[str] = [
    "home_goals", "away_goals", "result",
    "home_shots", "away_shots",
    "home_shots_ot", "away_shots_ot",
    "home_corners", "away_corners",
]

#: Fallback spellings for the columns above, in lookup order. The ``target_*``
#: names are what a features CSV written before this change carries; the
#: ``*_score`` names are the StatsBomb/spec spelling the helpers accept.
MATCH_HISTORY_ALIASES: dict[str, tuple[str, ...]] = {
    "home_goals":    ("home_goals", "target_home_goals", "home_score"),
    "away_goals":    ("away_goals", "target_away_goals", "away_score"),
    "result":        ("result", "target_result"),
    "home_shots":    ("home_shots",),
    "away_shots":    ("away_shots",),
    "home_shots_ot": ("home_shots_ot",),
    "away_shots_ot": ("away_shots_ot",),
    "home_corners":  ("home_corners", "target_home_corners", "home_crs"),
    "away_corners":  ("away_corners", "target_away_corners", "away_crs"),
}


# --------------------------------------------------------------------------- #
# Main 1X2 / goals / corners model: 108 features
# --------------------------------------------------------------------------- #

#: Match-derived features (rolling form, venue form, Elo, H2H).
BASE_FEATURE_COLS = [
    "home_form_wins_5", "home_form_draws_5",
    "home_form_losses_5", "home_form_points_5",
    "home_form_gf_5", "home_form_ga_5",
    "home_form_gd_5", "home_form_shots_5",
    "home_form_shots_ot_5", "home_form_corners_5",
    "home_form_wins_10", "home_form_draws_10",
    "home_form_losses_10", "home_form_points_10",
    "home_form_gf_10", "home_form_ga_10",
    "home_form_gd_10", "home_home_wins_5",
    "home_home_draws_5", "home_home_losses_5",
    "home_home_points_5", "home_home_gf_5",
    "home_home_ga_5", "home_home_gd_5",
    "home_home_shots_5", "home_home_corners_5",
    "away_form_wins_5", "away_form_draws_5",
    "away_form_losses_5", "away_form_points_5",
    "away_form_gf_5", "away_form_ga_5",
    "away_form_gd_5", "away_form_shots_5",
    "away_form_shots_ot_5", "away_form_corners_5",
    "away_form_wins_10", "away_form_draws_10",
    "away_form_losses_10", "away_form_points_10",
    "away_form_gf_10", "away_form_ga_10",
    "away_form_gd_10", "away_away_wins_5",
    "away_away_draws_5", "away_away_losses_5",
    "away_away_points_5", "away_away_gf_5",
    "away_away_ga_5", "away_away_gd_5",
    "away_away_shots_5", "away_away_corners_5",
    "home_elo", "away_elo",
    "elo_difference", "h2h_home_wins_5",
    "h2h_draws_5", "h2h_away_wins_5",
    "h2h_home_gf_5", "h2h_away_gf_5",
]


#: Canonical model input order. Do not reorder: the trainer fits on this order
#: and the API scores with it, so a reordering silently attaches every weight
#: to the wrong feature.
FEATURE_COLS: list[str] = (
    BASE_FEATURE_COLS
    + ["league_encoded", "season_encoded"]
    + FBREF_FEATURE_COLS
)

# --------------------------------------------------------------------------- #
# Specialist classifiers
# --------------------------------------------------------------------------- #

BTTS_FEATURE_COLS = [
    "home_btts_rate_5", "away_btts_rate_5",
    "home_cs_rate_5", "away_cs_rate_5",
    "home_scored_rate_5", "away_scored_rate_5",
    "home_form_gf_5", "away_form_gf_5",
    "home_form_ga_5", "away_form_ga_5",
    "home_fbref_goals_against", "away_fbref_goals_against",
    "home_fbref_clean_sheets", "away_fbref_clean_sheets",
    "home_elo", "away_elo",
    "league_encoded",
]

OVER25_FEATURE_COLS = [
    "home_over25_rate_5", "away_over25_rate_5",
    "home_high_scoring_rate_5", "away_high_scoring_rate_5",
    "home_form_gf_5", "away_form_gf_5",
    "home_form_ga_5", "away_form_ga_5",
    "home_form_gf_10", "away_form_gf_10",
    "home_fbref_goals_against", "away_fbref_goals_against",
    "home_fbref_shots_on_target", "away_fbref_shots_on_target",
    "home_elo", "away_elo",
    "elo_difference", "league_encoded",
]

#: The six corner-specific rolling rates the Over 9.5 model cannot be scored
#: without. Shared by the trainer, the builder-facing validator and predict.
OVER95_CORNERS_REQUIRED_COLS: list[str] = [
    "home_corners_for_pg_5", "away_corners_for_pg_5",
    "home_corners_against_pg_5", "away_corners_against_pg_5",
    "home_over95_corners_rate_5", "away_over95_corners_rate_5",
]

OVER95_CORNERS_FEATURE_COLS = [
    "home_corners_for_pg_5", "away_corners_for_pg_5",
    "home_corners_against_pg_5", "away_corners_against_pg_5",
    "home_over95_corners_rate_5", "away_over95_corners_rate_5",
    "home_home_corners_5", "away_away_corners_5",
    "home_form_corners_5", "away_form_corners_5",
    "home_fbref_shots_on_target", "away_fbref_shots_on_target",
    "home_fbref_crosses", "away_fbref_crosses",
    "home_elo", "away_elo",
    "league_encoded",
]


# --------------------------------------------------------------------------- #
# Contract checks
# --------------------------------------------------------------------------- #

#: Expected length of each list. A change here is a model-contract change: the
#: existing artifacts would no longer match, so both the trainer and the API
#: raise on import instead of silently scoring a different vector.
EXPECTED_FEATURE_COUNTS: dict[str, int] = {
    "FEATURE_COLS": 108,
    "BTTS_FEATURE_COLS": 17,
    "OVER25_FEATURE_COLS": 18,
    "OVER95_CORNERS_FEATURE_COLS": 17,
}


def validate_feature_column_counts() -> None:
    """Raise ``RuntimeError`` when a list no longer has its contract length.

    Called at import time of the trainer and the API, and by the parity test,
    so a half-finished edit cannot reach production unnoticed.
    """
    actual = {
        "FEATURE_COLS": len(FEATURE_COLS),
        "BTTS_FEATURE_COLS": len(BTTS_FEATURE_COLS),
        "OVER25_FEATURE_COLS": len(OVER25_FEATURE_COLS),
        "OVER95_CORNERS_FEATURE_COLS": len(OVER95_CORNERS_FEATURE_COLS),
    }
    wrong = {
        name: (actual[name], expected)
        for name, expected in EXPECTED_FEATURE_COUNTS.items()
        if actual[name] != expected
    }
    if wrong:
        details = ", ".join(
            f"{name}={got} (expected {expected})"
            for name, (got, expected) in wrong.items()
        )
        raise RuntimeError(f"feature column contract violated: {details}")
    for name, columns in (
        ("FEATURE_COLS", FEATURE_COLS),
        ("BTTS_FEATURE_COLS", BTTS_FEATURE_COLS),
        ("OVER25_FEATURE_COLS", OVER25_FEATURE_COLS),
        ("OVER95_CORNERS_FEATURE_COLS", OVER95_CORNERS_FEATURE_COLS),
    ):
        if len(set(columns)) != len(columns):
            duplicates = sorted({c for c in columns if columns.count(c) > 1})
            raise RuntimeError(f"{name} contains duplicate columns: {duplicates}")


validate_feature_column_counts()
