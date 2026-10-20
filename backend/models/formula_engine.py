"""
Formula-based prediction engine.

Replaces ML models with a statistical approach:
  1. Compute team attack rating from xG,
     shots, progressive actions, pressing
  2. Compute team defence rating from xGA,
     saves, clean sheets, pressing allowed
  3. Apply home advantage factor
  4. Use Poisson distribution for all markets

This engine uses live current-season data
from team_season_stats and player_season_stats
rather than historical rolling averages.

Data sources (all three "new" tables are read):
  team_season_stats    FBref / Understat / StatsBomb team stats
  player_season_stats  FBref player stats (squad quality + injuries)
  statsbomb_match_stats  head-to-head xG from event data (optional)

Run standalone from the backend/ directory:
    python models/formula_engine.py "Arsenal" "Chelsea" "ENG-Premier League"
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import poisson

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, TeamSeasonStats, PlayerSeasonStats
from sqlalchemy import and_, text

# ═══════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════

# Name reported in the "engine" field of a prediction.
ENGINE_NAME = "formula_v1"

# Home advantage multiplier
# Teams score ~30% more at home on average
HOME_ADVANTAGE = 1.15
AWAY_ADJUSTMENT = 0.90

# League average xG per game (fallback)
LEAGUE_AVG_XG = 1.35

# League average team-level goal-creating actions per 90 (fallback).
# FBref team GCA/90 sits around 5-7; this is only used when a league has
# no rows to measure it from.
DEFAULT_GCA90 = 5.5

# Minimum games played to use a team's stats
MIN_GAMES = 3

# Season preference order — most recent first
SEASON_ORDER = [
    "2026-2027", "2025-2026", "2024-2025",
    "2023-2024", "2026", "2025", "2024", "2023",
]

# Number of games after which current season
# gets full weight (no blending)
FULL_WEIGHT_GAMES = 20

# Games assumed for the blend weight when a row does not say how many
# games it covers (per-game aggregates). Mid-season, so an even blend.
DEFAULT_SEASON_GAMES = 10

# UCL performance is supplementary to each team's domestic baseline.
UCL_LEAGUE_NAME = "UEFA-Champions League"
UCL_MAX_WEIGHT = 0.30
UCL_FULL_GAMES = 10

# Stat keys copied from the current season untouched by the season blend:
# the keeper W-D-L columns are a record (or a five-game form window) that
# games_played() reads, so blending them would corrupt the games estimate.
_UNBLENDED_KEYS = (
    "form_wins", "form_draws", "form_losses",
    "games", "team", "league", "season",
)

# Max goals to consider in Poisson sum
MAX_GOALS = 10

# Corner rate multiplier — corners per xG unit
CORNER_RATE = 5.2

# Attack weight factors
ATTACK_WEIGHTS = {
    "xg_per_game":      0.35,
    "sot_per_game":     0.20,
    "g_per_shot":       0.15,
    "progressive":      0.15,
    "press_succ":       0.15,
}

# --- Shot quality ------------------------------------------------------ #

# Typical goals per shot. FBref rows sit between ~0.05 and ~0.25, so a
# stored value outside GOALS_PER_SHOT_RANGE is not measuring goals per shot
# at all (the API-Football collector derives a placeholder from goals and
# games) and the neutral value is used instead of a meaningless ratio.
DEFAULT_G_PER_SHOT = 0.10
GOALS_PER_SHOT_RANGE = (0.02, 0.40)

# Defence weight factors
DEFENCE_WEIGHTS = {
    "xga_per_game":     0.35,
    "saves_per_game":   0.20,
    "clean_sheet_pct":  0.15,
    "press_allowed":    0.15,
    "blocks_per_game":  0.15,
}

# --- Expected-goals clamps --------------------------------------------- #

# A Poisson mean outside this range is never realistic for one team in one
# match, so clamping keeps a bad data row from producing a 99% favourite.
MIN_LAMBDA = 0.15
MAX_LAMBDA = 5.0

# League-average goals per team per game is read from the data, but only
# inside this window (real top divisions sit between ~1.15 and ~1.65).
MIN_BASE_GOALS = 0.90
MAX_BASE_GOALS = 1.80

# Relative ratings (team rating / league mean rating) are clamped too, so a
# single half-scraped team cannot become unbeatable.
MIN_RELATIVE = 0.40
MAX_RELATIVE = 2.50

# --- Goalkeeper -------------------------------------------------------- #

# PSxG+/- per 90 is the cleanest measure of keeper shot-stopping value.
# A keeper worth +0.30 PSxG/90 (elite) removes ~5% of the goals conceded;
# -0.30 (poor) adds ~5%. Deliberately gentle — the defence rating already
# carries save% and clean sheets.
GK_SENSITIVITY = 0.16
GK_MIN_MODIFIER = 0.90
GK_MAX_MODIFIER = 1.10
GK_TYPICAL_PSXG90 = 0.30

# --- Injuries ---------------------------------------------------------- #

# How strongly a lost share of output moves the ratings.
INJURY_ATTACK_SENSITIVITY = 0.60   # share of squad xG+xAG lost
INJURY_DEFENCE_SENSITIVITY = 0.45  # share of defensive minutes lost

INJURY_MIN_ATTACK = 0.75           # attack modifier floor
INJURY_MAX_DEFENCE = 1.30          # defence modifier ceiling

# Used when a missing player cannot be matched to player_season_stats:
# every unknown absence costs a flat amount.
INJURY_UNKNOWN_ATTACK_PENALTY = 0.015
INJURY_UNKNOWN_DEFENCE_PENALTY = 0.010
INJURY_MAX_UNKNOWN = 5

# Midfielders count half as much as defenders/keepers for the defensive
# share, because their minutes protect the goal less directly.
INJURY_MIDFIELD_WEIGHT = 0.5

DEFENSIVE_POSITIONS = ("GK", "DF")
MIDFIELD_POSITIONS = ("MF",)

# Player-name match confidence, see _name_match_score().
NAME_SCORE_EXACT = 3
NAME_SCORE_CONTAINED = 2
NAME_SCORE_SURNAME = 1

# --- Corners ----------------------------------------------------------- #

# Empirical average total corners per match across the big five leagues.
# The corner expectation below is built from crosses / attacking-third
# touches / carries into the box, which is proportional to corners but not
# in corner units, so it is rescaled so that two league-average teams land
# on this number.
CORNER_LEAGUE_TOTAL = 10.4
CORNER_MIN_SCALE = 0.50
CORNER_MAX_SCALE = 2.50
CORNER_HOME_TILT = 1.05            # home side wins slightly more corners
CORNER_AWAY_TILT = 0.95            # (sums to 2.0, so the total is kept)
MIN_EXPECTED_CORNERS = 1.0
MAX_EXPECTED_CORNERS = 12.0

# --- Head to head ------------------------------------------------------ #

H2H_TABLE = "statsbomb_match_stats"
H2H_MAX_MATCHES = 6
H2H_MIN_MATCHES = 2
H2H_WEIGHT = 0.08                  # max tilt in either direction
H2H_MIN_MODIFIER = 0.90
H2H_MAX_MODIFIER = 1.10

# --- Caching ----------------------------------------------------------- #

# League averages and league baselines are identical for every fixture in a
# league, and the daily fixtures job predicts dozens of matches in a row, so
# both are cached in-process for a short while.
CACHE_TTL_SECONDS = 900.0
CACHE_MAX_ENTRIES = 128

# Team-name scale detection: a row whose totals are within this factor of a
# single game's worth of stats is already stored per game (StatsBomb
# aggregates), not per season (FBref / Understat).
RATE_SCALE_TOLERANCE = 1.05

# How many games a stat row can cover: a domestic season plus a cup run.
# An estimate above this means two columns of different scales were paired
# up, which is not a game count.
MAX_GAMES_ESTIMATE = 60.0

# The keeper W-D-L columns hold a *last-five form window* in the
# API-Football rows and a full-season record in the FBref / Understat rows,
# so a total this small is not evidence of a short season.
FORM_WINDOW = 5

# Fallback league averages, all per game, used when a league has no rows.
DEFAULT_LEAGUE_AVERAGES = {
    "avg_xg":    LEAGUE_AVG_XG,
    "avg_xga":   LEAGUE_AVG_XG,
    "avg_shots": 12.0,
    "avg_sot":   4.5,
    "avg_saves": 3.5,
    "avg_gca90": DEFAULT_GCA90,
    # Used as the "missing value" stand-in inside the rating formulas, so a
    # half-scraped row reads as an average team instead of an outlier.
    "avg_prgc":    13.0,
    "avg_prgp":    45.0,
    "avg_att3rd":  150.0,
    "avg_tkl_int": 20.0,
}

# Stat keys that are season *totals* and must be divided by games played
# before they are compared with the fixed denominators in the rating
# formulas. Everything not listed here is already a rate, a percentage or a
# per-90 value.
#
# A total whose RATE_KEYS twin is stored (goals against, shots, shots on
# target) is overwritten by that rate in :func:`to_per_game`, so the
# division below is only the fallback for rows with no rate column.
PER_GAME_KEYS = (
    "xg", "npxg", "shots", "sot", "goals",
    "prgp", "prgc", "passes_att", "passes_cmp", "kp", "ppa",
    "gca", "sca",
    "press", "press_succ", "tkl", "int_", "tkl_int",
    "blocks", "clr", "err",
    "touch_att3rd", "cpa",
    "ga", "sota", "saves", "cs", "psxg", "psxg_diff",
    "misc_crs", "corners",
)

# Total / rate pairs used to recover games played when W-D-L is missing.
GAMES_RATE_PAIRS = (
    ("shots", "sh_per90"),
    ("sot",   "sot_per90"),
    ("ga",    "ga90"),
    ("sca",   "sca90"),
    ("gca",   "gca90"),
)

# Totals that also have a stored per-90 column. These rate columns are the
# values this engine trusts for a per-game figure: a collector writes them
# from a single row (FBref's GA90 / Sh/90, the API-Football standings rate),
# they are already per game, and unlike ``total / games_played`` they cannot
# be corrupted by a bad games-played estimate. See :func:`to_per_game`.
RATE_KEYS = {
    "ga":    "ga90",
    "shots": "sh_per90",
    "sot":   "sot_per90",
}

# Shared cache: key -> (expires_at_monotonic, value)
_CACHE: dict[tuple, tuple[float, Any]] = {}

# Keys we have already logged a warning for, so a missing table or an
# absent league does not spam the Render log once per fixture.
_WARNED: set[str] = set()


# ═══════════════════════════════════════════
# SMALL HELPERS
# ═══════════════════════════════════════════


def _safe(val, default: float = 0.0) -> float:
    """Return float or default if None/NaN."""
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _num(val, digits: int = 4) -> float:
    """
    Round to ``digits`` and return a plain Python float.

    Every number that leaves this module in a prediction dict goes through
    here, so the response never contains a numpy scalar (FastAPI's JSON
    encoder does not know those).
    """
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return round(f, digits)


def _warn_once(key: str, message: str) -> None:
    """Print a warning the first time ``key`` is seen."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(message)


def _cache_get(key: tuple) -> Any:
    """Return a cached value, or None when absent/expired."""
    hit = _CACHE.get(key)
    if hit is None:
        return None
    expires_at, value = hit
    if time.monotonic() > expires_at:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: tuple, value: Any) -> None:
    """Store a value in the shared cache."""
    if len(_CACHE) >= CACHE_MAX_ENTRIES:
        _CACHE.clear()
    _CACHE[key] = (time.monotonic() + CACHE_TTL_SECONDS, value)


def clear_cache() -> None:
    """Drop every cached league average / baseline (e.g. after a re-scrape)."""
    _CACHE.clear()


def _new_session():
    """
    Open a database session, or return None when that is impossible.

    ``database.py`` builds SessionLocal even when the engine could not be
    created, so session construction itself can raise (no DATABASE_URL, no
    driver, pool exhausted). Every loader here degrades to its own neutral
    fallback instead of letting the prediction request blow up.
    """
    try:
        return SessionLocal()
    except Exception as exc:
        _warn_once(
            "session",
            f"[formula] ERROR: cannot open a database session: {exc}",
        )
        return None


def _close_session(db) -> None:
    """Close a session, ignoring errors from an already-broken connection."""
    if db is None:
        return
    try:
        db.close()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[formula] WARNING: could not close DB session: {exc}")


# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════

# Common club suffixes and provider aliases stripped when deduplicating
# league teams across Understat, FBref and API-Football.
_TEAM_SUFFIXES = (
    " & hove albion",
    " and hove albion",
    " hotspur",
    " wanderers",
    " united",
    " utd",
    " city",
    " town",
    " albion",
    " athletic",
    " rovers",
    " county",
    " fc",
    " afc",
    " sc",
    " cf",
    " fco",
    " hsc",
    " sco",
    " ac",
    " foot",
)

# Two-club city stems where stripping "city" / "united" / "fc" would
# collapse two distinct clubs into one.
_PROTECTED_TEAM_STEMS = frozenset({
    "manchester",
    "sheffield",
    "bristol",
    "paris",
    "real",
})

_TEAM_ALIASES = {
    # FotMob's short forms. Without them the suffix strip below turns
    # "man city" and "man utd" into the same key ("man") — two different
    # clubs collapsed into one — while leaving FBref's "manchester city"
    # / "manchester utd" as two more keys for the same two clubs.
    "man city":                 "manchester city",
    "man utd":                  "manchester united",
    "man united":               "manchester united",
    "manchester utd":           "manchester united",
    "spurs":                    "tottenham",
    "wolverhampton":            "wolves",
    "wolverhampton wanderers":  "wolves",
    "nott'm forest":            "nottingham forest",
    "sheffield utd":            "sheffield united",
    "afc bournemouth":          "bournemouth",
    "west bromwich":            "west brom",
    "deportivo alaves":         "alaves",
    "real betis":               "betis",
    "espanol":                  "espanyol",
    "bayern munchen":           "bayern munich",
    "bayer leverkusen":         "leverkusen",
    "borussia dortmund":        "dortmund",
    "borussia m.gladbach":      "gladbach",
    "borussia monchengladbach": "gladbach",
    "m'gladbach":               "gladbach",
    "1. fsv mainz 05":          "mainz",
    "mainz 05":                 "mainz",
    "fc koln":                  "koln",
    "1. fc koln":               "koln",
    "tsg hoffenheim":           "hoffenheim",
    "spvgg greuther furth":     "greuther furth",
    "ac milan":                 "milan",
    "inter milan":              "inter",
    "internazionale":           "inter",
    "hellas verona":            "verona",
    "as roma":                  "roma",
    "ss lazio":                 "lazio",
    "ssc napoli":               "napoli",
    "paris saint-germain":      "paris saint germain",
    "paris sg":                 "paris saint germain",
    "saint-etienne":            "saint etienne",
    "olympique de marseille":   "marseille",
    "olympique lyonnais":       "lyon",
    "ogc nice":                 "nice",
    "rc lens":                  "lens",
    "stade rennais":            "rennes",
    "stade rennais fc":         "rennes",
    "stade brestois 29":        "brest",
    "stade brestois":           "brest",
    "fc nantes":                "nantes",
    "rc strasbourg":            "strasbourg",
    "fc metz":                  "metz",
    "stade de reims":           "reims",
    "fc lorient":               "lorient",
    "girondins de bordeaux":    "bordeaux",
    "en avant guingamp":        "guingamp",
    "aj auxerre":               "auxerre",
    "as monaco":                "monaco",
    "losc lille":               "lille",
    "lille osc":                "lille",
}


def _dedup_team_key(name: str) -> str:
    """
    Canonical team key after stripping common club suffixes and normalising
    provider aliases (Understat vs API-Football).
    """
    key = (name or "").lower().strip()
    key = _TEAM_ALIASES.get(key, key)
    for suffix in _TEAM_SUFFIXES:
        if key.endswith(suffix):
            stem = key[: -len(suffix)].strip()
            if stem and stem not in _PROTECTED_TEAM_STEMS:
                key = stem
                break
    return _TEAM_ALIASES.get(key, key)


def _row_to_stats_dict(row) -> dict:
    """
    Map one TeamSeasonStats ORM row onto the flat stat dict used by every
    rating function in this module.

    Values are exactly as stored: FBref and Understat write season totals,
    while the StatsBomb aggregator writes per-game averages. Use
    :func:`to_per_game` to normalise before comparing teams.
    """
    return {
        # Shooting / Attack
        "xg":           row.sh_xg,
        "npxg":         row.sh_npxg,
        "shots":        row.sh_shots,
        "sot":          row.sh_sot,
        "sot_pct":      row.sh_sot_pct,
        "sh_per90":     row.sh_sh_per90,
        "sot_per90":    row.sh_sot_per90,
        "g_per_sh":     row.sh_g_per_sh,
        "goals":        row.sh_goals,
        # Passing / Progressive
        "prgp":         row.ps_prgp,
        "prgc":         row.pos_prgc,
        "passes_att":   row.ps_att,
        "passes_cmp":   row.ps_cmp,
        "passes_pct":   row.ps_cmp_pct,
        "kp":           row.ps_kp,
        "ppa":          row.ps_ppa,
        # Goal creation
        "gca":          row.gsc_gca,
        "gca90":        row.gsc_gca90,
        "sca":          row.gsc_sca,
        "sca90":        row.gsc_sca90,
        # Defence
        "press":        row.def_press,
        "press_succ":   row.def_press_succ,
        "press_pct":    row.def_press_pct,
        "tkl":          row.def_tkl,
        "int_":         row.def_int,
        "tkl_int":      row.def_tkl_int,
        "blocks":       row.def_blocks,
        "clr":          row.def_clr,
        "err":          row.def_err,
        # Possession
        "poss":         row.pos_poss,
        "touch_att3rd": row.pos_touch_att3rd,
        "cpa":          row.pos_cpa,
        "to_succ_pct":  row.pos_to_succ_pct,
        # Goalkeeping
        "ga":           row.gk_ga,
        "ga90":         row.gk_ga90,
        "sota":         row.gk_sota,
        "saves":        row.gk_saves,
        "save_pct":     row.gk_save_pct,
        "cs":           row.gk_cs,
        "cs_pct":       row.gk_cs_pct,
        # Advanced GK
        "psxg":         row.gka_psxg,
        "psxg_diff":    row.gka_psxg_diff,
        "psxg_diff90":  row.gka_psxg_diff90,
        "opa90":        row.gka_opa90,
        # Misc
        "misc_crs":     row.misc_crs,
        "corners":      row.misc_corners,
        "aerial_pct":   row.misc_aerial_pct,
        # Form stored fields
        "form_wins":    row.gk_w,
        "form_draws":   row.gk_d,
        "form_losses":  row.gk_l,
        # Row metadata (not a stat — never normalised)
        "team":         row.team,
        "league":       row.league,
        "season":       row.season,
    }


def _merge_stat_dicts(primary: dict, secondary: dict) -> dict:
    """Fill missing (None) stat entries in ``primary`` from ``secondary``."""
    merged = dict(primary)
    for k, v in secondary.items():
        if merged.get(k) is None and v is not None:
            merged[k] = v
    return merged


def _blend_stats(
    current: dict,
    previous: dict | None,
    games_current: float,
) -> dict:
    """
    Blend current and previous season stats
    using a weight based on games played.

    At FULL_WEIGHT_GAMES games the current
    season gets 100% weight. Below that
    the previous season fills the gap.

    Only numeric values are blended.
    Non-numeric values come from current.
    None values fall back to the other season.
    """
    if previous is None or games_current >= FULL_WEIGHT_GAMES:
        return current

    w_cur = max(0.0, min(games_current / FULL_WEIGHT_GAMES, 1.0))
    w_pre = 1.0 - w_cur

    blended = {}
    all_keys = set(current) | set(previous)

    for key in all_keys:
        cur_val = current.get(key)
        pre_val = previous.get(key)

        if cur_val is None and pre_val is None:
            blended[key] = None
            continue

        if cur_val is None:
            blended[key] = pre_val
            continue

        if pre_val is None:
            blended[key] = cur_val
            continue

        try:
            blended[key] = round(
                w_cur * float(cur_val) +
                w_pre * float(pre_val),
                4
            )
        except (TypeError, ValueError):
            blended[key] = cur_val

    return blended


def _previous_season(season: str | None) -> str | None:
    """
    The season before ``season`` in SEASON_ORDER, or None.

    SEASON_ORDER mixes two labelling schemes (European "2023-2024" and
    MLS calendar "2026"). The list is not a timeline across that boundary,
    so "2023-2024" must not be blended with "2026". Only a neighbour with
    the same label style counts as the previous season.
    """
    if not season or season not in SEASON_ORDER:
        return None
    idx = SEASON_ORDER.index(season)
    if idx + 1 >= len(SEASON_ORDER):
        return None
    previous = SEASON_ORDER[idx + 1]
    if ("-" in season) != ("-" in previous):
        return None
    return previous


def _per_game_values(stats: dict, games: float | None) -> dict:
    """
    Copy of ``stats`` with every season total divided by ``games``.

    A row whose games are unknown (None) already holds per-game values, so
    it is returned as a copy unchanged. Blending has to happen on per-game
    values: a 6-game season total and a 38-game season total cannot be
    averaged directly.
    """
    per_game = dict(stats)
    if games and games > 0:
        for key in PER_GAME_KEYS:
            value = per_game.get(key)
            if value is not None:
                per_game[key] = _safe(value) / games
    return per_game


def _load_season_stats(
    db,
    team: str,
    league: str,
    season: str,
) -> dict | None:
    """
    Stat dict for one team in one season, merged across provider-variant
    team names the same way load_team_stats does for the current season.
    Returns None when the team has no row in that season.
    """
    exact = (
        db.query(TeamSeasonStats)
        .filter(and_(
            TeamSeasonStats.team   == team,
            TeamSeasonStats.league == league,
            TeamSeasonStats.season == season,
        )).first()
    )

    target_key = _dedup_team_key(team)
    variants = [
        r for r in db.query(TeamSeasonStats).filter(and_(
            TeamSeasonStats.league == league,
            TeamSeasonStats.season == season,
        )).all()
        if _dedup_team_key(r.team) == target_key
    ]

    rows = ([exact] if exact is not None else []) + [
        r for r in variants if exact is None or r.id != exact.id
    ]
    if not rows:
        return None

    stats = _row_to_stats_dict(rows[0])
    for extra in rows[1:]:
        stats = _merge_stat_dicts(stats, _row_to_stats_dict(extra))
    return stats


def _blend_with_previous_season(
    db,
    team: str,
    league: str,
    stats: dict,
) -> dict:
    """
    Blend the current season's stats with the previous season's, weighted
    by games played so far (see _blend_stats).

    ``stats`` must already carry ``games`` (set by load_team_stats). The
    result stays in the same format as the input: season totals stay totals
    and per-game rows stay per-game, so to_per_game() downstream works
    unchanged. The result also records ``blend_weight_current`` (the share
    of the current season, 0-1) and ``blend_previous_season``.
    """
    stats["blend_weight_current"] = 1.0
    stats["blend_previous_season"] = None

    games_cur = games_played(stats)
    # Per-game rows (StatsBomb) do not say how many games they cover.
    weight_games = (
        games_cur if games_cur is not None and games_cur >= 1
        else float(DEFAULT_SEASON_GAMES)
    )
    if weight_games >= FULL_WEIGHT_GAMES:
        return stats

    prev_season = _previous_season(stats.get("season"))
    if prev_season is None:
        return stats

    previous = _load_season_stats(db, team, league, prev_season)
    if previous is None:
        return stats

    games_prev = games_played(previous)
    blended = _blend_stats(
        _per_game_values(stats, games_cur),
        _per_game_values(previous, games_prev),
        weight_games,
    )

    # Back to season totals on the current row's games, so the downstream
    # to_per_game() divides them by the same count and gets the blend.
    if games_cur:
        for key in PER_GAME_KEYS:
            if blended.get(key) is not None:
                blended[key] = round(_safe(blended[key]) * games_cur, 4)

    for key in _UNBLENDED_KEYS:
        blended[key] = stats.get(key)

    w_cur = max(0.0, min(weight_games / FULL_WEIGHT_GAMES, 1.0))
    blended["blend_weight_current"] = round(w_cur, 4)
    blended["blend_previous_season"] = prev_season

    games_label = (
        f"{weight_games:.0f} games" if games_cur is not None
        else f"games unknown (assuming {weight_games:.0f})"
    )
    print(f"[formula] Blending {team} {games_label}: "
          f"{w_cur:.0%} current + "
          f"{1.0 - w_cur:.0%} previous")
    return blended


def load_ucl_stats(
    team: str,
    season: str | None = None,
) -> dict | None:
    """
    Load UCL season stats for a team.
    Returns stats dict or None if not found.
    """
    db = _new_session()
    if db is None:
        return None
    try:
        query = db.query(TeamSeasonStats).filter(
            and_(
                TeamSeasonStats.team   == team,
                TeamSeasonStats.league == UCL_LEAGUE_NAME,
            )
        )
        if season:
            query = query.filter(
                TeamSeasonStats.season == season)
        else:
            query = query.order_by(
                TeamSeasonStats.season.desc())
        row = query.first()
        if row is None:
            return None
        return _row_to_stats_dict(row)
    except Exception as exc:
        print(f"[formula] WARNING: could not "
              f"load UCL stats for {team}: {exc}")
        return None
    finally:
        _close_session(db)


def blend_ucl_with_domestic(
    domestic_stats: dict,
    ucl_stats: dict | None,
    domestic_games: float,
) -> dict:
    """
    Blend UCL stats with domestic league stats.

    UCL weight = min(ucl_games / UCL_FULL_GAMES, UCL_MAX_WEIGHT). At 2 UCL
    games, UCL contributes 20%; at 5 or more games the contribution caps
    at 30%, keeping the domestic league as the representative baseline.

    The cap accounts for UCL's different opposition quality and the small
    sample sizes early in the season.
    """
    if ucl_stats is None:
        return domestic_stats

    ucl_games = games_played(ucl_stats) or 0
    if ucl_games < 1:
        return domestic_stats

    w_ucl = min(
        ucl_games / UCL_FULL_GAMES,
        UCL_MAX_WEIGHT,
    )
    w_dom = 1.0 - w_ucl

    print(f"[formula] UCL blend: "
          f"{ucl_games} UCL games → "
          f"{w_ucl:.0%} UCL + "
          f"{w_dom:.0%} domestic")

    return _blend_stats(
        _per_game_values(domestic_stats, domestic_games),
        _per_game_values(ucl_stats, ucl_games),
        w_dom * FULL_WEIGHT_GAMES,
    )


def load_team_stats(
    team: str,
    league: str,
    season: str | None = None,
) -> dict | None:
    """
    Load team stats from team_season_stats.
    If season is None, walks PREFERRED_SEASONS (newest first, European
    labels then MLS calendar years) and uses the first that has a row,
    falling back to the newest season present. Returns dict of stats or
    None if not found.
    """
    db = _new_session()
    if db is None:
        return None
    try:
        query = db.query(TeamSeasonStats).filter(
            and_(
                TeamSeasonStats.team   == team,
                TeamSeasonStats.league == league,
            )
        )
        if season:
            query = query.filter(
                TeamSeasonStats.season == season)
        else:
            query = query.order_by(
                TeamSeasonStats.season.desc())

        # Prefer most recent season with data
        PREFERRED_SEASONS = [
            "2026-2027", "2025-2026", "2024-2025",
            "2026", "2025", "2024",
        ]
        row = None
        if season:
            row = query.filter(
                TeamSeasonStats.season == season
            ).first()
        if row is None:
            for preferred in PREFERRED_SEASONS:
                row = (
                    db.query(TeamSeasonStats)
                    .filter(and_(
                        TeamSeasonStats.team   == team,
                        TeamSeasonStats.league == league,
                        TeamSeasonStats.season == preferred,
                    )).first()
                )
                if row is not None:
                    break
        if row is None:
            row = query.first()

        # Also check for a duplicate row under a provider-variant name
        # (e.g. Understat "Wolverhampton Wanderers" vs API-Football "Wolves")
        # so neither xG nor standings columns are missed.
        target_key = _dedup_team_key(team)
        # Variant-name rows are scoped to the season that was just
        # resolved (or to the requested season when nothing was found).
        season_filter = (
            row.season if row is not None else season
        )
        league_query = db.query(TeamSeasonStats).filter(
            TeamSeasonStats.league == league
        )
        if season_filter:
            league_query = league_query.filter(
                TeamSeasonStats.season == season_filter
            )
        else:
            league_query = league_query.order_by(
                TeamSeasonStats.season.desc()
            )
        matching_rows = [
            r for r in league_query.all()
            if _dedup_team_key(r.team) == target_key
        ]

        # No exact-name row and no season to pin the variant rows to:
        # keep only the newest season the variant name appears in, so
        # stats from two different seasons are never merged into one
        # (per-game) stat line.
        if row is None and season_filter is None and matching_rows:
            newest = max(r.season for r in matching_rows if r.season)
            matching_rows = [
                r for r in matching_rows if r.season == newest
            ]

        if row is None and not matching_rows:
            return None

        stats = _row_to_stats_dict(row if row is not None else matching_rows[0])
        for extra in matching_rows:
            if row is not None and extra.id == row.id:
                continue
            stats = _merge_stat_dicts(stats, _row_to_stats_dict(extra))
        stats["games"] = games_played(stats)
        # Early in the season, lean on last season so one thin row does not
        # carry the whole prediction (see _blend_with_previous_season).
        stats = _blend_with_previous_season(db, team, league, stats)
        return stats

    except Exception as exc:
        print(f"[formula] ERROR loading team stats "
              f"{team} {league}: {exc}")
        return None
    finally:
        _close_session(db)


def league_seasons(league: str) -> list[str]:
    """Every season present in team_season_stats for a league, newest first."""
    cache_key = ("seasons", league)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    db = _new_session()
    if db is None:
        return []
    try:
        rows = (
            db.query(TeamSeasonStats.season)
            .filter(TeamSeasonStats.league == league)
            .distinct()
            .all()
        )
        seasons = sorted({row[0] for row in rows if row[0]}, reverse=True)
        if seasons:
            _cache_set(cache_key, seasons)
        return seasons
    except Exception as exc:
        print(f"[formula] WARNING: could not list seasons for "
              f"{league}: {exc}")
        return []
    finally:
        _close_session(db)


def resolve_season(league: str, season: str | None = None) -> str | None:
    """
    Return the season to predict with.

    An explicit ``season`` is used when the league actually has rows for it.
    Otherwise — and when no season is given — the most recent season in
    team_season_stats is used, so both teams, the league averages and the
    calibration baselines all describe the same campaign. Predicting a
    season the league has no data for would silently leave the baselines
    neutral and the ratings uncalibrated.
    """
    seasons = league_seasons(league)

    if season:
        if not seasons or season in seasons:
            return season
        print(f"[formula] NOTE: no team_season_stats rows for {league} "
              f"{season} — using {seasons[0]}")
        return seasons[0]

    return seasons[0] if seasons else None


# Columns that make a row a real team season rather than an empty shell (a
# standings-only API-Football row, a FotMob row whose deep-stats fetch
# failed) or a mis-keyed artefact (a numeric team id, a blank name). A row
# must carry at least one of them to take part in league calibration.
_MEASURED_TEAM_COLUMNS = (
    "sh_xg", "sh_npxg", "sh_shots", "sh_sot", "sh_goals",
    "gk_ga", "gk_saves",
    "ps_prgp", "pos_prgc", "gsc_gca",
    "def_press", "def_tkl_int",
    "misc_crs", "pos_poss", "pos_touch_att3rd",
)

# Most clubs a single top-flight season can hold — the expanded training set
# tops out at 24 (ENG-Championship). A larger count means team_season_stats
# holds rows that are not that league's clubs for that season, so the
# calibration means are computed over teams that do not play in it.
MAX_PLAUSIBLE_LEAGUE_TEAMS = 30


def _is_real_team_row(row) -> bool:
    """
    Whether one ``team_season_stats`` row really describes a club's season.

    Collectors upsert a row as soon as they see a team, so the table also
    holds empty shells and artefacts whose "team" is a numeric id. Counting
    those inflates ``league_teams`` and drags the league means towards the
    neutral defaults, so they are dropped here.
    """
    name = str(getattr(row, "team", "") or "").strip()
    if not name or not any(ch.isalpha() for ch in name):
        return False
    return any(
        getattr(row, column, None) is not None
        for column in _MEASURED_TEAM_COLUMNS
    )


def _league_season_rows(
    league: str,
    season: str | None,
) -> tuple[list, str | None]:
    """
    Every team row for exactly one league and one season.

    Both halves of the scope are applied together, and a missing season is
    resolved from the league rather than dropped: filtering on the league
    alone aggregates every campaign on file plus every club that ever
    played in it, which is how a 20-club league could be reported as
    hundreds of teams.

    Rows are deduplicated by club (:func:`_dedup_team_key`) after the
    unusable ones are dropped, so one club stored under two provider
    spellings is still one team. Returns ``(rows, season_used)``; ``rows``
    is empty when the league has no rows for that season, or when no
    session can be opened.
    """
    if not season:
        season = (league_seasons(league) or [None])[0]
    if not season:
        return [], None

    db = _new_session()
    if db is None:
        return [], season
    try:
        query = db.query(TeamSeasonStats).filter(and_(
            TeamSeasonStats.league == league,
            TeamSeasonStats.season == season,
        ))

        rows: list = []
        seen: set[str] = set()
        for row in query.all():
            # Defence in depth: the SQL filter is the guarantee, this is the
            # check that proves it. A row that somehow slipped through (a
            # look-alike label, a driver that ignored the bind parameters)
            # must never reach league calibration.
            if str(row.league) != league or str(row.season) != season:
                continue
            if not _is_real_team_row(row):
                continue
            key = _dedup_team_key(str(row.team).lower().strip())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

        if len(rows) > MAX_PLAUSIBLE_LEAGUE_TEAMS:
            # Name the first few offenders: an inflated count is always a
            # data problem (another league's clubs, a stale campaign, player
            # rows, ...), and the Render log is the only place it surfaces.
            sample = ", ".join(str(r.team) for r in rows[:10])
            _warn_once(
                f"league_size:{league}:{season}",
                f"[formula] WARNING: {league} {season} holds {len(rows)} "
                f"distinct teams — a top flight has at most "
                f"{MAX_PLAUSIBLE_LEAGUE_TEAMS}, so team_season_stats keeps "
                f"rows that are not that league's clubs. Rating "
                f"normalisation for it is unreliable. First rows: {sample}",
            )
        return rows, season
    except Exception as exc:
        print(f"[formula] ERROR loading {league} {season} rows: {exc}")
        return [], season
    finally:
        _close_session(db)


def load_league_averages(
    league: str,
    season: str | None = None,
) -> dict:
    """
    Compute league-average stats from every team
    in team_season_stats for one league and one season.
    Used to normalise team ratings.

    Every average is per game: rows are converted with :func:`to_per_game`
    first, which is what the fixed denominators in the rating formulas and
    the defaults in ``DEFAULT_LEAGUE_AVERAGES`` assume.

    The league and the season are both mandatory: see
    :func:`_league_season_rows`. A season that the league has no rows for
    yields the neutral defaults, never an average taken across seasons or
    across other leagues.
    """
    cache_key = ("league_avgs", league, season)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    rows, season = _league_season_rows(league, season)

    if not rows:
        return dict(DEFAULT_LEAGUE_AVERAGES, teams=0, season=season)

    per_game = [to_per_game(_row_to_stats_dict(row)) for row in rows]

    xg_vals    = [p["xg"]    for p in per_game if _safe(p.get("xg"))    > 0]
    xga_vals   = [p["ga"]    for p in per_game if _safe(p.get("ga"))    > 0]
    shots_vals = [p["shots"] for p in per_game if _safe(p.get("shots")) > 0]
    sot_vals   = [p["sot"]   for p in per_game if _safe(p.get("sot"))   > 0]
    saves_vals = [p["saves"] for p in per_game if _safe(p.get("saves")) > 0]
    gca90_vals = [p["gca90"] for p in per_game if _safe(p.get("gca90")) > 0]
    prgc_vals  = [p["prgc"]  for p in per_game if _safe(p.get("prgc"))  > 0]
    prgp_vals  = [p["prgp"]  for p in per_game if _safe(p.get("prgp"))  > 0]
    att3_vals  = [
        p["touch_att3rd"] for p in per_game
        if _safe(p.get("touch_att3rd")) > 0
    ]
    tklint_vals = [
        p["tkl_int"] for p in per_game if _safe(p.get("tkl_int")) > 0
    ]

    def avg(lst, default=LEAGUE_AVG_XG):
        return sum(lst) / len(lst) if lst else default

    averages = {
        "avg_xg":    avg(xg_vals),
        "avg_xga":   avg(xga_vals),
        "avg_shots": avg(shots_vals),
        "avg_sot":   avg(sot_vals),
        "avg_saves": avg(saves_vals),
        "avg_gca90": avg(gca90_vals, DEFAULT_GCA90),
        "avg_prgc":    avg(prgc_vals,   DEFAULT_LEAGUE_AVERAGES["avg_prgc"]),
        "avg_prgp":    avg(prgp_vals,   DEFAULT_LEAGUE_AVERAGES["avg_prgp"]),
        "avg_att3rd":  avg(att3_vals,   DEFAULT_LEAGUE_AVERAGES["avg_att3rd"]),
        "avg_tkl_int": avg(tklint_vals, DEFAULT_LEAGUE_AVERAGES["avg_tkl_int"]),
        "teams":     len(rows),
        "season":    season,
    }
    _cache_set(cache_key, averages)
    return averages


def _empty_squad_summary() -> dict:
    """Squad summary used when a team has no player rows."""
    return {
        "top_scorer_goals":  0,
        "top_scorer_xg":     0,
        "top_assister_ast":  0,
        "squad_depth_goals": 0,
        "squad_xg_total":    0,
    }


def _summarise_squad(players: list[dict]) -> dict:
    """
    Reduce a squad's player rows to the five numbers the attack rating uses.

    The five most prolific players stand in for squad quality, matching the
    ``ORDER BY goals DESC LIMIT 5`` intent of the original query but sorting
    in Python so that NULL goal counts cannot win the ordering (Postgres
    sorts NULLs first for DESC).
    """
    if not players:
        return _empty_squad_summary()

    ordered = sorted(
        players,
        key=lambda p: _safe(p.get("goals")),
        reverse=True,
    )[:5]

    top_scorer = ordered[0]

    return {
        "top_scorer_goals":  _safe(top_scorer.get("goals")),
        "top_scorer_xg":     _safe(top_scorer.get("xg")),
        "top_assister_ast":  max(_safe(p.get("assists")) for p in ordered),
        "squad_depth_goals": sum(_safe(p.get("goals")) for p in ordered),
        "squad_xg_total":    sum(_safe(p.get("xg"))    for p in ordered),
    }


def _player_to_dict(row) -> dict:
    """Map one PlayerSeasonStats row onto a flat dict."""
    return {
        "player":   row.player,
        "team":     row.team,
        "league":   row.league,
        "season":   row.season,
        "position": row.position,
        "minutes":  row.minutes,
        "mp":       row.mp,
        "goals":    row.goals,
        "assists":  row.assists,
        "xg":       row.xg,
        "xag":      row.xag,
        "npxg":     row.npxg,
    }


def load_squad_players(
    team: str,
    league: str,
    season: str | None = None,
) -> list[dict]:
    """
    Load every player row for one team.

    Used both for the squad-quality summary and for matching injured
    players to the output they were contributing.
    """
    db = _new_session()
    if db is None:
        return []
    try:
        query = (
            db.query(PlayerSeasonStats)
            .filter(
                PlayerSeasonStats.team == team,
                PlayerSeasonStats.league == league,
                PlayerSeasonStats.goals.isnot(None),
            )
        )
        if season:
            query = query.filter(PlayerSeasonStats.season == season)
        rows = (
            query
            .order_by(
                PlayerSeasonStats.goals.desc().nulls_last()
            )
            .all()
        )
        return [_player_to_dict(row) for row in rows]
    except Exception as exc:
        _warn_once(
            f"players:{team}:{league}",
            f"[formula] WARNING: could not load players for "
            f"{team} ({league}): {exc}",
        )
        return []
    finally:
        _close_session(db)


def load_key_players(
    team: str,
    league: str,
    season: str | None = None,
) -> dict:
    """
    Load top players for a team.
    Returns top scorer and top assister.
    """
    db = _new_session()
    if db is None:
        return _empty_squad_summary()
    try:
        query = (
            db.query(PlayerSeasonStats)
            .filter(
                PlayerSeasonStats.team == team,
                PlayerSeasonStats.league == league,
                PlayerSeasonStats.goals.isnot(None),
            )
        )
        if season:
            query = query.filter(PlayerSeasonStats.season == season)
        rows = (
            query
            .order_by(
                PlayerSeasonStats.goals.desc().nulls_last()
            )
            .limit(5)
            .all()
        )
        return _summarise_squad([_player_to_dict(row) for row in rows])
    except Exception as exc:
        print(f"[formula] WARNING: could not load "
              f"players for {team}: {exc}")
        return _empty_squad_summary()
    finally:
        _close_session(db)


def load_league_squads(
    league: str,
    season: str | None = None,
) -> dict[str, list[dict]]:
    """
    Load every player row in a league in one query, grouped by team.

    The league calibration pass needs a squad summary per team; doing that
    one team at a time would mean 20-100 extra round trips.

    Scoped to one season like the team rows it accompanies: a missing season
    is resolved from the league, so a squad is never merged across campaigns
    (a player who changed club would otherwise strengthen two of them).
    """
    if not season:
        season = (league_seasons(league) or [None])[0]
    if not season:
        return {}

    db = _new_session()
    if db is None:
        return {}
    try:
        query = (
            db.query(PlayerSeasonStats)
            .filter(
                PlayerSeasonStats.league == league,
                PlayerSeasonStats.season == season,
                PlayerSeasonStats.goals.isnot(None),
            )
        )
        rows = (
            query
            .order_by(
                PlayerSeasonStats.goals.desc().nulls_last()
            )
            .all()
        )

        squads: dict[str, list[dict]] = {}
        for row in rows:
            squads.setdefault(row.team, []).append(_player_to_dict(row))
        return squads
    except Exception as exc:
        _warn_once(
            f"league_squads:{league}:{season}",
            f"[formula] WARNING: could not load league squads for "
            f"{league} {season}: {exc}",
        )
        return {}
    finally:
        _close_session(db)


# ═══════════════════════════════════════════
# PER-GAME NORMALISATION
# ═══════════════════════════════════════════


def games_played(
    stats: dict | None,
    league_avgs: dict | None = None,
) -> float | None:
    """
    Best estimate of how many games a stat row covers.

    Returns ``None`` when the row already stores per-game averages rather
    than season totals, which is what the StatsBomb aggregator writes.
    Detection order:

      1. total / per-90 pairs (shots, SoT, GA, SCA, GCA). These columns
         measure the row's own scale, so they win over W-D-L.
      2. W + D + L, when the total is larger than a five-game form window.
      3. total / league-average-per-game, which needs no rate column, used
         before a five-game W-D-L total that may just be recent form.
      4. W + D + L as the last resort.

    The keeper W-D-L columns are *last-five form* in the API-Football rows
    and a season record in the FBref / Understat rows, so trusting W + D + L
    first turns a five-game form window into a "five-game season" and
    divides every season total by five — the ``home_ga_per_game: 6.8``
    report (34 goals against, last-five record 3W-1D-1L).

    An estimate below MIN_GAMES is still returned when the row is clearly
    multi-game, so the caller can reject it as insufficient data instead of
    silently treating two games' totals as a per-game rate.
    """
    if not stats:
        return None

    wins   = _safe(stats.get("form_wins"))
    draws  = _safe(stats.get("form_draws"))
    losses = _safe(stats.get("form_losses"))
    total  = wins + draws + losses
    form_window = 0 < total <= FORM_WINDOW

    # 1. Total / per-90 pairs. A ratio near 1.0 is the StatsBomb per-game
    #    signature (both columns hold the same number), not a one-game
    #    season, so it is remembered instead of returned.
    plausible: list[float] = []
    short: list[float] = []
    ga_pair: float | None = None
    already_per_game = False

    for total_key, rate_key in GAMES_RATE_PAIRS:
        season_total = _safe(stats.get(total_key))
        per_game = _safe(stats.get(rate_key))
        if season_total <= 0 or per_game <= 0:
            continue
        estimate = season_total / per_game
        if estimate <= RATE_SCALE_TOLERANCE:
            already_per_game = True
            continue
        if estimate > MAX_GAMES_ESTIMATE:
            # Two columns of different scales were paired up; not a count.
            continue
        if estimate >= MIN_GAMES:
            plausible.append(estimate)
            if (total_key, rate_key) == ("ga", "ga90"):
                ga_pair = estimate
        else:
            short.append(estimate)

    if plausible:
        # GA90 is written from the same row as GA by every collector, while
        # sh_per90 can hold a proxy (the API-Football rows store a
        # goals-per-game-derived stand-in), so the keeper pair is the most
        # trustworthy single estimate. Otherwise the median of the pairs.
        if ga_pair is not None:
            estimate = ga_pair
        else:
            plausible.sort()
            estimate = plausible[len(plausible) // 2]
        return float(round(estimate))

    if already_per_game:
        return None

    if short:
        short.sort()
        return float(round(short[len(short) // 2]))

    # 2. A season record (more than a five-game form window) is exact.
    if total >= MIN_GAMES and not form_window:
        return float(total)

    # 3. League-average-per-game: the only signal left for a season-total
    #    row with no rate column at all. Estimates below MIN_GAMES are not
    #    returned — a per-game row measured against a per-game average
    #    lands there too, and treating it as "not enough games" would
    #    reject a perfectly good row.
    if league_avgs and total < MIN_GAMES:
        for total_key, avg_key in (
            ("shots", "avg_shots"),
            ("sot",   "avg_sot"),
            ("xg",    "avg_xg"),
        ):
            season_total = _safe(stats.get(total_key))
            average = _safe(league_avgs.get(avg_key))
            if season_total <= 0 or average <= 0:
                continue
            estimate = season_total / average
            if estimate > MAX_GAMES_ESTIMATE:
                continue
            if estimate >= MIN_GAMES:
                return float(round(estimate))

    # 4. Whatever W + D + L says, including the ambiguous form window.
    if total > 0:
        return float(total)
    return None


def to_per_game(
    stats: dict | None,
    league_avgs: dict | None = None,
) -> dict:
    """
    Return a copy of ``stats`` with every season total converted to a
    per-game rate.

    Rows that already hold per-game values (StatsBomb aggregates) are
    returned unchanged, so the whole engine can compare like with like.

    A total that has a stored per-90 column (goals against, shots, shots on
    target — see ``RATE_KEYS``) is taken from that column instead of being
    re-derived from the total, because the rate is computed by the collector
    from the same row and is already per game. That makes the reported
    per-game figures independent of the games-played estimate: reading
    ``ga90`` gives Arsenal 0.76 whether the row says 38 games or holds a
    five-game form window in W-D-L.
    """
    if not stats:
        return {}

    normalised = dict(stats)
    games = games_played(stats, league_avgs)
    normalised["games"] = games

    if games and games >= MIN_GAMES:
        for key in PER_GAME_KEYS:
            value = normalised.get(key)
            if value is None:
                continue
            normalised[key] = _safe(value) / games

    # Stored rates are the authoritative per-game values.
    for total_key, rate_key in RATE_KEYS.items():
        rate = _safe(normalised.get(rate_key))
        if rate > 0:
            normalised[total_key] = rate

    return normalised


# ═══════════════════════════════════════════
# RATING COMPUTATION
# ═══════════════════════════════════════════


def compute_attack_rating(
    stats: dict,
    league_avgs: dict,
    players: dict,
) -> float:
    """
    Compute a team's attacking strength as a
    multiplier relative to league average.

    Returns value around 1.0:
      > 1.0 = stronger than average
      < 1.0 = weaker than average
    """
    avg_xg    = league_avgs.get("avg_xg", LEAGUE_AVG_XG)
    avg_sot   = league_avgs.get("avg_sot", 4.5)

    if avg_xg <= 0:
        avg_xg = LEAGUE_AVG_XG

    # xG component — normalised to league average
    xg = _safe(stats.get("xg"), avg_xg)
    xg_rating = xg / avg_xg

    # Shots on target component
    sot = _safe(stats.get("sot"), avg_sot)
    sot_rating = sot / avg_sot if avg_sot > 0 else 1.0

    # Shot quality — goals per shot.
    # A stored value outside the goals-per-shot range is not measuring shot
    # quality (the API-Football rows write a placeholder derived from goals
    # and games played), so the league-typical value is used instead of a
    # ratio that would drag the whole attack rating down.
    g_per_sh = _safe(stats.get("g_per_sh"), DEFAULT_G_PER_SHOT)
    shot_min, shot_max = GOALS_PER_SHOT_RANGE
    if not shot_min <= g_per_sh <= shot_max:
        g_per_sh = DEFAULT_G_PER_SHOT
    shot_quality = g_per_sh / DEFAULT_G_PER_SHOT

    # Progressive actions — proxy for creativity.
    # A missing column falls back to the league's own per-game average, so a
    # partially scraped row (the StatsBomb and Understat collectors fill only
    # a handful of columns) reads as an average team rather than an
    # outstanding one.
    prgc  = _safe(stats.get("prgc"), _safe(
        league_avgs.get("avg_prgc"), DEFAULT_LEAGUE_AVERAGES["avg_prgc"]))
    prgp  = _safe(stats.get("prgp"), _safe(
        league_avgs.get("avg_prgp"), DEFAULT_LEAGUE_AVERAGES["avg_prgp"]))
    progressive = (prgc + prgp) / 150.0

    # Pressing in opponent half — attacking intensity
    press_att3rd = _safe(stats.get("touch_att3rd"), _safe(
        league_avgs.get("avg_att3rd"),
        DEFAULT_LEAGUE_AVERAGES["avg_att3rd"])) / 200.0

    # GCA — direct measure of chance creation.
    # Normalised against the league's own GCA/90 rather than a fixed 0.3:
    # FBref reports team-level GCA90 at ~4-7 per 90 (0.3 is a per-player
    # figure), so a hardcoded denominator would drive every team into the
    # 3.0 clamp below and leave the attack rating with no signal at all.
    avg_gca90 = _safe(league_avgs.get("avg_gca90"), 0.0)
    if avg_gca90 <= 0:
        avg_gca90 = DEFAULT_GCA90
    gca90 = _safe(stats.get("gca90"), avg_gca90)
    gca_rating = gca90 / avg_gca90 if gca90 > 0 else 1.0

    # Player quality boost
    top_goals = _safe(players.get("top_scorer_goals"), 10)
    squad_xg  = _safe(players.get("squad_xg_total"), 30)
    player_boost = min(
        1.0 + (top_goals / 100) + (squad_xg / 300), 1.3)

    # Weighted combination
    raw = (
        ATTACK_WEIGHTS["xg_per_game"]  * xg_rating +
        ATTACK_WEIGHTS["sot_per_game"] * sot_rating +
        ATTACK_WEIGHTS["g_per_shot"]   * shot_quality +
        ATTACK_WEIGHTS["progressive"]  * progressive +
        ATTACK_WEIGHTS["press_succ"]   * press_att3rd
    ) * player_boost

    # Blend with GCA
    final = 0.7 * raw + 0.3 * gca_rating

    # Clamp to reasonable range
    return float(np.clip(final, 0.3, 3.0))


def compute_defence_rating(
    stats: dict,
    league_avgs: dict,
) -> float:
    """
    Compute a team's defensive strength as a
    multiplier. LOWER = better defence.

    Returns value around 1.0:
      < 1.0 = better than average defence
      > 1.0 = worse than average defence

    ``stats`` is expected to be per-game normalised, so ``ga`` is goals
    against *per game*: :func:`to_per_game` fills it from the stored GA90
    column (0.76 for 29 conceded in 38 games), never from GA / 38 alone.
    """
    avg_xga = league_avgs.get("avg_xga", LEAGUE_AVG_XG)

    if avg_xga <= 0:
        avg_xga = LEAGUE_AVG_XG

    # xGA — how much the team concedes, per game
    xga = _safe(stats.get("ga"), avg_xga)
    xga_rating = xga / avg_xga  # higher = worse

    # GK save % — higher save % = better defence
    save_pct = _safe(stats.get("save_pct"), 70.0)
    save_rating = (100 - save_pct) / 30.0

    # GK PSxG+/- — positive means GK saves more than expected
    psxg_diff = _safe(stats.get("psxg_diff"), 0.0)
    gk_bonus = 1.0 - (psxg_diff / 20.0)
    gk_bonus = float(np.clip(gk_bonus, 0.7, 1.3))

    # Clean sheet % — direct measure of shutouts
    cs_pct = _safe(stats.get("cs_pct"), 25.0)
    cs_rating = (100 - cs_pct) / 75.0

    # Pressing — teams that press well concede fewer
    press_pct = _safe(stats.get("press_pct"), 30.0)
    press_rating = (100 - press_pct) / 70.0

    # Blocks + interceptions — defensive actions (league-average fallback,
    # as in compute_attack_rating)
    tkl_int = _safe(stats.get("tkl_int"), _safe(
        league_avgs.get("avg_tkl_int"),
        DEFAULT_LEAGUE_AVERAGES["avg_tkl_int"]))
    defensive_actions = 50.0 / max(tkl_int, 1.0)

    # Weighted combination
    raw = (
        DEFENCE_WEIGHTS["xga_per_game"]   * xga_rating +
        DEFENCE_WEIGHTS["saves_per_game"] * save_rating +
        DEFENCE_WEIGHTS["clean_sheet_pct"]* cs_rating +
        DEFENCE_WEIGHTS["press_allowed"]  * press_rating +
        DEFENCE_WEIGHTS["blocks_per_game"]* defensive_actions
    ) * gk_bonus

    return float(np.clip(raw, 0.3, 3.0))


def compute_form_modifier(stats: dict) -> float:
    """
    Compute a recent form modifier from last 5 results.
    Returns multiplier around 1.0.
    Poor form → below 1.0, good form → above 1.0.
    """
    wins   = _safe(stats.get("form_wins"),   2.0)
    draws  = _safe(stats.get("form_draws"),  1.5)
    losses = _safe(stats.get("form_losses"), 1.5)
    total  = wins + draws + losses

    if total == 0:
        return 1.0

    pts = wins * 3 + draws
    max_pts = total * 3
    form_pct = pts / max_pts if max_pts > 0 else 0.5

    # Convert to multiplier: 0.9 to 1.1
    return float(0.9 + 0.2 * form_pct)


def compute_goalkeeper_modifier(stats: dict) -> float:
    """
    Multiplier on the goals this team concedes from keeper quality alone.

    ``compute_defence_rating`` reads PSxG+/- on a season scale, which the
    per-game normalisation in this module cancels out, so shot-stopping
    value is re-applied here from PSxG+/- per 90 with a deliberately small
    sensitivity: < 1.0 for a keeper who saves more than he faces.
    """
    psxg_per90 = stats.get("psxg_diff90")
    if psxg_per90 is None:
        # Fall back to the per-game PSxG+/- when the per-90 column is empty.
        psxg_per90 = stats.get("psxg_diff")

    value = _safe(psxg_per90, 0.0)
    if value == 0.0:
        return 1.0

    modifier = 1.0 - GK_SENSITIVITY * (value / GK_TYPICAL_PSXG90)
    return float(np.clip(modifier, GK_MIN_MODIFIER, GK_MAX_MODIFIER))


# ═══════════════════════════════════════════
# LEAGUE CALIBRATION
# ═══════════════════════════════════════════


def relative_rating(value: float, mean: float) -> float:
    """
    Express a rating relative to the league mean.

    The attack and defence formulas are weighted sums of ratios, so they
    land *near* 1.0 but not *on* it — their fixed denominators (150
    progressive actions, 50 tackles + interceptions, ...) leave an offset
    that is the same for every team. Dividing by the league mean removes
    that offset, which keeps an average team exactly average and stops the
    offset from being multiplied into every expected-goals figure.
    """
    if not mean or mean <= 0:
        mean = 1.0
    return float(np.clip(value / mean, MIN_RELATIVE, MAX_RELATIVE))


def _raw_corner_expectation(stats: dict) -> float:
    """
    One team's corner expectation from the cross / attacking-third /
    carries-into-the-box formula, clipped to a realistic per-team range.
    """
    crs   = _safe(stats.get("misc_crs"),     20.0)
    att3  = _safe(stats.get("touch_att3rd"), 200.0)
    cpa   = _safe(stats.get("cpa"),          10.0)

    expected = (
        (crs  / 60)  * 5.0 +
        (att3 / 600) * 3.0 +
        (cpa  / 30)  * 2.0
    )
    return float(np.clip(expected, 2.0, 9.0))


def compute_league_baselines(
    league: str,
    season: str | None = None,
) -> dict:
    """
    League-wide reference values used to calibrate one fixture.

    Runs the exact same rating formulas over every team in the league and
    reports their means, plus the league's real goals-per-game and the
    scale factor that turns the raw cross / attacking-third corner formula
    into corners. ``corner_scale`` is a diagnostic only now:
    :func:`compute_corner_markets` estimates corners directly and is not
    rescaled by it.

    The league *and* the season are both part of the scope: the means are
    taken over the clubs of that one competition in that one campaign (see
    :func:`_league_season_rows`), so a team's relative rating is always
    measured against its own league.

    Returns neutral baselines (all multipliers 1.0) when the league has no
    rows, so the engine still predicts instead of failing.
    """
    cache_key = ("baselines", league, season)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Resolve the season first so the baselines, the league averages and
    # the squads below are all read from the same campaign.
    rows, season = _league_season_rows(league, season)
    league_avgs = load_league_averages(league, season)
    fallback = {
        "mean_attack":   1.0,
        "mean_defence":  1.0,
        "mean_corners":  5.0,
        "corner_scale":  1.0,
        "goals_per_game": float(
            np.clip(
                _safe(league_avgs.get("avg_xg"), LEAGUE_AVG_XG),
                MIN_BASE_GOALS,
                MAX_BASE_GOALS,
            )
        ),
        "teams":         0,
        "season":        season,
    }

    # Snapshot to plain dicts: the ORM instances would be detached once the
    # session inside _league_season_rows closes.
    team_rows = [(_row_to_stats_dict(row), row.team) for row in rows]

    if not team_rows:
        _warn_once(
            f"baselines:{league}:{season}",
            f"[formula] WARNING: no team_season_stats rows for "
            f"{league} {season} — using neutral baselines",
        )
        return fallback

    squads = load_league_squads(league, season)

    attacks: list[float] = []
    defences: list[float] = []
    corners: list[float] = []

    for raw_stats, team in team_rows:
        per_game = to_per_game(raw_stats, league_avgs)
        players = _summarise_squad(squads.get(team, []))

        attacks.append(
            compute_attack_rating(per_game, league_avgs, players))
        defences.append(
            compute_defence_rating(per_game, league_avgs))
        corners.append(_raw_corner_expectation(per_game))

    mean_corners = float(np.mean(corners)) if corners else 5.0
    corner_scale = CORNER_LEAGUE_TOTAL / max(2.0 * mean_corners, 1e-6)

    baselines = {
        "mean_attack":   float(np.mean(attacks))  if attacks  else 1.0,
        "mean_defence":  float(np.mean(defences)) if defences else 1.0,
        "mean_corners":  mean_corners,
        "corner_scale":  float(
            np.clip(corner_scale, CORNER_MIN_SCALE, CORNER_MAX_SCALE)),
        "goals_per_game": float(
            np.clip(
                _safe(league_avgs.get("avg_xg"), LEAGUE_AVG_XG),
                MIN_BASE_GOALS,
                MAX_BASE_GOALS,
            )
        ),
        "teams":         len(team_rows),
        "season":        season,
    }

    if baselines["mean_attack"] <= 0:
        baselines["mean_attack"] = 1.0
    if baselines["mean_defence"] <= 0:
        baselines["mean_defence"] = 1.0

    _cache_set(cache_key, baselines)
    return baselines


# ═══════════════════════════════════════════
# INJURY ADJUSTMENTS
# ═══════════════════════════════════════════


def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    cleaned = "".join(
        ch if ch.isalnum() or ch.isspace() else " "
        for ch in str(name or "")
    )
    return " ".join(cleaned.lower().split())


def _name_tokens(name: str) -> set[str]:
    """Name tokens long enough to identify a person or a club."""
    return {tok for tok in _normalise_name(name).split() if len(tok) > 2}


def _names_match(injured: str, stored: str) -> bool:
    """Whether two player names are considered the same person."""
    return _name_match_score(injured, stored) > 0


def _name_match_score(injured: str, stored: str) -> int:
    """
    Confidence that an injury-report name and a stored player name are the
    same person: 0 = no match, higher = more confident.

    Tolerates the usual differences between data providers (full name vs
    surname, initials, accents, spelling of compound names) without
    matching two *different* players who share a surname — Marcus and
    Khephren Thuram both play for the same club, so a shared surname alone
    is not enough. The best-scoring candidate wins, so an exact match is
    never stolen by a looser one.
    """
    injured_norm = _normalise_name(injured)
    stored_norm = _normalise_name(stored)
    if not injured_norm or not stored_norm:
        return 0

    if injured_norm == stored_norm:
        return NAME_SCORE_EXACT

    # Whole-token containment: "Salah" inside "Mohamed Salah".
    padded_injured = f" {injured_norm} "
    padded_stored = f" {stored_norm} "
    if len(injured_norm) >= 4 and padded_injured in padded_stored:
        return NAME_SCORE_CONTAINED
    if len(stored_norm) >= 4 and padded_stored in padded_injured:
        return NAME_SCORE_CONTAINED

    injured_tokens = injured_norm.split()
    stored_tokens = stored_norm.split()
    if len(injured_tokens) < 2 or len(stored_tokens) < 2:
        return 0

    # Same surname plus a compatible given name (shared initial, so
    # "M. Salah" matches "Mohamed Salah" but "Andre Ayew" does not match
    # "Jordan Ayew").
    if injured_tokens[-1] != stored_tokens[-1]:
        return 0
    if injured_tokens[0][0] == stored_tokens[0][0]:
        return NAME_SCORE_SURNAME
    return 0


def _iter_injury_names(value: Any) -> list[str]:
    """
    Pull player names out of one team's injury entry.

    Accepts a single name, a list of names, a list of dicts with a
    name/player/key, or a dict keyed by player name.
    """
    names: list[str] = []

    if value is None:
        return names

    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else names

    if isinstance(value, dict):
        if "players" in value or "injuries" in value or "out" in value:
            for key in ("players", "injuries", "out"):
                names.extend(_iter_injury_names(value.get(key)))
            return names
        for key, entry in value.items():
            names.append(str(key))
            if isinstance(entry, dict):
                names.extend(_iter_injury_names(entry))
        return names

    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, dict):
                for key in ("name", "player", "player_name", "name_"):
                    if item.get(key):
                        names.append(str(item[key]))
                        break
            else:
                names.extend(_iter_injury_names(item))
        return names

    return names


def extract_team_injuries(
    injuries: dict | None,
    team: str,
    role: str | None = None,
) -> list[str]:
    """
    Return the list of missing players for one team from an injuries dict.

    Supported shapes::

        {"Arsenal": ["Bukayo Saka"], "Chelsea": ["Reece James"]}
        {"home": ["Bukayo Saka"], "away": ["Reece James"]}
        {"Arsenal": {"players": [{"name": "Bukayo Saka", "reason": "..."}]}}

    Keys are matched case-insensitively, then by the ``home`` / ``away``
    role, then by partial name overlap, so a caller can pass whatever the
    upstream API-Football injury feed provides.
    """
    if not injuries or not isinstance(injuries, dict):
        return []

    team_norm = _normalise_name(team)
    team_tokens = _name_tokens(team)

    # 1. Exact (case-insensitive) team key.
    for key, value in injuries.items():
        if _normalise_name(key) == team_norm:
            return sorted(set(_iter_injury_names(value)))

    # 2. home / away role key.
    if role:
        for key, value in injuries.items():
            if _normalise_name(key) == role.lower():
                return sorted(set(_iter_injury_names(value)))

    # 3. Partial team-name overlap ("Tottenham" vs "Tottenham Hotspur").
    for key, value in injuries.items():
        key_tokens = _name_tokens(key)
        if not key_tokens or not team_tokens:
            continue
        if key_tokens <= team_tokens or team_tokens <= key_tokens:
            return sorted(set(_iter_injury_names(value)))

    return []


def _position_groups(position: str | None) -> tuple[bool, bool]:
    """Return (is_defensive, is_midfield) for an FBref position string."""
    tokens = {
        tok.strip().upper()
        for tok in str(position or "").replace("/", ",").split(",")
        if tok.strip()
    }
    is_defensive = bool(tokens & set(DEFENSIVE_POSITIONS))
    is_midfield = bool(tokens & set(MIDFIELD_POSITIONS))
    return is_defensive, is_midfield


def _defensive_weight(player: dict) -> float:
    """How much one player's minutes protect his own goal."""
    is_defensive, is_midfield = _position_groups(player.get("position"))
    if is_defensive:
        return 1.0
    if is_midfield:
        return INJURY_MIDFIELD_WEIGHT
    return 0.0


def compute_injury_adjustments(
    team: str,
    league: str,
    injuries: dict | None = None,
    role: str | None = None,
    season: str | None = None,
    squad: list[dict] | None = None,
) -> dict:
    """
    Turn an injury list into two multipliers for one team.

      attack  — < 1.0, scaled by the share of the squad's xG + xAG the
                missing players were producing
      defence — > 1.0 (this team concedes more), scaled by the share of
                defensive minutes lost

    Players are matched against player_season_stats so a missing star costs
    more than a missing squad player. Names that cannot be matched still
    cost a flat penalty, and a team with no player rows at all falls back
    to a purely count-based adjustment.
    """
    result = {
        "attack":              1.0,
        "defence":             1.0,
        "missing":             [],
        "matched":             [],
        "unmatched":           [],
        "attack_share_lost":   0.0,
        "defence_share_lost":  0.0,
    }

    missing = extract_team_injuries(injuries, team, role)
    result["missing"] = missing
    if not missing:
        return result

    players = squad if squad is not None else load_squad_players(
        team, league, season)

    if not players:
        count = min(len(missing), INJURY_MAX_UNKNOWN)
        result["unmatched"] = list(missing)
        result["attack"] = float(np.clip(
            1.0 - count * INJURY_UNKNOWN_ATTACK_PENALTY,
            INJURY_MIN_ATTACK, 1.0))
        result["defence"] = float(np.clip(
            1.0 + count * INJURY_UNKNOWN_DEFENCE_PENALTY,
            1.0, INJURY_MAX_DEFENCE))
        return result

    total_creativity = sum(
        _safe(p.get("xg")) + _safe(p.get("xag")) for p in players)
    total_output = sum(
        _safe(p.get("goals")) + _safe(p.get("assists")) for p in players)
    total_defensive = sum(
        _safe(p.get("minutes")) * _defensive_weight(p) for p in players)

    lost_creativity = 0.0
    lost_output = 0.0
    lost_defensive = 0.0
    used_ids: set[int] = set()
    matched: list[str] = []
    unmatched: list[str] = []

    for name in missing:
        hit = None
        best_score = 0
        for player in players:
            if id(player) in used_ids:
                continue
            score = _name_match_score(name, player.get("player", ""))
            if score > best_score:
                hit, best_score = player, score
        if hit is None:
            unmatched.append(name)
            continue

        used_ids.add(id(hit))
        matched.append(hit.get("player") or name)
        lost_creativity += _safe(hit.get("xg")) + _safe(hit.get("xag"))
        lost_output += _safe(hit.get("goals")) + _safe(hit.get("assists"))
        lost_defensive += (
            _safe(hit.get("minutes")) * _defensive_weight(hit))

    attack_share = 0.0
    if total_creativity > 0:
        attack_share = lost_creativity / total_creativity
    elif total_output > 0:
        attack_share = lost_output / total_output

    defence_share = (
        lost_defensive / total_defensive if total_defensive > 0 else 0.0)

    # Unknown names still cost something.
    unknown = min(len(unmatched), INJURY_MAX_UNKNOWN)
    attack_modifier = (
        1.0
        - INJURY_ATTACK_SENSITIVITY * min(attack_share, 1.0)
        - unknown * INJURY_UNKNOWN_ATTACK_PENALTY
    )
    defence_modifier = (
        1.0
        + INJURY_DEFENCE_SENSITIVITY * min(defence_share, 1.0)
        + unknown * INJURY_UNKNOWN_DEFENCE_PENALTY
    )

    result.update({
        "matched":            matched,
        "unmatched":          unmatched,
        "attack_share_lost":  min(attack_share, 1.0),
        "defence_share_lost": min(defence_share, 1.0),
        "attack":  float(np.clip(
            attack_modifier, INJURY_MIN_ATTACK, 1.0)),
        "defence": float(np.clip(
            defence_modifier, 1.0, INJURY_MAX_DEFENCE)),
    })
    return result


# ═══════════════════════════════════════════
# HEAD TO HEAD (statsbomb_match_stats)
# ═══════════════════════════════════════════


def load_h2h_adjustment(
    home_team: str,
    away_team: str,
    limit: int = H2H_MAX_MATCHES,
) -> dict:
    """
    Small tilt from recent meetings stored in statsbomb_match_stats.

    Uses xG rather than goals, so a lucky 3-0 does not outweigh being
    outplayed. The effect is deliberately capped at ±H2H_WEIGHT: head to
    head is the weakest of the signals available here, and this table may
    not exist yet, in which case the adjustment is neutral.
    """
    neutral = {
        "home": 1.0, "away": 1.0, "meetings": 0,
        "home_xg": 0.0, "away_xg": 0.0,
    }

    home_norm = _normalise_name(home_team)
    away_norm = _normalise_name(away_team)
    if not home_norm or not away_norm or home_norm == away_norm:
        return neutral

    db = _new_session()
    if db is None:
        return neutral
    try:
        rows = db.execute(
            text(
                f"""
                SELECT home_team, away_team, home_xg, away_xg, date
                FROM {H2H_TABLE}
                WHERE lower(home_team) IN (:team_a, :team_b)
                  AND lower(away_team) IN (:team_a, :team_b)
                ORDER BY date DESC
                LIMIT :row_limit
                """
            ),
            {
                "team_a": home_norm,
                "team_b": away_norm,
                "row_limit": int(limit),
            },
        ).fetchall()
    except Exception as exc:
        _warn_once(
            "h2h_table",
            f"[formula] NOTE: head-to-head unavailable "
            f"({H2H_TABLE}): {exc}",
        )
        return neutral
    finally:
        _close_session(db)

    home_xg = 0.0
    away_xg = 0.0
    meetings = 0

    for row in rows:
        stored_home = _normalise_name(row[0] or "")
        stored_away = _normalise_name(row[1] or "")
        xg_home = _safe(row[2])
        xg_away = _safe(row[3])

        if stored_home == home_norm and stored_away == away_norm:
            home_xg += xg_home
            away_xg += xg_away
        elif stored_home == away_norm and stored_away == home_norm:
            home_xg += xg_away
            away_xg += xg_home
        else:
            continue
        meetings += 1

    if meetings < H2H_MIN_MATCHES:
        return dict(neutral, meetings=meetings)

    avg_home = home_xg / meetings
    avg_away = away_xg / meetings
    total = avg_home + avg_away
    if total <= 0:
        return dict(neutral, meetings=meetings)

    # 0.5 = perfectly even; above it the home side dominated the meetings.
    share = avg_home / total
    tilt = H2H_WEIGHT * (2.0 * share - 1.0)

    return {
        "home": float(np.clip(1.0 + tilt, H2H_MIN_MODIFIER, H2H_MAX_MODIFIER)),
        "away": float(np.clip(1.0 - tilt, H2H_MIN_MODIFIER, H2H_MAX_MODIFIER)),
        "meetings": meetings,
        "home_xg": _num(avg_home, 3),
        "away_xg": _num(avg_away, 3),
    }


# ═══════════════════════════════════════════
# EXPECTED GOALS
# ═══════════════════════════════════════════


def compute_expected_goals(
    base_goals: float,
    home_attack_rel: float,
    home_defence_rel: float,
    away_attack_rel: float,
    away_defence_rel: float,
    home_form: float = 1.0,
    away_form: float = 1.0,
    home_injury: dict | None = None,
    away_injury: dict | None = None,
    home_gk: float = 1.0,
    away_gk: float = 1.0,
    h2h: dict | None = None,
) -> tuple[float, float]:
    """
    Turn ratings into the two Poisson means (lambda) for the match.

    Classic attack-vs-defence construction, with every factor expressed
    relative to the league so the average fixture reproduces the league's
    own scoring rate::

        lambda_home = base * attack_rel(home) * defence_rel(away)
                      * HOME_ADVANTAGE * form * injuries * keeper * h2h

    A team's own attacking injuries and its opponent's defensive injuries
    both raise or lower the same lambda.
    """
    home_injury = home_injury or {}
    away_injury = away_injury or {}
    h2h = h2h or {}

    base = float(np.clip(base_goals, MIN_BASE_GOALS, MAX_BASE_GOALS))

    lambda_home = (
        base
        * home_attack_rel
        * away_defence_rel
        * HOME_ADVANTAGE
        * home_form
        * _safe(home_injury.get("attack"), 1.0)
        * _safe(away_injury.get("defence"), 1.0)
        * away_gk
        * _safe(h2h.get("home"), 1.0)
    )

    lambda_away = (
        base
        * away_attack_rel
        * home_defence_rel
        * AWAY_ADJUSTMENT
        * away_form
        * _safe(away_injury.get("attack"), 1.0)
        * _safe(home_injury.get("defence"), 1.0)
        * home_gk
        * _safe(h2h.get("away"), 1.0)
    )

    lambda_home = float(np.clip(lambda_home, MIN_LAMBDA, MAX_LAMBDA))
    lambda_away = float(np.clip(lambda_away, MIN_LAMBDA, MAX_LAMBDA))
    return lambda_home, lambda_away


# ═══════════════════════════════════════════
# POISSON MARKET COMPUTATION
# ═══════════════════════════════════════════


def compute_scoreline_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Compute P(home scores i, away scores j)
    for all i, j up to max_goals.
    Returns a (max_goals+1) x (max_goals+1) matrix.
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = (
                poisson.pmf(i, lambda_home) *
                poisson.pmf(j, lambda_away)
            )
    return matrix


def compute_markets(
    lambda_home: float,
    lambda_away: float,
) -> dict:
    """
    Compute all prediction markets from
    expected goals (lambda values).
    """
    matrix = compute_scoreline_matrix(
        lambda_home, lambda_away)

    # 1X2
    home_win = float(np.sum(
        np.tril(matrix, k=-1)))
    draw     = float(np.sum(
        np.diag(matrix)))
    away_win = float(np.sum(
        np.triu(matrix, k=1)))

    # Normalise to sum to 1
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw     /= total
        away_win /= total

    # Over/Under goals
    total_lambda = lambda_home + lambda_away
    over_1_5 = float(1 - poisson.cdf(1, total_lambda))
    over_2_5 = float(1 - poisson.cdf(2, total_lambda))
    over_3_5 = float(1 - poisson.cdf(3, total_lambda))

    # BTTS
    btts = float(
        (1 - poisson.cdf(0, lambda_home)) *
        (1 - poisson.cdf(0, lambda_away))
    )

    return {
        "home_win": round(home_win, 4),
        "draw":     round(draw,     4),
        "away_win": round(away_win, 4),
        "over_1_5": round(over_1_5, 4),
        "over_2_5": round(over_2_5, 4),
        "over_3_5": round(over_3_5, 4),
        "btts":     round(btts,     4),
    }


def top_scorelines(
    lambda_home: float,
    lambda_away: float,
    count: int = 5,
) -> list[dict]:
    """Most likely exact scores, best first."""
    matrix = compute_scoreline_matrix(lambda_home, lambda_away)
    flat = [
        {"score": f"{i}-{j}", "probability": _num(matrix[i, j], 4)}
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
    ]
    flat.sort(key=lambda item: item["probability"], reverse=True)
    return flat[:max(count, 0)]


def corner_markets_from_expected(
    expected_home: float,
    expected_away: float,
) -> dict:
    """
    Build the corner markets from two expected-corner figures.

    The shared Poisson builder for the raw cross / attacking-third
    expectation path (:func:`calibrate_corners`).
    :func:`compute_corner_markets` no longer routes through it: that
    estimator is calibrated in corner units and emits its own markets.
    """
    exp_home = float(np.clip(
        expected_home, MIN_EXPECTED_CORNERS, MAX_EXPECTED_CORNERS))
    exp_away = float(np.clip(
        expected_away, MIN_EXPECTED_CORNERS, MAX_EXPECTED_CORNERS))
    total_corners = exp_home + exp_away

    over_8_5  = float(1 - poisson.cdf(8,  total_corners))
    over_9_5  = float(1 - poisson.cdf(9,  total_corners))
    over_10_5 = float(1 - poisson.cdf(10, total_corners))
    over_11_5 = float(1 - poisson.cdf(11, total_corners))

    return {
        "expected_home":  _num(exp_home,       2),
        "expected_away":  _num(exp_away,       2),
        "expected_total": _num(total_corners,  2),
        "over_8_5":       _num(over_8_5,  4),
        "over_9_5":       _num(over_9_5,  4),
        "over_10_5":      _num(over_10_5, 4),
        "over_11_5":      _num(over_11_5, 4),
    }


def compute_corner_markets(
    home_stats: dict,
    away_stats: dict,
    league_avgs: dict | None = None,
) -> dict:
    """Build corner markets, preferring observed corners taken per game.

    Inputs from the main prediction path are already normalised by
    :func:`to_per_game`. Each side is selected independently, so a fixture
    can use a real rate for one team and the established pressure proxy for
    the other while a season is being backfilled.
    """
    BASE_CORNERS = 5.3
    AWAY_PENALTY = 0.85

    def pressure_proxy(stats: dict, is_home: bool) -> float:
        # Retain the previous fallback calibration for rows without corners.
        shots = _safe(stats.get("sh_per90") or stats.get("shots"), 0.0)
        sot = _safe(stats.get("sot_per90") or stats.get("sot"), 0.0)
        goals = _safe(stats.get("goals") or stats.get("sh_goals"), 0.0)
        xg = _safe(stats.get("xg") or stats.get("sh_xg"), 0.0)
        if shots > 20:
            shots /= 38
        if sot > 10:
            sot /= 38
        if goals > 10:
            goals /= 38
        if xg > 10:
            xg /= 38

        ratios = []
        if shots > 0:
            ratios.append(shots / 12.0)
        if sot > 0:
            ratios.append(sot / 4.5)
        if goals > 0:
            ratios.append(goals / 1.5)
        if xg > 0:
            ratios.append(xg / 1.4)
        pressure = sum(ratios) / len(ratios) if ratios else 1.0
        estimate = BASE_CORNERS * pressure
        if not is_home:
            estimate *= AWAY_PENALTY
        return estimate

    def estimate(stats: dict, is_home: bool) -> float:
        observed = _safe(stats.get("corners"), 0.0)
        if observed > 0:
            # A modest venue split preserves a realistic total: equal 5.5
            # rates become 5.78 + 5.23 = 11.0, rather than being rescaled.
            estimate_value = observed * (
                CORNER_HOME_TILT if is_home else CORNER_AWAY_TILT
            )
        else:
            estimate_value = pressure_proxy(stats, is_home)
        return float(np.clip(
            estimate_value, MIN_EXPECTED_CORNERS, MAX_EXPECTED_CORNERS
        ))

    return corner_markets_from_expected(
        estimate(home_stats, True),
        estimate(away_stats, False),
    )


def calibrate_corners(raw_corners: dict, baselines: dict) -> dict:
    """
    Rescale the raw corner expectations onto real corner units.

    The cross / attacking-third formula is proportional to corners but not
    measured in them, so both sides are multiplied by the league scale
    factor (which makes two average teams total CORNER_LEAGUE_TOTAL) and
    tilted slightly towards the home side before the markets are recomputed.

    Not used by the main prediction path any more: the output of
    :func:`compute_corner_markets` is already in corner units and already
    home/away split, so feeding it through here would apply the league
    scale and the tilt a second time. Kept exported for the raw
    ``_raw_corner_expectation`` path and for callers that rescale their own
    expectations.
    """
    scale = _safe(baselines.get("corner_scale"), 1.0)
    if scale <= 0:
        scale = 1.0

    expected_home = _safe(raw_corners.get("expected_home"), 5.0)
    expected_away = _safe(raw_corners.get("expected_away"), 5.0)

    return corner_markets_from_expected(
        expected_home * scale * CORNER_HOME_TILT,
        expected_away * scale * CORNER_AWAY_TILT,
    )


# ═══════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ═══════════════════════════════════════════


def predict_match(
    home_team: str,
    away_team: str,
    league: str,
    season: str | None = None,
    injuries: dict | None = None,
    original_league: str | None = None,
) -> dict | None:
    """
    Public entry point: full prediction dict, or None when the fixture
    cannot be predicted (unknown team, too little data, database down).

    The engine is called from request handlers and from the daily fixtures
    loop, so an unexpected error is logged with its traceback and turned
    into None rather than propagated as a 500.
    """
    try:
        return _predict_match(
            home_team, away_team, league, season, injuries,
            original_league)
    except Exception as exc:
        print(f"[formula] ERROR: prediction failed for {home_team} vs "
              f"{away_team} ({league}): {exc}")
        traceback.print_exc()
        return None


def _predict_match(
    home_team: str,
    away_team: str,
    league: str,
    season: str | None = None,
    injuries: dict | None = None,
    original_league: str | None = None,
) -> dict | None:
    """
    Generate a full match prediction using the
    formula engine.

    Parameters:
        home_team : team name as stored in DB
        away_team : team name as stored in DB
        league          : domestic league used to load team stats
        season          : season label e.g. "2024-2025" or None
        injuries        : dict of injured players per team
        original_league : source tournament, used to enable UCL blending

    Returns full prediction dict or None if
    insufficient data.

    The returned dict has the same shape as the ML response from
    ``POST /predict/match`` (home_team, away_team, league, season, result,
    goals, corners, feature_vector_used) plus additive extras the formula
    engine can explain: engine, predicted_result, confidence,
    top_scorelines and ratings.
    """
    original_league = original_league or league
    print(f"[formula] Predicting: {home_team} vs "
          f"{away_team} ({league})")

    # Pin the season once, so the two teams, the league averages and the
    # calibration baselines are all read from the same campaign.
    season = resolve_season(league, season)

    # Load data
    home_stats = load_team_stats(home_team, league, season)
    away_stats = load_team_stats(away_team, league, season)

    if (home_stats is None or away_stats is None) and season is not None:
        # The requested season may not exist for one side (a promoted club,
        # a mid-season scrape, MLS calendar years) — fall back to each
        # team's most recent season before giving up.
        if home_stats is None:
            home_stats = load_team_stats(home_team, league, None)
        if away_stats is None:
            away_stats = load_team_stats(away_team, league, None)

    if home_stats is None or away_stats is None:
        missing = []
        if home_stats is None:
            missing.append(home_team)
        if away_stats is None:
            missing.append(away_team)
        print(f"[formula] WARNING: no stats found for: "
              f"{', '.join(missing)}")
        return None

    # A UCL fixture keeps its domestic league for the baseline stats,
    # while adding a capped contribution from each side's UCL row.
    if original_league == UCL_LEAGUE_NAME:
        home_ucl = load_ucl_stats(home_team, season)
        away_ucl = load_ucl_stats(away_team, season)
        home_games_count = games_played(home_stats) or 10
        away_games_count = games_played(away_stats) or 10
        if home_ucl:
            home_stats = blend_ucl_with_domestic(
                home_stats, home_ucl, home_games_count)
        if away_ucl:
            away_stats = blend_ucl_with_domestic(
                away_stats, away_ucl, away_games_count)

    league_avgs = load_league_averages(league, season)

    # Enough history to trust the row?
    home_games = games_played(home_stats, league_avgs)
    away_games = games_played(away_stats, league_avgs)
    for team, games in ((home_team, home_games), (away_team, away_games)):
        if games is not None and games < MIN_GAMES:
            print(f"[formula] WARNING: only {games:.0f} game(s) of data "
                  f"for {team} — below MIN_GAMES={MIN_GAMES}")
            return None

    # Season totals -> per-game rates, so every team is compared on the
    # same basis regardless of which collector wrote the row.
    home_raw, away_raw = home_stats, away_stats
    home_stats = to_per_game(home_stats, league_avgs)
    away_stats = to_per_game(away_stats, league_avgs)

    # Player data: one query per team, reused for squad quality and for
    # matching the injury list against the output those players were
    # actually contributing.
    home_squad = load_squad_players(home_team, league, season)
    away_squad = load_squad_players(away_team, league, season)

    home_players = load_key_players(home_team, league, season)
    away_players = load_key_players(away_team, league, season)

    # Compute ratings
    home_attack  = compute_attack_rating(
        home_stats, league_avgs, home_players)
    home_defence = compute_defence_rating(
        home_stats, league_avgs)
    away_attack  = compute_attack_rating(
        away_stats, league_avgs, away_players)
    away_defence = compute_defence_rating(
        away_stats, league_avgs)

    # Form modifiers
    home_form = compute_form_modifier(home_raw)
    away_form = compute_form_modifier(away_raw)

    # League baselines: the means the ratings are divided by, the league's
    # real goals per game, and the corner scale factor.
    baselines = compute_league_baselines(league, season)

    # Goalkeeper quality (per-90 PSxG+/-, which normalisation preserves).
    home_gk = compute_goalkeeper_modifier(home_stats)
    away_gk = compute_goalkeeper_modifier(away_stats)

    # Injuries passed in by the caller (reuses the squads loaded above).
    home_injury = compute_injury_adjustments(
        home_team, league, injuries, "home", season, home_squad)
    away_injury = compute_injury_adjustments(
        away_team, league, injuries, "away", season, away_squad)

    # Head to head from statsbomb_match_stats (neutral if unavailable).
    h2h = load_h2h_adjustment(home_team, away_team)

    # Ratings relative to the league mean.
    home_attack_rel  = relative_rating(
        home_attack,  baselines["mean_attack"])
    away_attack_rel  = relative_rating(
        away_attack,  baselines["mean_attack"])
    home_defence_rel = relative_rating(
        home_defence, baselines["mean_defence"])
    away_defence_rel = relative_rating(
        away_defence, baselines["mean_defence"])

    # Expected goals
    lambda_home, lambda_away = compute_expected_goals(
        base_goals=baselines["goals_per_game"],
        home_attack_rel=home_attack_rel,
        home_defence_rel=home_defence_rel,
        away_attack_rel=away_attack_rel,
        away_defence_rel=away_defence_rel,
        home_form=home_form,
        away_form=away_form,
        home_injury=home_injury,
        away_injury=away_injury,
        home_gk=home_gk,
        away_gk=away_gk,
        h2h=h2h,
    )

    # Markets
    markets = compute_markets(lambda_home, lambda_away)
    # The corner estimator is already calibrated in real corner units and
    # carries its own home/away split (AWAY_PENALTY), so the raw-formula
    # league scale and tilt from calibrate_corners must NOT be re-applied
    # on top of it — doing that multiplied the expectations by the league
    # scale a second time (e.g. 14.3 -> 21.0 expected corners).
    corners = compute_corner_markets(home_stats, away_stats, league_avgs)

    home_win = markets["home_win"]
    draw     = markets["draw"]
    away_win = markets["away_win"]

    predicted_result = max(
        (("H", home_win), ("D", draw), ("A", away_win)),
        key=lambda item: item[1],
    )[0]

    used_season = (
        season
        or home_raw.get("season")
        or away_raw.get("season")
        or ""
    )

    print(
        f"[formula] {home_team} {lambda_home:.2f} - "
        f"{lambda_away:.2f} {away_team} | "
        f"{home_win:.1%} / {draw:.1%} / {away_win:.1%} | "
        f"pick={predicted_result}"
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "league":    league,
        "season":    used_season,
        "result": {
            "home_win": _num(home_win),
            "draw":     _num(draw),
            "away_win": _num(away_win),
        },
        "goals": {
            "expected_home": _num(lambda_home, 3),
            "expected_away": _num(lambda_away, 3),
            "over_1_5":      markets["over_1_5"],
            "over_2_5":      markets["over_2_5"],
            "over_3_5":      markets["over_3_5"],
            "btts":          markets["btts"],
        },
        "corners": corners,
        "feature_vector_used": {
            # League context
            "league_goals_per_game": _num(baselines["goals_per_game"]),
            "league_mean_attack":    _num(baselines["mean_attack"]),
            "league_mean_defence":   _num(baselines["mean_defence"]),
            "league_teams":          _num(baselines["teams"], 0),
            "league_avg_xg":         _num(league_avgs.get("avg_xg")),
            "league_avg_sot":        _num(league_avgs.get("avg_sot")),
            "corner_scale":          _num(baselines["corner_scale"]),
            # Games behind each row
            "home_games":            _num(home_games or 0, 1),
            "away_games":            _num(away_games or 0, 1),
            # Season blend: how much of each row is the current season
            "home_season_games":     _num(home_games or 0, 1),
            "home_blend_weight":     _num(
                home_stats.get("blend_weight_current", 1.0)),
            "away_season_games":     _num(away_games or 0, 1),
            "away_blend_weight":     _num(
                away_stats.get("blend_weight_current", 1.0)),
            # Raw ratings
            "home_attack_rating":    _num(home_attack),
            "home_defence_rating":   _num(home_defence),
            "away_attack_rating":    _num(away_attack),
            "away_defence_rating":   _num(away_defence),
            # League-relative ratings
            "home_attack_rel":       _num(home_attack_rel),
            "home_defence_rel":      _num(home_defence_rel),
            "away_attack_rel":       _num(away_attack_rel),
            "away_defence_rel":      _num(away_defence_rel),
            # Modifiers
            "home_form_modifier":    _num(home_form),
            "away_form_modifier":    _num(away_form),
            "home_gk_modifier":      _num(home_gk),
            "away_gk_modifier":      _num(away_gk),
            "home_injury_attack":    _num(home_injury["attack"]),
            "home_injury_defence":   _num(home_injury["defence"]),
            "away_injury_attack":    _num(away_injury["attack"]),
            "away_injury_defence":   _num(away_injury["defence"]),
            "h2h_home_modifier":     _num(h2h["home"]),
            "h2h_away_modifier":     _num(h2h["away"]),
            "home_advantage":        _num(HOME_ADVANTAGE),
            "away_adjustment":       _num(AWAY_ADJUSTMENT),
            # Per-game inputs actually used. ga / shots / sot come straight
            # from the stored per-90 columns (ga90, sh_per90, sh_sot_per90);
            # goals per game is sh_goals divided by the games estimate.
            "home_xg_per_game":      _num(home_stats.get("xg")),
            "away_xg_per_game":      _num(away_stats.get("xg")),
            "home_ga_per_game":      _num(home_stats.get("ga")),
            "away_ga_per_game":      _num(away_stats.get("ga")),
            "home_goals_per_game":   _num(home_stats.get("goals")),
            "away_goals_per_game":   _num(away_stats.get("goals")),
            "home_shots_per_game":   _num(home_stats.get("shots")),
            "away_shots_per_game":   _num(away_stats.get("shots")),
            "home_sot_per_game":     _num(home_stats.get("sot")),
            "away_sot_per_game":     _num(away_stats.get("sot")),
            "home_save_pct":         _num(home_stats.get("save_pct"), 2),
            "away_save_pct":         _num(away_stats.get("save_pct"), 2),
            "home_possession":       _num(home_stats.get("poss"), 2),
            "away_possession":       _num(away_stats.get("poss"), 2),
            "home_gca90":            _num(home_stats.get("gca90")),
            "away_gca90":            _num(away_stats.get("gca90")),
            # Outputs
            "lambda_home":           _num(lambda_home, 3),
            "lambda_away":           _num(lambda_away, 3),
        },
        # --- Additive extras (safe for existing clients) ---
        "engine": ENGINE_NAME,
        "predicted_result": predicted_result,
        "confidence": _num(max(home_win, draw, away_win)),
        "top_scorelines": top_scorelines(lambda_home, lambda_away),
        "ratings": {
            "home": {
                "team":           home_team,
                "attack_rating":  _num(home_attack),
                "defence_rating": _num(home_defence),
                "attack_rel":     _num(home_attack_rel),
                "defence_rel":    _num(home_defence_rel),
                "form":           _num(home_form),
                "goalkeeper":     _num(home_gk),
                "games":          _num(home_games or 0, 1),
                "season":         home_raw.get("season"),
                "expected_goals":   _num(lambda_home, 3),
                "expected_corners": corners["expected_home"],
                "top_scorer_goals": _num(home_players["top_scorer_goals"], 1),
                "squad_xg_total":   _num(home_players["squad_xg_total"], 1),
                "injuries": {
                    "attack_modifier":     _num(home_injury["attack"]),
                    "defence_modifier":    _num(home_injury["defence"]),
                    "missing":             home_injury["missing"],
                    "matched":             home_injury["matched"],
                    "unmatched":           home_injury["unmatched"],
                    "attack_share_lost":   _num(
                        home_injury["attack_share_lost"]),
                    "defence_share_lost":  _num(
                        home_injury["defence_share_lost"]),
                },
            },
            "away": {
                "team":           away_team,
                "attack_rating":  _num(away_attack),
                "defence_rating": _num(away_defence),
                "attack_rel":     _num(away_attack_rel),
                "defence_rel":    _num(away_defence_rel),
                "form":           _num(away_form),
                "goalkeeper":     _num(away_gk),
                "games":          _num(away_games or 0, 1),
                "season":         away_raw.get("season"),
                "expected_goals":   _num(lambda_away, 3),
                "expected_corners": corners["expected_away"],
                "top_scorer_goals": _num(away_players["top_scorer_goals"], 1),
                "squad_xg_total":   _num(away_players["squad_xg_total"], 1),
                "injuries": {
                    "attack_modifier":     _num(away_injury["attack"]),
                    "defence_modifier":    _num(away_injury["defence"]),
                    "missing":             away_injury["missing"],
                    "matched":             away_injury["matched"],
                    "unmatched":           away_injury["unmatched"],
                    "attack_share_lost":   _num(
                        away_injury["attack_share_lost"]),
                    "defence_share_lost":  _num(
                        away_injury["defence_share_lost"]),
                },
            },
            "head_to_head": h2h,
            "league": {
                "name":            league,
                "season":          used_season,
                "teams":           baselines["teams"],
                "goals_per_game":  _num(baselines["goals_per_game"]),
                "mean_attack":     _num(baselines["mean_attack"]),
                "mean_defence":    _num(baselines["mean_defence"]),
                "corner_scale":    _num(baselines["corner_scale"]),
            },
        },
    }


__all__ = [
    "ENGINE_NAME",
    "calibrate_corners",
    "clear_cache",
    "compute_attack_rating",
    "compute_corner_markets",
    "compute_defence_rating",
    "compute_expected_goals",
    "compute_form_modifier",
    "compute_goalkeeper_modifier",
    "compute_injury_adjustments",
    "compute_league_baselines",
    "compute_markets",
    "compute_scoreline_matrix",
    "corner_markets_from_expected",
    "extract_team_injuries",
    "games_played",
    "league_seasons",
    "load_h2h_adjustment",
    "load_key_players",
    "load_league_averages",
    "load_squad_players",
    "load_team_stats",
    "predict_match",
    "relative_rating",
    "resolve_season",
    "to_per_game",
    "top_scorelines",
]


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run the formula engine for one fixture.",
    )
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("league")
    parser.add_argument(
        "--season", default=None,
        help="Season label, e.g. 2024-2025 (default: most recent)",
    )
    parser.add_argument(
        "--injuries", default=None,
        help='JSON injuries dict, e.g. \'{"Arsenal": ["Bukayo Saka"]}\'',
    )
    args = parser.parse_args()

    injury_input = json.loads(args.injuries) if args.injuries else None
    prediction = predict_match(
        args.home_team,
        args.away_team,
        args.league,
        args.season,
        injury_input,
    )

    if prediction is None:
        print("\nNo prediction: not enough data for that fixture.")
        sys.exit(1)

    print(json.dumps(prediction, indent=2))


# ═══════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════


def get_engine_status() -> dict[str, Any]:
    """
    Report whether the formula engine can serve predictions.

    ``ready`` is True when the database engine was created and a session
    factory exists. It is a configuration check only: no query is run, so
    the call is cheap and never raises. A ready engine can still return
    None for an individual fixture (unknown team, too little data, database
    down), and the router then falls back to the ML models.
    """
    try:
        from database import engine as db_engine
    except Exception as exc:
        return {
            "ready": False,
            "engine": ENGINE_NAME,
            "reason": f"database module unavailable: {exc}",
            "database_configured": False,
            "cache_entries": len(_CACHE),
        }

    database_configured = db_engine is not None
    session_factory_ok = SessionLocal is not None

    if not database_configured:
        reason = "no database engine configured (check DATABASE_URL)"
    elif not session_factory_ok:
        reason = "session factory unavailable"
    else:
        reason = "ok"

    return {
        "ready": database_configured and session_factory_ok,
        "engine": ENGINE_NAME,
        "reason": reason,
        "database_configured": database_configured,
        "cache_entries": len(_CACHE),
    }
