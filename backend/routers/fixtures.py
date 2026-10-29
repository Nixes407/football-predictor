"""Endpoints for daily football fixtures and their stored predictions."""

import json
import math
import sys
import threading
import time
import traceback
import unicodedata
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache, wraps
# Imported by name: ``from sqlalchemy import inspect`` below shadows the module.
from inspect import signature as _signature
from pathlib import Path

import pandas as pd
import routers.predict as predict_module
from fastapi import APIRouter, Query
from sqlalchemy import Column, Float, inspect, text
from sqlalchemy.orm import Session

from data.fixtures import fetch_fixtures

# Imported the same defensive way data/fixtures.py does: the spillover pass
# must keep working (without MLS) if the collector is unavailable.
try:
    from data.fotmob_collector import fetch_fixtures_for_date
except Exception:  # pragma: no cover - optional dependency
    try:
        # A script run from backend/data/ lands here.
        from fotmob_collector import fetch_fixtures_for_date
    except Exception:
        fetch_fixtures_for_date = None
from database import DailyMatch, SessionLocal, TeamSeasonStats
from routers.predict import (
    _away_corners,
    _away_goals,
    _blend_predictions,
    _build_feature_vector,
    _encode_label,
    _features_df,
    _formula_engine_available,
    _get_runtime_feature_cols,
    _home_corners,
    _home_goals,
    _le_league,
    _le_season,
    _poisson_btts,
    _poisson_over,
    _result_model,
    compute_first_goal_probs,
    formula_predict,
)

try:  # SQLAlchemy 2.x
    from sqlalchemy.orm import add_mapped_attribute
except ImportError:  # pragma: no cover - SQLAlchemy 1.4 fallback
    add_mapped_attribute = None


router = APIRouter()

_FOTMOB_NAMES_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "fotmob_team_names.json"
)

# Season used when a stored fixture carries none (same rule as
# predict.MatchRequest). It must be the CURRENT season: the model encodes the
# season, so a stale fixed label fed the classifier a value that described a
# different year. The formula engine resolves it to the newest season the
# league actually has data for.
DEFAULT_SEASON = predict_module.feature_inference.current_season_label()

UCL_LEAGUE = "UEFA-Champions League"

# Fixture spelling (football-data.org, after data.fixtures.map_team_name) and
# FotMob spelling -> the domestic league the club's stats live in.
#
# football-data.org names the same club differently from FotMob ("Bayern
# Munich" vs "Bayern München", "Paris SG" vs "Paris Saint-Germain"), and the
# UCL feed re-uses the national feed's names, so every variant either side can
# send is listed here. Keys are matched exactly first and then accent-, case-,
# punctuation- and whitespace-insensitively (see _get_ucl_domestic_league), so
# a variant that only differs in a dotted abbreviation still resolves.
UCL_TEAM_DOMESTIC_LEAGUE = {
    # England
    "Arsenal":                  "ENG-Premier League",
    "Aston Villa":              "ENG-Premier League",
    "Chelsea":                  "ENG-Premier League",
    "Liverpool":                "ENG-Premier League",
    "Manchester City":          "ENG-Premier League",
    "Man City":                 "ENG-Premier League",
    "Manchester United":        "ENG-Premier League",
    "Man United":               "ENG-Premier League",
    "Man Utd":                  "ENG-Premier League",
    "Newcastle United":         "ENG-Premier League",
    "Newcastle":                "ENG-Premier League",
    "Tottenham Hotspur":        "ENG-Premier League",
    "Tottenham":                "ENG-Premier League",
    "Spurs":                    "ENG-Premier League",

    # Spain
    "Real Madrid":              "ESP-La Liga",
    "Barcelona":                "ESP-La Liga",
    "FC Barcelona":             "ESP-La Liga",
    "Atletico Madrid":          "ESP-La Liga",
    "Atlético Madrid":          "ESP-La Liga",
    "Atletico de Madrid":       "ESP-La Liga",
    "Atlético de Madrid":       "ESP-La Liga",   # football-data.org spelling
    "Club Atletico de Madrid":  "ESP-La Liga",
    "Athletic Club":            "ESP-La Liga",
    "Athletic Bilbao":          "ESP-La Liga",
    "Real Betis":               "ESP-La Liga",
    "Betis":                    "ESP-La Liga",
    "Real Betis Balompié":      "ESP-La Liga",
    "Villarreal":               "ESP-La Liga",
    "Villarreal CF":            "ESP-La Liga",
    "Real Sociedad":            "ESP-La Liga",
    "Sevilla":                  "ESP-La Liga",
    "Sevilla FC":               "ESP-La Liga",
    "Girona":                   "ESP-La Liga",

    # Germany
    "Bayern München":           "GER-Bundesliga",
    "Bayern Munich":            "GER-Bundesliga",
    "Bayern Munchen":           "GER-Bundesliga",
    "FC Bayern München":        "GER-Bundesliga",
    "FC Bayern Munich":         "GER-Bundesliga",
    "Borussia Dortmund":        "GER-Bundesliga",
    "Dortmund":                 "GER-Bundesliga",
    "BVB":                      "GER-Bundesliga",
    "RB Leipzig":               "GER-Bundesliga",
    "RasenBallsport Leipzig":   "GER-Bundesliga",
    "Bayer Leverkusen":         "GER-Bundesliga",
    "Bayer 04 Leverkusen":      "GER-Bundesliga",
    "Leverkusen":               "GER-Bundesliga",
    "VfB Stuttgart":            "GER-Bundesliga",
    "Stuttgart":                "GER-Bundesliga",
    "Eintracht Frankfurt":      "GER-Bundesliga",
    "Frankfurt":                "GER-Bundesliga",
    "Borussia Mönchengladbach": "GER-Bundesliga",
    "Gladbach":                 "GER-Bundesliga",
    "Union Berlin":             "GER-Bundesliga",
    "Wolfsburg":                "GER-Bundesliga",

    # Italy
    "Inter":                    "ITA-Serie A",
    "Internazionale":           "ITA-Serie A",
    "Inter Milan":              "ITA-Serie A",
    "FC Internazionale":        "ITA-Serie A",
    "AC Milan":                 "ITA-Serie A",
    "Milan":                    "ITA-Serie A",
    "Juventus":                 "ITA-Serie A",
    "Juventus FC":              "ITA-Serie A",
    "Napoli":                   "ITA-Serie A",
    "SSC Napoli":               "ITA-Serie A",
    "Roma":                     "ITA-Serie A",
    "AS Roma":                  "ITA-Serie A",
    "Atalanta":                 "ITA-Serie A",
    "Atalanta BC":              "ITA-Serie A",
    "Lazio":                    "ITA-Serie A",
    "SS Lazio":                 "ITA-Serie A",
    "Bologna":                  "ITA-Serie A",
    "Bologna FC":               "ITA-Serie A",
    "Bologna FC 1909":          "ITA-Serie A",
    "Fiorentina":               "ITA-Serie A",
    "ACF Fiorentina":           "ITA-Serie A",
    "Como":                     "ITA-Serie A",

    # France
    "Paris Saint-Germain":      "FRA-Ligue 1",
    "Paris Saint Germain":      "FRA-Ligue 1",
    "Paris SG":                 "FRA-Ligue 1",
    "PSG":                      "FRA-Ligue 1",
    "Marseille":                "FRA-Ligue 1",
    "Olympique de Marseille":   "FRA-Ligue 1",
    "Monaco":                   "FRA-Ligue 1",
    "AS Monaco":                "FRA-Ligue 1",
    "Lens":                     "FRA-Ligue 1",
    "RC Lens":                  "FRA-Ligue 1",
    "Racing Club de Lens":      "FRA-Ligue 1",
    "Lille":                    "FRA-Ligue 1",
    "LOSC Lille":               "FRA-Ligue 1",
    "Lyon":                     "FRA-Ligue 1",
    "Olympique Lyonnais":       "FRA-Ligue 1",
    "Brest":                    "FRA-Ligue 1",
    "Stade Brestois 29":        "FRA-Ligue 1",
    "Nice":                     "FRA-Ligue 1",
    "OGC Nice":                 "FRA-Ligue 1",
    "Rennes":                   "FRA-Ligue 1",
    "Stade Rennais FC":         "FRA-Ligue 1",
    "RC Strasbourg":            "FRA-Ligue 1",
    "Strasbourg":               "FRA-Ligue 1",

    # Portugal
    "Benfica":                  "POR-Primeira Liga",
    "SL Benfica":               "POR-Primeira Liga",
    "FC Porto":                 "POR-Primeira Liga",
    "Porto":                    "POR-Primeira Liga",
    "Sporting CP":              "POR-Primeira Liga",
    "Sporting Clube de Portugal": "POR-Primeira Liga",
    "Braga":                    "POR-Primeira Liga",
    "SC Braga":                 "POR-Primeira Liga",

    # Netherlands
    "Feyenoord":                "NED-Eredivisie",
    "Feyenoord Rotterdam":      "NED-Eredivisie",
    "PSV Eindhoven":            "NED-Eredivisie",
    "PSV":                      "NED-Eredivisie",
    "Ajax":                     "NED-Eredivisie",
    "AFC Ajax":                 "NED-Eredivisie",
    "Twente":                   "NED-Eredivisie",
    "FC Twente":                "NED-Eredivisie",
    "AZ":                       "NED-Eredivisie",
    "AZ Alkmaar":               "NED-Eredivisie",

    # Turkey
    "Fenerbahçe":               "TUR-Süper Lig",
    "Fenerbahce":               "TUR-Süper Lig",
    "Fenerbahçe SK":            "TUR-Süper Lig",
    "Galatasaray":              "TUR-Süper Lig",
    "Galatasaray SK":           "TUR-Süper Lig",
    "Beşiktaş":                 "TUR-Süper Lig",
    "Besiktas":                 "TUR-Süper Lig",
    "Beşiktaş JK":              "TUR-Süper Lig",
    "Trabzonspor":              "TUR-Süper Lig",
    "Başakşehir":               "TUR-Süper Lig",
    "Istanbul Basaksehir":      "TUR-Süper Lig",

    # Norway
    "Bodø/Glimt":               "NOR-Eliteserien",
    "Bodo/Glimt":               "NOR-Eliteserien",
    "FK Bodø/Glimt":            "NOR-Eliteserien",
    "Viking":                   "NOR-Eliteserien",
    "Viking FK":                "NOR-Eliteserien",
    "Rosenborg":                "NOR-Eliteserien",
    "Rosenborg BK":             "NOR-Eliteserien",
    "Molde":                    "NOR-Eliteserien",
    "Molde FK":                 "NOR-Eliteserien",

    # Austria
    "LASK":                     "AUT-Bundesliga",
    "LASK Linz":                "AUT-Bundesliga",
    "FC LASK Linz":             "AUT-Bundesliga",
    "Salzburg":                 "AUT-Bundesliga",
    "Red Bull Salzburg":        "AUT-Bundesliga",
    "FC Red Bull Salzburg":     "AUT-Bundesliga",
    "FC Salzburg":              "AUT-Bundesliga",
    "RB Salzburg":              "AUT-Bundesliga",
    "Rapid Wien":               "AUT-Bundesliga",
    "SK Rapid Wien":            "AUT-Bundesliga",
    "Sturm Graz":               "AUT-Bundesliga",
    "SK Sturm Graz":            "AUT-Bundesliga",
}


# Letters that are their own character rather than an accented Latin one, so
# NFKD leaves them alone (ß is already folded to "ss" by casefold()).
_RELAXED_CHARS = str.maketrans({
    "ø": "o", "æ": "ae", "œ": "oe", "đ": "d", "ð": "d", "þ": "th",
    "ł": "l", "ı": "i", "ĸ": "k",
})


def _relaxed_name(name: str) -> str:
    """
    Comparison key for team names: accents, case, punctuation and repeated
    whitespace removed.

    Providers disagree on more than spelling — "Union St.Gilloise" vs
    "Union St. Gilloise", "Atlético" vs "Atletico", "Bodø/Glimt" vs
    "Bodo/Glimt" — and none of that changes which club is meant. Used for
    lookups only; the original spelling is what gets stored and displayed.
    """
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    # Accents decompose, but letters like ø and đ are letters in their own
    # right, so they need folding by hand to match their ASCII spelling.
    folded = stripped.casefold().translate(_RELAXED_CHARS)
    # Treat punctuation ("St.Gilloise", "Bodø/Glimt", "Paris Saint-Germain")
    # as a separator, so the tokens line up whichever variant is used.
    cleaned = "".join(
        " " if not char.isalnum() else char for char in folded
    )
    return " ".join(cleaned.split())


# Relaxed key -> domestic league, so a spelling that only differs in accents,
# case or punctuation still resolves. Built once: the table above is static,
# and the first spelling of a club wins if two ever relax to the same key.
def _build_ucl_relaxed_index() -> dict[str, str]:
    """Map every relaxed club name to its domestic league."""
    index: dict[str, str] = {}
    for team, league in UCL_TEAM_DOMESTIC_LEAGUE.items():
        key = _relaxed_name(team)
        if key:
            index.setdefault(key, league)
    return index


UCL_DOMESTIC_LEAGUE_BY_RELAXED_NAME = _build_ucl_relaxed_index()

# A stored prediction is refreshed at most this long after it was computed.
FIXTURE_STALE_HOURS = 2

# 1X2 probabilities: the fields that make a stored fixture "predicted".
RESULT_PREDICTION_FIELDS = (
    "pred_home_win",
    "pred_draw",
    "pred_away_win",
)

# Prediction fields daily_matches has always had columns for.
STORED_PREDICTION_FIELDS = RESULT_PREDICTION_FIELDS + (
    "pred_home_goals",
    "pred_away_goals",
    "pred_home_corners",
    "pred_away_corners",
)

# Markets the API also answers with, which the table predates: they are
# derived from the stored expectations (same Poisson arithmetic the formula
# engine uses) and persisted whenever the columns can be created.
DERIVED_PREDICTION_FIELDS = (
    "pred_over_2_5",
    "pred_btts",
    "pred_corners_over_9_5",
)

PREDICTION_FIELDS = STORED_PREDICTION_FIELDS + DERIVED_PREDICTION_FIELDS

# Set once the derived-market columns are known to exist *and* have been
# mapped onto DailyMatch (see ensure_derived_prediction_columns).
_derived_columns_registered = False


def _current_predictor_value(name: str, imported_value):
    """Use the latest predictor state, including artifacts loaded after import."""
    current_value = getattr(predict_module, name, None)
    return imported_value if current_value is None else current_value


def _empty_prediction_values() -> dict:
    """Every prediction field, unset."""
    return {name: None for name in PREDICTION_FIELDS}


def _prediction_values(row: DailyMatch) -> dict:
    """Current prediction fields of a stored fixture (None where unset)."""
    return {name: getattr(row, name, None) for name in PREDICTION_FIELDS}


def _has_result_prediction(values: dict) -> bool:
    """A fixture counts as predicted once its 1X2 probabilities are set."""
    return all(
        values.get(name) is not None for name in RESULT_PREDICTION_FIELDS
    )


def _derived_markets(values: dict) -> dict:
    """
    Over 2.5, BTTS and corners over 9.5 implied by the expected goals and
    corners of a prediction.

    The formula engine publishes these markets itself; for predictions that
    only carry expectations (rows written by the ML models, or a response
    built from an older row) they are recomputed with exactly the same
    Poisson arithmetic the engine uses — ``predict._poisson_over`` and
    ``predict._poisson_btts``. Fields whose expectation is missing stay None.
    """
    markets = {name: None for name in DERIVED_PREDICTION_FIELDS}

    home_goals = values.get("pred_home_goals")
    away_goals = values.get("pred_away_goals")
    home_corners = values.get("pred_home_corners")
    away_corners = values.get("pred_away_corners")

    if home_goals is not None and away_goals is not None:
        total_goals = float(home_goals) + float(away_goals)
        markets["pred_over_2_5"] = round(
            _poisson_over(total_goals, 2.5), 4)
        markets["pred_btts"] = round(
            _poisson_btts(float(home_goals), float(away_goals)), 4)

    if home_corners is not None and away_corners is not None:
        total_corners = float(home_corners) + float(away_corners)
        markets["pred_corners_over_9_5"] = round(
            _poisson_over(total_corners, 9.5), 4)

    return markets


def _with_derived_markets(values: dict) -> dict:
    """Fill any missing derived market of ``values`` in place and return it."""
    for name, market in _derived_markets(values).items():
        if values.get(name) is None and market is not None:
            values[name] = market
    return values


def _mapped_prediction_values(values: dict) -> dict:
    """Keep only the prediction keys daily_matches has a column for."""
    mapped = set(DailyMatch.__table__.columns.keys())
    return {
        name: value for name, value in values.items() if name in mapped
    }


def _ml_models_available() -> bool:
    """Whether the ML artifacts needed for a prediction are loaded."""
    return all(
        _current_predictor_value(name, imported) is not None
        for name, imported in (
            ("_result_model", _result_model),
            ("_home_goals", _home_goals),
            ("_away_goals", _away_goals),
            ("_home_corners", _home_corners),
            ("_away_corners", _away_corners),
            ("_features_df", _features_df),
        )
    )


def _compute_prediction_values(
    home_team: str,
    away_team: str,
    league: str,
    season: str,
) -> dict[str, float]:
    """Run the same feature construction and model inference as /predict/match."""
    result_model = _current_predictor_value("_result_model", _result_model)
    home_goals_model = _current_predictor_value("_home_goals", _home_goals)
    away_goals_model = _current_predictor_value("_away_goals", _away_goals)
    home_corners_model = _current_predictor_value("_home_corners", _home_corners)
    away_corners_model = _current_predictor_value("_away_corners", _away_corners)
    league_encoder = _current_predictor_value("_le_league", _le_league)
    season_encoder = _current_predictor_value("_le_season", _le_season)
    features_df = _current_predictor_value("_features_df", _features_df)

    if any(
        model is None
        for model in (
            result_model,
            home_goals_model,
            away_goals_model,
            home_corners_model,
            away_corners_model,
        )
    ):
        raise RuntimeError("Prediction models are not loaded.")
    if features_df is None:
        raise RuntimeError("Features data is not loaded.")

    league_encoded = _encode_label(league_encoder, league)
    season_encoded = _encode_label(season_encoder, season)
    # Use the feature list of the loaded model, exactly like /predict/match,
    # so a model trained on a different column set is not fed the hardcoded
    # one (which sklearn rejects on a feature-name mismatch).
    runtime_cols = _get_runtime_feature_cols()
    vector = _build_feature_vector(
        home_team,
        away_team,
        league_encoded,
        season_encoded,
        features_df,
        runtime_cols,
        league=league,
        season=season,
    )
    X = pd.DataFrame([vector])[runtime_cols]

    result_probabilities = result_model.predict_proba(X)[0]
    result_map = dict(zip(list(result_model.classes_), result_probabilities))

    values = {
        "pred_home_win": round(float(result_map.get("H", 0.0)), 4),
        "pred_draw": round(float(result_map.get("D", 0.0)), 4),
        "pred_away_win": round(float(result_map.get("A", 0.0)), 4),
        "pred_home_goals": round(
            max(float(home_goals_model.predict(X)[0]), 0.0), 3
        ),
        "pred_away_goals": round(
            max(float(away_goals_model.predict(X)[0]), 0.0), 3
        ),
        "pred_home_corners": round(
            max(float(home_corners_model.predict(X)[0]), 0.0), 3
        ),
        "pred_away_corners": round(
            max(float(away_corners_model.predict(X)[0]), 0.0), 3
        ),
    }
    return _with_derived_markets(values)


# formula_engine.predict_match takes ONE league for both teams: it loads the
# home team's season stats, squad and key players from it, and the away team's
# from that same league. A UCL fixture between clubs from two countries would
# therefore look the away club up in the home club's league, find nothing and
# return no prediction at all — the single biggest source of UCL fixtures
# without one. The engine resolves every team-scoped row through the
# module-level loaders below, so the away club is pointed at its own domestic
# league for the duration of the call (see _engine_team_league_overrides).
_ENGINE_TEAM_LOADERS = (
    "load_team_stats",
    "load_squad_players",
    "load_key_players",
)

# Serialises the calls that patch the engine, so two cross-league predictions
# never rewrite the module's loaders on top of each other.
_ENGINE_LEAGUE_OVERRIDE_LOCK = threading.RLock()

# Engine parameters that could carry a per-team league directly, most
# explicit first. Checked at call time: if the engine ever grows one, it is
# used instead of patching, and this module keeps working unchanged.
_AWAY_LEAGUE_PARAMETERS = (
    "away_league",
    "away_domestic_league",
    "away_team_league",
)


def _formula_engine_module():
    """
    The imported ``formula_engine`` module, or None when it is not loaded.

    Looked up through the predictor's own ``predict_match`` function, so the
    module that actually serves predictions is the one that gets patched.
    """
    predictor = _current_predictor_value("formula_predict", formula_predict)
    module_name = getattr(predictor, "__module__", None)
    if not module_name:
        return None
    return sys.modules.get(module_name)


def _engine_away_league_parameter() -> str | None:
    """
    Name of the engine parameter that takes the away team's league, if any.

    Returns None for the current engine, whose ``predict_match`` accepts a
    single league — the caller then falls back to the loader override.
    """
    predictor = _current_predictor_value("formula_predict", formula_predict)
    if predictor is None:
        return None
    try:
        parameters = _signature(predictor).parameters
    except (TypeError, ValueError):
        return None
    for name in _AWAY_LEAGUE_PARAMETERS:
        if name in parameters:
            return name
    return None


@contextmanager
def _engine_team_league_overrides(team_leagues: dict[str, str]):
    """
    Read the listed teams from their own league while the engine predicts.

    ``formula_engine`` loads a match's teams through module-level functions
    (``load_team_stats``, ``load_squad_players``, ``load_key_players``) that
    all take the competition as their second argument. Wrapping them for the
    duration of one prediction is what lets a UCL fixture use each club's own
    domestic league — the season blend, the squad and the key players of the
    away club all come from the league it actually plays in.

    Only teams named in ``team_leagues`` are redirected, always towards the
    league they belong to, so a lookup that another thread happens to run
    meanwhile can only ever get the right league instead of none. Yields True
    when the override is in place and False when the engine module is not
    loaded (the prediction then just runs unpatched).
    """
    if not team_leagues:
        yield False
        return

    engine = _formula_engine_module()
    if engine is None:
        yield False
        return

    def _redirect(loader):
        @wraps(loader)
        def wrapper(team, league, season=None, *args, **kwargs):
            return loader(
                team, team_leagues.get(team, league), season,
                *args, **kwargs,
            )

        return wrapper

    with _ENGINE_LEAGUE_OVERRIDE_LOCK:
        originals: dict[str, object] = {}
        try:
            for name in _ENGINE_TEAM_LOADERS:
                loader = getattr(engine, name, None)
                if not callable(loader):
                    continue
                originals[name] = loader
                setattr(engine, name, _redirect(loader))
            yield bool(originals)
        finally:
            for name, loader in originals.items():
                setattr(engine, name, loader)


def _run_formula_predict(prediction_args: dict, away_league: str | None):
    """
    Call the engine with both domestic leagues where it can take them.

    The away league is passed as a parameter when the engine defines one, and
    otherwise applied by overriding the engine's team loaders for this call.
    """
    if not away_league or away_league == prediction_args["league"]:
        return formula_predict(**prediction_args)

    parameter = _engine_away_league_parameter()
    if parameter:
        return formula_predict(**{**prediction_args, parameter: away_league})

    with _engine_team_league_overrides(
        {prediction_args["away_team"]: away_league}
    ) as applied:
        if not applied:
            print("[fixtures] NOTE: formula engine module not loaded — "
                  f"predicting {prediction_args['home_team']} vs "
                  f"{prediction_args['away_team']} with "
                  f"{prediction_args['league']} for both teams")
        return formula_predict(**prediction_args)


def _predict_with_formula(
    home_team: str,
    away_team: str,
    league: str,
    season: str | None,
    original_league: str | None = None,
    away_league: str | None = None,
) -> dict | None:
    """
    Prediction values from the formula engine, or None when it cannot
    produce one (engine unavailable, unknown teams, too little data).

    ``league`` is the competition the Poisson baseline is taken from (for a
    UCL fixture: the home team's domestic league) and, with ``away_league``,
    the league the away team is read from when the two differ.

    The engine's response is mapped onto the fixtures prediction fields, so
    storing a prediction never has to know the engine's response shape.
    """
    if not _formula_engine_available or formula_predict is None:
        print(f"[fixtures] Formula engine unavailable — cannot predict "
              f"{home_team} vs {away_team}")
        return None
    if not (home_team and away_team and league):
        print(f"[fixtures] Formula engine skipped for {home_team!r} vs "
              f"{away_team!r}: home_team, away_team and league must all "
              f"be set (league={league!r})")
        return None

    try:
        print(f"[fixtures] formula_predict: {home_team!r} vs "
              f"{away_team!r} league={league!r} "
              f"away_league={away_league!r} "
              f"season={(season or DEFAULT_SEASON)!r}")
        prediction_args = {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "season": season or DEFAULT_SEASON,
        }
        if original_league is not None:
            prediction_args["original_league"] = original_league
        prediction = _run_formula_predict(prediction_args, away_league)
        print(f"[fixtures] formula_predict returned: {bool(prediction)}")
    except Exception as exc:
        print(f"[fixtures] WARNING: formula prediction failed for "
              f"{home_team} vs {away_team}: {exc}")
        traceback.print_exc()
        return None

    if not prediction:
        return None

    result = prediction.get("result") or {}
    goals = prediction.get("goals") or {}
    corners = prediction.get("corners") or {}

    values = {
        "pred_home_win": result.get("home_win"),
        "pred_draw": result.get("draw"),
        "pred_away_win": result.get("away_win"),
        "pred_home_goals": goals.get("expected_home"),
        "pred_away_goals": goals.get("expected_away"),
        "pred_home_corners": corners.get("expected_home"),
        "pred_away_corners": corners.get("expected_away"),
        "pred_over_2_5": goals.get("over_2_5"),
        "pred_btts": goals.get("btts"),
        "pred_corners_over_9_5": corners.get("over_9_5"),
    }

    if not _has_result_prediction(values):
        # Without 1X2 probabilities the response is not a usable prediction.
        print(f"[fixtures] WARNING: formula engine returned an incomplete "
              f"prediction for {home_team} vs {away_team}")
        return None

    return _with_derived_markets(values)


def _get_ucl_domestic_league(team: str) -> str | None:
    """
    Return the domestic league for a UCL team.

    The exact spelling is tried first, then the accent/case/punctuation
    insensitive spelling, so a variant that is not listed verbatim (a trailing
    "FC", an accent, a dotted abbreviation) still maps to the right league
    instead of hiding a coverable fixture.

    Returns None if the team is from a league
    we do not cover — those fixtures are hidden.
    """
    if not team:
        return None
    league = UCL_TEAM_DOMESTIC_LEAGUE.get(team)
    if league:
        return league
    league = UCL_DOMESTIC_LEAGUE_BY_RELAXED_NAME.get(_relaxed_name(team))
    if league:
        print(f"[fixtures] UCL: {team!r} resolved by normalised name to "
              f"{league}")
        return league
    return None


def _parse_utc(value) -> datetime | None:
    """A stored ISO timestamp (or datetime) as an aware UTC datetime."""
    if value is None or value == "":
        return None
    try:
        parsed = (value if isinstance(value, datetime)
                  else datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@lru_cache(maxsize=2)
def _fotmob_names_from_db(_time_bucket: int) -> dict[str, list[str]]:
    """Use committed stats when the collector's JSON is on another host."""
    db = SessionLocal()
    try:
        rows = db.query(
            TeamSeasonStats.team, TeamSeasonStats.league,
        ).distinct().all()
        names: dict[str, list[str]] = {}
        for team, league in rows:
            if team and league:
                league_names = names.setdefault(league, [])
                if team not in league_names:
                    league_names.append(team)
        return names
    finally:
        db.close()


def _load_fotmob_names() -> dict[str, list[str]]:
    """Load FotMob names from JSON, or the DB if the JSON is not deployed."""
    try:
        if _FOTMOB_NAMES_FILE.exists():
            names = json.loads(_FOTMOB_NAMES_FILE.read_text())
            if isinstance(names, dict) and names:
                return names
    except (OSError, ValueError):
        pass
    try:
        # The collector runs in CI; its local JSON may not reach the API host.
        return _fotmob_names_from_db(int(time.monotonic() // 300))
    except Exception as exc:
        print(f"[fixtures] WARNING: could not load FotMob names: {exc}")
        return {}


def _find_fotmob_name(
    name: str,
    league: str,
    fotmob_names: dict,
) -> str:
    """Resolve a fixture spelling to a FotMob name in its league.

    Try exact, case-insensitive, suffix-stripped, partial and token matches,
    in that order. Ambiguous matches prefer the longest candidate; unknown
    names are left unchanged.
    """
    candidates = fotmob_names.get(league, [])
    if not isinstance(candidates, list):
        return name
    candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
    if name in candidates:
        return name

    name_lower = name.casefold()
    for candidate in candidates:
        if candidate.casefold() == name_lower:
            return candidate

    def clean(value: str) -> str:
        value = " ".join(value.casefold().split())
        suffixes = (
            " fc", " afc", " cf", " sc", " city", " united", " town",
            " rovers", " wanderers", " athletic", " albion", " hotspur",
            " county",
        )
        # Only remove trailing club suffixes, not those inside a team name.
        while any(value.endswith(suffix) for suffix in suffixes):
            value = next(
                value.removesuffix(suffix).strip()
                for suffix in suffixes if value.endswith(suffix)
            )
        return value

    name_clean = clean(name)
    if not name_clean:
        return name

    def unique(matches: list[str]) -> str | None:
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return max(matches, key=len)
        return None

    clean_matches = [c for c in candidates if clean(c) == name_clean]
    if clean_matches:
        return unique(clean_matches) or name

    partial_matches = [
        c for c in candidates
        if clean(c) and (clean(c) in name_clean or name_clean in clean(c))
    ]
    if partial_matches:
        return unique(partial_matches) or name

    name_tokens = set(name_clean.replace("-", " ").split())
    if len(name_tokens) >= 2:
        token_matches = []
        for candidate in candidates:
            tokens = set(clean(candidate).replace("-", " ").split())
            overlap = name_tokens & tokens
            if len(overlap) >= 2 and len(overlap) == min(
                len(name_tokens), len(tokens)
            ):
                token_matches.append(candidate)
        if token_matches:
            return unique(token_matches) or name

    # Spellings whose provider name shares no token with the stored one:
    # abbreviation ("PSG"), translated name ("Bayern Munich" vs "Bayern
    # München"), or a different word order ("Feyenoord Rotterdam"). UCL
    # fixtures hit these hardest, because football-data.org re-uses the
    # national feed's name for the same club FotMob lists once.
    KNOWN_ALIASES = {
        # England
        "man city":                  "Manchester City",
        "man united":                "Manchester United",
        "man utd":                   "Manchester United",
        "qpr":                       "Queens Park Rangers",
        "west brom":                 "West Bromwich Albion",
        "spurs":                     "Tottenham",
        "wolves":                    "Wolverhampton Wanderers",
        # Spain
        "atletico":                  "Atletico Madrid",
        "atletico madrid":           "Atlético Madrid",
        "atlético madrid":           "Atlético Madrid",
        "atletico de madrid":        "Atlético Madrid",
        "club atletico de madrid":   "Atlético Madrid",
        "athletic bilbao":           "Athletic Club",
        "betis":                     "Real Betis",
        # Germany
        "bayern munich":             "Bayern München",
        "fc bayern munich":          "Bayern München",
        "bayern munchen":            "Bayern München",
        "fc bayern munchen":         "Bayern München",
        "bvb":                       "Borussia Dortmund",
        "dortmund":                  "Borussia Dortmund",
        "rb leipzig":                "RB Leipzig",
        "rasenballsport leipzig":    "RB Leipzig",
        "leverkusen":                "Bayer Leverkusen",
        "bayer 04 leverkusen":       "Bayer Leverkusen",
        "borussia m.gladbach":       "Borussia Mönchengladbach",
        "gladbach":                  "Borussia Mönchengladbach",
        "frankfurt":                 "Eintracht Frankfurt",
        "stuttgart":                 "VfB Stuttgart",
        # Italy
        "inter":                     "Inter",
        "inter milan":               "Inter",
        "internazionale":            "Inter",
        "ac milan":                  "AC Milan",
        "milan":                     "AC Milan",
        "roma":                      "AS Roma",
        "lazio":                     "SS Lazio",
        "napoli":                    "SSC Napoli",
        "atalanta":                  "Atalanta BC",
        "bologna":                   "Bologna FC 1909",
        "fiorentina":                "ACF Fiorentina",
        # France
        "psg":                       "Paris Saint-Germain",
        "paris sg":                  "Paris Saint-Germain",
        "paris saint germain":       "Paris Saint-Germain",
        "paris saint-germain":       "Paris Saint-Germain",
        "racing club de lens":       "Lens",
        "rc lens":                   "Lens",
        "olympique de marseille":    "Marseille",
        "as monaco":                 "Monaco",
        "losc lille":                "Lille",
        "olympique lyonnais":        "Lyon",
        "ogc nice":                  "Nice",
        "stade brestois 29":         "Brest",
        "stade rennais fc":          "Rennes",
        # Portugal
        "sporting clube de portugal": "Sporting CP",
        "sl benfica":                "Benfica",
        # Netherlands
        "feyenoord rotterdam":       "Feyenoord",
        "afc ajax":                  "Ajax",
        "psv eindhoven":             "PSV",
        "az alkmaar":                "AZ",
        # Belgium
        "club brugge kv":            "Club Brugge",
        "rsc anderlecht":            "Anderlecht",
        "krc genk":                  "Genk",
        "kaa gent":                  "Gent",
        "union st.gilloise":         "Union St.Gilloise",
        "union saint-gilloise":      "Union St.Gilloise",
        "royal antwerp":             "Antwerp",
        # Turkey
        "fenerbahce":                "Fenerbahçe",
        "fenerbahçe sk":             "Fenerbahçe",
        "galatasaray sk":            "Galatasaray",
        "besiktas":                  "Beşiktaş",
        "beşiktaş jk":               "Beşiktaş",
        "istanbul basaksehir":       "Başakşehir",
        # Norway
        "bodo/glimt":                "Bodø/Glimt",
        "fk bodo/glimt":             "Bodø/Glimt",
        "viking fk":                 "Viking",
        "rosenborg bk":              "Rosenborg",
        "molde fk":                  "Molde",
        # Austria
        "lask linz":                 "LASK",
        "fc lask linz":              "LASK",
        "red bull salzburg":         "FC Salzburg",
        "fc red bull salzburg":      "FC Salzburg",
        "rb salzburg":               "FC Salzburg",
        "sk rapid wien":             "Rapid Wien",
        "sk sturm graz":             "Sturm Graz",
    }
    alias_result = KNOWN_ALIASES.get(name_lower)
    if alias_result:
        # An alias only counts when the league really holds that club: the
        # list is global, the candidates are this league's teams. Matching is
        # on the relaxed name (accents and punctuation) and, failing that, on
        # whole tokens, so "Tottenham" still finds "Tottenham Hotspur" while
        # never colliding two different clubs that share a word.
        alias_key = alias_result.casefold()
        alias_tokens = set(_relaxed_name(alias_result).split())

        exact = [c for c in candidates if c.casefold() == alias_key]
        if exact:
            return exact[0]

        relaxed = [
            c for c in candidates
            if _relaxed_name(c) == _relaxed_name(alias_result)
        ]
        if relaxed:
            return unique(relaxed) or name

        if alias_tokens:
            token_matches = []
            for candidate in candidates:
                tokens = set(_relaxed_name(candidate).split())
                if tokens and (
                    alias_tokens <= tokens or tokens <= alias_tokens
                ):
                    token_matches.append(candidate)
            if token_matches:
                return unique(token_matches) or name

    return name


def _fixture_values_to_nested(values: dict) -> dict:
    """Convert flat pred_* fixture values into the nested shape expected by _blend_predictions."""
    return {
        "result": {
            "home_win": values["pred_home_win"],
            "draw": values["pred_draw"],
            "away_win": values["pred_away_win"],
        },
        "goals": {
            "expected_home": values["pred_home_goals"],
            "expected_away": values["pred_away_goals"],
            "over_2_5": values.get("pred_over_2_5"),
            "btts": values.get("pred_btts"),
        },
        "corners": {
            "expected_home": values["pred_home_corners"],
            "expected_away": values["pred_away_corners"],
            "over_9_5": values.get("pred_corners_over_9_5"),
        },
    }


def _predict_fixture_values(
    home_team: str,
    away_team: str,
    league: str | None,
    season: str | None,
) -> dict:
    """
    Best available prediction for one fixture.

    Resolve team names to FotMob spellings before asking the formula engine.
    A UCL fixture also gets each club resolved in its own domestic league:
    the home one is the baseline the engine predicts from, the away one is
    where the away club's own stats, squad and key players are read from.
    When both formula and ML predictions are available, blends them with
    _blend_predictions; otherwise returns whichever engine succeeded.
    """
    # Preserve the tournament before replacing it with each team's
    # domestic league for team-name resolution and domestic stats.
    fixture_league = league
    fotmob_names = _load_fotmob_names()
    away_dom: str | None = None
    if fixture_league == UCL_LEAGUE:
        home_dom = _get_ucl_domestic_league(home_team)
        away_dom = _get_ucl_domestic_league(away_team)
        if not home_dom or not away_dom:
            print(f"[fixtures] UCL: skipping {home_team} vs {away_team} "
                  "— domestic league not found")
            return {}
        home_team = _find_fotmob_name(home_team, home_dom, fotmob_names)
        away_team = _find_fotmob_name(away_team, away_dom, fotmob_names)
        league = home_dom
    else:
        home_team = _find_fotmob_name(home_team, league, fotmob_names)
        away_team = _find_fotmob_name(away_team, league, fotmob_names)

    if fixture_league == UCL_LEAGUE:
        formula_result = _predict_with_formula(
            home_team, away_team, league, season,
            original_league=fixture_league,
            away_league=away_dom,
        )
    else:
        formula_result = _predict_with_formula(
            home_team, away_team, league, season
        )

    ml_result = None
    # Use current predictor state: model artifacts may load after this module.
    features_df = _current_predictor_value("_features_df", _features_df)
    if (
        _ml_models_available()
        and features_df is not None
        and {"home_team", "away_team"}.issubset(features_df.columns)
    ):
        home_known = bool((
            (features_df["home_team"] == home_team)
            | (features_df["away_team"] == home_team)
        ).any())
        away_known = bool((
            (features_df["home_team"] == away_team)
            | (features_df["away_team"] == away_team)
        ).any())
        if home_known and away_known:
            try:
                ml_result = _compute_prediction_values(
                    home_team, away_team, league, season or DEFAULT_SEASON
                )
            except Exception as exc:
                print(f"[fixtures] WARNING: prediction failed for "
                      f"{home_team} vs {away_team}: {exc}")
                traceback.print_exc()

    if formula_result and ml_result:
        try:
            blended = _blend_predictions(
                _fixture_values_to_nested(formula_result),
                _fixture_values_to_nested(ml_result),
            )
            blended_values = {
                "pred_home_win": blended["result"]["home_win"],
                "pred_draw": blended["result"]["draw"],
                "pred_away_win": blended["result"]["away_win"],
                "pred_home_goals": blended["goals"]["expected_home"],
                "pred_away_goals": blended["goals"]["expected_away"],
                "pred_home_corners": blended["corners"]["expected_home"],
                "pred_away_corners": blended["corners"]["expected_away"],
                "pred_over_2_5": blended["goals"]["over_2_5"],
                "pred_btts": blended["goals"]["btts"],
                "pred_corners_over_9_5": blended["corners"]["over_9_5"],
            }
            print(f"[fixtures] Blended formula + ML: {home_team} vs {away_team}")
            return _with_derived_markets(blended_values)
        except Exception as exc:
            print(f"[fixtures] WARNING: blend failed for "
                  f"{home_team} vs {away_team}: {exc} — using formula")
            return formula_result

    if formula_result:
        return formula_result

    if ml_result:
        print(f"[fixtures] ML fallback: {home_team} vs {away_team}")
        return ml_result

    print(f"[fixtures] No data: {home_team} vs {away_team}")
    return {}


def _register_derived_columns() -> None:
    """
    Map the derived-market columns onto DailyMatch.

    ``daily_matches`` was created before over/under, BTTS and corners
    markets were part of the fixtures response, so the columns are attached
    to the model here (database.py is not touched). Only called once the
    physical columns are known to exist, which keeps every INSERT and UPDATE
    valid even when the table cannot be migrated.
    """
    global _derived_columns_registered
    if _derived_columns_registered:
        return

    for name in DERIVED_PREDICTION_FIELDS:
        if name in DailyMatch.__table__.columns:
            continue
        column = Column(name, Float, nullable=True)
        DailyMatch.__table__.append_column(column)
        if add_mapped_attribute is not None:
            add_mapped_attribute(DailyMatch, name, column)
        else:  # pragma: no cover - SQLAlchemy 1.4
            DailyMatch.__mapper__.add_property(name, column)

    _derived_columns_registered = True


def _clear_statement_caches(db: Session) -> None:
    """
    Drop statements that were compiled before the columns were mapped.

    SQLAlchemy caches a compiled SELECT per query shape, so a shape compiled
    while daily_matches still lacked the derived-market columns would keep
    loading rows without them for the lifetime of the process. Clearing the
    caches is safe — every statement is recompiled on demand — and runs once,
    right after registration.
    """
    seen: set[int] = set()
    bind = None
    try:
        bind = db.get_bind()
    except Exception:  # pragma: no cover - defensive
        bind = None

    for cache in (
        getattr(bind, "_compiled_cache", None),
        getattr(DailyMatch.__mapper__, "_compiled_cache", None),
        getattr(getattr(bind, "engine", None), "_compiled_cache", None),
    ):
        clear = getattr(cache, "clear", None)
        if cache is None or id(cache) in seen or not callable(clear):
            continue
        seen.add(id(cache))
        try:
            clear()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[fixtures] NOTE: could not clear a SQLAlchemy "
                  f"statement cache: {exc}")


def _existing_columns(db: Session) -> set[str] | None:
    """Column names of daily_matches, or None when it cannot be inspected."""
    try:
        inspector = inspect(db.get_bind())
        return {
            column["name"]
            for column in inspector.get_columns(DailyMatch.__tablename__)
        }
    except Exception as exc:
        print(f"[fixtures] NOTE: could not inspect "
              f"{DailyMatch.__tablename__}: {exc}")
        return None


def ensure_derived_prediction_columns(db: Session) -> bool:
    """
    Make sure daily_matches has columns for the derived markets.

    Idempotent, and only ever run once per process: the columns are created
    with ``ALTER TABLE ... ADD COLUMN`` the first time a fixtures request
    needs them and then mapped onto the model, so both the fixture inserts
    and the stored predictions carry every field the frontend reads.

    Every failure is non-fatal — the caller keeps working and the markets are
    computed on the fly, they are just not persisted.
    """
    if _derived_columns_registered:
        return True

    existing = _existing_columns(db)
    if existing is None:
        return False

    missing = [
        name for name in DERIVED_PREDICTION_FIELDS if name not in existing
    ]

    if missing:
        try:
            for name in missing:
                db.execute(
                    text(
                        f"ALTER TABLE {DailyMatch.__tablename__} "
                        f"ADD COLUMN {name} FLOAT"
                    )
                )
            db.commit()
        except Exception as exc:
            db.rollback()
            # Another worker may have added the column in the meantime, so
            # re-check before giving up.
            still_missing = [
                name
                for name in DERIVED_PREDICTION_FIELDS
                if name not in (_existing_columns(db) or set())
            ]
            if still_missing:
                print(f"[fixtures] WARNING: could not add "
                      f"{still_missing} to {DailyMatch.__tablename__}: {exc} "
                      f"— predictions will still be returned, but not "
                      f"persisted")
                return False

    _register_derived_columns()
    _clear_statement_caches(db)
    return True


def _apply_prediction_values(row: DailyMatch, values: dict) -> None:
    """
    Write prediction values onto a row.

    Mapped fields (the seven daily_matches columns, plus the derived markets
    once their columns exist) are persisted on the next commit; a derived
    market without a column is still set as an attribute, so the response
    built from this row carries it too.
    """
    for name in PREDICTION_FIELDS:
        value = values.get(name)
        if value is None:
            continue
        setattr(row, name, value)


def compute_and_store_predictions(fixtures: list, db: Session) -> None:
    """Compute predictions for unseen fixtures and persist every valid fixture."""
    ensure_derived_prediction_columns(db)

    for fixture in fixtures:
        home_team = fixture.get("home_team", "Unknown")
        away_team = fixture.get("away_team", "Unknown")
        try:
            external_id = fixture.get("external_id")
            if external_id is None:
                raise ValueError("fixture has no external_id")
            external_id = str(external_id)

            existing = (
                db.query(DailyMatch)
                .filter(DailyMatch.external_id == external_id)
                .first()
            )
            if existing is not None:
                continue
        except Exception as exc:
            db.rollback()
            print(
                f"[fixtures] WARNING: could not check {home_team} vs "
                f"{away_team}: {exc}"
            )
            traceback.print_exc()
            continue

        predictions = _predict_fixture_values(
            home_team=home_team,
            away_team=away_team,
            league=fixture.get("league"),
            season=fixture.get("season"),
        )

        try:
            row = DailyMatch(
                external_id=external_id,
                match_date=fixture.get("match_date"),
                kickoff_time=fixture.get("kickoff_time"),
                home_team=home_team,
                away_team=away_team,
                league=fixture.get("league"),
                season=fixture.get("season"),
                status=fixture.get("status"),
                **_mapped_prediction_values(predictions),
                prediction_computed_at=datetime.now(timezone.utc).isoformat(),
            )
            db.add(row)
            db.commit()
        except Exception as exc:
            db.rollback()
            print(
                f"[fixtures] WARNING: could not store {home_team} vs "
                f"{away_team}: {exc}"
            )
            traceback.print_exc()


def fill_missing_predictions(rows: list[DailyMatch], db: Session) -> int:
    """
    Compute and store the predictions of cached fixtures that have none.

    ``/fixtures/today`` serves stored rows, so a fixture cached before the
    predictor could answer it (models still loading, no team stats for a
    promoted club yet) used to stay prediction-less forever. Every fixture
    whose 1X2 probabilities are still NULL gets a formula-engine prediction
    stored back into ``daily_matches``; fixtures that only miss the derived
    markets (rows the ML models wrote) get those filled from the expectations
    already stored with them.

    Each row is committed on its own, so one unusable fixture cannot roll
    back the predictions already stored for the others. A fixture whose
    prediction still cannot be produced is left exactly as it is — its
    ``prediction_computed_at`` included. Returns how many rows actually
    gained a prediction value.
    """
    if not rows:
        return 0

    ensure_derived_prediction_columns(db)
    filled = 0

    for row in rows:
        try:
            values = _prediction_values(row)
            if _has_result_prediction(values) and all(
                values.get(name) is not None
                for name in DERIVED_PREDICTION_FIELDS
            ):
                continue

            if not _has_result_prediction(values):
                computed = _predict_fixture_values(
                    home_team=row.home_team,
                    away_team=row.away_team,
                    league=row.league,
                    season=row.season,
                )
                for name in PREDICTION_FIELDS:
                    if values.get(name) is None:
                        values[name] = computed.get(name)

            _with_derived_markets(values)

            if not _has_result_prediction(values):
                print(f"[fixtures] WARNING: no prediction available for "
                      f"{row.home_team} vs {row.away_team} — leaving the "
                      f"fixture without one")
                continue

            # Only values that actually change are written back. A row that
            # still misses a market (no expected corners to derive one from,
            # say) keeps its old timestamp: stamping it would report a fill
            # that never happened and would keep the day looking fresh, so
            # the staleness check could never refresh it again.
            added = {
                name: value
                for name, value in values.items()
                if value is not None
                and getattr(row, name, None) != value
            }
            if not added:
                continue

            _apply_prediction_values(row, added)
            row.prediction_computed_at = datetime.now(timezone.utc).isoformat()
            db.commit()
            filled += 1
        except Exception as exc:
            db.rollback()
            print(
                f"[fixtures] WARNING: prediction failed for "
                f"{row.home_team} vs {row.away_team}: {exc}"
            )
            traceback.print_exc()
            continue

    return filled


def _row_to_dict(row: DailyMatch) -> dict:
    """Serialize a DailyMatch row using the fixtures API response shape."""
    data = {
        "id": row.id,
        "external_id": row.external_id,
        "match_date": row.match_date,
        "kickoff_time": row.kickoff_time,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "league": row.league,
        "season": row.season,
        "status": row.status,
    }

    values = _prediction_values(row)
    _with_derived_markets(values)
    for name in PREDICTION_FIELDS:
        data[name] = values.get(name)

    # Daily fixtures store their own (possibly blended) expected goals. Derive
    # this additive market without changing or writing any stored fields. An
    # absent or invalid rate is not evidence of a guaranteed 0-0 draw.
    home_goals = values["pred_home_goals"]
    away_goals = values["pred_away_goals"]
    if all(
        rate is not None and math.isfinite(rate) and rate >= 0
        for rate in (home_goals, away_goals)
    ):
        data["first_goal"] = compute_first_goal_probs(home_goals, away_goals)

    data["prediction_computed_at"] = row.prediction_computed_at
    data["has_prediction"] = (
        row.pred_home_win is not None and
        row.pred_draw is not None and
        row.pred_away_win is not None
    )
    return data


def _late_mls_fixtures(previous_date_str: str) -> list[dict]:
    """Yesterday's USA-MLS fixtures, for the late-UTC spillover pass.

    Asks FotMob for the MLS rows directly. The previous implementation called
    ``fetch_fixtures`` here, which re-fetched every football-data.org
    competition for that date just to keep the USA-MLS subset — one provider
    request per league on every ``/fixtures/today`` call, and the 429s the
    rate limiter in front of this endpoint could not absorb.
    """
    if fetch_fixtures_for_date is not None:
        return fetch_fixtures_for_date(previous_date_str) or []

    # FotMob collector unavailable: fall back to the full feed, filtered to
    # MLS, so the spillover still works at its old (expensive) cost.
    return [
        fixture
        for fixture in (fetch_fixtures(previous_date_str) or [])
        if isinstance(fixture, dict) and fixture.get("league") == "USA-MLS"
    ]


# Dates whose late USA-MLS spillover has already been chased in this process.
# A day with no MLS matches legitimately has no rows, so re-running the fetch
# for it on every /fixtures/today call was pure wasted provider traffic; one
# attempt per date per process is enough to pick up a transient failure on a
# later day.
_MLS_SPILLOVER_ATTEMPTED: set[str] = set()


def _load_matches(
    target_date: str,
    db: Session,
    *,
    fetch_spillover: bool = True,
) -> list[DailyMatch]:
    """Every stored fixture for one date, plus late-UTC USA-MLS spillover.

    USA-MLS kickoffs from 22:00 UTC onwards are stored under the previous
    day's ``match_date``, so without this pass they would never appear in
    the response for the day they are played. The previous day's late
    fixtures are therefore merged in as well: rows whose ``match_date`` is
    the day before ``target_date``, whose league is USA-MLS and whose
    kickoff is at or after 22:00 UTC that previous day. No other league is
    affected.
    """
    rows = (
        db.query(DailyMatch)
        .filter(DailyMatch.match_date == target_date)
        .all()
    )

    # Second query: the previous day's late-UTC USA-MLS fixtures. Any
    # failure here leaves the date's own fixtures untouched.
    try:
        previous_date = date.fromisoformat(target_date) - timedelta(days=1)
        previous_date_str = previous_date.isoformat()
        kickoff_cutoff = datetime(
            previous_date.year, previous_date.month, previous_date.day,
            22, 0, tzinfo=timezone.utc,
        )
        spillover = (
            db.query(DailyMatch)
            .filter(DailyMatch.match_date == previous_date_str)
            .filter(DailyMatch.league == "USA-MLS")
            .all()
        )
        if (
            not spillover
            and fetch_spillover
            and previous_date_str not in _MLS_SPILLOVER_ATTEMPTED
        ):
            # Marked before fetching: a day that genuinely has no MLS
            # fixtures must not re-cost a provider request on every call.
            _MLS_SPILLOVER_ATTEMPTED.add(previous_date_str)
            try:
                previous_fixtures = _late_mls_fixtures(previous_date_str)
                usa_mls_fixtures = [
                    {
                        **fixture,
                        "match_date": (
                            fixture.get("match_date") or previous_date_str
                        ),
                    }
                    for fixture in (previous_fixtures or [])
                    if isinstance(fixture, dict)
                    and fixture.get("league") == "USA-MLS"
                ]
                compute_and_store_predictions(usa_mls_fixtures, db)
                spillover = (
                    db.query(DailyMatch)
                    .filter(DailyMatch.match_date == previous_date_str)
                    .filter(DailyMatch.league == "USA-MLS")
                    .all()
                )
            except Exception as exc:
                print(
                    f"[fixtures] WARNING: could not fetch or store late "
                    f"USA-MLS fixtures for {previous_date_str}: {exc}"
                )
                traceback.print_exc()

        # Parse before comparing: stored strings may differ in format
        # ("...Z" vs "...+00:00"), same as _fixtures_are_stale.
        late_usa_mls = [
            row
            for row in spillover
            if (kickoff := _parse_utc(row.kickoff_time)) is not None
            and kickoff >= kickoff_cutoff
        ]
    except Exception as exc:
        print(f"[fixtures] WARNING: could not load late USA-MLS fixtures "
              f"from the day before {target_date}: {exc}")
        traceback.print_exc()
        return rows

    # Merge without duplicates: a fixture is identified by external_id.
    merged = list(rows)
    seen_external_ids = {row.external_id for row in merged}
    for row in late_usa_mls:
        if row.external_id in seen_external_ids:
            continue
        seen_external_ids.add(row.external_id)
        merged.append(row)
    return merged


def _fixtures_are_stale(rows: list) -> bool:
    """
    Returns True if fixtures should be
    refreshed — when the newest
    prediction_computed_at is more than
    FIXTURE_STALE_HOURS old or missing.
    """
    if not rows:
        return False
    # Parse before comparing: stored strings may differ in format.
    timestamps = [
        parsed
        for parsed in (_parse_utc(r.prediction_computed_at) for r in rows)
        if parsed is not None
    ]
    if not timestamps:
        return True
    newest = max(timestamps)
    age = datetime.now(timezone.utc) - newest
    return age > timedelta(hours=FIXTURE_STALE_HOURS)


def _has_open_fixtures(rows: list) -> bool:
    """True if any stored fixture has not kicked off yet."""
    now = datetime.now(timezone.utc)
    return any(
        (kickoff := _parse_utc(r.kickoff_time)) is not None and kickoff > now
        for r in rows
    )


def refresh_open_predictions(
    rows: list[DailyMatch],
    fresh_fixtures: list,
    db: Session,
) -> int:
    """
    Re-predict fixtures that have not kicked off yet, using the latest stats.

    ``compute_and_store_predictions`` skips fixtures already stored, so a
    refresh needs this separate pass. Status and kickoff come from the
    fresh feed. A fixture that has kicked off keeps the prediction it was
    made with. A fixture whose new prediction is unavailable keeps its old
    one. Every open fixture is stamped as checked, so the staleness check
    does not re-run predictions on every request. Returns how many
    predictions were replaced.
    """
    fresh_by_id = {
        str(f.get("external_id")): f for f in fresh_fixtures
        if f.get("external_id") is not None
    }
    now = datetime.now(timezone.utc)
    refreshed = 0

    for row in rows:
        try:
            fresh = fresh_by_id.get(str(row.external_id))
            if fresh:
                if fresh.get("status"):
                    row.status = fresh["status"]
                if fresh.get("kickoff_time"):
                    row.kickoff_time = fresh["kickoff_time"]

            kickoff = _parse_utc(row.kickoff_time)
            if kickoff is None or kickoff <= now:
                db.commit()
                continue

            values = _predict_fixture_values(
                home_team=row.home_team,
                away_team=row.away_team,
                league=row.league,
                season=row.season,
            )
            if _has_result_prediction(values):
                _apply_prediction_values(row, values)
                refreshed += 1
            row.prediction_computed_at = now.isoformat()
            db.commit()
        except Exception as exc:
            db.rollback()
            print(f"[fixtures] WARNING: refresh failed for "
                  f"{row.home_team} vs {row.away_team}: {exc}")
            traceback.print_exc()

    return refreshed


def _get_or_fetch_matches(target_date: str, db: Session) -> list[dict]:
    """
    Return the stored matches for a date, predicting and caching the ones
    that are missing — either the whole day, or just its predictions.
    """
    ensure_derived_prediction_columns(db)

    rows = _load_matches(target_date, db)
    # ``_load_matches`` also merges in the previous day's late-UTC USA-MLS
    # spillover, so ``rows`` can be non-empty even when this date has no
    # stored fixtures of its own yet. The cache checks below are about the
    # date's own fixtures only, so a day without any is still fetched.
    day_rows = [row for row in rows if row.match_date == target_date]
    if not day_rows:
        fixtures = fetch_fixtures(target_date)
        compute_and_store_predictions(fixtures, db)
        rows = _load_matches(target_date, db, fetch_spillover=False)
    elif _fixtures_are_stale(day_rows) and _has_open_fixtures(day_rows):
        # Only dates with fixtures still to play are refreshed. Re-predicting
        # a finished match would overwrite the forecast it was judged on.
        print(f"[fixtures] Stale — refreshing "
              f"{target_date}...")
        new_fixtures = fetch_fixtures(target_date)
        compute_and_store_predictions(new_fixtures, db)
        refresh_open_predictions(day_rows, new_fixtures, db)
        rows = _load_matches(target_date, db, fetch_spillover=False)

    # Always try to fill missing predictions regardless of the staleness
    # timer: a fixture cached while no prediction could be produced (models
    # still loading, no team stats yet) gets one on the next API call, even
    # if the day's stored predictions are otherwise fresh.
    filled = fill_missing_predictions(rows, db)
    if filled:
        print(f"[fixtures] Filled {filled} "
              f"missing predictions")
        rows = _load_matches(target_date, db, fetch_spillover=False)

    return [_row_to_dict(row) for row in rows]


def _close_session(db) -> None:
    if db is not None:
        try:
            db.close()
        except Exception as exc:
            print(f"[fixtures] WARNING: could not close DB session: {exc}")


@router.get("/today")
def get_today_fixtures() -> list[dict]:
    """Return today's UTC fixtures, fetching them if they are not cached."""
    db = None
    try:
        target_date = datetime.now(timezone.utc).date().isoformat()
        db = SessionLocal()
        return _get_or_fetch_matches(target_date, db)
    except Exception as exc:
        print(f"[fixtures] WARNING: could not load today's fixtures: {exc}")
        traceback.print_exc()
        return []
    finally:
        _close_session(db)


@router.get("/date")
def get_fixtures_for_date(
    target_date: str = Query(..., alias="date", description="Date in YYYY-MM-DD format"),
) -> list[dict]:
    """Return fixtures for the requested date, fetching them if not cached."""
    db = None
    try:
        parsed_date = date.fromisoformat(target_date)
        if parsed_date.isoformat() != target_date:
            raise ValueError("date must be in YYYY-MM-DD format")
        db = SessionLocal()
        return _get_or_fetch_matches(target_date, db)
    except Exception as exc:
        print(f"[fixtures] WARNING: could not load fixtures for {target_date}: {exc}")
        traceback.print_exc()
        return []
    finally:
        _close_session(db)
