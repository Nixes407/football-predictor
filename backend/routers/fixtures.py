"""Endpoints for daily football fixtures and their stored predictions."""

import json
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import routers.predict as predict_module
from fastapi import APIRouter, Query
from sqlalchemy import Column, Float, inspect, text
from sqlalchemy.orm import Session

from data.fixtures import fetch_fixtures
from database import DailyMatch, SessionLocal, TeamSeasonStats
from routers.predict import (
    _away_corners,
    _away_goals,
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

# Season used when a stored fixture carries none (matches the default of
# predict.MatchRequest). The formula engine resolves it to the newest season
# the league actually has data for.
DEFAULT_SEASON = "2024-2025"

UCL_LEAGUE = "UEFA-Champions League"

UCL_TEAM_DOMESTIC_LEAGUE = {
    # England
    "Arsenal":              "ENG-Premier League",
    "Aston Villa":          "ENG-Premier League",
    "Chelsea":              "ENG-Premier League",
    "Liverpool":            "ENG-Premier League",
    "Manchester City":      "ENG-Premier League",
    "Manchester United":    "ENG-Premier League",
    "Newcastle United":     "ENG-Premier League",
    "Tottenham":            "ENG-Premier League",
    # Spain
    "Real Madrid":          "ESP-La Liga",
    "Barcelona":            "ESP-La Liga",
    "Atletico Madrid":      "ESP-La Liga",
    "Atlético de Madrid":   "ESP-La Liga",
    "Real Betis":           "ESP-La Liga",
    "Betis":                "ESP-La Liga",          # football-data.org normaliser
    "Real Betis Balompié":  "ESP-La Liga",
    "Real Sociedad":        "ESP-La Liga",
    "Athletic Club":        "ESP-La Liga",
    "Girona":               "ESP-La Liga",
    "Sevilla":              "ESP-La Liga",
    "Villarreal":           "ESP-La Liga",
    "Villarreal CF":        "ESP-La Liga",
    "Club Atletico de Madrid":  "ESP-La Liga",
    # Germany
    "Bayern Munchen":       "GER-Bundesliga",
    "Bayern München":       "GER-Bundesliga",
    "Bayern Munich":        "GER-Bundesliga",
    "Borussia Dortmund":    "GER-Bundesliga",
    "Dortmund":             "GER-Bundesliga",       # football-data.org normaliser
    "RB Leipzig":           "GER-Bundesliga",
    "VfB Stuttgart":        "GER-Bundesliga",
    "Stuttgart":            "GER-Bundesliga",       # football-data.org normaliser
    "Leverkusen":           "GER-Bundesliga",
    "Bayer Leverkusen":     "GER-Bundesliga",
    "Bayer 04 Leverkusen":  "GER-Bundesliga",
    "FC Bayern München":    "GER-Bundesliga",
    "Borussia Mönchengladbach": "GER-Bundesliga",
    "Gladbach":             "GER-Bundesliga",       # football-data.org normaliser
    "Frankfurt":            "GER-Bundesliga",
    "Union Berlin":         "GER-Bundesliga",
    "Wolfsburg":            "GER-Bundesliga",
    # Italy
    "Internazionale":       "ITA-Serie A",
    "Inter":                "ITA-Serie A",
    "Inter Milan":          "ITA-Serie A",
    "Napoli":               "ITA-Serie A",
    "AS Roma":              "ITA-Serie A",
    "Roma":                 "ITA-Serie A",
    "Como":                 "ITA-Serie A",
    "Juventus":             "ITA-Serie A",
    "Milan":                "ITA-Serie A",
    "AC Milan":             "ITA-Serie A",
    "Atalanta":             "ITA-Serie A",
    "Bologna":              "ITA-Serie A",
    "Lazio":                "ITA-Serie A",
    "SS Lazio":             "ITA-Serie A",
    "SSC Napoli":           "ITA-Serie A",
    "Juventus FC":          "ITA-Serie A",
    "Atalanta BC":          "ITA-Serie A",
    "Bologna FC 1909":      "ITA-Serie A",
    # France
    "Paris Saint-Germain":  "FRA-Ligue 1",
    "Paris Saint Germain":  "FRA-Ligue 1",
    "Paris SG":             "FRA-Ligue 1",          # football-data.org normaliser
    "RC Lens":              "FRA-Ligue 1",
    "Lens":                 "FRA-Ligue 1",
    "Lille":                "FRA-Ligue 1",
    "LOSC Lille":           "FRA-Ligue 1",
    "Monaco":               "FRA-Ligue 1",
    "AS Monaco":            "FRA-Ligue 1",
    "Marseille":            "FRA-Ligue 1",
    "Lyon":                 "FRA-Ligue 1",
    "Rennes":               "FRA-Ligue 1",
    "Racing Club de Lens":  "FRA-Ligue 1",
    "RC Strasbourg":        "FRA-Ligue 1",
    "Stade Brestois 29":    "FRA-Ligue 1",
    # Portugal
    "FC Porto":             "POR-Primeira Liga",
    "Porto":                "POR-Primeira Liga",
    "Sporting CP":          "POR-Primeira Liga",
    "Benfica":              "POR-Primeira Liga",
    "SL Benfica":           "POR-Primeira Liga",
    "Braga":                "POR-Primeira Liga",
    "SC Braga":             "POR-Primeira Liga",
    "Sporting Clube de Portugal": "POR-Primeira Liga",
    # Netherlands
    "Feyenoord":            "NED-Eredivisie",
    "PSV Eindhoven":        "NED-Eredivisie",
    "PSV":                  "NED-Eredivisie",
    "Ajax":                 "NED-Eredivisie",
    "AFC Ajax":             "NED-Eredivisie",
}

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


def _predict_with_formula(
    home_team: str,
    away_team: str,
    league: str,
    season: str | None,
) -> dict | None:
    """
    Prediction values from the formula engine, or None when it cannot
    produce one (engine unavailable, unknown teams, too little data).

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
              f"season={(season or DEFAULT_SEASON)!r}")
        prediction = formula_predict(
            home_team=home_team,
            away_team=away_team,
            league=league,
            season=season or DEFAULT_SEASON,
        )
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

    Returns None if the team is from a league
    we do not cover — those fixtures are hidden.
    """
    return UCL_TEAM_DOMESTIC_LEAGUE.get(team)


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
    """Resolve a fixture spelling to an unambiguous FotMob name in its league.

    Try exact, case-insensitive, suffix-stripped, partial and token matches,
    in that order. Leave unknown or ambiguous names unchanged.
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
        return matches[0] if len(matches) == 1 else None

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
        return unique(token_matches) or name

    return name


def _predict_fixture_values(
    home_team: str,
    away_team: str,
    league: str | None,
    season: str | None,
) -> dict:
    """
    Best available prediction for one fixture.

    Resolve team names to FotMob spellings before asking the formula engine.
    ML is only a fallback when both teams occur in the training history;
    otherwise return no prediction rather than a default-valued forecast.
    """
    # Dynamically resolve team names to FotMob spelling. For UCL matches,
    # find each side's domestic league before resolving its name.
    fotmob_names = _load_fotmob_names()
    if league == UCL_LEAGUE:
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

    formula_result = _predict_with_formula(
        home_team, away_team, league, season
    )
    if formula_result:
        return formula_result

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
            print(f"[fixtures] ML fallback: {home_team} vs {away_team}")
            try:
                return _compute_prediction_values(
                    home_team, away_team, league, season or DEFAULT_SEASON
                )
            except Exception as exc:
                print(f"[fixtures] WARNING: prediction failed for "
                      f"{home_team} vs {away_team}: {exc}")
                traceback.print_exc()

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

    data["prediction_computed_at"] = row.prediction_computed_at
    data["has_prediction"] = (
        row.pred_home_win is not None and
        row.pred_draw is not None and
        row.pred_away_win is not None
    )
    return data


def _load_matches(target_date: str, db: Session) -> list[DailyMatch]:
    """Every stored fixture for one date."""
    return (
        db.query(DailyMatch)
        .filter(DailyMatch.match_date == target_date)
        .all()
    )


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
    if not rows:
        fixtures = fetch_fixtures(target_date)
        compute_and_store_predictions(fixtures, db)
        rows = _load_matches(target_date, db)
    elif _fixtures_are_stale(rows) and _has_open_fixtures(rows):
        # Only dates with fixtures still to play are refreshed. Re-predicting
        # a finished match would overwrite the forecast it was judged on.
        print(f"[fixtures] Stale — refreshing "
              f"{target_date}...")
        new_fixtures = fetch_fixtures(target_date)
        compute_and_store_predictions(new_fixtures, db)
        refresh_open_predictions(rows, new_fixtures, db)
        rows = _load_matches(target_date, db)

    # Always try to fill missing predictions regardless of the staleness
    # timer: a fixture cached while no prediction could be produced (models
    # still loading, no team stats yet) gets one on the next API call, even
    # if the day's stored predictions are otherwise fresh.
    filled = fill_missing_predictions(rows, db)
    if filled:
        print(f"[fixtures] Filled {filled} "
              f"missing predictions")
        rows = _load_matches(target_date, db)

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
