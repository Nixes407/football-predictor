"""
FotMob advanced stats collector.

Replaces Sofascore as the primary advanced team stats source. Talks to
FotMob's public JSON API directly: no Chrome or browser automation is
required on GitHub Actions.

Why the previous version got 404s
---------------------------------
FotMob moved its JSON API: the old ``https://www.fotmob.com/api/<endpoint>``
routes now answer 404 and the current routes live under
``https://www.fotmob.com/api/data/<endpoint>`` (confirmed by live endpoint
probes, 2026-09-30). This collector therefore tries the ``/api/data`` hub
first and falls back to the legacy ``/api`` hub, so it works under either
layout.

Endpoints used
--------------
Team season deep stats (xG etc.):
    GET /api/data/leagueseasondeepstats
        id={league_id} season={season} type=teams stat={stat_name}

Player season deep stats (goals, assists, xG, rating):
    GET /api/data/leagueseasondeepstats
        id={league_id} season={season} type=players stat={stat_name}
        also legacy: teamOrPlayer=player variants

``season`` must be FotMob's internal numeric season id (e.g. 27110), not the
readable "2026/2027" label, so labels are resolved through the season
directory that the same endpoint exposes. The legacy parameter style
(``teamOrPlayer=team|player`` with unsuffixed stat names) is also attempted,
and the first successful raw response per endpoint is printed so the exact JSON
structure is visible in the GitHub Actions logs.

League table / standings:
    GET /api/data/leagues
        id={league_id} ccode3={ccode} season={season}

Data flow
---------
For every league in FOTMOB_LEAGUES, both seasons (current + previous) are
collected. The deep-stats endpoint supplies the advanced metrics, the league
table supplies matches played / wins / draws / losses / goals, and the
combined result is upserted into team_season_stats:

    expectedGoals -> sh_xg      shotsOnTarget -> sh_sot
    possession    -> pos_poss   goals          -> sh_goals
    saves         -> gk_saves   cleanSheets    -> gk_cs

Player deep-stats are collected in parallel and upserted into
player_season_stats:

    goals          -> goals      assists        -> assists
    expected_goals -> xg,sh_xg   expected_assists -> xag
    matchesPlayed  -> mp         minutes        -> minutes
    position       -> position

Missing statistics never overwrite existing values.

Run from backend/ directory:
    python data/fotmob_collector.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, TeamSeasonStats, PlayerSeasonStats
from sqlalchemy import and_


FOTMOB_NAMES_FILE = _BACKEND / "data" / "fotmob_team_names.json"


# FotMob tournament ids, country codes (ccode3) and season labels. Season
# labels use FotMob's readable "YYYY/YYYY" (European) or "YYYY" (MLS and
# Eliteserien) form; they are converted to the database's dash format on upsert.
#
# Every id below was checked against FotMob's live API on 2026-10-14
# (GET /api/data/leagues?id=... and /api/data/leagueseasondeepstats?id=...).
# Ids that were used or proposed before and are wrong:
#   BEL 57 -> NED Eredivisie, 55 -> ITA Serie A (Italian rows stored as BEL),
#       62 -> no league (null). The Belgian First Division A is id 40.
#   NOR 107 -> no league (null). Eliteserien is id 59 and publishes
#       calendar-year seasons ("2026", "2025") like MLS; asking for
#       "2025/2026" silently returns the 2026 table.
#   AUT 44 -> Copa America. The Austrian Bundesliga is id 38.
# A wrong id now fails loudly: league tables and deep stats from a country
# other than the league-key prefix are rejected (see _check_league).
FOTMOB_LEAGUES = {
    "ENG-Premier League": {
        "id": 47,
        "ccode": "GBR_ENG",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "ENG-Championship": {
        "id": 48,
        "ccode": "GBR_ENG",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "ESP-La Liga": {
        "id": 87,
        "ccode": "ESP",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "GER-Bundesliga": {
        "id": 54,
        "ccode": "DEU",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "ITA-Serie A": {
        "id": 55,
        "ccode": "ITA",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "FRA-Ligue 1": {
        "id": 53,
        "ccode": "FRA",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "POR-Primeira Liga": {
        "id": 61,
        "ccode": "PRT",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "NED-Eredivisie": {
        "id": 57,
        "ccode": "NLD",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "BEL-Jupiler Pro League": {
        "id": 40,
        "ccode": "BEL",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "TUR-Süper Lig": {
        "id": 71,
        "ccode": "TUR",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "NOR-Eliteserien": {
        "id": 59,
        "ccode": "NOR",
        "season": "2026",
        "prev_season": "2025",
    },
    "AUT-Bundesliga": {
        "id": 38,
        "ccode": "AUT",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "UEFA-Champions League": {
        "id": 42,
        "ccode": "EUR",
        "season": "2026/2027",
        "prev_season": "2025/2026",
    },
    "USA-MLS": {
        "id": 130,
        "ccode": "USA",
        "season": "2026",
        "prev_season": "2025",
    },
}
LEAGUES = list(FOTMOB_LEAGUES)

BASE_URL = "https://www.fotmob.com/api"

# FotMob moved the JSON API from /api/<endpoint> to /api/data/<endpoint>;
# the legacy hub answers 404 (verified by live probes on 2026-09-30). Try the
# current hub first and keep the legacy hub from the task spec as a fallback,
# so the collector works under either layout.
API_BASES = (f"{BASE_URL}/data", BASE_URL)

REQUEST_DELAY = 2.0        # seconds between requests
REQUEST_TIMEOUT = (10, 60)
SAMPLE_CHARS = 4000        # raw sample response truncation for the logs

# Browser-like headers for all requests.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.fotmob.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.fotmob.com",
}

# One deep-stats query per logical stat. Each entry lists request variants to
# try, most likely first:
#   * current API style: type=teams with "<stat>_team" stat names
#   * legacy task-spec style: teamOrPlayer=team with bare stat names
# goals_team_match is a per-match average, so it is scaled by matches played.
DEEP_STAT_QUERIES = {
    "expectedGoals": [
        {"stat": "expected_goals_team", "type": "teams"},
        {"stat": "expected_goals", "teamOrPlayer": "team"},
    ],
    "shotsOnTarget": [
        {"stat": "ontarget_scoring_att_team", "type": "teams"},
        {"stat": "shots_on_target", "teamOrPlayer": "team"},
    ],
    "possession": [
        {"stat": "possession_percentage_team", "type": "teams"},
        {"stat": "possession_percentage", "teamOrPlayer": "team"},
    ],
    "goals": [
        {"stat": "goals_team_match", "type": "teams"},
        {"stat": "goals", "teamOrPlayer": "team"},
    ],
    "saves": [
        {"stat": "saves_team", "type": "teams"},
        {"stat": "saves", "teamOrPlayer": "team"},
    ],
    "cleanSheets": [
        {"stat": "clean_sheet_team", "type": "teams"},
        {"stat": "clean_sheets", "teamOrPlayer": "team"},
    ],
}

# Deep-stats response key -> TeamSeasonStats column (task spec mapping).
STAT_COLUMNS = {
    "expectedGoals": "sh_xg",
    "shotsOnTarget": "sh_sot",
    "possession": "pos_poss",
    "goals": "sh_goals",
    "saves": "gk_saves",
    "cleanSheets": "gk_cs",
}

# ---------------------------------------------------------------------------
# Player deep-stats config (type=players)
# ---------------------------------------------------------------------------
# One query per logical player stat. Variants mirror the team collector:
#   * current API style: type=players with "<stat>_player" stat names
#   * bare stat with type=players
#   * legacy: teamOrPlayer=player with bare stat names
# Also try alternative spellings seen in FotMob URLs and worldfootballR docs
# (e.g. expected_goals, xG, fotmob_rating, assists vs assist).
DEEP_PLAYER_QUERIES = {
    "goals": [
        {"stat": "goals_player", "type": "players"},
        {"stat": "goals", "type": "players"},
        {"stat": "goals", "teamOrPlayer": "player"},
        {"stat": "goals_player_match", "type": "players"},
    ],
    "assists": [
        {"stat": "assists_player", "type": "players"},
        {"stat": "assists", "type": "players"},
        {"stat": "assist_player", "type": "players"},
        {"stat": "assists", "teamOrPlayer": "player"},
        {"stat": "assists_player_match", "type": "players"},
    ],
    "rating": [
        {"stat": "rating_player", "type": "players"},
        {"stat": "rating", "type": "players"},
        {"stat": "fotmob_rating_player", "type": "players"},
        {"stat": "player_rating", "type": "players"},
        {"stat": "rating", "teamOrPlayer": "player"},
    ],
    "expectedGoals": [
        {"stat": "expected_goals_player", "type": "players"},
        {"stat": "expected_goals", "type": "players"},
        {"stat": "expected_goals", "teamOrPlayer": "player"},
        {"stat": "xg_player", "type": "players"},
        {"stat": "expectedGoals", "type": "players"},
    ],
    "expectedAssists": [
        {"stat": "expected_assists_player", "type": "players"},
        {"stat": "expected_assists", "type": "players"},
        {"stat": "expected_assists", "teamOrPlayer": "player"},
        {"stat": "expected_assist_player", "type": "players"},
        {"stat": "xa_player", "type": "players"},
    ],
}

# Player logical -> PlayerSeasonStats column. Rating has no column in the
# current schema, so it is kept in-memory only and not persisted (do NOT
# invent schema columns without updating database.py).
PLAYER_STAT_COLUMNS = {
    "goals": "goals",
    "assists": "assists",
    "expectedGoals": "xg",
    "expectedAssists": "xag",
}


def _safe(val) -> float | None:
    """Parse FotMob numbers, percentages and missing values safely."""
    try:
        if isinstance(val, str):
            val = val.strip().rstrip("%").replace(",", "")
        number = float(val)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _norm_team(name) -> str:
    """Case-insensitive, whitespace-normalised team name for joining."""
    return " ".join(str(name).split()).casefold()


def _db_season(season: str) -> str:
    """Convert FotMob season labels to the dash format used in the database."""
    return str(season).replace("/", "-")


def _fotmob_season(label, calendar_year: bool) -> str:
    """A season label in the league's own FotMob form, keeping the start year.

    Calendar-year leagues publish "2026", split-year leagues "2026/2027".
    A label in the wrong form does not fail: FotMob's league table silently
    falls back to the latest season, which would store the current table
    under another season's label.
    """
    text = str(label).strip().replace("-", "/")
    start = text.split("/", 1)[0]
    if len(start) != 4 or not start.isdigit():
        return text
    year = int(start)
    return str(year) if calendar_year else f"{year}/{year + 1}"


def _get_seasons(league: str) -> list[str]:
    """Return configured seasons in previous-to-current order for a league.

    Labels are normalised to the league's FotMob season form, so a
    calendar-year league is never asked for a split-year season (or the
    reverse) even if FOTMOB_LEAGUES is edited with the wrong form.
    """
    CALENDAR_YEAR_LEAGUES = {
        "USA-MLS",
        # Eliteserien runs March-December. FotMob lists its seasons as
        # "2026", "2025", ... (verified 2026-10-14), and a "2025/2026"
        # request silently returns the 2026 table.
        "NOR-Eliteserien",
    }
    calendar_year = league in CALENDAR_YEAR_LEAGUES
    config = FOTMOB_LEAGUES.get(league, {})
    seasons = []
    for key in ("prev_season", "season"):
        if config.get(key):
            label = _fotmob_season(config[key], calendar_year)
            if label not in seasons:
                seasons.append(label)
    return seasons or [_fotmob_season("2026/2027", calendar_year)]


def _league_country(league: str) -> str:
    """FotMob country code expected for a league key ("BEL-..." -> "BEL").

    FotMob files UEFA competitions under "INT"; for every other configured
    league the key prefix equals FotMob's country code (verified 2026-10-14).
    """
    prefix = league.split("-", 1)[0].strip().upper()
    return {"UEFA": "INT"}.get(prefix, prefix)


def _check_league(details, key: str, league_id: int, expected: str | None) -> None:
    """Reject a FotMob response that belongs to another country's league.

    A wrong id does not fail on its own: it quietly returns some other
    competition (id 55 is Serie A, so Italian clubs were stored as Belgian).
    Comparing the country code FotMob reports turns that into an error.
    """
    if not expected or not isinstance(details, dict):
        return
    actual = str(details.get(key) or "").strip().upper()
    if actual and actual != expected:
        raise ValueError(
            f"FotMob league id {league_id} is {details.get('name')!r} "
            f"({actual}), not a {expected} league - fix its id in FOTMOB_LEAGUES"
        )


def _season_aliases(value) -> set[str]:
    """Season label spellings: 2026/2027 == 2026/27."""
    text = str(value or "").strip().replace(" ", "")
    aliases = {text} if text else set()
    if "/" in text:
        left, right = text.split("/", 1)
        if len(left) == 4 and right.isdigit():
            if len(right) == 2:
                aliases.add(f"{left}/{left[:2]}{right}")
            elif len(right) == 4:
                aliases.add(f"{left}/{right[2:]}")
    return aliases


def _match_season(seasons: list, season: str):
    """Find the internal season id for a readable label in a seasons list."""
    wanted = _season_aliases(season)
    for entry in seasons:
        if not isinstance(entry, dict):
            continue
        for field in ("name", "id"):
            if _season_aliases(entry.get(field)) & wanted:
                return entry.get("id")
    return None


def _row_team(row: dict) -> str | None:
    """Team name from a deep-stats row, tolerating alternative layouts."""
    for key in ("name", "teamName", "ParticipantName", "participant_name", "team_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
        if isinstance(value, dict):
            # e.g. {"name": "Arsenal"}
            for k in ("name", "teamName", "label"):
                v = value.get(k)
                if isinstance(v, str) and v.strip():
                    return " ".join(v.split())
    # nested participant
    for nested_key in ("participant", "Participant"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for k in ("name", "teamName", "participant_name"):
                v = nested.get(k)
                if isinstance(v, str) and v.strip():
                    return " ".join(v.split())
    return None


def _row_value(row: dict) -> float | None:
    """Stat value from a deep-stats row: {"value": x} or a bare number.

    Handles both camelCase (statValue) and snake_case (stat_value) as well
    as bare numbers, and tolerates the statValue being a dict with various
    key casings. Player rows from worldfootballR use stat_value etc.
    """
    # Try known value container keys in priority order
    for container_key in ("statValue", "stat_value", "StatValue", "statvalue"):
        if container_key in row:
            val = row[container_key]
            if isinstance(val, dict):
                for k in ("value", "StatValue", "stat_value", "val", "Value"):
                    if k in val:
                        return _safe(val[k])
                # fallback: first numeric inside dict
                for kk, vv in val.items():
                    s = _safe(vv)
                    if s is not None:
                        return s
                return None
            return _safe(val)
    # Legacy bare keys used by some endpoints
    for k in ("value", "Value", "stat", "StatValue"):
        if k in row:
            s = _safe(row[k])
            if s is not None:
                return s
    return None


def _row_matches(row: dict) -> float | None:
    """Matches played: explicit fields first, then the substat value."""
    for key in ("matchesPlayed", "MatchesPlayed", "matches_played", "matchesPlayed", "matches_played"):
        number = _safe(row.get(key))
        if number is not None:
            return number
    # also try generic mp keys
    for key in ("mp", "MP"):
        number = _safe(row.get(key))
        if number is not None:
            return number
    sub = row.get("substatValue")
    if sub is None:
        sub = row.get("sub_stat_value")
    if sub is None:
        sub = row.get("SubstatValue")
    if isinstance(sub, dict):
        for key in ("value", "StatValue", "stat_value"):
            if key in sub:
                return _safe(sub[key])
        # first numeric
        for kk, vv in sub.items():
            s = _safe(vv)
            if s is not None:
                return s
        return None
    return _safe(sub)


def _row_minutes(row: dict) -> float | None:
    """Minutes played from a player row, handling multiple key variants."""
    for key in ("minutesPlayed", "minutes_played", "minutes", "Minutes", "mins", "Mins", "minutesplayed", "MinutesPlayed"):
        number = _safe(row.get(key))
        if number is not None:
            return number
    # nested participant
    for nested_key in ("participant", "Participant", "player"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for k in ("minutesPlayed", "minutes_played", "minutes"):
                v = nested.get(k)
                s = _safe(v)
                if s is not None:
                    return s
    return None


def _row_position(row: dict) -> str | None:
    """Position string from a player row."""
    for key in ("position", "Position", "pos", "Pos", "role", "Role", "positionName", "position_name", "PositionName"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
        if isinstance(value, dict):
            for k in ("label", "name", "title", "key", "fallback"):
                vv = value.get(k)
                if isinstance(vv, str) and vv.strip():
                    # keys like "leftwinger_short" are not useful as display, but better than nothing
                    # prefer label over key
                    if k == "label":
                        return " ".join(vv.split())
            # if dict has label, return it
            lab = value.get("label")
            if isinstance(lab, str) and lab.strip():
                return " ".join(lab.split())
    for nested_key in ("participant", "Participant", "player"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for k in ("position", "pos", "role", "positionName"):
                vv = nested.get(k)
                if isinstance(vv, str) and vv.strip():
                    return " ".join(vv.split())
                if isinstance(vv, dict):
                    lab = vv.get("label") or vv.get("name") or vv.get("fallback")
                    if isinstance(lab, str) and lab.strip():
                        return " ".join(lab.split())
    return None


def _row_player(row: dict) -> str | None:
    """Player name from a deep-stats row, tolerating many layouts."""
    for key in ("name", "playerName", "ParticipantName", "participant_name", "participantName", "player_name", "fullName", "player"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
        if isinstance(value, dict):
            for k in ("name", "playerName", "fullName", "label"):
                vv = value.get(k)
                if isinstance(vv, str) and vv.strip():
                    return " ".join(vv.split())
    # nested participant / player objects
    for nested_key in ("participant", "Participant", "player", "Player"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for k in ("name", "playerName", "ParticipantName", "fullName", "participant_name"):
                vv = nested.get(k)
                if isinstance(vv, str) and vv.strip():
                    return " ".join(vv.split())
            # sometimes participant is {"id": 123, "name": "Haaland"}
            if isinstance(nested.get("name"), str) and nested["name"].strip():
                return " ".join(nested["name"].split())
    return None


def _row_player_team(row: dict) -> str | None:
    """Team name for a player row (must match team_season_stats spellings)."""
    for key in ("teamName", "team_name", "TeamName", "ParticipantTeam", "team", "team_title", "teamTitle", "teamNameStr"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
        if isinstance(value, dict):
            for k in ("name", "teamName", "title", "label"):
                vv = value.get(k)
                if isinstance(vv, str) and vv.strip():
                    return " ".join(vv.split())
    # worldfootballR uses team_id numeric + participant_name; team_name may be under "teamName" or "team_title"
    for nested_key in ("participant", "Participant", "player", "Player"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for k in ("teamName", "team_name", "team", "teamTitle", "team_title"):
                vv = nested.get(k)
                if isinstance(vv, str) and vv.strip():
                    return " ".join(vv.split())
                if isinstance(vv, dict):
                    for kk in ("name", "teamName", "label"):
                        vvv = vv.get(kk)
                        if isinstance(vvv, str) and vvv.strip():
                            return " ".join(vvv.split())
    # fallback: if row has explicit teamId but no name, we cannot map without extra lookup — skip
    return None


def _parse_scores(scores) -> tuple[float | None, float | None]:
    """Split a standings scoresStr such as '70-24' into goals for/against."""
    if not isinstance(scores, str) or "-" not in scores:
        return None, None
    scored, _, conceded = scores.partition("-")
    return _safe(scored), _safe(conceded)


def _upsert_team(db, team, league, season, attrs) -> None:
    """Upsert mapped team stats without overwriting with missing values."""
    existing = (
        db.query(TeamSeasonStats)
        .filter(and_(
            TeamSeasonStats.team == team,
            TeamSeasonStats.league == league,
            TeamSeasonStats.season == season,
        ))
        .first()
    )
    if existing is None:
        existing = TeamSeasonStats(team=team, league=league, season=season)
        db.add(existing)
    for key, value in attrs.items():
        # SQLAlchemy permits arbitrary Python attributes: reject misspellings
        # rather than silently losing a value on commit.
        if key not in TeamSeasonStats.__table__.columns:
            raise ValueError(f"Unknown TeamSeasonStats field: {key}")
        value = _safe(value)
        if value is not None:
            setattr(existing, key, value)


def _upsert_player(db, player, team, league, season, position, attrs) -> None:
    """Upsert mapped player stats without overwriting with missing values.

    Mirrors _upsert_team and fbref_player_collector._upsert_player: existing
    non-null values are never cleared by a missing fetch. Position is stored
    separately as it may come from a different field than the stat values.
    """
    existing = (
        db.query(PlayerSeasonStats)
        .filter(and_(
            PlayerSeasonStats.player == player,
            PlayerSeasonStats.team == team,
            PlayerSeasonStats.league == league,
            PlayerSeasonStats.season == season,
        ))
        .first()
    )
    if existing is None:
        existing = PlayerSeasonStats(player=player, team=team, league=league, season=season)
        db.add(existing)
    # position first (do not clear existing with None)
    if position is not None:
        # normalise whitespace
        pos_norm = " ".join(str(position).split())
        if pos_norm:
            existing.position = pos_norm
    for key, value in attrs.items():
        if key not in PlayerSeasonStats.__table__.columns:
            raise ValueError(f"Unknown PlayerSeasonStats field: {key}")
        # mp is Integer, handling via _safe -> float then int
        if key == "mp" and value is not None:
            try:
                iv = int(float(value))
                if existing.mp is None or iv is not None:
                    setattr(existing, key, iv)
                continue
            except (TypeError, ValueError):
                pass
        safe = _safe(value)
        if safe is not None:
            # preserve integer nature for mp
            if key == "mp":
                setattr(existing, key, int(safe))
            else:
                setattr(existing, key, safe)


def _team_attrs(entry: dict, standing: dict) -> dict:
    """Map one team's FotMob data onto TeamSeasonStats columns."""
    attrs = {}
    for logical, column in STAT_COLUMNS.items():
        value = _safe(entry.get(logical))
        if value is not None:
            attrs[column] = round(value, 1 if column == "pos_poss" else 3)
    # Standings supply exact season totals and results where columns exist.
    for column in ("gk_ga", "gk_w", "gk_d", "gk_l"):
        value = _safe(standing.get(column))
        if value is not None:
            attrs[column] = value
    goals_for = _safe(standing.get("goalsFor"))
    if goals_for is not None:
        # The exact table total beats the per-match estimate.
        attrs["sh_goals"] = goals_for
    matches = _safe(entry.get("matchesPlayed"))
    if matches is None:
        matches = _safe(standing.get("matchesPlayed"))
    if "sh_goals" in attrs and "sh_xg" in attrs:
        attrs["sh_g_minus_xg"] = round(attrs["sh_goals"] - attrs["sh_xg"], 3)
    if "gk_cs" in attrs and matches:
        attrs["gk_cs_pct"] = round(attrs["gk_cs"] / matches * 100, 1)
    return attrs


def _player_attrs(entry: dict) -> dict:
    """Map one player's FotMob merged entry onto PlayerSeasonStats columns."""
    attrs = {}
    for logical, column in PLAYER_STAT_COLUMNS.items():
        # entry holds logical keys: goals, assists, expectedGoals, expectedAssists
        if logical not in entry:
            continue
        val = _safe(entry.get(logical))
        if val is not None:
            # extra parity: expectedGoals also stored as sh_xg if column exists
            attrs[column] = val
            if logical == "expectedGoals" and "sh_xg" in PlayerSeasonStats.__table__.columns:
                attrs["sh_xg"] = val
            if logical == "expectedAssists" and "xag" in PlayerSeasonStats.__table__.columns:
                # already mapped to xag
                pass
    # matchesPlayed -> mp, minutes -> minutes
    # Prefer explicit keys entry[mp/minutes] first; fallback to alternative casings
    mp_val = None
    for k in ("matchesPlayed", "matches_played", "mp", "MatchesPlayed"):
        if k in entry and entry[k] is not None:
            mp_val = _safe(entry[k])
            if mp_val is not None:
                try:
                    attrs["mp"] = int(mp_val)
                except (TypeError, ValueError):
                    attrs["mp"] = mp_val
                break
    mins_val = None
    for k in ("minutes", "minutesPlayed", "minutes_played"):
        if k in entry and entry[k] is not None:
            mins_val = _safe(entry[k])
            if mins_val is not None:
                attrs["minutes"] = mins_val
                break
    # position is handled by _upsert_player, not here
    return attrs


def _merge_teams(stats: dict, table: dict) -> dict:
    """Deep-stats entries keyed by team, with standings-only teams appended."""
    teams = dict(stats)
    known = {_norm_team(name) for name in teams}
    for name in table:
        if _norm_team(name) not in known:
            teams[name] = {}
    return teams


class _FotMobClient:
    """Rate-limited HTTP client for FotMob's JSON API with hub fallback."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        retry = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"], respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._last_request = 0.0
        self._base = None            # API hub that answered last
        self._samples = set()        # endpoints whose response was sampled
        self._seasons = {}           # (league_id, season label) -> season param
        self._deep_style = None      # team-param style that worked
        self._player_style = None    # player-param style that worked
        self._player_samples = set() # player deep-stats sample keys

    # -- HTTP ----------------------------------------------------------

    def _get(self, endpoint: str, params: dict) -> dict:
        """GET {hub}/{endpoint}, preferring the hub that answered last.

        404/405/410 means the route does not exist on that hub, so the other
        known hub is tried before giving up.
        """
        candidates = [self._base] if self._base else []
        candidates += [base for base in API_BASES if base not in candidates]
        last_error = None
        for base in candidates:
            time.sleep(max(0.0, REQUEST_DELAY - (time.monotonic() - self._last_request)))
            try:
                response = self.session.get(
                    f"{base}/{endpoint}", params=params, timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_error = exc
                continue
            finally:
                self._last_request = time.monotonic()
            if response.status_code in (404, 405, 410):
                last_error = RuntimeError(
                    f"HTTP {response.status_code}: {base}/{endpoint}"
                )
                continue
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError(f"Non-JSON reply from {base}/{endpoint}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Unexpected FotMob {endpoint} response")
            if self._base != base:
                print(f"[fotmob] Using API hub: {base}")
                self._base = base
            self._print_sample(response, payload, endpoint)
            return payload
        if last_error is None:
            last_error = RuntimeError(f"FotMob {endpoint} request failed")
        raise last_error

    def _print_sample(self, response, payload: dict, endpoint: str) -> None:
        """Print the raw response of the first successful call per endpoint."""
        if endpoint in self._samples:
            return
        self._samples.add(endpoint)
        print(f"[fotmob] Sample response: GET {response.url} -> HTTP {response.status_code}")
        print(f"[fotmob] Sample response: top-level keys: {list(payload)}")
        print(f"[fotmob] Sample response: {json.dumps(payload)[:SAMPLE_CHARS]}")

    def _print_player_sample(self, response, payload: dict, endpoint: str, style: str) -> None:
        """Print first successful player response variant (separate from team)."""
        key = f"{endpoint}:{style}"
        if key in self._player_samples:
            return
        self._player_samples.add(key)
        print(f"[fotmob] Player sample response: GET {response.url} -> HTTP {response.status_code}")
        print(f"[fotmob] Player sample response: top-level keys: {list(payload)}")
        print(f"[fotmob] Player sample response: {json.dumps(payload)[:SAMPLE_CHARS]}")

    # -- deep stats ----------------------------------------------------

    def _deep_fetch(self, league_id: int, season_param, query: dict,
                    country: str | None = None):
        """One deep-stats request; returns (payload, rows or None).

        ``country`` is the FotMob country code the league must belong to;
        a response for another country's league raises ValueError.
        """
        payload = self._get("leagueseasondeepstats", {
            "id": league_id, "season": season_param, **query,
        })
        _check_league(payload.get("leagueDetails"), "countryCode",
                      league_id, country)
        rows = payload.get("statsData")
        return payload, rows if isinstance(rows, list) and rows else None

    def _ordered_queries(self, queries: list) -> list:
        """Try the parameter style that already worked first (team)."""
        if not self._deep_style:
            return list(queries)
        style = self._deep_style

        def matches(query):
            return 0 if all(query.get(k) == v for k, v in style.items()) else 1

        return sorted(queries, key=matches)

    def _ordered_player_queries(self, queries: list) -> list:
        """Try the player parameter style that already worked first."""
        if not self._player_style:
            return list(queries)
        style = self._player_style

        def matches(query):
            return 0 if all(query.get(k) == v for k, v in style.items()) else 1

        return sorted(queries, key=matches)

    def _resolve_season(self, league_id: int, season: str,
                        country: str | None = None):
        """Map a readable season label to the deep-stats season parameter.

        The deep-stats endpoint wants FotMob's internal numeric season id
        (e.g. 27110), so a label like "2026/2027" is resolved through the
        season directory the endpoint itself returns. Results are memoized.
        """
        key = (league_id, str(season))
        if key in self._seasons:
            return self._seasons[key]
        payload = None
        for query in self._ordered_queries(DEEP_STAT_QUERIES["expectedGoals"]):
            payload, rows = self._deep_fetch(league_id, season, query, country)
            if rows:
                self._seasons[key] = season
                return season
        # Also try player queries for season resolution if team queries yielded
        # no rows but the season listing is still available in the payload.
        if payload is None:
            # try one player query as probe - cheap, may populate seasons list
            for query in self._ordered_player_queries(DEEP_PLAYER_QUERIES["goals"]):
                try:
                    payload, rows = self._deep_fetch(league_id, season, query, country)
                    if rows:
                        self._seasons[key] = season
                        return season
                    if payload and payload.get("seasons"):
                        break
                except Exception:
                    continue
        # No rows for the label: resolve the internal id via the seasons list.
        season_id = _match_season((payload or {}).get("seasons") or [], season)
        if season_id is None:
            directory = self._get("leagueseasondeepstats", {"id": league_id})
            _check_league(directory.get("leagueDetails"), "countryCode",
                          league_id, country)
            season_id = _match_season(directory.get("seasons") or [], season)
        if season_id is None:
            raise ValueError(
                f"FotMob publishes no season {season} for league {league_id}"
            )
        self._seasons[key] = season_id
        return season_id

    def get_team_xg_stats(self, league_id: int, season: str,
                          country: str | None = None) -> dict:
        """Team season stats for one league and season from the deep-stats API.

        Combines one request per stat (expected goals, shots on target and
        possession percentage, plus the goals, saves and clean sheets needed
        for the database mapping) into a single dict keyed by team name:

            {"Arsenal": {"expectedGoals": 35.4, "shotsOnTarget": 120.0,
                         "possession": 58.3, "goals": 70.0, "saves": 41.0,
                         "cleanSheets": 14.0, "matchesPlayed": 38}}
        """
        season_param = self._resolve_season(league_id, season, country)
        if str(season_param) != str(season):
            print(f"[fotmob] Season {season} -> internal id {season_param}")
        teams = {}
        for logical, queries in DEEP_STAT_QUERIES.items():
            rows = None
            used = None
            for query in self._ordered_queries(queries):
                _, rows = self._deep_fetch(
                    league_id, season_param, query, country,
                )
                if rows:
                    used = query
                    break
            if not rows:
                print(f"[fotmob] WARNING: no {logical} stats for "
                      f"league {league_id} {season}")
                continue
            style = {k: v for k, v in used.items() if k != "stat"}
            if self._deep_style != style:
                self._deep_style = style
                print(f"[fotmob] Deep-stats params that worked: {style}")
            per_match = str(used.get("stat", "")).endswith("_match")
            unscaled = 0
            for row in rows:
                team = _row_team(row)
                if team is None:
                    continue
                entry = teams.setdefault(team, {})
                matches = _row_matches(row) or entry.get("matchesPlayed")
                if matches is not None:
                    entry["matchesPlayed"] = matches
                value = _row_value(row)
                if value is None:
                    continue
                if per_match:
                    if matches:
                        value = value * matches
                    else:
                        unscaled += 1
                entry[logical] = value
            if unscaled:
                print(f"[fotmob] WARNING: {logical} kept as a per-match value "
                      f"for {unscaled} teams (no match counts available)")
        return teams

    def get_player_stats(self, league_id: int, season: str,
                         country: str | None = None) -> dict:
        """Player season stats for one league and season.

        Combines one request per logical player stat (goals, assists, rating,
        expected_goals, expected_assists) into a single dict keyed by
        (team, player):

            {("Arsenal", "Bukayo Saka"): {"goals": 12, "assists": 8,
              "expectedGoals": 9.4, "rating": 7.45, "matchesPlayed": 30,
              "minutes": 2550, "position": "FW"}, ...}

        Stats are merged across the multiple stat calls, so a player who
        appears in any of the endpoints is retained with whatever of their
        rows were available. Team names are normalised for whitespace but kept
        with FotMob's original casing to match team_season_stats.
        """
        season_param = self._resolve_season(league_id, season, country)
        if str(season_param) != str(season):
            print(f"[fotmob] Season {season} -> internal id {season_param} (players)")
        players: dict[tuple[str, str], dict] = {}
        # Keep raw payload sample for first successful player call
        first_payload_logged = False
        for logical, queries in DEEP_PLAYER_QUERIES.items():
            payload = None
            rows = None
            used = None
            for query in self._ordered_player_queries(queries):
                try:
                    payload, rows = self._deep_fetch(
                        league_id, season_param, query, country,
                    )
                except Exception as exc:
                    # let caller handle per-logical? continue trying variants
                    continue
                if rows:
                    used = query
                    break
            if not rows:
                print(f"[fotmob] WARNING: no {logical} player stats for "
                      f"league {league_id} {season}")
                continue
            style = {k: v for k, v in used.items() if k != "stat"}
            if self._player_style != style:
                self._player_style = style
                print(f"[fotmob] Player deep-stats params that worked: {style}")
            # Log first successful player payload variant separately from team
            if not first_payload_logged and payload is not None:
                try:
                    # use _print_player_sample via _get info — we don't have response object here,
                    # so we manually log a trimmed sample similar to _print_sample
                    key = f"leagueseasondeepstats:players:{logical}"
                    if key not in self._player_samples:
                        self._player_samples.add(key)
                        print(f"[fotmob] Player sample: league {league_id} {season} {logical} -> "
                              f"stat={used.get('stat')} rows={len(rows)} top_keys={list(payload.keys())[:8]}")
                        print(f"[fotmob] Player sample row keys: {list(rows[0].keys()) if rows else []}")
                        print(f"[fotmob] Player sample row: {json.dumps(rows[0])[:SAMPLE_CHARS] if rows else ''}")
                except Exception:
                    pass
                first_payload_logged = True
            per_match = str(used.get("stat", "")).endswith("_match") or str(used.get("stat", "")).endswith("_per90")
            unscaled = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                player = _row_player(row)
                team = _row_player_team(row)
                # Some payloads nest player/team under different keys; if one is missing,
                # try alternative extraction: team may be under participant team, player under participant name.
                if player is None or team is None:
                    # try reading team from row if player extraction used team key incorrectly
                    # fallback: if player still None skip row
                    if player is None:
                        continue
                    if team is None:
                        # ambiguous row without team — skip to avoid cross-team contamination
                        continue
                # Normalise whitespace (team names must match team_season_stats spellings)
                team = " ".join(team.split())
                player = " ".join(player.split())
                key = (team, player)
                entry = players.setdefault(key, {"team": team, "player": player})
                # capture auxiliary fields merged across calls
                # matchesPlayed / mp
                m = _row_matches(row)
                if m is not None and entry.get("matchesPlayed") is None:
                    entry["matchesPlayed"] = m
                # minutes
                mins = _row_minutes(row)
                if mins is not None and entry.get("minutes") is None:
                    entry["minutes"] = mins
                # position
                pos = _row_position(row)
                if pos and not entry.get("position"):
                    entry["position"] = pos
                # Also capture minutes/matches that might be under snake_case keys not caught above
                # (worldfootballR shows minutes_played, matches_played)
                for alt_key, target in (("minutes_played", "minutes"), ("minutesPlayed", "minutes"),
                                        ("matches_played", "matchesPlayed"), ("matchesPlayed", "matchesPlayed")):
                    if alt_key in row and target not in entry:
                        v = _safe(row.get(alt_key))
                        if v is not None:
                            entry[target] = v
                value = _row_value(row)
                if value is None:
                    continue
                if per_match:
                    # scale per-match average to total if we have matches
                    mm = m or entry.get("matchesPlayed")
                    if mm:
                        value = value * mm
                    else:
                        unscaled += 1
                entry[logical] = value
            if unscaled:
                print(f"[fotmob] WARNING: {logical} player kept as per-match value "
                      f"for {unscaled} players (no match counts available)")
        return players

    # -- standings -----------------------------------------------------

    def get_league_table(self, league_id: int, ccode: str, season: str,
                         country: str | None = None) -> dict:
        """League standings: matches played, results and goals for/against.

        Raises ValueError when FotMob answers with another country's league
        (wrong id) or with a different season than requested: for an
        unknown season label FotMob silently returns the latest table.
        """
        payload = self._get("leagues", {
            "id": league_id, "ccode3": ccode, "season": season,
        })
        details = payload.get("details")
        _check_league(details, "country", league_id, country)
        selected = details.get("selectedSeason") if isinstance(details, dict) else None
        if selected and not (_season_aliases(selected) & _season_aliases(season)):
            raise ValueError(
                f"FotMob returned season {selected} for league {league_id} "
                f"instead of {season}"
            )
        teams = {}
        entries = payload.get("table")
        if not isinstance(entries, list):
            raise ValueError("FotMob league response has no table")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            data = entry.get("data") or {}
            # Split tables are "composite": MLS conferences and the Belgian /
            # Austrian play-off groups keep their rows in data["tables"][i]
            # rather than data[\"table\"]. Their rows carry full-season totals.
            groups = [data] + [
                group for group in (data.get("tables") or [])
                if isinstance(group, dict)
            ]
            for group in groups:
                table = group.get("table")
                rows = table.get("all") if isinstance(table, dict) else None
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    name = row.get("name") if isinstance(row, dict) else None
                    if not isinstance(name, str) or not name.strip():
                        continue
                    name = " ".join(name.split())
                    if name in teams:
                        continue
                    goals_for, goals_against = _parse_scores(row.get("scoresStr"))
                    teams[name] = {
                        "matchesPlayed": _safe(row.get("played")),
                        "gk_w": _safe(row.get("wins")),
                        "gk_d": _safe(row.get("draws")),
                        "gk_l": _safe(row.get("losses")),
                        "goalsFor": goals_for,
                        "gk_ga": goals_against,
                    }
        if not teams:
            raise ValueError("FotMob league table is empty")
        return teams

    def close(self) -> None:
        self.session.close()


def _store_fotmob_name(db, fotmob_name: str, league: str) -> None:
    """The upsert already stores the API spelling in TeamSeasonStats.team.

    The league/name pairs are exported from committed rows at the end of
    collection, so there is no separate table or additional write here.
    """
    pass


def collect_team_stats() -> int:
    """Collect both seasons for all leagues; return successfully committed rows."""
    # Clean up potentially corrupted Belgian data
    db_clean = SessionLocal()
    try:
        deleted = db_clean.query(
            TeamSeasonStats
        ).filter(
            TeamSeasonStats.league ==
            "BEL-Jupiler Pro League"
        ).delete()
        db_clean.commit()
        if deleted:
            print(f"[fotmob] Cleaned up "
                  f"{deleted} corrupted "
                  f"Belgian league rows")
    except Exception as exc:
        db_clean.rollback()
        print(f"[fotmob] WARNING: Belgian cleanup failed: {exc}")
    finally:
        db_clean.close()

    print("[fotmob] Collecting team stats via the FotMob JSON API...")
    total = 0
    client = _FotMobClient()
    try:
        for league, config in FOTMOB_LEAGUES.items():
            league_id = config["id"]
            country = _league_country(league)
            for api_season in _get_seasons(league):
                db_season = _db_season(api_season)
                print(f"[fotmob] {league} {api_season}...")
                try:
                    try:
                        table = client.get_league_table(
                            league_id, config["ccode"], api_season, country,
                        )
                    except Exception as exc:
                        print(f"[fotmob] WARNING: league table failed for "
                              f"{league} {api_season}: {exc}")
                        table = {}
                    try:
                        stats = client.get_team_xg_stats(
                            league_id, api_season, country,
                        )
                    except Exception as exc:
                        print(f"[fotmob] WARNING: deep stats failed for "
                              f"{league} {api_season}: {exc}")
                        stats = {}
                    if not stats and not table:
                        print(f"[fotmob] ERROR: no data for {league} {api_season}")
                        continue
                    if not stats:
                        print(f"[fotmob] WARNING: storing standings only for "
                              f"{league} {api_season}")
                    teams = _merge_teams(stats, table)
                    norm_table = {
                        _norm_team(name): rec for name, rec in table.items()
                    }
                    print(f"[fotmob] {league} {api_season}: {len(stats)} teams "
                          f"with deep stats, {len(teams)} to upsert")
                    db = SessionLocal()
                    committed = 0
                    try:
                        for team, entry in teams.items():
                            standing = norm_table.get(_norm_team(team), {})
                            attrs = _team_attrs(entry, standing)
                            if attrs:
                                _upsert_team(db, team, league, db_season, attrs)
                                # The exact API name is now in team_season_stats.
                                _store_fotmob_name(db, team, league)
                                committed += 1
                        db.commit()
                        total += committed
                        print(f"[fotmob] {league} {api_season}: "
                              f"committed {committed} teams")
                    except Exception:
                        db.rollback()
                        raise
                    finally:
                        db.close()
                except Exception as exc:
                    print(f"[fotmob] ERROR {league} {api_season}: {exc}")
    finally:
        client.close()

    # Export persisted team names for fixture matching. The database is
    # authoritative for every league it holds, so stale names (such as the
    # Italian clubs once stored as Belgian) are dropped; leagues missing
    # from the database keep their previously exported names.
    try:
        existing = {}
        if FOTMOB_NAMES_FILE.exists():
            existing = json.loads(FOTMOB_NAMES_FILE.read_text())
        # Rebuilt from the database only, after the cleanup above.
        existing.pop("BEL-Jupiler Pro League", None)
        current = {}
        db_check = SessionLocal()
        try:
            rows = db_check.query(
                TeamSeasonStats.team, TeamSeasonStats.league,
            ).distinct().all()
            for team, league in rows:
                if league not in current:
                    current[league] = []
                if team not in current[league]:
                    current[league].append(team)
        finally:
            db_check.close()
        existing.update(current)
        FOTMOB_NAMES_FILE.write_text(json.dumps(existing, indent=2))
        print(f"[fotmob] Saved team names to {FOTMOB_NAMES_FILE.name}")
    except Exception as exc:
        print(f"[fotmob] WARNING: could not save team names: {exc}")

    print(f"[fotmob] Done. Total team rows: {total}")
    return total


def collect_player_stats() -> int:
    """Collect both seasons player stats for all leagues; return committed rows.

    Loops all FOTMOB_LEAGUES × seasons, fetches player deep-stats (goals,
    assists, rating, expected_goals, expected_assists) via the same
    leagueseasondeepstats hub as team stats, merges them per (team, player),
    and upserts into player_season_stats. Never crashes the whole run on one
    league failure — logs ERROR and continues. Returns total players committed.
    """
    print("[fotmob] Collecting PLAYER stats via the FotMob JSON API...")
    total = 0
    client = _FotMobClient()
    try:
        for league, config in FOTMOB_LEAGUES.items():
            league_id = config["id"]
            country = _league_country(league)
            for api_season in _get_seasons(league):
                db_season = _db_season(api_season)
                print(f"[fotmob] {league} {api_season} players...")
                try:
                    try:
                        players = client.get_player_stats(
                            league_id, api_season, country,
                        )
                    except Exception as exc:
                        print(f"[fotmob] WARNING: player deep stats failed for "
                              f"{league} {api_season}: {exc}")
                        import traceback
                        traceback.print_exc()
                        players = {}
                    if not players:
                        print(f"[fotmob] WARNING: no player stats for {league} {api_season}")
                        continue
                    print(f"[fotmob] {league} {api_season}: {len(players)} players with data, "
                          f"upserting...")
                    db = SessionLocal()
                    committed = 0
                    skipped = 0
                    try:
                        for (team, player), entry in players.items():
                            # Team name already normalised; keep FotMob spelling
                            # which must match team_season_stats for the same league.
                            position = entry.get("position")
                            attrs = _player_attrs(entry)
                            # Skip rows with no meaningful stats (e.g. only rating)
                            if not attrs:
                                skipped += 1
                                continue
                            # Do not create rows with empty player/team/league/season
                            if not player or not team:
                                skipped += 1
                                continue
                            _upsert_player(db, player, team, league, db_season, position, attrs)
                            committed += 1
                        db.commit()
                        total += committed
                        print(f"[fotmob] {league} {api_season}: "
                              f"committed {committed} players (skipped {skipped} empty)")
                    except Exception:
                        db.rollback()
                        raise
                    finally:
                        db.close()
                except Exception as exc:
                    print(f"[fotmob] ERROR {league} {api_season} players: {exc}")
                    import traceback
                    traceback.print_exc()
                    continue
    finally:
        client.close()

    print(f"[fotmob] Done. Total player rows: {total}")
    return total


def collect_fotmob() -> None:
    """Run FotMob stats collection (teams + players) and enforce success rules."""
    print("=" * 60)
    print("FOTMOB — ADVANCED STATS COLLECTION")
    print("=" * 60)
    print("\n--- TEAM STATS ---")
    team_total = collect_team_stats()
    print("\n--- PLAYER STATS ---")
    try:
        player_total = collect_player_stats()
    except Exception as exc:
        print(f"[fotmob] ERROR: player collection crashed: {exc}")
        import traceback
        traceback.print_exc()
        player_total = 0

    print("\n" + "=" * 60)
    print("FOTMOB COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Team rows updated: {team_total}")
    print(f"Player rows updated: {player_total}")

    # Team failure is fatal (keeps current behaviour)
    if team_total == 0:
        print("\nERROR: Zero team rows collected. Check FotMob availability and database access.")
        sys.exit(1)

    # Player collection failure is loud but non-fatal unless teams succeeded and
    # zero players were written across all leagues (broken player endpoints).
    if player_total == 0:
        print("\nERROR: Zero player rows collected despite team success. "
              "Check FotMob player endpoints (type=players).")
        sys.exit(1)

    print("\nFOTMOB collection succeeded (teams + players).")


if __name__ == "__main__":
    collect_fotmob()
