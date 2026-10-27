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

Date-based fixture listing (USA-MLS only):
    GET /api/matches?vts={epoch ms of midnight UTC of the target date}

    This route still lives on the legacy /api hub; _get() prefers /api/data
    and falls back on 404, so it resolves under either layout. FotMob answers
    with every competition playing that day, so fetch_fixtures_for_date()
    filters the payload down to USA-MLS and no other league is served here.

Public API
----------
    collect_fotmob()                    full team + player stats run
    collect_team_stats(client=None)     team_season_stats upserts
    collect_player_stats(client=None)   player_season_stats upserts
    fetch_fixtures_for_date(date_str)   USA-MLS fixtures for one UTC day

    There is deliberately no module-level client instance: requests.Session is
    not thread-safe, so every entry point builds and closes its own.

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
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import (
    SessionLocal, TeamSeasonStats, PlayerSeasonStats,
    cleanup_non_team_rows, ensure_team_corners_column,
    purge_bel_jupiler_rows, upsert_team_season_stats,
)
from sqlalchemy import and_


FOTMOB_NAMES_FILE = _BACKEND / "data" / "fotmob_team_names.json"


# FotMob tournament ids, country codes (ccode3) and season labels. Season
# labels use FotMob's readable "YYYY/YYYY" (European) or "YYYY" (MLS and
# Eliteserien) form; they are converted to the database's dash format on upsert.
#
# Every id below was checked against FotMob's live API on 2026-10-14
# (GET /api/data/leagues?id=... and /api/data/leagueseasondeepstats?id=...).
# Ids that were used or proposed before and are wrong:
#   Belgian ids (40/57/55/62) are NOT supported: those ids point at other
#   countries' competitions and were the source of mislabelled team rows.
#   Never add a BEL league key back to FOTMOB_LEAGUES.
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

# ---------------------------------------------------------------------------
# Date-based fixture listing (see fetch_fixtures_for_date)
# ---------------------------------------------------------------------------
# FotMob's day listing is "GET {hub}/matches?vts={epoch ms of midnight UTC}".
# It is the one endpoint here that is not keyed by league id, so the MLS rows
# have to be selected out of a response that lists every competition playing
# that day.
MATCHES_ENDPOINT = "matches"

# The only league this collector serves fixtures for.
MLS_LEAGUE = "USA-MLS"

# FotMob labels the competition "MLS" and occasionally "Major League Soccer".
MLS_NAME_MARKERS = ("mls", "major league soccer")

# Sub-competitions that repeat the marker and must not be mistaken for the
# first division: matching them would inject reserve/academy clubs into the
# USA-MLS rows that the predictor joins against team_season_stats.
MLS_NAME_EXCLUSIONS = (
    "next", "youth", "academy", "reserve", "u17", "u19", "u21", "u23",
    "is back", "legends", "e-sports", "esports", "virtual",
)

# FotMob's status vocabulary mapped onto the lowercase football-data.org
# tokens (SCHEDULED / IN_PLAY / PAUSED / FINISHED / AWARDED / CANCELED /
# POSTPONED / ABANDONED), so a daily_matches row means the same thing
# whichever source wrote it. Unrecognised values pass through lowercased.
MLS_STATUS_ALIASES = {
    "ns": "scheduled", "timed": "scheduled", "scheduled": "scheduled",
    "not started": "scheduled", "not yet started": "scheduled",
    "follow": "scheduled", "following": "scheduled", "upcoming": "scheduled",
    "live": "in_play", "in play": "in_play", "ongoing": "in_play",
    "1h": "in_play", "2h": "in_play", "et": "in_play", "extra time": "in_play",
    "ht": "paused", "break": "paused", "halftime": "paused",
    "ft": "finished", "finished": "finished", "match finished": "finished",
    "aet": "finished", "pens": "finished", "penalty shootout": "finished",
    "forfeit": "awarded", "awarded": "awarded", "walkover": "awarded",
    "canc": "canceled", "canceled": "canceled", "cancelled": "canceled",
    "postp": "postponed", "postponed": "postponed",
    "int": "interrupted", "interrupted": "interrupted",
    "abd": "abandoned", "abandoned": "abandoned",
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
    "corners": [
        {"stat": "corner_taken_team", "type": "teams"},
        {"stat": "corners_team", "type": "teams"},
        {"stat": "corner_taken", "teamOrPlayer": "team"},
        {"stat": "corners", "teamOrPlayer": "team"},
    ],
    "crosses": [
        {"stat": "accurate_cross_team", "type": "teams"},
        {"stat": "crosses_team", "type": "teams"},
        {"stat": "accurate_cross", "teamOrPlayer": "team"},
        {"stat": "crosses", "teamOrPlayer": "team"},
    ],
    "touchesOppBox": [
        {"stat": "touches_in_opp_box_team", "type": "teams"},
        {"stat": "touches_in_opp_box", "teamOrPlayer": "team"},
    ],
    "shotsTotal": [
        {"stat": "total_scoring_att_team", "type": "teams"},
        {"stat": "shots_team", "type": "teams"},
        {"stat": "total_scoring_att", "teamOrPlayer": "team"},
        {"stat": "shots", "teamOrPlayer": "team"},
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
    "corners": "misc_corners",
    "crosses": "misc_crs",
    "touchesOppBox": "pos_touch_attpen",
    "shotsTotal": "sh_shots",
}

# ---------------------------------------------------------------------------
# Player deep-stats config (type=players)
# ---------------------------------------------------------------------------
# One query per logical player stat. Variants mirror the team collector:
#   * current API style: type=players with "<stat>_player" stat names
#   * bare stat with type=players
#   * legacy: teamOrPlayer=player with bare stat names
# Also try alternative spellings seen in FotMob URLs and worldfootballR docs
# (e.g. expected_goals, expected_assists, xG, fotmob_rating). FotMob's current
# assists key is ``goal_assist`` (title: "Assists").
DEEP_PLAYER_QUERIES = {
    "goals": [
        {"stat": "goals_player", "type": "players"},
        {"stat": "goals", "type": "players"},
        {"stat": "goals", "teamOrPlayer": "player"},
        {"stat": "goals_player_match", "type": "players"},
    ],
    "assists": [
        {"stat": "goal_assist", "type": "players"},
        {"stat": "goal_assist", "teamOrPlayer": "player"},
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
        # Last-resort availability probe only; per-90 values are deliberately
        # not stored in the season-total xG field (see get_player_stats).
        {"stat": "expected_goals_per_90", "type": "players"},
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


def _is_player_leaderboard_row(row: dict) -> bool:
    """True when a deep-stats row describes a player rather than a club.

    FotMob's ``/api/data/leagueseasondeepstats`` ignores ``teamOrPlayer=team``
    when the ``stat`` parameter is a bare player stat name (e.g.
    ``total_scoring_att``, ``accurate_cross``, ``expected_goals``, ``goals``)
    and returns the *player* leaderboard, where ``name`` is the player and
    ``teamName``/``teamId`` is their club. Rejecting those rows prevents
    player names from polluting ``team_season_stats``.
    """
    if not isinstance(row, dict):
        return False
    if any(k in row for k in (
        "playerName", "player_name", "minutesPlayed", "minutes_played",
        "position", "positionId",
    )):
        return True
    name = (
        row.get("name")
        or row.get("ParticipantName")
        or row.get("participant_name")
    )
    team_name = (
        row.get("teamName")
        or row.get("team_name")
        or row.get("ParticipantTeam")
    )
    if isinstance(name, str) and isinstance(team_name, str):
        if name.strip() and team_name.strip():
            if _norm_team(name) != _norm_team(team_name):
                return True
    row_id = _coerce_team_id(row.get("id"))
    team_id = _coerce_team_id(row.get("teamId") or row.get("team_id"))
    if row_id is not None and team_id is not None and row_id != team_id:
        return True
    return False


def _row_team(row: dict) -> str | None:
    """Team name from a deep-stats row, tolerating alternative layouts."""
    if _is_player_leaderboard_row(row):
        return None
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


def _coerce_team_id(value) -> int | None:
    """Parse a FotMob team id without accepting arbitrary display text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            try:
                return int(text)
            except ValueError:
                return None
    if isinstance(value, dict):
        for key in ("id", "teamId", "team_id", "value"):
            if key in value:
                return _coerce_team_id(value[key])
    return None


def _row_team_id(row: dict, *, include_generic_id: bool = False) -> int | None:
    """Team id from a row, excluding a player's own generic ``id`` by default."""
    for key in ("teamId", "team_id", "teamID", "teamid"):
        if key in row:
            team_id = _coerce_team_id(row.get(key))
            if team_id is not None:
                return team_id
    for nested_key in ("team", "Team"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for key in ("teamId", "team_id", "id"):
                if key in nested:
                    team_id = _coerce_team_id(nested.get(key))
                    if team_id is not None:
                        return team_id
    # A participant object in a player leaderboard is commonly the player,
    # so only accept explicit team-id fields from it, never its generic id.
    for nested_key in ("participant", "Participant"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for key in ("teamId", "team_id"):
                if key in nested:
                    team_id = _coerce_team_id(nested.get(key))
                    if team_id is not None:
                        return team_id
    if include_generic_id:
        return _coerce_team_id(row.get("id"))
    return None


def _iter_payload_dicts(payload):
    """Yield nested JSON objects so optional player top-three team names can be used."""
    stack = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            yield value
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def _stats_list_names(payload: dict) -> list[str]:
    """Extract available FotMob stat keys from a payload's statsList field."""
    stats_list = None
    for obj in _iter_payload_dicts(payload):
        for key, value in obj.items():
            if str(key).casefold() == "statslist":
                stats_list = value
                break
        if stats_list is not None:
            break
    names = []

    def collect(value):
        if isinstance(value, str):
            name = value.strip()
            if name and name not in names:
                names.append(name)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for key in ("name", "stat", "statName", "stat_name", "key"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    name = candidate.strip()
                    if name not in names:
                        names.append(name)
                    return
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    collect(child)
                elif isinstance(child, str) and not str(key).isdigit():
                    # Some payloads encode the list as {stat_key: title}.
                    stat_key = str(key).strip()
                    if stat_key and stat_key.casefold() not in {
                        "title", "label", "description", "format",
                    } and stat_key not in names:
                        names.append(stat_key)

    if stats_list is not None:
        collect(stats_list)
    return names


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


def _team_stat_is_per_match(logical: str, stat_name: str, rows: list) -> bool:
    """Detect rate-valued pressure leaderboards that must become totals."""
    if str(stat_name).casefold().endswith("_match"):
        return True
    thresholds = {
        "corners": 15.0,
        "crosses": 30.0,
        "shotsTotal": 30.0,
        "touchesOppBox": 100.0,
    }
    threshold = thresholds.get(logical)
    if threshold is None:
        return False
    values = sorted(
        value for row in rows if isinstance(row, dict)
        for value in [_row_value(row)] if value is not None
    )
    if not values:
        return False
    median = values[len(values) // 2]
    return median < threshold


def _row_matches(row: dict, *, allow_substat: bool = False) -> float | None:
    """Matches played from explicit fields, optionally falling back to substat.

    Team rows may use their secondary substat for matches played and opt in
    with ``allow_substat=True``. Player leaderboards often use that field for
    assists or another metric, so the safe default is to ignore it.
    """
    for key in (
        "matchesPlayed", "MatchesPlayed", "matches_played", "matches-played",
        "mp", "MP", "gamesPlayed", "games_played", "appearances",
    ):
        number = _safe(row.get(key))
        if number is not None:
            return number
    if not allow_substat:
        return None
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
    """Position label from a player row; unknown numeric FotMob codes are omitted."""
    for key in ("position", "Position", "pos", "Pos", "role", "Role", "positionName", "position_name", "PositionName"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
        # FotMob uses numeric position codes (for example 115); without a
        # reliable mapping, do not persist the raw code as position text.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            continue
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
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    continue
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
    # Team-id-only player rows are resolved by the per-league/season map in
    # _FotMobClient.get_player_stats; never use the numeric id as a team name.
    return None


def _parse_scores(scores) -> tuple[float | None, float | None]:
    """Split a standings scoresStr such as '70-24' into goals for/against."""
    if not isinstance(scores, str) or "-" not in scores:
        return None, None
    scored, _, conceded = scores.partition("-")
    return _safe(scored), _safe(conceded)


def _upsert_team(db, team, league, season, attrs) -> None:
    """Upsert mapped team stats through the shared guarded database write.

    The guard rejects blocked leagues before any DB work; the canonical
    implementation also skips None values so a partial feed never clears a
    stat another collector stored.
    """
    upsert_team_season_stats(db, team, league, season, attrs)


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
    # Position labels are strings; unknown numeric FotMob position codes must
    # never be persisted as display text, and None must not clear old values.
    if isinstance(position, str):
        pos_norm = " ".join(position.split())
        if pos_norm:
            existing.position = pos_norm
    for key, value in attrs.items():
        if key not in PlayerSeasonStats.__table__.columns:
            raise ValueError(f"Unknown PlayerSeasonStats field: {key}")
        # Missing or invalid values never clear an existing non-null value.
        safe = _safe(value)
        if safe is None:
            continue
        if key == "mp":
            # mp is Integer; _safe rejects NaN/infinity before conversion.
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
    for k in (
        "matchesPlayed", "MatchesPlayed", "matches_played", "matches-played",
        "mp", "MP", "gamesPlayed", "games_played", "appearances",
    ):
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
    """Deep-stats entries keyed by team, with standings-only teams appended.

    When the authoritative league table is available, any deep-stats key not
    in ``table`` (e.g. a player name that leaked from a fallback endpoint)
    is dropped so ``team_season_stats`` only ever receives actual clubs in
    that league and season.
    """
    if table:
        table_by_norm = {_norm_team(name): name for name in table}
        teams: dict = {}
        for name, entry in stats.items():
            canonical = table_by_norm.get(_norm_team(name))
            if canonical is not None:
                teams[canonical] = entry
        for name in table:
            if name not in teams:
                teams[name] = {}
        return teams
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
        self._team_id_names = {}     # (league_id, season label/id) -> {team_id: name}
        self._team_id_map_attempted = set()
        self._team_id_map_probe_attempted = set()
        self._team_table_row_counts = {}
        self._deep_style = None      # team-param style that worked
        self._player_style = None    # player-param style that worked
        self._player_samples = set() # player deep-stats sample keys
        self._player_statslist_logged = set()
        self._player_team_map_logged = set()
        self._team_statslist_logged = set()

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

    def _team_id_map_keys(self, league_id: int, season) -> list[tuple[int, str]]:
        """Cache keys for both a readable season label and its internal id."""
        keys = [(league_id, str(season))]
        season_param = self._seasons.get((league_id, str(season)))
        if season_param is not None:
            internal_key = (league_id, str(season_param))
            if internal_key not in keys:
                keys.append(internal_key)
        return keys

    def _cache_team_id_name(self, league_id: int, season, team_id, name,
                            *, overwrite: bool = False) -> None:
        """Remember a normalized team name under label and resolved season id."""
        team_id = _coerce_team_id(team_id)
        if team_id is None or not isinstance(name, str) or not name.strip():
            return
        name = " ".join(name.split())
        for key in self._team_id_map_keys(league_id, season):
            names = self._team_id_names.setdefault(key, {})
            if overwrite or team_id not in names:
                names[team_id] = name

    def _cache_team_table_size(self, league_id: int, season, count: int) -> None:
        for key in self._team_id_map_keys(league_id, season):
            self._team_table_row_counts[key] = count

    def _team_id_map_for(self, league_id: int, season) -> dict[int, str]:
        names = {}
        for key in self._team_id_map_keys(league_id, season):
            names.update(self._team_id_names.get(key, {}))
        return names

    def _build_team_id_map(self, league_id: int, season: str,
                           ccode: str | None = None,
                           country: str | None = None) -> dict[int, str]:
        """Build the season's team-id map from standings and available team stats."""
        self._resolve_season(league_id, season, country)
        if not ccode:
            ccode = next((
                config.get("ccode") for config in FOTMOB_LEAGUES.values()
                if config.get("id") == league_id
            ), None)

        key = (league_id, str(season))
        if ccode and key not in self._team_id_map_attempted:
            self._team_id_map_attempted.add(key)
            try:
                # get_league_table also records id/name pairs while retaining
                # its existing name-keyed return shape for team collection.
                self.get_league_table(league_id, ccode, season, country)
            except Exception as exc:
                print(f"[fotmob] WARNING: teamId map standings lookup failed for "
                      f"league {league_id} {season}: {exc}")

        names = self._team_id_map_for(league_id, season)
        expected_count = max((
            self._team_table_row_counts.get(cache_key, 0)
            for cache_key in self._team_id_map_keys(league_id, season)
        ), default=0)
        if expected_count == 0 or len(names) < expected_count or not names:
            if key not in self._team_id_map_probe_attempted:
                self._team_id_map_probe_attempted.add(key)
                season_param = self._seasons.get(key, season)
                # One team-stat logical query is a cheap fallback when the
                # standings payload lacks ids or is unavailable.
                for query in self._ordered_queries(DEEP_STAT_QUERIES["expectedGoals"]):
                    try:
                        _, rows = self._deep_fetch(
                            league_id, season_param, query, country,
                        )
                    except Exception:
                        continue
                    if not rows:
                        continue
                    resolved = 0
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        team = _row_team(row)
                        team_id = _row_team_id(row, include_generic_id=True)
                        if team is not None and team_id is not None:
                            self._cache_team_id_name(
                                league_id, season, team_id, team, overwrite=True,
                            )
                            resolved += 1
                    if resolved:
                        print(f"[fotmob] Team deep-stats id/name probe: "
                              f"league {league_id} {season}, mapped {resolved} rows")
                        break
                names = self._team_id_map_for(league_id, season)
        return names

    def _log_player_stats_list(self, payload: dict, league_id: int,
                               season: str) -> None:
        """Log available stat keys once for each league's first player response."""
        if league_id in self._player_statslist_logged:
            return
        self._player_statslist_logged.add(league_id)
        names = _stats_list_names(payload)
        print(f"[fotmob] Player statsList names for league {league_id} {season}: "
              f"{names if names else 'not present in payload'}")

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
        """Reuse known styles without demoting FotMob's canonical stat keys."""
        queries = list(queries)
        # Keep both goal_assist variants first, in their configured order. The
        # style learned from goals must not move old assists fallbacks ahead of
        # FotMob's confirmed current assists key.
        canonical = [q for q in queries if q.get("stat") == "goal_assist"]
        # Per-90 xG is an availability fallback only, never a season-total value.
        per90 = [
            q for q in queries
            if "per_90" in str(q.get("stat", "")) or "_per90" in str(q.get("stat", ""))
        ]
        remainder = [q for q in queries if q not in canonical and q not in per90]
        if self._player_style:
            style = self._player_style

            def matches(query):
                return 0 if all(query.get(k) == v for k, v in style.items()) else 1

            remainder = sorted(remainder, key=matches)
        return canonical + remainder + per90

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
            payload = None
            rows = None
            used = None
            for query in self._ordered_queries(queries):
                payload, candidate_rows = self._deep_fetch(
                    league_id, season_param, query, country,
                )
                if not candidate_rows:
                    continue
                # Reject player leaderboards returned by FotMob when a legacy
                # teamOrPlayer=team fallback uses a bare player stat name.
                if any(
                    _is_player_leaderboard_row(r)
                    for r in candidate_rows
                    if isinstance(r, dict)
                ):
                    continue
                rows = candidate_rows
                used = query
                break
            if not rows:
                print(f"[fotmob] WARNING: no {logical} stats for "
                      f"league {league_id} {season}")
                if logical in {"corners", "crosses", "shotsTotal"}:
                    print(f"[fotmob] {logical} rows fetched: 0; merged teams: 0")
                continue
            style = {k: v for k, v in used.items() if k != "stat"}
            if style.get("type") == "teams" and self._deep_style != style:
                self._deep_style = style
                print(f"[fotmob] Deep-stats params that worked: {style}")
            if league_id not in self._team_statslist_logged and payload is not None:
                self._team_statslist_logged.add(league_id)
                names = _stats_list_names(payload)
                print(f"[fotmob] Team statsList names for league {league_id} {season}: "
                      f"{names if names else 'not present in payload'}")
            stat_name = str(used.get("stat", ""))
            per_match = _team_stat_is_per_match(logical, stat_name, rows)
            if logical in {"corners", "crosses", "touchesOppBox", "shotsTotal"}:
                units = "per-match (scaling by matchesPlayed)" if per_match else "season totals"
                print(f"[fotmob] {logical}: stat={stat_name} worked; detected {units}")
            skipped_unscaled = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                team = _row_team(row)
                team_id = _row_team_id(row, include_generic_id=True)
                if team is not None and team_id is not None:
                    # Deep team rows are a second teamId/name source and use
                    # the same whitespace normalization as team upserts.
                    self._cache_team_id_name(
                        league_id, season, team_id, team, overwrite=True,
                    )
                if team is None:
                    continue
                entry = teams.setdefault(team, {})
                matches = _row_matches(row, allow_substat=True) or entry.get("matchesPlayed")
                if matches is not None:
                    entry["matchesPlayed"] = matches
                value = _row_value(row)
                if value is None:
                    continue
                if per_match:
                    if matches:
                        value = value * matches
                    else:
                        # A rate in a season-total column is worse than no
                        # value; preserve any existing DB value instead.
                        skipped_unscaled += 1
                        continue
                entry[logical] = value
            if skipped_unscaled:
                print(f"[fotmob] WARNING: {logical} skipped for "
                      f"{skipped_unscaled} teams (per-match values without matchesPlayed)")
            if logical in {"corners", "crosses", "shotsTotal"}:
                merged = sum(1 for entry in teams.values() if entry.get(logical) is not None)
                print(f"[fotmob] {logical} rows fetched: {len(rows)}; merged teams: {merged}")
        corners_non_null = sum(
            1 for entry in teams.values() if entry.get("corners") is not None
        )
        print(f"[fotmob] league {league_id} {season}: "
              f"{corners_non_null} teams got misc_corners non-null after merge")
        return teams

    def get_player_stats(self, league_id: int, season: str,
                         country: str | None = None,
                         ccode: str | None = None) -> dict:
        """Fetch and merge player deep-stats, resolving teamId-only rows.

        Results remain keyed by (team name, player name). Team names are
        resolved from the league table and team deep-stats caches; rows whose
        team id remains unknown are skipped rather than stored under a number.
        """
        season_param = self._resolve_season(league_id, season, country)
        if str(season_param) != str(season):
            print(f"[fotmob] Season {season} -> internal id {season_param} (players)")
        team_id_names = self._build_team_id_map(
            league_id, season, ccode=ccode, country=country,
        )
        players: dict[tuple[str, str], dict] = {}
        first_payload_logged = False
        team_id_resolved_rows = 0
        unresolved_team_id_rows = 0
        missing_team_rows = 0
        total_rows_merged = 0

        for logical, queries in DEEP_PLAYER_QUERIES.items():
            payload = None
            rows = None
            used = None
            for query in self._ordered_player_queries(queries):
                try:
                    payload, rows = self._deep_fetch(
                        league_id, season_param, query, country,
                    )
                except Exception:
                    # Try the remaining variants for this stat; a later
                    # logical stat must still be allowed to succeed.
                    continue
                if rows:
                    used = query
                    break
            if not rows:
                print(f"[fotmob] {logical} player rows fetched: 0; "
                      "merged after team resolve: 0")
                print(f"[fotmob] WARNING: no {logical} player stats for "
                      f"league {league_id} {season}")
                continue

            style = {k: v for k, v in used.items() if k != "stat"}
            if self._player_style != style:
                self._player_style = style
                print(f"[fotmob] Player deep-stats params that worked: {style}")

            if not first_payload_logged and payload is not None:
                try:
                    key = f"leagueseasondeepstats:players:{logical}"
                    if key not in self._player_samples:
                        self._player_samples.add(key)
                        print(f"[fotmob] Player sample: league {league_id} {season} {logical} -> "
                              f"stat={used.get('stat')} rows={len(rows)} "
                              f"top_keys={list(payload.keys())[:8]}")
                        if isinstance(rows[0], dict):
                            print(f"[fotmob] Player sample row keys: {list(rows[0].keys())}")
                            print(f"[fotmob] Player sample row: "
                                  f"{json.dumps(rows[0])[:SAMPLE_CHARS]}")
                except (TypeError, ValueError):
                    pass
                first_payload_logged = True
            if payload is not None:
                self._log_player_stats_list(payload, league_id, season)

            stat_name = str(used.get("stat", "")).casefold()
            per_match = stat_name.endswith("_match")
            per90 = "_per_90" in stat_name or "_per90" in stat_name
            if per90:
                print(f"[fotmob] WARNING: {logical} returned per-90 values from "
                      f"{used.get('stat')}; not storing them as season totals")

            unscaled = 0
            rows_merged = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                player = _row_player(row)
                if player is None:
                    continue
                team = _row_player_team(row)
                if team is None:
                    team_id = _row_team_id(row)
                    if team_id is None:
                        missing_team_rows += 1
                        continue
                    team = team_id_names.get(team_id)
                    if team is None:
                        unresolved_team_id_rows += 1
                        continue
                    team_id_resolved_rows += 1

                # Use the same whitespace normalization as _row_team and team
                # upserts, retaining FotMob's original capitalization.
                team = " ".join(team.split())
                player = " ".join(player.split())
                key = (team, player)
                entry = players.setdefault(key, {"team": team, "player": player})
                rows_merged += 1

                # Only explicit appearance fields count for player matches;
                # leaderboard substatValue may instead be assists or another stat.
                matches = _row_matches(row, allow_substat=False)
                if matches is not None and entry.get("matchesPlayed") is None:
                    entry["matchesPlayed"] = matches
                minutes = _row_minutes(row)
                if minutes is not None and entry.get("minutes") is None:
                    entry["minutes"] = minutes
                position = _row_position(row)
                if position and not entry.get("position"):
                    entry["position"] = position

                # A per-90 fallback is intentionally not converted into a
                # season total, even if minutes or appearances are present.
                if per90:
                    continue
                value = _row_value(row)
                if value is None:
                    continue
                if per_match:
                    matches = matches or entry.get("matchesPlayed")
                    if matches:
                        value *= matches
                    else:
                        unscaled += 1
                entry[logical] = value

            total_rows_merged += rows_merged
            print(f"[fotmob] {logical} player rows fetched: {len(rows)}; "
                  f"merged after team resolve: {rows_merged}")
            if unscaled:
                print(f"[fotmob] WARNING: {logical} player kept as a per-match value "
                      f"for {unscaled} players (no match counts available)")

        summary_key = (league_id, str(season))
        if summary_key not in self._player_team_map_logged:
            self._player_team_map_logged.add(summary_key)
            print(f"[fotmob] {league_id} {season}: teamId→name map resolved "
                  f"{len(team_id_names)} teamIds; player rows resolved via teamId "
                  f"{team_id_resolved_rows}; skipped {unresolved_team_id_rows} rows "
                  f"with unresolved teamId ({missing_team_rows} rows had no team name/id)")
            print(f"[fotmob] {league_id} {season}: {len(players)} merged player rows "
                  f"after team resolve from {total_rows_merged} stat rows")
        return players

    # -- standings -----------------------------------------------------

    def get_league_table(self, league_id: int, ccode: str, season: str,
                         country: str | None = None) -> dict:
        """League standings: matches played, results and goals for/against.

        Raises ValueError when FotMob answers with another country's league
        (wrong id) or with a different season than requested: for an
        unknown season label FotMob silently returns the latest table.
        """
        self._team_id_map_attempted.add((league_id, str(season)))
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
        # The leagues response can also carry stats.players.topThree rows with
        # explicit teamId/teamName. Use those names only as a supplementary
        # source; standings and team deep-stats rows remain authoritative.
        for candidate in _iter_payload_dicts(payload):
            team_id = _row_team_id(candidate)
            team_name = _row_player_team(candidate)
            if team_id is not None and team_name:
                self._cache_team_id_name(
                    league_id, season, team_id, team_name, overwrite=False,
                )
        teams = {}
        entries = payload.get("table")
        if not isinstance(entries, list):
            raise ValueError("FotMob league response has no table")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            data = entry.get("data") or {}
            if not isinstance(data, dict):
                continue
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
                    if not isinstance(row, dict):
                        continue
                    name = row.get("name")
                    if not isinstance(name, str) or not name.strip():
                        continue
                    name = " ".join(name.split())
                    team_id = _row_team_id(row, include_generic_id=True)
                    if team_id is not None:
                        self._cache_team_id_name(
                            league_id, season, team_id, name, overwrite=True,
                        )
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
        self._cache_team_table_size(league_id, season, len(teams))
        return teams

    def close(self) -> None:
        self.session.close()


# ---------------------------------------------------------------------------
# Fixtures for one UTC day (USA-MLS)
# ---------------------------------------------------------------------------


def _midnight_utc_ms(day: date) -> int:
    """Epoch milliseconds of midnight UTC for ``day`` (FotMob's ``vts``)."""
    return int(
        datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        .timestamp() * 1000
    )


def _fixture_date(value) -> date | None:
    """Parse the leading ``YYYY-MM-DD`` of a date or ISO timestamp."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _match_name(*values):
    """A FotMob team name from a string field or a nested team object."""
    for value in values:
        if isinstance(value, str):
            text = " ".join(value.split())
            if text:
                return text
        elif isinstance(value, dict):
            inner = _match_name(
                value.get("name"), value.get("team"), value.get("text"),
                value.get("rawName"),
            )
            if inner:
                return inner
    return None


def _league_entry_name(entry: dict) -> str | None:
    """League name of a ``leagues[]`` entry, nested or flat."""
    for candidate in (entry.get("league"), entry):
        if isinstance(candidate, dict):
            name = _match_name(
                candidate.get("name"), candidate.get("shortName"),
                candidate.get("short"), candidate.get("label"),
            )
            if name:
                return name
    return None


def _is_mls_league(name: str | None) -> bool:
    """True only for the top-flight USA-MLS competition."""
    if not name:
        return False
    text = name.casefold()
    if not any(marker in text for marker in MLS_NAME_MARKERS):
        return False
    return not any(excluded in text for excluded in MLS_NAME_EXCLUSIONS)


def _match_kickoff(match: dict) -> str | None:
    """Kickoff as an ISO-8601 UTC string.

    ``utcTime`` is the documented field; a bare epoch (seconds or
    milliseconds) is accepted as a fallback because the late-UTC USA-MLS
    spillover compares kickoff times, and a row without one is invisible to
    that comparison.
    """
    for key in ("utcTime", "utcDate", "kickoffTime", "kickoff"):
        value = match.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("start", "kickoffEpoch", "matchTime"):
        value = match.get(key)
        number = _safe(value)
        if number is None:
            continue
        seconds = number / 1000.0 if abs(number) > 1e11 else number
        try:
            return (
                datetime.fromtimestamp(seconds, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (OverflowError, OSError, ValueError):
            continue
    return None


def _looks_like_score(text: str) -> bool:
    """True for "2 - 1" / "2:1" / "2" — a score, not a match status.

    FotMob frequently puts the scoreline in ``status.statusStr`` for finished
    and live matches, so a score-shaped token must never be stored as a
    fixture's status.
    """
    if text.isdigit():
        return True
    flattened = text.replace("\u2013", "-").replace(":", "-")
    parts = [part.strip() for part in flattened.split("-")]
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def _match_status(match: dict) -> str | None:
    """FotMob's match status as the lowercase token football-data.org uses."""
    raw = match.get("status")
    if isinstance(raw, dict):
        # ``description`` and ``code`` come first: ``statusStr`` is often the
        # scoreline rather than a state.
        raw = (
            raw.get("description") or raw.get("code") or raw.get("short")
            or raw.get("statusStr") or raw.get("status")
        )
    if raw is None or raw == "":
        raw = match.get("statusStr")
    if raw is None or raw == "":
        return None
    token = " ".join(str(raw).split()).casefold()
    if not token or _looks_like_score(token):
        return None
    return MLS_STATUS_ALIASES.get(token, token) or None


def _matches_of_entry(entry: dict) -> list:
    """The match rows of one ``leagues[]`` entry, nested or flat."""
    rows = entry.get("matches")
    if isinstance(rows, list):
        return rows
    nested = entry.get("league")
    if isinstance(nested, dict) and isinstance(nested.get("matches"), list):
        return nested["matches"]
    return []


def fetch_fixtures_for_date(target_date: str) -> list[dict]:
    """Return USA-MLS fixtures kicking off on ``target_date`` (YYYY-MM-DD, UTC).

    Queries FotMob's public date-based listing
    (``GET {hub}/api/matches?vts={epoch ms of midnight UTC}``) and keeps the
    MLS rows, so the day's other competitions are ignored and no other league
    is served from this endpoint.

    Returns a list of dicts in ``data.fixtures``' own fixture shape, ready for
    ``compute_and_store_predictions``::

        {"external_id": "5058691",
         "match_date":  "2026-10-17",          # the kickoff's UTC date
         "kickoff_time":"2026-10-17T23:30Z",
         "home_team":   "Atlanta United",
         "away_team":   "Inter Miami CF",
         "league":      "USA-MLS",
         "season":      "2026",                 # calendar-year league
         "status":      "scheduled"}

    Any failure is logged and answered with ``[]``: a day of missing MLS
    fixtures must never break the football-data.org leagues.
    """
    day = _fixture_date(target_date)
    if day is None:
        print(f"[fotmob] WARNING: ignoring an unparsable fixture date "
              f"{target_date!r}")
        return []

    target_date = day.isoformat()
    # MLS is a calendar-year league and FotMob stores its seasons as "2026",
    # so the season is the target day's year. derive_season()'s July-based
    # "2025-2026" form is wrong here: no MLS row in team_season_stats
    # carries it, and the lookup would silently miss every spring fixture.
    season = str(day.year)

    fixtures: list[dict] = []
    client = _FotMobClient()
    try:
        payload = client._get(MATCHES_ENDPOINT, {"vts": _midnight_utc_ms(day)})
        entries = payload.get("leagues")
        if not isinstance(entries, list):
            print(f"[fotmob] WARNING: FotMob {MATCHES_ENDPOINT} response for "
                  f"{target_date} has no 'leagues' list "
                  f"(keys: {list(payload)[:12]})")
            return []

        seen: set[str] = set()
        skipped = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not _is_mls_league(_league_entry_name(entry)):
                continue
            for match in _matches_of_entry(entry):
                if not isinstance(match, dict):
                    skipped += 1
                    continue
                home = _match_name(match.get("home"), match.get("homeTeam"),
                                   match.get("homeTeamName"))
                away = _match_name(match.get("away"), match.get("awayTeam"),
                                   match.get("awayTeamName"))
                external_id = match.get("id") or match.get("matchId")
                if not home or not away or external_id in (None, ""):
                    skipped += 1
                    continue
                external_id = str(external_id).strip()
                if not external_id or external_id in seen:
                    continue
                seen.add(external_id)
                kickoff = _match_kickoff(match)
                match_day = _fixture_date(kickoff) if kickoff else None
                fixtures.append({
                    "external_id": external_id,
                    # The kickoff's own UTC date, so a match that spills past
                    # midnight lands on the day /fixtures/today will query.
                    "match_date": (match_day or day).isoformat(),
                    "kickoff_time": kickoff,
                    "home_team": home,
                    "away_team": away,
                    "league": MLS_LEAGUE,
                    "season": season,
                    "status": _match_status(match),
                })
        if skipped:
            print(f"[fotmob] Skipped {skipped} unusable FotMob MLS rows for "
                  f"{target_date} (missing team, id or kickoff)")
    except Exception as exc:
        print(f"[fotmob] WARNING: USA-MLS fixture fetch for {target_date} "
              f"failed: {exc}")
        return []
    finally:
        client.close()

    print(f"[fotmob] Found {len(fixtures)} USA-MLS fixtures for {target_date}")
    return fixtures


def _store_fotmob_name(db, fotmob_name: str, league: str) -> None:
    """The upsert already stores the API spelling in TeamSeasonStats.team.

    The league/name pairs are exported from committed rows at the end of
    collection, so there is no separate table or additional write here.
    """
    pass


def collect_team_stats(client: _FotMobClient | None = None) -> int:
    """Collect both seasons for all leagues; return successfully committed rows.

    A caller may pass a shared client so its season and team-id caches are
    reused by player collection; no-argument public use remains supported.
    """
    # Daily jobs do not necessarily pass through FastAPI's init_db startup.
    # Evolve the existing production table before ORM writes reference the
    # newly mapped column.
    if not ensure_team_corners_column():
        raise RuntimeError("team_season_stats.misc_corners is not ready")

    # Clean up potentially corrupted Belgian data and any non-team/player rows.
    # The purge is the same one-time cleanup the API runs at startup; after the
    # first clean run it is a no-op (and the guarded upsert below prevents any
    # blocked league from being re-inserted).
    cleanup_non_team_rows()
    purge_bel_jupiler_rows()

    print("[fotmob] Collecting team stats via the FotMob JSON API...")
    total = 0
    owns_client = client is None
    if client is None:
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
        if owns_client:
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


def collect_player_stats(client: _FotMobClient | None = None) -> int:
    """Collect both seasons player stats for all leagues; return committed rows.

    Loops all FOTMOB_LEAGUES × seasons, fetches player deep-stats (goals,
    assists, rating, expected_goals, expected_assists) via the same
    leagueseasondeepstats hub as team stats, merges them per (team, player),
    and upserts into player_season_stats. Never crashes the whole run on one
    league failure — logs ERROR and continues. Returns total players committed.
    """
    print("[fotmob] Collecting PLAYER stats via the FotMob JSON API...")
    total = 0
    owns_client = client is None
    if client is None:
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
                            ccode=config["ccode"],
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
        if owns_client:
            client.close()

    print(f"[fotmob] Done. Total player rows: {total}")
    return total


def collect_fotmob() -> None:
    """Run FotMob stats collection (teams + players) and enforce success rules."""
    print("=" * 60)
    print("FOTMOB — ADVANCED STATS COLLECTION")
    print("=" * 60)
    client = _FotMobClient()
    try:
        print("\n--- TEAM STATS ---")
        try:
            team_total = collect_team_stats(client=client)
        except Exception as exc:
            print(f"[fotmob] ERROR: team collection crashed: {exc}")
            import traceback
            traceback.print_exc()
            team_total = 0
        print("\n--- PLAYER STATS ---")
        try:
            player_total = collect_player_stats(client=client)
        except Exception as exc:
            print(f"[fotmob] ERROR: player collection crashed: {exc}")
            import traceback
            traceback.print_exc()
            player_total = 0
    finally:
        client.close()

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
