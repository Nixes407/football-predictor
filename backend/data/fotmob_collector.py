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

``season`` must be FotMob's internal numeric season id (e.g. 27110), not the
readable "2026/2027" label, so labels are resolved through the season
directory that the same endpoint exposes. The legacy parameter style
(``teamOrPlayer=team`` with unsuffixed stat names) is also attempted, and the
first successful raw response per endpoint is printed so the exact JSON
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

from database import SessionLocal, TeamSeasonStats
from sqlalchemy import and_


FOTMOB_NAMES_FILE = _BACKEND / "data" / "fotmob_team_names.json"


# FotMob tournament ids, country codes (ccode3) and season labels. Season
# labels use FotMob's readable "YYYY/YYYY" (European) or "YYYY" (MLS and
# Eliteserien) form; they are converted to the database's dash format on upsert.
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
        "id": 55,
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
        "id": 107,
        "ccode": "NOR",
        "season": "2026",
        "prev_season": "2025",
    },
    "AUT-Bundesliga": {
        "id": 44,
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


def _get_seasons(league: str) -> list[str]:
    """Return configured seasons in previous-to-current order for a league."""
    CALENDAR_YEAR_LEAGUES = {
        "USA-MLS",
        "NOR-Eliteserien",
    }
    config = FOTMOB_LEAGUES.get(league, {})
    seasons = []
    if config.get("prev_season"):
        seasons.append(config["prev_season"])
    if config.get("season"):
        seasons.append(config["season"])
    return seasons if seasons else ["2026/2027"]


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
    for key in ("name", "teamName", "ParticipantName"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return None


def _row_value(row: dict) -> float | None:
    """Stat value from a deep-stats row: {"value": x} or a bare number."""
    value = row.get("statValue")
    if isinstance(value, dict):
        for key in ("value", "StatValue"):
            if key in value:
                return _safe(value[key])
        return None
    return _safe(value)


def _row_matches(row: dict) -> float | None:
    """Matches played: explicit fields first, then the substat value."""
    for key in ("matchesPlayed", "MatchesPlayed"):
        number = _safe(row.get(key))
        if number is not None:
            return number
    sub = row.get("substatValue")
    if isinstance(sub, dict):
        for key in ("value", "StatValue"):
            if key in sub:
                return _safe(sub[key])
        return None
    return _safe(sub)


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

    # -- deep stats ----------------------------------------------------

    def _deep_fetch(self, league_id: int, season_param, query: dict):
        """One deep-stats request; returns (payload, rows or None)."""
        payload = self._get("leagueseasondeepstats", {
            "id": league_id, "season": season_param, **query,
        })
        rows = payload.get("statsData")
        return payload, rows if isinstance(rows, list) and rows else None

    def _ordered_queries(self, queries: list) -> list:
        """Try the parameter style that already worked first."""
        if not self._deep_style:
            return list(queries)
        style = self._deep_style

        def matches(query):
            return 0 if all(query.get(k) == v for k, v in style.items()) else 1

        return sorted(queries, key=matches)

    def _resolve_season(self, league_id: int, season: str):
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
            payload, rows = self._deep_fetch(league_id, season, query)
            if rows:
                self._seasons[key] = season
                return season
        # No rows for the label: resolve the internal id via the seasons list.
        season_id = _match_season((payload or {}).get("seasons") or [], season)
        if season_id is None:
            directory = self._get("leagueseasondeepstats", {"id": league_id})
            season_id = _match_season(directory.get("seasons") or [], season)
        if season_id is None:
            raise ValueError(
                f"FotMob publishes no season {season} for league {league_id}"
            )
        self._seasons[key] = season_id
        return season_id

    def get_team_xg_stats(self, league_id: int, season: str) -> dict:
        """Team season stats for one league and season from the deep-stats API.

        Combines one request per stat (expected goals, shots on target and
        possession percentage, plus the goals, saves and clean sheets needed
        for the database mapping) into a single dict keyed by team name:

            {"Arsenal": {"expectedGoals": 35.4, "shotsOnTarget": 120.0,
                         "possession": 58.3, "goals": 70.0, "saves": 41.0,
                         "cleanSheets": 14.0, "matchesPlayed": 38}}
        """
        season_param = self._resolve_season(league_id, season)
        if str(season_param) != str(season):
            print(f"[fotmob] Season {season} -> internal id {season_param}")
        teams = {}
        for logical, queries in DEEP_STAT_QUERIES.items():
            rows = None
            used = None
            for query in self._ordered_queries(queries):
                _, rows = self._deep_fetch(league_id, season_param, query)
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

    # -- standings -----------------------------------------------------

    def get_league_table(self, league_id: int, ccode: str, season: str) -> dict:
        """League standings: matches played, results and goals for/against."""
        payload = self._get("leagues", {
            "id": league_id, "ccode3": ccode, "season": season,
        })
        teams = {}
        entries = payload.get("table")
        if not isinstance(entries, list):
            raise ValueError("FotMob league response has no table")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            data = entry.get("data") or {}
            rows = (data.get("table") or {}).get("all") or []
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
    print("[fotmob] Collecting team stats via the FotMob JSON API...")
    total = 0
    client = _FotMobClient()
    try:
        for league, config in FOTMOB_LEAGUES.items():
            league_id = config["id"]
            for api_season in _get_seasons(league):
                db_season = _db_season(api_season)
                print(f"[fotmob] {league} {api_season}...")
                try:
                    try:
                        table = client.get_league_table(
                            league_id, config["ccode"], api_season,
                        )
                    except Exception as exc:
                        print(f"[fotmob] WARNING: league table failed for "
                              f"{league} {api_season}: {exc}")
                        table = {}
                    try:
                        stats = client.get_team_xg_stats(league_id, api_season)
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

    # Export persisted team names for fixture matching. Keep previously
    # collected leagues when a league's API request fails this time.
    try:
        existing = {}
        if FOTMOB_NAMES_FILE.exists():
            existing = json.loads(FOTMOB_NAMES_FILE.read_text())
        db_check = SessionLocal()
        try:
            rows = db_check.query(
                TeamSeasonStats.team, TeamSeasonStats.league,
            ).distinct().all()
            for team, league in rows:
                if league not in existing:
                    existing[league] = []
                if team not in existing[league]:
                    existing[league].append(team)
        finally:
            db_check.close()
        FOTMOB_NAMES_FILE.write_text(json.dumps(existing, indent=2))
        print(f"[fotmob] Saved team names to {FOTMOB_NAMES_FILE.name}")
    except Exception as exc:
        print(f"[fotmob] WARNING: could not save team names: {exc}")

    print(f"[fotmob] Done. Total rows: {total}")
    return total


def collect_fotmob() -> None:
    """Run FotMob stats collection and fail if no rows were committed."""
    print("=" * 60)
    print("FOTMOB — ADVANCED STATS COLLECTION")
    print("=" * 60)
    print("\n--- TEAM STATS ---")
    team_total = collect_team_stats()
    print("\n" + "=" * 60)
    print("FOTMOB COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Team rows updated: {team_total}")
    if team_total == 0:
        print("\nERROR: Zero rows collected. Check FotMob availability and database access.")
        sys.exit(1)


if __name__ == "__main__":
    collect_fotmob()
