"""
API-Football data collector.

Run from backend/ directory:

    python data/api_football_collector.py

Requests are paced to the key's per-minute limit (10/min on the free plan)
and API errors (plan, quota, parameters), which arrive as HTTP 200 with an
empty response, are printed instead of looking like "0 items".
"""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, TeamSeasonStats, PlayerSeasonStats
from sqlalchemy import and_

API_KEY  = os.environ.get("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"

LEAGUE_MAP = {
    "ENG-Premier League": 39,
    "ENG-Championship":   40,
    "ESP-La Liga":        140,
    "GER-Bundesliga":     78,
    "ITA-Serie A":        135,
    "FRA-Ligue 1":        61,
    "POR-Primeira Liga":  94,
    "NED-Eredivisie":     88,
    "BEL-Jupiler Pro League": 144,
    "TUR-Süper Lig":      203,
    "NOR-Eliteserien":    103,
    "AUT-Bundesliga":     218,
    "USA-MLS":            253,
    "UEFA-Champions League": 2,
}

# API-Football seasons are 4-digit start years: 2025 is the 2025-2026
# season of a European league and the 2025 calendar season of MLS and
# Eliteserien. Free plans cannot read recent seasons; the API then answers
# HTTP 200 with no rows and errors={"plan": ...}, which _get() prints.
SEASON_MAP = {
    "ENG-Premier League": [2025, 2026],
    "ENG-Championship":   [2025, 2026],
    "ESP-La Liga":        [2025, 2026],
    "GER-Bundesliga":     [2025, 2026],
    "ITA-Serie A":        [2025, 2026],
    "FRA-Ligue 1":        [2025, 2026],
    "POR-Primeira Liga":  [2025, 2026],
    "NED-Eredivisie":     [2025, 2026],
    "BEL-Jupiler Pro League": [2025, 2026],
    "TUR-Süper Lig":      [2025, 2026],
    "NOR-Eliteserien":    [2025, 2026],
    "AUT-Bundesliga":     [2025, 2026],
    "USA-MLS":            [2025, 2026],
    "UEFA-Champions League": [2025, 2026],
}

SEASON_LABEL = {
    (39,  2025): "2025-2026", (39,  2026): "2026-2027",
    (40,  2025): "2025-2026", (40,  2026): "2026-2027",
    (140, 2025): "2025-2026", (140, 2026): "2026-2027",
    (78,  2025): "2025-2026", (78,  2026): "2026-2027",
    (135, 2025): "2025-2026", (135, 2026): "2026-2027",
    (61,  2025): "2025-2026", (61,  2026): "2026-2027",
    (94,  2025): "2025-2026", (94,  2026): "2026-2027",
    (88,  2025): "2025-2026", (88,  2026): "2026-2027",
    (253, 2025): "2025",      (253, 2026): "2026",
    (2,   2025): "2025-2026", (2,   2026): "2026-2027",
    (144, 2025): "2025-2026",
    (144, 2026): "2026-2027",
    (203, 2025): "2025-2026",
    (203, 2026): "2026-2027",
    (103, 2025): "2025",
    (103, 2026): "2026",
    (218, 2025): "2025-2026",
    (218, 2026): "2026-2027",
}

# Leagues whose injuries and recent form are fetched. The newer leagues
# (BEL, TUR, NOR, AUT, MLS) are left out to stay inside the request budget;
# their form still comes from the standings read by collect_team_stats.
INJURY_LEAGUES = {
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "ENG-Championship",
    "POR-Primeira Liga",
    "NED-Eredivisie",
    "UEFA-Champions League",
}

# API-Football limits requests per minute as well as per day: Free 10/min,
# Pro 300/min, Ultra 450/min, Mega 900/min
# (https://www.api-football.com/news/post/how-ratelimit-works).
# A fixed 1.2 s gap sends ~50 requests a minute, so free keys got HTTP 429
# from the eleventh call on. Requests are now spaced by 60 / limit seconds
# using the X-RateLimit-Limit header, at the free-plan pace until the first
# response reports the key's real limit. A 429 is retried once the minute
# window has passed.
MIN_REQUEST_GAP  = 1.2   # seconds; floor for paid plans
FREE_PLAN_GAP    = 6.5   # seconds; 10 requests a minute plus a margin
RATE_LIMIT_WAIT  = 65    # seconds to wait after HTTP 429
MAX_RATE_RETRIES = 2

_calls_made      = 0
_request_gap     = FREE_PLAN_GAP
_last_request    = 0.0
_quota_exhausted = False
# Standings by (league id, season): collect_team_stats and
# collect_team_form read the same tables, so each is fetched once per run.
_standings_cache: dict[tuple[int, int], dict] = {}


def _safe(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _pct(val) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


# API-Football reports a player's role as a full word ("Midfielder", in
# statistics[].games.position), while player_season_stats.position holds
# the FBref codes (GK/DF/MF/FW) that the prediction engine matches on
# (see formula_engine._position_groups). Collected roles are normalised to
# the FBref codes so both sources agree.
POSITION_MAP = {
    "goalkeeper": "GK",
    "defender":   "DF",
    "midfielder": "MF",
    "attacker":   "FW",
    "forward":    "FW",
}


def _norm_position(value) -> str | None:
    """FBref-style position code for an API-Football role string."""
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.upper() in ("GK", "DF", "MF", "FW"):
        return raw.upper()
    return POSITION_MAP.get(raw.lower())


def _api_errors(payload) -> dict:
    """The ``errors`` object of an API-Football reply, as a dict.

    The API reports quota, plan and parameter problems with HTTP 200, an
    empty ``response`` and the reason in ``errors`` (``[]`` when fine),
    e.g. {"requests": "You have reached the request limit for the day"}
    or {"plan": "Free plans do not have access to this season, ..."}.
    """
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if isinstance(errors, dict):
        return errors
    if isinstance(errors, list) and errors:
        return {"errors": errors}
    return {}


def _pace(headers) -> None:
    """Space requests to the per-minute limit the API reports for this key."""
    global _request_gap
    limit = _safe(headers.get("x-ratelimit-limit"))
    if limit and limit > 0:
        _request_gap = max(MIN_REQUEST_GAP, 60.0 / limit + 0.5)


def _get(endpoint: str, params: dict) -> dict | None:
    global _calls_made, _last_request, _quota_exhausted
    if not API_KEY:
        print("[api] ERROR: API_FOOTBALL_KEY not set")
        sys.exit(1)
    if _quota_exhausted:
        print(f"[api] SKIP {endpoint}: daily request quota exhausted")
        return None
    headers = {"x-apisports-key": API_KEY}
    url     = f"{BASE_URL}/{endpoint}"
    for attempt in range(MAX_RATE_RETRIES + 1):
        wait = _request_gap - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(
                url, headers=headers,
                params=params, timeout=30)
        except Exception as exc:
            print(f"[api] ERROR: {endpoint}: {exc}")
            return None
        finally:
            _last_request = time.monotonic()
        _calls_made += 1
        _pace(resp.headers)
        try:
            data = resp.json()
        except ValueError:
            data = None
        errors = _api_errors(data)

        if resp.status_code == 429 and "requests" not in errors:
            if attempt < MAX_RATE_RETRIES:
                print(f"[api] WARNING: HTTP 429 for {endpoint} "
                      f"(per-minute limit) - retrying in "
                      f"{RATE_LIMIT_WAIT}s "
                      f"({attempt + 1}/{MAX_RATE_RETRIES})")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            print(f"[api] WARNING: HTTP 429 for {endpoint} - "
                  f"giving up after {MAX_RATE_RETRIES} retries")
            return None
        if "requests" in errors:
            # Daily quota used up: every further call would fail as well.
            _quota_exhausted = True
            print(f"[api] ERROR: {errors['requests']} - "
                  f"skipping the remaining API calls")
        if resp.status_code != 200 or not isinstance(data, dict):
            print(f"[api] WARNING: HTTP "
                  f"{resp.status_code} for {endpoint}")
            return None

        remaining = resp.headers.get(
            "x-ratelimit-requests-remaining", "?")
        n = len(data.get("response") or [])
        print(f"[api] [{_calls_made}] {endpoint} "
              f"-> {n} items "
              f"(remaining: {remaining})")
        if errors and "requests" not in errors:
            print(f"[api] API error for {endpoint} "
                  f"{params}: {errors}")
        return data
    return None


def _get_standings(league_id: int, season: int) -> dict | None:
    """Standings for one league season, fetched at most once per run."""
    key = (league_id, season)
    if key not in _standings_cache:
        data = _get("standings", {
            "league": league_id,
            "season": season,
        })
        if data is None:
            return None
        _standings_cache[key] = data
    return _standings_cache[key]


def check_status() -> bool:
    data = _get("status", {})
    if not data:
        return False
    errors = _api_errors(data)
    r = data.get("response")
    if errors or not isinstance(r, dict):
        print(f"[api] ERROR: status check failed: "
              f"{errors or 'no account data'}")
        return False
    acct = r.get("account") or {}
    sub  = r.get("subscription") or {}
    reqs = r.get("requests") or {}
    print(f"[api] Account: {acct.get('firstname')} "
          f"{acct.get('lastname')}")
    print(f"[api] Plan: {sub.get('plan')}")
    print(f"[api] Calls today: "
          f"{reqs.get('current')} / "
          f"{reqs.get('limit_day')}")
    return True


def _apply_attrs(existing, attrs) -> None:
    """
    Write collected values onto an ORM row, skipping ``None``.

    A collector that only fills part of a row (the standings feed has no
    clean sheets, shots or cards) passes None for what it could not read.
    Writing those would wipe a value another collector had stored — the
    daily API-Football run was clearing the clean-sheet column that FBref
    fills — so None means "unknown, leave what is there".
    """
    for k, v in attrs.items():
        if v is None:
            continue
        setattr(existing, k, v)


def _upsert_team(db, team, league, season, attrs):
    existing = (
        db.query(TeamSeasonStats)
        .filter(and_(
            TeamSeasonStats.team   == team,
            TeamSeasonStats.league == league,
            TeamSeasonStats.season == season,
        )).first()
    )
    if existing is None:
        existing = TeamSeasonStats(
            team=team, league=league, season=season)
        db.add(existing)
    _apply_attrs(existing, attrs)


def _upsert_player(db, player, team,
                   league, season, attrs):
    existing = (
        db.query(PlayerSeasonStats)
        .filter(and_(
            PlayerSeasonStats.player == player,
            PlayerSeasonStats.team   == team,
            PlayerSeasonStats.league == league,
            PlayerSeasonStats.season == season,
        )).first()
    )
    if existing is None:
        existing = PlayerSeasonStats(
            player=player, team=team,
            league=league, season=season)
        db.add(existing)
    _apply_attrs(existing, attrs)


def collect_fixtures_and_results() -> dict:
    """2 API calls."""
    print("[fixtures] Fetching fixtures and results...")
    today     = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    output    = {"fixtures": [], "results": []}

    data = _get("fixtures", {"date": today})
    if data:
        for f in data.get("response", []):
            lid = f["league"]["id"]
            ln  = next((k for k, v in LEAGUE_MAP.items()
                        if v == lid), None)
            if not ln:
                continue
            output["fixtures"].append({
                "fixture_id": f["fixture"]["id"],
                "league":     ln,
                "home_team":  f["teams"]["home"]["name"],
                "away_team":  f["teams"]["away"]["name"],
                "kickoff":    f["fixture"]["date"],
                "status":     f["fixture"]["status"]["short"],
                "home_id":    f["teams"]["home"]["id"],
                "away_id":    f["teams"]["away"]["id"],
            })

    data = _get("fixtures", {
        "date": yesterday, "status": "FT"})
    if data:
        for f in data.get("response", []):
            lid = f["league"]["id"]
            ln  = next((k for k, v in LEAGUE_MAP.items()
                        if v == lid), None)
            if not ln:
                continue
            output["results"].append({
                "fixture_id": f["fixture"]["id"],
                "league":     ln,
                "home_team":  f["teams"]["home"]["name"],
                "away_team":  f["teams"]["away"]["name"],
                "home_goals": f["goals"]["home"],
                "away_goals": f["goals"]["away"],
                "date":       f["fixture"]["date"][:10],
            })

    print(f"[fixtures] Today: {len(output['fixtures'])} | "
          f"Yesterday: {len(output['results'])}")
    return output


def collect_team_stats() -> int:
    """
    26 API calls — standings per domestic league per season.
    Standings returns all teams at once so we never
    need per-team requests. Stays within 100/day.
    """
    print("[team] Collecting team statistics "
          "via standings...")
    total = 0

    for league_name, league_id in LEAGUE_MAP.items():
        if league_name in (
            "UEFA-Champions League",
        ):
            continue
        for season_year in SEASON_MAP[league_name]:
            season_label = SEASON_LABEL.get(
                (league_id, season_year),
                str(season_year))
            print(f"[team] {league_name} "
                  f"{season_year}...")

            data = _get_standings(league_id, season_year)
            if not data:
                continue

            standings = data.get("response", [])
            if not standings:
                print(f"[team] No standings for "
                      f"{league_name} {season_year}")
                continue

            db = SessionLocal()
            try:
                for group in standings:
                    league_info = group.get(
                        "league", {})
                    all_groups  = league_info.get(
                        "standings", [[]])
                    if not all_groups:
                        continue

                    for team_entry in all_groups[0]:
                        team_name = team_entry.get(
                            "team", {}).get("name", "")
                        if not team_name:
                            continue

                        all_s   = team_entry.get("all", {})
                        form_str= team_entry.get("form","") or ""

                        played = _safe(all_s.get("played")) or 1
                        gf     = _safe(all_s.get("goals", {}).get("for"))
                        ga     = _safe(all_s.get("goals", {}).get("against"))

                        recent = form_str[-5:]
                        fw = recent.count("W")
                        fd = recent.count("D")
                        fl = recent.count("L")

                        attrs = {
                            # Goals — the only real numbers the standings
                            # feed carries. Shots / SoT are deliberately not
                            # estimated from goals here: the old
                            # gf / played * 4 style proxies overwrote the
                            # real per-90 columns FBref writes.
                            "sh_goals":     gf,
                            "gk_ga":        ga,
                            "gk_ga90":      round(ga / played, 2) if ga else None,
                            # The keeper W-D-L columns are the last-five form
                            # window in this table (see collect_team_form).
                            "gk_w":         float(fw),
                            "gk_d":         float(fd),
                            "gk_l":         float(fl),
                            # The standings feed has no clean sheets; the
                            # None is skipped by _upsert_team, so a value
                            # FBref stored is not wiped.
                            "gk_cs":        None,
                        }

                        _upsert_team(
                            db, team_name, league_name,
                            season_label, attrs)
                        total += 1

                db.commit()
                print(f"[team] {league_name} "
                      f"{season_year}: committed")

            except Exception as exc:
                db.rollback()
                print(f"[team] ERROR {league_name} "
                      f"{season_year}: {exc}")
                raise
            finally:
                db.close()

    print(f"[team] Done. Total rows: {total}")
    return total

def collect_injuries() -> dict:
    """8 API calls — one per INJURY_LEAGUES league (UCL skipped: not stored)."""
    print("[injuries] Collecting injuries...")
    all_injuries = {}
    today = date.today().isoformat()

    for league_name, league_id in LEAGUE_MAP.items():
        if league_name not in INJURY_LEAGUES:
            continue
        if league_name == "UEFA-Champions League":
            continue
        data = _get("injuries", {
            "league": league_id,
            # The API requires "season" whenever "league" is given.
            "season": max(SEASON_MAP[league_name]),
            "date":   today,
        })
        injuries = []
        if data:
            for e in data.get("response", []):
                player = e.get("player", {})
                team   = e.get("team", {})
                injuries.append({
                    "player": player.get("name"),
                    "team":   team.get("name"),
                    "type":   player.get("type"),
                    "reason": player.get("reason"),
                })
        all_injuries[league_name] = injuries
        print(f"[injuries] {league_name}: "
              f"{len(injuries)} injuries")

    return all_injuries


def collect_team_form() -> int:
    """
    Recent form for the INJURY_LEAGUES leagues (UCL skipped).

    Reuses the latest-season standings collect_team_stats already fetched
    in this run: no extra calls inside run_all, 8 when run on its own.
    """
    print("[form] Collecting standings/form...")
    updated = 0

    for league_name, league_id in LEAGUE_MAP.items():
        if league_name not in INJURY_LEAGUES:
            continue
        if league_name == "UEFA-Champions League":
            continue
        latest   = max(SEASON_MAP[league_name])
        season_l = SEASON_LABEL.get(
            (league_id, latest), str(latest))

        data = _get_standings(league_id, latest)
        if not data:
            continue

        standings = data.get("response", [])
        if not standings:
            continue

        db = SessionLocal()
        try:
            for group in standings:
                league_info   = group.get("league", {})
                all_standings = league_info.get(
                    "standings", [[]])
                if not all_standings:
                    continue
                for entry in all_standings[0]:
                    team_name = entry.get(
                        "team", {}).get("name", "")
                    form_str  = entry.get("form", "") or ""
                    recent    = form_str[-5:]
                    fw = recent.count("W")
                    fd = recent.count("D")
                    fl = recent.count("L")

                    existing = (
                        db.query(TeamSeasonStats)
                        .filter(and_(
                            TeamSeasonStats.team   == team_name,
                            TeamSeasonStats.league == league_name,
                            TeamSeasonStats.season == season_l,
                        )).first()
                    )
                    if existing is None:
                        existing = TeamSeasonStats(
                            team=team_name,
                            league=league_name,
                            season=season_l,
                        )
                        db.add(existing)

                    existing.gk_w = float(fw)
                    existing.gk_d = float(fd)
                    existing.gk_l = float(fl)
                    updated += 1

            db.commit()
            print(f"[form] {league_name}: updated")
        except Exception as exc:
            db.rollback()
            print(f"[form] ERROR {league_name}: {exc}")
            raise
        finally:
            db.close()

    print(f"[form] Done. Updated: {updated}")
    return updated


def collect_player_stats() -> int:
    """
    Fetch players for each team using
    the /players endpoint with team parameter:
    one /teams call per league for the team ids,
    then one /players call per team.

    The league-wide page (league + season, no team / player) comes back
    empty on the free tier, which is why this collector used to store no
    rows. /teams supplies the team ids per league, then page 1 of
    /players is read for every team.

    Budget: 1 /teams call per league plus ~1 /players call per team
    across thirteen domestic leagues. That is more than the
    100/day free-tier quota, so the loop stops as soon as the API reports
    the limit (a quota or plan error arrives in a 200 body) and the rows
    collected up to that point are committed.
    """
    print("[player] Collecting player stats...")
    total = 0

    for league_name, league_id in LEAGUE_MAP.items():
        if league_name == "UEFA-Champions League":
            continue
        latest   = max(SEASON_MAP[league_name])
        season_l = SEASON_LABEL.get(
            (league_id, latest), str(latest))
        league_total = 0

        # Get teams for this league
        teams_data = _get("teams", {
            "league": league_id,
            "season": latest,
        })
        if not teams_data:
            continue

        # API-Football reports quota and plan errors in a 200 body, with an
        # empty response — stop rather than spend the rest of the loop on
        # calls that cannot succeed.
        errors = teams_data.get("errors")
        if errors:
            print(f"[player] Stopping: {errors}")
            print(f"[player] Total rows: {total}")
            return total

        teams = teams_data.get("response", [])
        if not teams:
            print(f"[player] No teams for "
                  f"{league_name} {latest}")
            continue

        print(f"[player] {league_name}: "
              f"{len(teams)} teams")

        for team_entry in teams:
            team_info = team_entry.get("team", {})
            tid   = team_info.get("id")
            tname = team_info.get("name", "Unknown")
            if not tid:
                continue

            data = _get("players", {
                "team":   tid,
                "season": latest,
                "page":   1,
            })
            if not data:
                continue

            errors = data.get("errors")
            if errors:
                print(f"[player] Stopping: {errors}")
                print(f"[player] Total rows: {total}")
                return total

            players = data.get("response", [])
            if not players:
                continue

            pages = _safe(
                (data.get("paging") or {}).get("total"))
            if pages and pages > 1:
                print(f"[player] NOTE: {tname} has "
                      f"{int(pages)} pages — only "
                      f"page 1 collected")

            db = SessionLocal()
            try:
                for entry in players:
                    p    = entry.get("player", {})
                    name = p.get("name", "Unknown")
                    age  = p.get("age")

                    stats = entry.get(
                        "statistics", [])
                    if not stats:
                        continue
                    # A mid-season transfer leaves one statistics entry per
                    # club, so prefer the one for the team just requested.
                    s = next(
                        (st for st in stats
                         if (st.get("team") or {}).get("id") == tid),
                        stats[0])

                    games    = s.get("games", {})
                    goals_s  = s.get("goals", {})
                    shots_s  = s.get("shots", {})
                    tackles_s= s.get("tackles", {})
                    duels_s  = s.get("duels", {})
                    dribbles_s=s.get("dribbles",{})
                    fouls_s  = s.get("fouls", {})
                    cards_s  = s.get("cards", {})
                    penalty_s= s.get("penalty", {})

                    # The role lives in the statistics block (the player
                    # object has no position) and arrives as a full word
                    # ("Midfielder"); it is normalised to the FBref codes
                    # (GK/DF/MF/FW) the prediction engine matches on.
                    pos = _norm_position(
                        p.get("position")
                        or games.get("position"))

                    mins  = _safe(games.get("minutes")) or 0
                    per90 = (mins/90) if mins>0 else 1

                    gls = _safe(goals_s.get("total"))
                    ast = _safe(goals_s.get("assists"))
                    sh  = _safe(shots_s.get("total"))
                    sot = _safe(shots_s.get("on"))
                    dw  = _safe(duels_s.get("won"))
                    dt  = _safe(duels_s.get("total"))

                    attrs = {
                        "position":   pos,
                        "age":        age,
                        # API-Football spells this field "appearences".
                        "mp":         (
                            games.get("appearences")
                            or games.get("appearances")
                        ),
                        "starts":     games.get("lineups"),
                        "minutes":    _safe(
                            games.get("minutes")),
                        "minutes_per90": round(
                            per90, 2),
                        "goals":      gls,
                        "assists":    ast,
                        "g_plus_a":   (
                            (gls or 0)+(ast or 0)
                        ) or None,
                        "crdy":       _safe(
                            cards_s.get("yellow")),
                        "crdr":       _safe(
                            cards_s.get("red")),
                        "sh_shots":   sh,
                        "sh_sot":     sot,
                        "sh_sot_pct": round(
                            sot/sh*100,1
                        ) if sh and sot else None,
                        "sh_g_per_sh":round(
                            gls/sh,3
                        ) if sh and gls else None,
                        "sh_sh_per90":round(
                            sh/per90,2
                        ) if sh else None,
                        "sh_sot_per90":round(
                            sot/per90,2
                        ) if sot else None,
                        # NOTE: the API also returns passing numbers
                        # (passes.total / key / accuracy), but
                        # player_season_stats has no column for them —
                        # ps_att / ps_kp / ps_cmp_pct belong to
                        # TeamSeasonStats — so they are not read here.
                        "tkl":        _safe(
                            tackles_s.get("total")),
                        "int_":       _safe(
                            tackles_s.get(
                                "interceptions")),
                        "tkl_plus_int": (
                            (_safe(tackles_s.get("total")) or 0) +
                            (_safe(tackles_s.get(
                                "interceptions")) or 0)
                        ) or None,
                        "fls":        _safe(
                            fouls_s.get("committed")),
                        "fld":        _safe(
                            fouls_s.get("drawn")),
                        "pkwon":      _safe(
                            penalty_s.get("won")),
                        "pkcon":      _safe(
                            penalty_s.get("commited")),
                        "aerial_won": dw,
                        "aerial_lost":(
                            (dt or 0)-(dw or 0)
                        ) or None,
                        "aerial_pct": round(
                            dw/dt*100,1
                        ) if dt and dw else None,
                        "prgc_carries":_safe(
                            dribbles_s.get("success")),
                        "gls_per90":  round(
                            gls/per90,3
                        ) if gls else None,
                        "ast_per90":  round(
                            ast/per90,3
                        ) if ast else None,
                        "g_plus_a_per90": round(
                            ((gls or 0)+(ast or 0))/per90, 3
                        ) if (gls or ast) else None,
                    }

                    _upsert_player(
                        db, name, tname,
                        league_name, season_l,
                        attrs)
                    total += 1
                    league_total += 1

                db.commit()

            except Exception as exc:
                db.rollback()
                print(f"[player] ERROR "
                      f"{tname}: {exc}")
                raise
            finally:
                db.close()

        if league_total:
            print(f"[player] {league_name}: "
                  f"{league_total} players")
        else:
            print(f"[player] WARNING: no players for "
                  f"{league_name} — /players returned "
                  f"nothing for every team")

    print(f"[player] Total rows: {total}")
    return total


def run_all() -> None:
    """
    Run all collectors in sequence.
    Exits with code 1 on any failure.

    API call budget (100/day free tier), in run order:
      1   status check (does not count against the daily quota)
      2   fixtures + yesterday results
      26  standings (13 domestic leagues x 2 seasons)
      8   injuries (INJURY_LEAGUES, UCL skipped)
      0   team form (reuses the standings fetched above)
      13  /teams for the player collector (1 per league)
      ~250 /players calls (one per team; approximate)
      Injuries and form run before the player collector, which spends
      whatever quota is left. A full player pass is over 250 calls, more
      than the daily quota, so collect_player_stats stops as soon as the
      API reports the limit and commits the teams it did reach; raise the
      plan or trim LEAGUE_MAP (or run that collector on its own) to cover
      more of the list. Calls are paced to the per-minute limit (about
      6.5 s apart on the free plan, 10 requests/minute).
    """
    print("=" * 60)
    print("API-FOOTBALL DAILY COLLECTION")
    print(f"Date: {date.today()}")
    print("=" * 60)

    if not API_KEY:
        print("FATAL: API_FOOTBALL_KEY not set")
        sys.exit(1)

    print("\n--- STATUS CHECK ---")
    if not check_status():
        print("FATAL: API key invalid or unreachable")
        sys.exit(1)

    print("\n--- FIXTURES AND RESULTS ---")
    collect_fixtures_and_results()

    print("\n--- TEAM STATISTICS ---")
    collect_team_stats()

    print("\n--- INJURIES ---")
    collect_injuries()

    print("\n--- TEAM FORM ---")
    collect_team_form()

    print("\n--- PLAYER STATISTICS ---")
    collect_player_stats()

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print(f"Total API calls made: {_calls_made}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
