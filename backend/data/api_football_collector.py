"""
API-Football data collector.

Run from backend/ directory:

    python data/api_football_collector.py

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
    "USA-MLS":            253,
}

SEASON_MAP = {
    "ENG-Premier League": [2023, 2024],
    "ENG-Championship":   [2023, 2024],
    "ESP-La Liga":        [2023, 2024],
    "GER-Bundesliga":     [2023, 2024],
    "ITA-Serie A":        [2023, 2024],
    "FRA-Ligue 1":        [2023, 2024],
    "POR-Primeira Liga":  [2023, 2024],
    "NED-Eredivisie":     [2023, 2024],
    "USA-MLS":            [2024, 2025],
}

SEASON_LABEL = {
    (39,  2023): "2023-2024", (39,  2024): "2024-2025",
    (40,  2023): "2023-2024", (40,  2024): "2024-2025",
    (140, 2023): "2023-2024", (140, 2024): "2024-2025",
    (78,  2023): "2023-2024", (78,  2024): "2024-2025",
    (135, 2023): "2023-2024", (135, 2024): "2024-2025",
    (61,  2023): "2023-2024", (61,  2024): "2024-2025",
    (94,  2023): "2023-2024", (94,  2024): "2024-2025",
    (88,  2023): "2023-2024", (88,  2024): "2024-2025",
    (253, 2024): "2024",      (253, 2025): "2025",
}

_calls_made = 0


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


def _get(endpoint: str, params: dict) -> dict | None:
    global _calls_made
    if not API_KEY:
        print("[api] ERROR: API_FOOTBALL_KEY not set")
        sys.exit(1)
    headers = {"x-apisports-key": API_KEY}
    url     = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(
            url, headers=headers,
            params=params, timeout=30)
        _calls_made += 1
        if resp.status_code == 200:
            data      = resp.json()
            remaining = resp.headers.get(
                "x-ratelimit-requests-remaining", "?")
            n = len(data.get("response", []))
            print(f"[api] [{_calls_made}] {endpoint} "
                  f"-> {n} items "
                  f"(remaining: {remaining})")
            time.sleep(1.2)
            return data
        else:
            print(f"[api] WARNING: HTTP "
                  f"{resp.status_code} for {endpoint}")
            return None
    except Exception as exc:
        print(f"[api] ERROR: {endpoint}: {exc}")
        return None


def check_status() -> bool:
    data = _get("status", {})
    if not data:
        return False
    r    = data.get("response", {})
    acct = r.get("account", {})
    reqs = r.get("requests", {})
    print(f"[api] Account: {acct.get('firstname')} "
          f"{acct.get('lastname')}")
    print(f"[api] Calls today: "
          f"{reqs.get('current')} / "
          f"{reqs.get('limit_day')}")
    return True


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
    for k, v in attrs.items():
        setattr(existing, k, v)


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
    for k, v in attrs.items():
        setattr(existing, k, v)


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
    """~18 API calls."""
    print("[team] Collecting team statistics...")
    total = 0

    for league_name, league_id in LEAGUE_MAP.items():
        for season_year in SEASON_MAP[league_name]:
            season_label = SEASON_LABEL.get(
                (league_id, season_year), str(season_year))
            print(f"[team] {league_name} {season_year}...")

            teams_data = _get("teams", {
                "league": league_id,
                "season": season_year,
            })
            if not teams_data:
                continue
            teams = teams_data.get("response", [])
            if not teams:
                continue

            db = SessionLocal()
            try:
                for team_entry in teams:
                    tid  = team_entry["team"]["id"]
                    name = team_entry["team"]["name"]

                    sd = _get("teams/statistics", {
                        "league": league_id,
                        "season": season_year,
                        "team":   tid,
                    })
                    if not sd:
                        continue
                    s = sd.get("response", {})
                    if not s:
                        continue

                    played  = (s.get("fixtures", {})
                                .get("played", {})
                                .get("total") or 1)
                    goals   = s.get("goals", {})
                    shots   = s.get("shots", {})
                    passes_ = s.get("passes", {})
                    tackles = s.get("tackles", {})
                    fouls   = s.get("fouls", {})
                    corners = s.get("corners", {})
                    cards   = s.get("cards", {})
                    form_str= s.get("form", "") or ""

                    recent = form_str[-5:]
                    fw = recent.count("W")
                    fd = recent.count("D")
                    fl = recent.count("L")

                    gf = _safe(goals.get("for", {})
                               .get("total", {})
                               .get("total"))
                    ga = _safe(goals.get("against", {})
                               .get("total", {})
                               .get("total"))
                    sh = _safe(shots.get("total", {})
                               .get("total"))
                    so = _safe(shots.get("on", {})
                               .get("total"))
                    cs = _safe(s.get("clean_sheet", {})
                               .get("total"))

                    attrs = {
                        "sh_goals":     gf,
                        "sh_shots":     sh,
                        "sh_sot":       so,
                        "sh_sot_pct":   _pct(shots.get("on", {}).get("percentage")),
                        "sh_sh_per90":  round(sh / played, 2) if sh else None,
                        "sh_sot_per90": round(so / played, 2) if so else None,
                        "sh_g_per_sh":  round(gf / sh, 3) if sh and gf else None,
                        "sh_g_per_sot": round(gf / so, 3) if so and gf else None,
                        "ps_att":       _safe(passes_.get("total", {}).get("total")),
                        "ps_cmp":       _safe(passes_.get("accurate", {}).get("total")),
                        "ps_cmp_pct":   _pct(passes_.get("accuracy")),
                        "def_tkl":      _safe(tackles.get("total", {}).get("total")),
                        "def_int":      _safe(tackles.get("interceptions", {}).get("total")),
                        "def_blk_sh":   _safe(tackles.get("blocks", {}).get("total")),
                        "def_tkl_int":  (
                            (_safe(tackles.get("total", {}).get("total")) or 0) +
                            (_safe(tackles.get("interceptions", {}).get("total")) or 0)
                        ) or None,
                        "misc_fls":     _safe(fouls.get("committed", {}).get("total")),
                        "misc_fld":     _safe(fouls.get("drawn", {}).get("total")),
                        "misc_crdy":    _safe(cards.get("yellow", {}).get("total", {}).get("total")),
                        "misc_crdr":    _safe(cards.get("red", {}).get("total", {}).get("total")),
                        "pos_cpa":      _safe(corners.get("total", {}).get("total")),
                        "gk_ga":        ga,
                        "gk_ga90":      round(ga / played, 2) if ga else None,
                        "gk_cs":        cs,
                        "gk_cs_pct":    round(cs / played * 100, 1) if cs else None,
                        "gk_w":         float(fw),
                        "gk_d":         float(fd),
                        "gk_l":         float(fl),
                    }

                    _upsert_team(
                        db, name, league_name,
                        season_label, attrs)
                    total += 1

                db.commit()
                print(f"[team] {league_name} "
                      f"{season_year}: "
                      f"{len(teams)} teams committed")
            except Exception as exc:
                db.rollback()
                print(f"[team] ERROR "
                      f"{league_name}: {exc}")
                raise
            finally:
                db.close()

    print(f"[team] Done. Total: {total}")
    return total


def collect_injuries() -> dict:
    """9 API calls — one per league."""
    print("[injuries] Collecting injuries...")
    all_injuries = {}
    today = date.today().isoformat()

    for league_name, league_id in LEAGUE_MAP.items():
        data = _get("injuries", {
            "league": league_id,
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
    """9 API calls — one per league."""
    print("[form] Collecting standings/form...")
    updated = 0

    for league_name, league_id in LEAGUE_MAP.items():
        latest   = max(SEASON_MAP[league_name])
        season_l = SEASON_LABEL.get(
            (league_id, latest), str(latest))

        data = _get("standings", {
            "league": league_id,
            "season": latest,
        })
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
    9 API calls — one page per league.
    Limited to 1 page to stay within budget.
    """
    print("[player] Collecting player statistics...")
    total = 0

    for league_name, league_id in LEAGUE_MAP.items():
        latest   = max(SEASON_MAP[league_name])
        season_l = SEASON_LABEL.get(
            (league_id, latest), str(latest))
        print(f"[player] {league_name} {latest}...")

        data = _get("players", {
            "league": league_id,
            "season": latest,
            "page":   1,
        })
        if not data:
            continue

        players = data.get("response", [])
        if not players:
            continue

        db = SessionLocal()
        try:
            for entry in players:
                p    = entry.get("player", {})
                name = p.get("name", "Unknown")
                age  = p.get("age")
                pos  = p.get("position")

                stats = entry.get("statistics", [])
                if not stats:
                    continue
                s    = stats[0]
                team = s.get("team", {}).get(
                    "name", "Unknown")

                games     = s.get("games", {})
                goals_s   = s.get("goals", {})
                shots_s   = s.get("shots", {})
                passes_s  = s.get("passes", {})
                tackles_s = s.get("tackles", {})
                duels_s   = s.get("duels", {})
                dribbles_s= s.get("dribbles", {})
                fouls_s   = s.get("fouls", {})
                cards_s   = s.get("cards", {})
                penalty_s = s.get("penalty", {})

                mins  = _safe(games.get("minutes")) or 0
                per90 = (mins / 90) if mins > 0 else 1

                gls = _safe(goals_s.get("total"))
                ast = _safe(goals_s.get("assists"))
                sh  = _safe(shots_s.get("total"))
                sot = _safe(shots_s.get("on"))
                dw  = _safe(duels_s.get("won"))
                dt  = _safe(duels_s.get("total"))

                attrs = {
                    "position":       pos,
                    "age":            age,
                    "mp":             games.get("appearences"),
                    "starts":         games.get("lineups"),
                    "minutes":        _safe(games.get("minutes")),
                    "minutes_per90":  round(per90, 2),
                    "goals":          gls,
                    "assists":        ast,
                    "g_plus_a":       (gls or 0) + (ast or 0) or None,
                    "crdy":           _safe(cards_s.get("yellow")),
                    "crdr":           _safe(cards_s.get("red")),
                    "sh_shots":       sh,
                    "sh_sot":         sot,
                    "sh_sot_pct":     round(sot / sh * 100, 1) if sh and sot else None,
                    "sh_g_per_sh":    round(gls / sh, 3) if sh and gls else None,
                    "sh_sh_per90":    round(sh / per90, 2) if sh else None,
                    "sh_sot_per90":   round(sot / per90, 2) if sot else None,
                    "ps_att":         _safe(passes_s.get("total")),
                    "ps_kp":          _safe(passes_s.get("key")),
                    "ps_cmp_pct":     _pct(passes_s.get("accuracy")),
                    "tkl":            _safe(tackles_s.get("total")),
                    "int_":           _safe(tackles_s.get("interceptions")),
                    "tkl_plus_int":   (
                        (_safe(tackles_s.get("total")) or 0) +
                        (_safe(tackles_s.get("interceptions")) or 0)
                    ) or None,
                    "fls":            _safe(fouls_s.get("committed")),
                    "fld":            _safe(fouls_s.get("drawn")),
                    "pkwon":          _safe(penalty_s.get("won")),
                    "pkcon":          _safe(penalty_s.get("commited")),
                    "aerial_won":     dw,
                    "aerial_lost":    ((dt or 0) - (dw or 0)) or None,
                    "aerial_pct":     round(dw / dt * 100, 1) if dt and dw else None,
                    "prgc_carries":   _safe(dribbles_s.get("success")),
                    "gls_per90":      round(gls / per90, 3) if gls else None,
                    "ast_per90":      round(ast / per90, 3) if ast else None,
                    "g_plus_a_per90": round(
                        ((gls or 0) + (ast or 0)) / per90, 3
                    ) if (gls or ast) else None,
                }

                _upsert_player(
                    db, name, team,
                    league_name, season_l, attrs)
                total += 1

            db.commit()
            print(f"[player] {league_name}: "
                  f"{len(players)} players committed")

        except Exception as exc:
            db.rollback()
            print(f"[player] ERROR {league_name}: {exc}")
            raise
        finally:
            db.close()

    print(f"[player] Done. Total: {total}")
    return total


def run_all() -> None:
    """
    Run all collectors in sequence.
    Exits with code 1 on any failure.

    API call budget (100/day free tier):
      1  status check
      2  fixtures + yesterday results
      18 team stats (9 leagues x 2 seasons)
      9  player stats (1 page per league)
      9  injuries
      9  team form
      Total: ~48 calls — well within budget
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

    print("\n--- PLAYER STATISTICS ---")
    collect_player_stats()

    print("\n--- INJURIES ---")
    collect_injuries()

    print("\n--- TEAM FORM ---")
    collect_team_form()

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print(f"Total API calls made: {_calls_made}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
