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
