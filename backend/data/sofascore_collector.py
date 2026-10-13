"""
Sofascore direct API collector.

Uses curl_cffi to impersonate Chrome TLS
fingerprint when calling api.sofascore.com.
Bypasses bot detection on cloud servers
including GitHub Actions.

Primary data source for all advanced stats.
Run from backend/ directory:
    python data/sofascore_collector.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, TeamSeasonStats, PlayerSeasonStats
from sqlalchemy import and_


# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

BASE_URL = "https://api.sofascore.com/api/v1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer":         "https://www.sofascore.com/",
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          "https://www.sofascore.com",
}

LEAGUE_CONFIG = {
    "ENG-Premier League":    {"tid": 17},
    "ENG-Championship":      {"tid": 18},
    "ESP-La Liga":           {"tid": 8},
    "GER-Bundesliga":        {"tid": 35},
    "ITA-Serie A":           {"tid": 23},
    "FRA-Ligue 1":           {"tid": 34},
    "POR-Primeira Liga":     {"tid": 238},
    "NED-Eredivisie":        {"tid": 37},
    "USA-MLS":               {"tid": 242},
    "UEFA-Champions League": {"tid": 7},
}

TARGET_SEASONS     = [2025, 2026]
MLS_TARGET_SEASONS = [2025, 2026]
REQUEST_DELAY      = 2.0


# ═══════════════════════════════════════════
# HTTP CLIENT
# ═══════════════════════════════════════════

def _get_client():
    """
    Create curl_cffi session impersonating Chrome.
    Falls back to plain requests if not installed.
    """
    try:
        from curl_cffi import requests as cffi_req
        session = cffi_req.Session(
            impersonate="chrome120")
        session.headers.update(HEADERS)
        print("[sofascore] Using curl_cffi "
              "(Chrome TLS impersonation)")
        return session, True
    except ImportError:
        import requests
        session = requests.Session()
        session.headers.update(HEADERS)
        print("[sofascore] WARNING: curl_cffi "
              "not found — using plain requests")
        return session, False


def _get(
    session,
    url: str,
    params: dict = None,
) -> dict | None:
    """Make one GET request. Returns JSON or None."""
    try:
        resp = session.get(
            url,
            params=params or {},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print("[sofascore] Rate limited — "
                  "waiting 30s...")
            time.sleep(30)
            return None
        elif resp.status_code == 403:
            print(f"[sofascore] 403 Forbidden: "
                  f"{url}")
            return None
        else:
            print(f"[sofascore] HTTP "
                  f"{resp.status_code}: {url}")
            return None
    except Exception as exc:
        print(f"[sofascore] ERROR: {url}: {exc}")
        return None


# ═══════════════════════════════════════════
# SEASON DISCOVERY
# ═══════════════════════════════════════════

def get_seasons(
    session,
    tid: int,
    target_years: list[int],
) -> list[dict]:
    """
    Fetch available seasons for a tournament.
    Returns list of {sid, year, name} dicts
    matching target_years.
    """
    url  = (f"{BASE_URL}/unique-tournament"
            f"/{tid}/seasons")
    data = _get(session, url)
    time.sleep(REQUEST_DELAY)

    if not data:
        return []

    seasons = data.get("seasons", [])
    matched = []

    for s in seasons:
        sid  = s.get("id")
        name = s.get("name", "")
        year = s.get("year")
        for ty in target_years:
            if year == ty or str(ty) in str(name):
                matched.append({
                    "sid":  sid,
                    "year": ty,
                    "name": name,
                })
                break

    print(f"[sofascore] tid={tid}: "
          f"{len(matched)} seasons matched: "
          f"{[s['name'] for s in matched]}")
    return matched


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

def _safe(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _season_label(
    league: str,
    year: int,
    name: str,
) -> str:
    if league == "USA-MLS":
        return str(year)
    return f"{year}-{year + 1}"


def _upsert_team(
    db, team, league, season, attrs):
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
            team=team, league=league,
            season=season)
        db.add(existing)
    for k, v in attrs.items():
        if v is not None:
            setattr(existing, k, v)


def _upsert_player(
    db, player, team, league, season, attrs):
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
        if v is not None:
            setattr(existing, k, v)
