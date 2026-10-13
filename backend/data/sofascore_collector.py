"""
Sofascore direct API collector.

Uses patchright to fetch api.sofascore.com through a stealth browser.
Falls back to curl_cffi TLS impersonation, then plain requests,
when the preferred client is not installed.

Primary data source for all advanced stats.
Run from backend/ directory:
    python data/sofascore_collector.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

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
    Returns a client with get(url, params)
    that uses patchright stealth browser.
    Falls back to curl_cffi then requests.
    """
    try:
        from patchright.sync_api import sync_playwright
        print("[sofascore] Using patchright "
              "(stealth Chrome browser)")
        return _PatchrightClient(), "patchright"
    except ImportError:
        pass

    try:
        from curl_cffi import requests as cffi_req
        session = cffi_req.Session(
            impersonate="chrome120")
        session.headers.update(HEADERS)
        print("[sofascore] Using curl_cffi "
              "(Chrome TLS impersonation)")
        return session, "curl_cffi"
    except ImportError:
        pass

    import requests
    session = requests.Session()
    session.headers.update(HEADERS)
    print("[sofascore] WARNING: using plain "
          "requests — likely to be blocked")
    return session, "requests"


class _PatchrightClient:
    """
    Stealth browser client using patchright.
    Opens one browser instance reused for
    all requests to avoid overhead.
    """

    def __init__(self):
        from patchright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features="
                    "AutomationControlled",
                ],
            )
            self._context = self._browser.new_context(
                user_agent=HEADERS["User-Agent"],
                extra_http_headers={
                    "Referer": HEADERS["Referer"],
                    "Accept-Language": HEADERS["Accept-Language"],
                },
            )
            self._page = self._context.new_page()
        except Exception:
            self.close()
            raise

        # Visit homepage first to get cookies
        try:
            self._page.goto(
                "https://www.sofascore.com",
                timeout=30000,
                wait_until="domcontentloaded",
            )
            time.sleep(3)
            print("[sofascore] Browser "
                  "initialised with cookies")
        except Exception as exc:
            print(f"[sofascore] WARNING: "
                  f"homepage visit failed: {exc}")

    def get(self, url: str,
            params: dict = None,
            timeout: int = 30,
            **kwargs) -> "_FakeResponse":
        """Fetch URL via stealth browser."""
        if params:
            qs = urlencode(params, doseq=True)
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{qs}"
        try:
            resp = self._page.goto(
                url,
                timeout=timeout * 1000,
                wait_until="networkidle",
            )
            body = self._page.inner_text("body")
            status = resp.status if resp else 0
            return _FakeResponse(status, body)
        except Exception as exc:
            print(f"[sofascore] Browser ERROR: "
                  f"{exc}")
            return _FakeResponse(0, "")

    def close(self):
        try:
            if hasattr(self, "_browser"):
                self._browser.close()
        except Exception:
            pass
        finally:
            try:
                self._pw.stop()
            except Exception:
                pass


class _FakeResponse:
    """Mimics requests.Response interface."""

    def __init__(self, status_code: int,
                 text: str):
        self.status_code = status_code
        self._text = text

    def json(self):
        import json
        return json.loads(self._text)


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
                  "waiting 60s...")
            time.sleep(60)
            # Retry once
            resp = session.get(
                url,
                params=params or {},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        elif resp.status_code == 403:
            print(f"[sofascore] 403 Forbidden: "
                  f"{url}")
            print("[sofascore] Sofascore is "
                  "blocking this request.")
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


# ═══════════════════════════════════════════
# TEAM STATS
# ═══════════════════════════════════════════


def _parse_team_stats(entry: dict, games: int) -> dict:
    s = entry.get("statistics", {})
    attrs = {}
    games = int(s.get("matches", games) or games)

    xg = _safe(s.get("expectedGoals"))
    if xg is not None:
        attrs["sh_xg"] = round(xg, 3)

    goals = _safe(s.get("goals"))
    if goals is not None:
        attrs["sh_goals"] = round(goals, 1)

    shots = _safe(s.get("shots") or s.get("totalShots"))
    if shots is not None:
        attrs["sh_shots"] = round(shots, 1)
        if games > 0:
            attrs["sh_sh_per90"] = round(shots / games, 2)

    sot = _safe(s.get("shotsOnTarget") or
                s.get("onTargetScoringAttempt"))
    if sot is not None:
        attrs["sh_sot"] = round(sot, 1)
        if games > 0:
            attrs["sh_sot_per90"] = round(sot / games, 2)

    if goals and shots and shots > 0:
        attrs["sh_g_per_sh"] = round(goals / shots, 3)

    poss = _safe(s.get("possession") or
                 s.get("ballPossession"))
    if poss is not None:
        attrs["pos_poss"] = round(poss, 1)

    ga = _safe(s.get("goalsConceded") or
               s.get("goalsAgainst"))
    if ga is not None:
        attrs["gk_ga"] = round(ga, 1)
        if games > 0:
            attrs["gk_ga90"] = round(ga / games, 2)

    saves = _safe(s.get("saves"))
    if saves is not None:
        attrs["gk_saves"] = round(saves, 1)
        if ga is not None and (ga + saves) > 0:
            attrs["gk_save_pct"] = round(
                saves / (ga + saves) * 100, 1)

    cs = _safe(s.get("cleanSheets") or
               s.get("cleanSheet"))
    if cs is not None:
        attrs["gk_cs"] = round(cs, 1)
        if games > 0:
            attrs["gk_cs_pct"] = round(
                cs / games * 100, 1)

    xa = _safe(s.get("expectedAssists") or s.get("xA"))
    if xa is not None:
        attrs["ps_xa"] = round(xa, 3)

    bcc = _safe(s.get("bigChancesCreated") or
                s.get("bigChances"))
    if bcc is not None:
        attrs["gsc_gca"] = round(bcc, 1)
        if games > 0:
            attrs["gsc_gca90"] = round(bcc / games, 2)

    tackles = _safe(s.get("tackles") or
                    s.get("tacklesWon"))
    if tackles is not None:
        attrs["def_tkl"] = round(tackles, 1)

    ints = _safe(s.get("interceptions"))
    if ints is not None:
        attrs["def_int"] = round(ints, 1)
        if tackles is not None:
            attrs["def_tkl_int"] = round(
                tackles + ints, 1)

    yellows = _safe(s.get("yellowCards"))
    if yellows is not None:
        attrs["misc_crdy"] = round(yellows, 1)

    reds = _safe(s.get("redCards"))
    if reds is not None:
        attrs["misc_crdr"] = round(reds, 1)

    passes = _safe(s.get("accuratePasses") or
                   s.get("totalPasses"))
    if passes is not None:
        attrs["ps_cmp"] = round(passes, 1)

    pass_pct = _safe(s.get("accuratePassesPercentage"))
    if pass_pct is not None:
        attrs["ps_cmp_pct"] = round(pass_pct, 1)

    fouls = _safe(s.get("fouls") or
                  s.get("foulsCommitted"))
    if fouls is not None:
        attrs["misc_fls"] = round(fouls, 1)

    return attrs


def collect_team_stats(
    session, league, tid, sid, season_label
) -> int:
    url  = (f"{BASE_URL}/unique-tournament/{tid}"
            f"/season/{sid}/statistics/overall")
    data = _get(session, url)
    time.sleep(REQUEST_DELAY)

    if not data:
        return 0

    results = data.get("results", [])
    if not results:
        print(f"[sofascore] No team results for "
              f"{league} {season_label}")
        return 0

    print(f"[sofascore] {league} {season_label}: "
          f"{len(results)} teams")

    db    = SessionLocal()
    total = 0
    try:
        for entry in results:
            team_name = entry.get(
                "team", {}).get("name", "")
            if not team_name:
                continue
            games = int(
                entry.get("statistics", {})
                     .get("matches", 0) or 0)
            attrs = _parse_team_stats(entry, games)
            if attrs:
                _upsert_team(
                    db, team_name,
                    league, season_label, attrs)
                total += 1
        db.commit()
        print(f"[sofascore] Committed {total} "
              f"team rows for "
              f"{league} {season_label}")
    except Exception as exc:
        db.rollback()
        print(f"[sofascore] ERROR team commit "
              f"{league} {season_label}: {exc}")
        raise
    finally:
        db.close()

    return total


# ═══════════════════════════════════════════
# PLAYER STATS
# ═══════════════════════════════════════════

POSITION_MAP = {
    "G": "GK", "GK": "GK",
    "D": "DF", "DF": "DF",
    "M": "MF", "MF": "MF",
    "F": "FW", "FW": "FW",
    "A": "FW",
}


def _parse_player(entry: dict):
    player_info = entry.get("player", {})
    team_info   = entry.get("team",   {})
    s           = entry.get("statistics", {})

    player_name = player_info.get("name", "")
    team_name   = team_info.get("name",   "")
    if not player_name or not team_name:
        return None

    pos_raw  = player_info.get("position", "")
    position = POSITION_MAP.get(
        str(pos_raw).upper(), pos_raw or None)

    attrs = {}

    goals   = _safe(s.get("goals"))
    assists = _safe(s.get("assists") or
                    s.get("goalAssist"))
    xg      = _safe(s.get("expectedGoals") or
                    s.get("xG"))
    xa      = _safe(s.get("expectedAssists") or
                    s.get("xA"))
    mins    = _safe(s.get("minutesPlayed") or
                    s.get("minutes"))

    if goals   is not None: attrs["goals"]   = goals
    if assists is not None: attrs["assists"]  = assists
    if xg      is not None: attrs["xg"]      = xg
    if xa      is not None: attrs["xag"]     = xa

    if goals is not None and assists is not None:
        attrs["g_plus_a"] = goals + assists

    if mins is not None:
        attrs["minutes"] = mins
        if mins > 0:
            per90 = mins / 90
            attrs["minutes_per90"] = round(per90, 2)
            if goals   is not None:
                attrs["gls_per90"] = round(
                    goals / per90, 3)
            if assists is not None:
                attrs["ast_per90"] = round(
                    assists / per90, 3)
            if goals is not None and assists is not None:
                attrs["g_plus_a_per90"] = round(
                    (goals + assists) / per90, 3)

    apps = _safe(s.get("appearances") or
                 s.get("matches"))
    if apps is not None:
        attrs["mp"] = int(apps)

    shots = _safe(s.get("totalShots") or
                  s.get("shots"))
    if shots is not None:
        attrs["sh_shots"] = shots

    sot = _safe(s.get("shotsOnTarget"))
    if sot is not None:
        attrs["sh_sot"] = sot
        if shots and shots > 0:
            attrs["sh_sot_pct"] = round(
                sot / shots * 100, 1)

    yellows = _safe(s.get("yellowCards"))
    if yellows is not None:
        attrs["crdy"] = yellows

    reds = _safe(s.get("redCards"))
    if reds is not None:
        attrs["crdr"] = reds

    tackles = _safe(s.get("tackles") or
                    s.get("tacklesWon"))
    if tackles is not None:
        attrs["tkl"] = tackles

    ints = _safe(s.get("interceptions"))
    if ints is not None:
        attrs["int_"] = ints
        if tackles is not None:
            attrs["tkl_plus_int"] = tackles + ints

    return player_name, team_name, position, attrs


def collect_player_stats(
    session, league, tid, sid, season_label
) -> int:
    total = 0
    db    = SessionLocal()
    seen  = set()

    try:
        for page in range(0, 5):
            url  = (
                f"{BASE_URL}/unique-tournament/{tid}"
                f"/season/{sid}/top-players/overall"
            )
            data = _get(session, url,
                        params={"page": page})
            time.sleep(REQUEST_DELAY)

            if not data:
                break

            top = data.get("topPlayers", {})
            if not top:
                break

            entries = []
            for category, plist in top.items():
                for entry in (plist or []):
                    pid = (entry.get("player", {})
                               .get("id"))
                    if pid and pid not in seen:
                        seen.add(pid)
                        entries.append(entry)

            if not entries:
                break

            print(f"[sofascore] {league} "
                  f"{season_label} page {page}: "
                  f"{len(entries)} players")

            for entry in entries:
                result = _parse_player(entry)
                if result is None:
                    continue
                pname, tname, pos, attrs = result
                if not attrs:
                    continue

                if pos:
                    attrs["position"] = pos

                _upsert_player(
                    db, pname, tname,
                    league, season_label, attrs)
                total += 1

            db.commit()
            if len(entries) < 20:
                break

    except Exception as exc:
        db.rollback()
        print(f"[sofascore] ERROR players "
              f"{league} {season_label}: {exc}")
        raise
    finally:
        db.close()

    print(f"[sofascore] {league} {season_label}: "
          f"{total} players committed")
    return total


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════


def collect_sofascore() -> dict:
    """
    Primary data collection from Sofascore.
    Exits with code 1 if zero rows collected
    so GitHub Actions workflow can fall back
    to Understat backup.
    """
    print("=" * 60)
    print("SOFASCORE — PRIMARY STATS COLLECTION")
    print("=" * 60)

    session, client_type = _get_client()
    if client_type != "patchright":
        print("[sofascore] WARNING: install "
              "patchright and Chromium for better results: "
              "pip install patchright && "
              "python -m patchright install chromium")

    try:
        team_total   = 0
        player_total = 0
        failed       = []

        for league, config in LEAGUE_CONFIG.items():
            tid         = config["tid"]
            is_mls      = league == "USA-MLS"
            target_yrs  = (MLS_TARGET_SEASONS
                           if is_mls else TARGET_SEASONS)

            print(f"\n[sofascore] === {league} ===")

            seasons = get_seasons(
                session, tid, target_yrs)
            if not seasons:
                print(f"[sofascore] WARNING: no "
                      f"seasons for {league}")
                failed.append(league)
                continue

            for si in seasons:
                sid   = si["sid"]
                year  = si["year"]
                label = _season_label(
                    league, year, si["name"])

                print(f"[sofascore] {league} "
                      f"{label} (sid={sid})...")

                try:
                    n = collect_team_stats(
                        session, league,
                        tid, sid, label)
                    team_total += n
                except Exception as exc:
                    print(f"[sofascore] team error "
                          f"{league} {label}: {exc}")
                    failed.append(
                        f"{league}/{label}/team")

                if year == max(target_yrs):
                    try:
                        n = collect_player_stats(
                            session, league,
                            tid, sid, label)
                        player_total += n
                    except Exception as exc:
                        print(
                            f"[sofascore] player "
                            f"error {league} "
                            f"{label}: {exc}")
                        failed.append(
                            f"{league}/{label}/players")

        print("\n" + "=" * 60)
        print("SOFASCORE COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Team rows   : {team_total}")
        print(f"Player rows : {player_total}")
        if failed:
            print(f"Failed      : {failed}")

        if team_total == 0 and player_total == 0:
            print("\nERROR: Zero rows collected.")
            print("Sofascore is blocking requests.")
            sys.exit(1)

        return {
            "team_total":   team_total,
            "player_total": player_total,
            "failed":       failed,
        }
    finally:
        # Close browser if patchright was used (also close fallback sessions).
        if hasattr(session, "close"):
            try:
                session.close()
                print("[sofascore] Browser closed" if client_type == "patchright"
                      else "[sofascore] Session closed")
            except Exception:
                pass



if __name__ == "__main__":
    collect_sofascore()
