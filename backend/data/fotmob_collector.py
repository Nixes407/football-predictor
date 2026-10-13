"""
FotMob advanced stats collector.

Replaces Sofascore as the primary advanced team stats source.
The published soccerdata package (1.9.1) has no FotMob scraper or
read_team_season_stats method for FotMob. Use its JSON HTTP API directly
instead: no Chrome or browser automation is required on GitHub Actions.
HTTP access is still subject to FotMob availability and access restrictions.

Collects team season stats including xG, shots, possession and more
for all 9 leagues. Missing statistics never overwrite existing values.

Run from backend/ directory:
    python data/fotmob_collector.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, TeamSeasonStats
from sqlalchemy import and_


# FotMob tournament IDs, not Sofascore tournament IDs.
LEAGUE_IDS = {
    "ENG-Premier League": 47,
    "ENG-Championship": 48,
    "ESP-La Liga": 87,
    "GER-Bundesliga": 54,
    "ITA-Serie A": 55,
    "FRA-Ligue 1": 53,
    "POR-Primeira Liga": 61,
    "NED-Eredivisie": 57,
    "USA-MLS": 130,
}
LEAGUES = list(LEAGUE_IDS)
SEASONS = ["2025-2026", "2026-2027"]
MLS_SEASONS = ["2025", "2026"]
BASE_URL = "https://www.fotmob.com/api"
REQUEST_DELAY = 1.0
REQUEST_TIMEOUT = (10, 60)

# Older responses supply English titles without machine-readable keys.
_STAT_TITLES = {
    "Ball possession": "BallPossesion",
    "Ball possesion": "BallPossesion",
    "Expected goals (xG)": "expected_goals",
    "Expected goals": "expected_goals",
    "Total shots": "total_shots",
    "Shots on target": "ShotsOnTarget",
    "Accurate passes": "accurate_passes",
    "Passes": "passes",
    "Pass success": "accurate_passes_percentage",
    "Fouls": "fouls",
    "Corners": "corners",
    "Offsides": "Offsides",
    "Tackles": "tackles",
    "Tackles won": "tackles_succeeded",
    "Interceptions": "interceptions",
    "Clearances": "clearances",
    "Blocks": "shot_blocks",
    "Saves": "keeper_saves",
    "Keeper saves": "keeper_saves",
    "Yellow cards": "yellow_cards",
    "Red cards": "red_cards",
}


def _get_seasons(league: str) -> list[str]:
    return MLS_SEASONS if league == "USA-MLS" else SEASONS


def _safe(val) -> float | None:
    """Parse FotMob numbers, percentages and missing values safely."""
    try:
        if isinstance(val, str):
            val = val.strip().rstrip("%").replace(",", "")
        number = float(val)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


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


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column, ignoring case as a fallback."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower = {col.lower(): col for col in df.columns if isinstance(col, str)}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _col_sum(df: pd.DataFrame, candidates: list[str]) -> float | None:
    """Sum only complete metrics; absent match stats are not zeroes."""
    col = _find_col(df, candidates)
    if col is None or df.empty:
        return None
    values = df[col].map(_safe)
    # Do not replace complete season totals with partial coverage totals.
    return _safe(values.sum(min_count=len(df)))


def _col_mean(df: pd.DataFrame, candidates: list[str]) -> float | None:
    """Average a fully covered metric, without treating missing as zero."""
    col = _find_col(df, candidates)
    if col is None or df.empty:
        return None
    values = df[col].map(_safe)
    return _safe(values.mean()) if values.notna().all() else None


def _extract_attrs(group: pd.DataFrame, games: int) -> dict:
    """Aggregate one row per team per completed match to season stats."""
    attrs = {}
    totals = {
        "sh_xg": ["expected_goals", "xg", "team_xg", "expectedGoals"],
        "sh_npxg": ["expected_goals_non_penalty"],
        "sh_goals": ["goals", "score", "goals_scored"],
        "sh_shots": ["total_shots", "shots", "shots_total", "shotsTotal"],
        "sh_sot": ["ShotsOnTarget", "shots_on_target", "on_target", "shots_on_goal"],
        "gk_ga": ["goals_conceded", "goals_against", "opponent_score", "goalsConceded"],
        "gk_saves": ["keeper_saves", "saves", "goalkeeper_saves", "savedShots"],
        "gk_sota": ["opponent_shots_on_target"],
        "gk_cs": ["clean_sheets", "cleanSheets", "clean_sheet"],
        "def_tkl": ["tackles", "total_tackles"],
        "def_tklw": ["tackles_succeeded", "tackles_won", "tacklesWon"],
        "def_int": ["interceptions"],
        "def_clr": ["clearances"],
        "def_blk_sh": ["shot_blocks", "blocked_shots"],
        "misc_crdy": ["yellow_cards", "yellowCards"],
        "misc_crdr": ["red_cards", "redCards"],
        "misc_fls": ["fouls", "fouls_committed", "foulsCommitted"],
        "misc_off": ["Offsides", "offsides"],
        "ps_cmp": ["accurate_passes", "accuratePasses"],
        "ps_att": ["passes", "total_passes"],
    }
    # Corners have no corresponding TeamSeasonStats column. In particular,
    # pos_cpa means carries into the penalty area, NOT corners.
    for field, candidates in totals.items():
        value = _col_sum(group, candidates)
        if value is not None:
            attrs[field] = round(value, 3)

    possession = _col_mean(group, [
        "BallPossesion", "BallPossession", "Ball possession", "Ball possesion",
        "possession", "ball_possession", "possessionPercent",
    ])
    if possession is not None:
        attrs["pos_poss"] = round(possession, 1)

    minutes = _col_sum(group, ["minutes"])
    nineties = minutes / 90 if minutes is not None else games
    for total, rate in (("sh_shots", "sh_sh_per90"),
                        ("sh_sot", "sh_sot_per90"), ("gk_ga", "gk_ga90")):
        if total in attrs and nineties > 0:
            attrs[rate] = round(attrs[total] / nineties, 2)

    for numerator, denominator, field, scale in (
        ("sh_goals", "sh_shots", "sh_g_per_sh", 1),
        ("sh_goals", "sh_sot", "sh_g_per_sot", 1),
        ("sh_sot", "sh_shots", "sh_sot_pct", 100),
        ("sh_npxg", "sh_shots", "sh_npxg_per_sh", 1),
        ("ps_cmp", "ps_att", "ps_cmp_pct", 100),
    ):
        if numerator in attrs and attrs.get(denominator, 0) > 0:
            attrs[field] = round(attrs[numerator] / attrs[denominator] * scale, 3)

    if "ps_cmp_pct" not in attrs:
        accuracy = _col_mean(group, [
            "accurate_passes_percentage", "pass_accuracy", "passAccuracy", "Pass success",
        ])
        if accuracy is not None:
            attrs["ps_cmp_pct"] = round(accuracy, 1)
    if "sh_goals" in attrs and "sh_xg" in attrs:
        attrs["sh_g_minus_xg"] = round(attrs["sh_goals"] - attrs["sh_xg"], 3)
    if "def_tkl" in attrs and "def_int" in attrs:
        attrs["def_tkl_int"] = attrs["def_tkl"] + attrs["def_int"]
    if "gk_saves" in attrs and attrs.get("gk_sota", 0) > 0:
        attrs["gk_save_pct"] = round(attrs["gk_saves"] / attrs["gk_sota"] * 100, 1)
    if "gk_cs" in attrs and games > 0:
        attrs["gk_cs_pct"] = round(attrs["gk_cs"] / games * 100, 1)
    return attrs


def _is_finished(status: dict) -> bool:
    """Exclude scheduled, live, cancelled, postponed and abandoned games."""
    reason = status.get("reason") or {}
    text = " ".join(str(value).lower() for value in reason.values())
    return bool(status.get("finished")) and not (
        status.get("cancelled") or status.get("awarded")
        or any(word in text for word in ("abandon", "abort", "cancel", "postpon", "award"))
    )


def _match_rows(payload: dict) -> list[dict]:
    """Flatten content.stats.Periods.All only, never add half-time twice."""
    header = payload.get("header") or {}
    status = header.get("status") or {}
    if not _is_finished(status):
        return []
    teams = header.get("teams") or []
    if len(teams) != 2:
        raise ValueError("FotMob match response has no home/away teams")
    scores = [_safe(team.get("score")) for team in teams]
    if any(score is None for score in scores):
        raise ValueError("Completed FotMob match has no numeric score")
    reason = status.get("reason") or {}
    extra_time = bool((payload.get("general") or {}).get("hasExtraTime")) or (
        reason.get("short") in {"AET", "AP", "Pen"}
    )
    rows = []
    for side, team in enumerate(teams):
        name = team.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("FotMob match response has no team name")
        conceded = scores[1 - side]
        rows.append({
            "team": name.strip(), "goals": scores[side],
            "goals_conceded": conceded, "clean_sheets": int(conceded == 0),
            "minutes": 120 if extra_time else 90,
        })

    stats = (payload.get("content") or {}).get("stats") or {}
    if "Periods" in stats:
        stats = (stats.get("Periods") or {}).get("All") or {}
    categories = stats.get("stats", []) if isinstance(stats, dict) else stats
    if not isinstance(categories, list):
        raise ValueError("Unexpected FotMob match statistics format")
    for category in categories:
        for stat in category.get("stats", []):
            values = stat.get("stats")
            if not isinstance(values, list) or len(values) != 2:
                continue
            title = stat.get("title")
            key = stat.get("key") or _STAT_TITLES.get(title, title)
            if not isinstance(key, str):
                continue
            for side, value in enumerate(values):
                rows[side][key] = _safe(value)
                if key.lower() in {"shotsontarget", "shots_on_target"}:
                    rows[1 - side]["opponent_shots_on_target"] = _safe(value)
    return rows


class _FotMobClient:
    """Bounded, rate-limited HTTP requests with retries; no browser required."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; FootballPredictor/1.0)",
            "Accept": "application/json",
            "Referer": "https://www.fotmob.com/",
        })
        retry = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"], respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._last_request = 0.0

    def _get(self, endpoint: str, **params) -> dict:
        time.sleep(max(0, REQUEST_DELAY - (time.monotonic() - self._last_request)))
        try:
            response = self.session.get(
                f"{BASE_URL}/{endpoint}", params=params, timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"Unexpected FotMob {endpoint} response")
            return payload
        finally:
            self._last_request = time.monotonic()

    def read_team_match_stats(self, league: str, season: str) -> pd.DataFrame:
        """Get the requested season's schedule, then completed match details."""
        api_season = season.replace("-", "/")
        payload = self._get("leagues", id=LEAGUE_IDS[league], season=api_season)
        available = payload.get("allAvailableSeasons")
        if available is not None and api_season not in available:
            raise ValueError(f"FotMob does not publish season {api_season}")
        selected = (payload.get("overview") or {}).get("selectedSeason")
        if selected is not None and selected != api_season:
            raise ValueError(f"FotMob returned season {selected}, expected {api_season}")
        matches = (payload.get("matches") or {}).get("allMatches")
        if not isinstance(matches, list):
            raise ValueError("FotMob league response has no full season match list")
        rows = []
        seen = set()
        for match in matches:
            if not _is_finished(match.get("status") or {}):
                continue
            match_id = match.get("id")
            if match_id is None:
                raise ValueError("Completed FotMob fixture has no ID")
            match_id = str(match_id)
            if match_id in seen:
                continue
            seen.add(match_id)
            details = self._get("matchDetails", matchId=match_id)
            general = details.get("general") or {}
            returned_season = general.get("parentLeagueSeason") or general.get("leagueSeason")
            if returned_season is not None and returned_season != api_season:
                raise ValueError(f"Match {match_id} belongs to season {returned_season}")
            match_rows = _match_rows(details)
            if not match_rows:
                raise ValueError(f"Completed fixture {match_id} has unfinished match details")
            rows.extend(match_rows)
        # Any failed match aborts this season instead of saving partial totals.
        return pd.DataFrame(rows)

    def close(self) -> None:
        self.session.close()


def collect_team_stats() -> int:
    """Collect all league/seasons; return only successfully committed rows."""
    print("[fotmob] Collecting team stats using the direct JSON API...")
    total = 0
    client = _FotMobClient()
    try:
        for league in LEAGUES:
            for season in _get_seasons(league):
                print(f"[fotmob] {league} {season}...")
                try:
                    df = client.read_team_match_stats(league, season)
                    if df.empty:
                        print(f"[fotmob] WARNING: no completed matches for {league} {season}")
                        continue
                    print(f"[fotmob] Got {len(df)} team-match rows. Columns: {list(df.columns)[:6]}...")
                    db = SessionLocal()
                    committed = 0
                    try:
                        for team, group in df.groupby("team"):
                            attrs = _extract_attrs(group, len(group))
                            if attrs:
                                _upsert_team(db, team, league, season, attrs)
                                committed += 1
                        db.commit()
                        total += committed
                        print(f"[fotmob] {league} {season}: committed {committed} teams")
                    except Exception:
                        db.rollback()
                        raise
                    finally:
                        db.close()
                except Exception as exc:
                    print(f"[fotmob] ERROR {league} {season}: {exc}")
    finally:
        client.close()
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
