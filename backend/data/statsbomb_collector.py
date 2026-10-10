"""
StatsBomb Open Data collector.

Downloads free StatsBomb event data directly
from their GitHub repository. No scraping,
no authentication, no CAPTCHA.

Data is stored in Supabase for use in
training the formula engine.

StatsBomb open data GitHub:
  https://github.com/statsbomb/open-data

Run from backend/ directory:
    python data/statsbomb_collector.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal
from sqlalchemy import text


# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

STATSBOMB_RAW = (
    "https://raw.githubusercontent.com"
    "/statsbomb/open-data/master/data"
)

# Competitions we want to download
# Format: (competition_id, season_id, label)
TARGET_COMPETITIONS = [
    # La Liga — Messi era (richest event data)
    (11, 90, "ESP-La Liga 2021-2022"),
    (11, 42, "ESP-La Liga 2020-2021"),
    (11, 4, "ESP-La Liga 2019-2020"),
    (11, 1, "ESP-La Liga 2018-2019"),
    # Premier League
    (2, 27, "ENG-Premier League 2015-2016"),
    # Champions League
    (16, 90, "UEFA Champions League 2021-2022"),
    (16, 44, "UEFA Champions League 2019-2020"),
    # FIFA World Cup
    (43, 3, "FIFA World Cup 2022"),
    (43, 106, "FIFA World Cup 2018"),
    # Women's Super League
    (37, 90, "Women's Super League 2021-2022"),
    (37, 42, "Women's Super League 2020-2021"),
]

# Maximum matches to download per competition
# to keep storage manageable
MAX_MATCHES_PER_COMP = 50


# ═══════════════════════════════════════════
# DOWNLOADER
# ═══════════════════════════════════════════


def _get_json(path: str) -> Any | None:
    """
    Download a JSON file from StatsBomb GitHub.
    Returns parsed JSON or None on failure.
    """
    url = f"{STATSBOMB_RAW}/{path}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"[sb] WARNING: HTTP {resp.status_code} for {path}")
            return None
    except Exception as exc:
        print(f"[sb] ERROR: {path}: {exc}")
        return None


def get_competitions() -> list[dict]:
    """Download the full competitions index."""
    data = _get_json("competitions.json")
    return data if data else []


def get_matches(competition_id: int, season_id: int) -> list[dict]:
    """Download match list for one competition/season."""
    data = _get_json(f"matches/{competition_id}/{season_id}.json")
    return data if data else []


def get_events(match_id: int) -> list[dict]:
    """Download all events for one match."""
    data = _get_json(f"events/{match_id}.json")
    return data if data else []


def get_lineups(match_id: int) -> list[dict]:
    """Download lineups for one match."""
    data = _get_json(f"lineups/{match_id}.json")
    return data if data else []


# ═══════════════════════════════════════════
# EVENT PROCESSING
# ═══════════════════════════════════════════


def _extract_shot_features(event: dict) -> dict | None:
    """
    Extract features from a shot event.
    Returns None if not a shot.
    """
    if event.get("type", {}).get("name") != "Shot":
        return None

    shot = event.get("shot", {})
    loc = event.get("location", [0, 0])

    return {
        "event_type": "shot",
        "player": event.get("player", {}).get("name"),
        "team": event.get("team", {}).get("name"),
        "x": loc[0] if len(loc) > 0 else None,
        "y": loc[1] if len(loc) > 1 else None,
        "xg": shot.get("statsbomb_xg"),
        "outcome": shot.get("outcome", {}).get("name"),
        "technique": shot.get("technique", {}).get("name"),
        "body_part": shot.get("body_part", {}).get("name"),
        "first_time": shot.get("first_time", False),
        "follows_dribble": shot.get("follows_dribble", False),
        "open_play_xg": shot.get("open_play_xg"),
        "deflected": shot.get("deflected", False),
        "minute": event.get("minute"),
    }


def _extract_pressure_features(event: dict) -> dict | None:
    """Extract features from a pressure event."""
    if event.get("type", {}).get("name") != "Pressure":
        return None

    loc = event.get("location", [0, 0])
    return {
        "event_type": "pressure",
        "player": event.get("player", {}).get("name"),
        "team": event.get("team", {}).get("name"),
        "x": loc[0] if len(loc) > 0 else None,
        "y": loc[1] if len(loc) > 1 else None,
        "minute": event.get("minute"),
    }


def _extract_carry_features(event: dict) -> dict | None:
    """Extract features from a carry (progressive) event."""
    if event.get("type", {}).get("name") != "Carry":
        return None

    carry = event.get("carry", {})
    loc = event.get("location", [0, 0])
    end = carry.get("end_location", [0, 0])

    # Progressive carry: moves toward goal by 10+ yards
    start_x = loc[0] if len(loc) > 0 else 0
    end_x = end[0] if len(end) > 0 else 0
    is_progressive = (end_x - start_x) >= 10

    if not is_progressive:
        return None

    return {
        "event_type": "progressive_carry",
        "player": event.get("player", {}).get("name"),
        "team": event.get("team", {}).get("name"),
        "start_x": start_x,
        "end_x": end_x,
        "distance": end_x - start_x,
        "minute": event.get("minute"),
    }


def _extract_gk_features(event: dict) -> dict | None:
    """Extract goalkeeper save features."""
    if event.get("type", {}).get("name") != "Goal Keeper":
        return None

    gk = event.get("goalkeeper", {})
    loc = event.get("location", [0, 0])

    return {
        "event_type": "goalkeeper_action",
        "player": event.get("player", {}).get("name"),
        "team": event.get("team", {}).get("name"),
        "action": gk.get("action", {}).get("name"),
        "outcome": gk.get("outcome", {}).get("name"),
        "position": gk.get("position", {}).get("name"),
        "x": loc[0] if len(loc) > 0 else None,
        "y": loc[1] if len(loc) > 1 else None,
        "minute": event.get("minute"),
    }


def process_match_events(events: list[dict], match_info: dict) -> dict:
    """
    Process all events in one match and extract
    team-level aggregate statistics.

    Returns a dict with stats for home and away team.
    """
    home_team = match_info.get("home_team", {}).get("home_team_name", "")
    away_team = match_info.get("away_team", {}).get("away_team_name", "")

    stats = {
        home_team: {
            "shots": 0,
            "shots_on_target": 0,
            "xg": 0.0,
            "goals": 0,
            "pressures": 0,
            "pressure_regains": 0,
            "progressive_carries": 0,
            "gk_saves": 0,
            "psxg": 0.0,
        },
        away_team: {
            "shots": 0,
            "shots_on_target": 0,
            "xg": 0.0,
            "goals": 0,
            "pressures": 0,
            "pressure_regains": 0,
            "progressive_carries": 0,
            "gk_saves": 0,
            "psxg": 0.0,
        },
    }

    for event in events:
        team = event.get("team", {}).get("name", "")
        if team not in stats:
            continue

        etype = event.get("type", {}).get("name", "")

        if etype == "Shot":
            shot = event.get("shot", {})
            xg = shot.get("statsbomb_xg", 0) or 0
            outcome = shot.get("outcome", {}).get("name", "")
            stats[team]["shots"] += 1
            stats[team]["xg"] += float(xg)
            if outcome in ("Goal", "Saved", "Saved To Post"):
                stats[team]["shots_on_target"] += 1
            if outcome == "Goal":
                stats[team]["goals"] += 1

        elif etype == "Pressure":
            stats[team]["pressures"] += 1

        elif etype == "Carry":
            carry = event.get("carry", {})
            loc = event.get("location", [0, 0])
            end = carry.get("end_location", [0, 0])
            if len(loc) > 0 and len(end) > 0:
                if (end[0] - loc[0]) >= 10:
                    stats[team]["progressive_carries"] += 1

        elif etype == "Goal Keeper":
            gk = event.get("goalkeeper", {})
            outcome = gk.get("outcome", {}).get("name", "")
            if outcome in ("Saved", "Saved To Post", "Touched Out"):
                stats[team]["gk_saves"] += 1

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_stats": stats.get(home_team, {}),
        "away_stats": stats.get(away_team, {}),
    }


# ═══════════════════════════════════════════
# DATABASE STORAGE
# ═══════════════════════════════════════════


def ensure_tables(db) -> None:
    """
    Create statsbomb_match_stats table if it
    does not exist. Stores processed match stats.
    """
    db.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS statsbomb_match_stats (
            id              SERIAL PRIMARY KEY,
            match_id        INTEGER UNIQUE NOT NULL,
            competition     TEXT,
            season          TEXT,
            date            TEXT,
            home_team       TEXT,
            away_team       TEXT,
            home_goals      INTEGER,
            away_goals      INTEGER,
            home_xg         FLOAT,
            away_xg         FLOAT,
            home_shots      INTEGER,
            away_shots      INTEGER,
            home_shots_ot   INTEGER,
            away_shots_ot   INTEGER,
            home_pressures  INTEGER,
            away_pressures  INTEGER,
            home_prog_carr  INTEGER,
            away_prog_carr  INTEGER,
            home_gk_saves   INTEGER,
            away_gk_saves   INTEGER,
            created_at      TIMESTAMP DEFAULT NOW()
        )
    """
        )
    )
    db.commit()
    print("[sb] Table statsbomb_match_stats ready")


def store_match_stats(
    db,
    match_id: int,
    competition: str,
    season: str,
    date_str: str,
    processed: dict,
    match_info: dict,
) -> None:
    """Upsert processed match stats into database."""
    home = processed["home_stats"]
    away = processed["away_stats"]

    score = match_info.get("score", {})
    home_goals = score.get("full_time", {}).get("home") or 0
    away_goals = score.get("full_time", {}).get("away") or 0

    db.execute(
        text(
            """
        INSERT INTO statsbomb_match_stats (
            match_id, competition, season, date,
            home_team, away_team,
            home_goals, away_goals,
            home_xg, away_xg,
            home_shots, away_shots,
            home_shots_ot, away_shots_ot,
            home_pressures, away_pressures,
            home_prog_carr, away_prog_carr,
            home_gk_saves, away_gk_saves
        ) VALUES (
            :match_id, :competition, :season, :date,
            :home_team, :away_team,
            :home_goals, :away_goals,
            :home_xg, :away_xg,
            :home_shots, :away_shots,
            :home_shots_ot, :away_shots_ot,
            :home_pressures, :away_pressures,
            :home_prog_carr, :away_prog_carr,
            :home_gk_saves, :away_gk_saves
        )
        ON CONFLICT (match_id) DO UPDATE SET
            home_xg        = EXCLUDED.home_xg,
            away_xg        = EXCLUDED.away_xg,
            home_shots     = EXCLUDED.home_shots,
            away_shots     = EXCLUDED.away_shots,
            home_shots_ot  = EXCLUDED.home_shots_ot,
            away_shots_ot  = EXCLUDED.away_shots_ot,
            home_pressures = EXCLUDED.home_pressures,
            away_pressures = EXCLUDED.away_pressures,
            home_prog_carr = EXCLUDED.home_prog_carr,
            away_prog_carr = EXCLUDED.away_prog_carr,
            home_gk_saves  = EXCLUDED.home_gk_saves,
            away_gk_saves  = EXCLUDED.away_gk_saves
    """
        ),
        {
            "match_id": match_id,
            "competition": competition,
            "season": season,
            "date": date_str,
            "home_team": processed["home_team"],
            "away_team": processed["away_team"],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": home["xg"],
            "away_xg": away["xg"],
            "home_shots": home["shots"],
            "away_shots": away["shots"],
            "home_shots_ot": home["shots_on_target"],
            "away_shots_ot": away["shots_on_target"],
            "home_pressures": home["pressures"],
            "away_pressures": away["pressures"],
            "home_prog_carr": home["progressive_carries"],
            "away_prog_carr": away["progressive_carries"],
            "home_gk_saves": home["gk_saves"],
            "away_gk_saves": away["gk_saves"],
        },
    )


# ═══════════════════════════════════════════
# TEAM AGGREGATE STATS
# ═══════════════════════════════════════════


def compute_team_season_aggregates() -> None:
    """
    After downloading match events, compute
    per-team per-season averages from
    statsbomb_match_stats and store in
    team_season_stats table.

    This gives us the rich features we need
    for the formula engine.
    """
    print("[sb] Computing team season aggregates...")

    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                """
            SELECT
                competition,
                season,
                home_team AS team,
                AVG(home_xg)         AS avg_xg,
                AVG(away_xg)         AS avg_xga,
                AVG(home_shots)      AS avg_shots,
                AVG(home_shots_ot)   AS avg_sot,
                AVG(home_pressures)  AS avg_pressures,
                AVG(home_prog_carr)  AS avg_prgc,
                AVG(home_gk_saves)   AS avg_gk_saves,
                AVG(home_goals)      AS avg_goals,
                COUNT(*)             AS games_played
            FROM statsbomb_match_stats
            GROUP BY competition, season, home_team

            UNION ALL

            SELECT
                competition,
                season,
                away_team AS team,
                AVG(away_xg)         AS avg_xg,
                AVG(home_xg)         AS avg_xga,
                AVG(away_shots)      AS avg_shots,
                AVG(away_shots_ot)   AS avg_sot,
                AVG(away_pressures)  AS avg_pressures,
                AVG(away_prog_carr)  AS avg_prgc,
                AVG(away_gk_saves)   AS avg_gk_saves,
                AVG(away_goals)      AS avg_goals,
                COUNT(*)             AS games_played
            FROM statsbomb_match_stats
            GROUP BY competition, season, away_team
        """
            )
        ).fetchall()

        from database import TeamSeasonStats
        from sqlalchemy import and_

        updated = 0
        for row in rows:
            existing = (
                db.query(TeamSeasonStats)
                .filter(
                    and_(
                        TeamSeasonStats.team == row.team,
                        TeamSeasonStats.league == row.competition,
                        TeamSeasonStats.season == row.season,
                    )
                )
                .first()
            )
            if existing is None:
                existing = TeamSeasonStats(
                    team=row.team,
                    league=row.competition,
                    season=row.season,
                )
                db.add(existing)

            # Map aggregates to TeamSeasonStats columns
            existing.sh_xg = row.avg_xg
            existing.sh_shots = row.avg_shots
            existing.sh_sot = row.avg_sot
            existing.sh_sh_per90 = row.avg_shots
            existing.sh_sot_per90 = row.avg_sot
            existing.def_press = row.avg_pressures
            existing.pos_prgc = row.avg_prgc
            existing.gk_saves = row.avg_gk_saves
            existing.sh_goals = row.avg_goals
            existing.gk_ga = row.avg_xga
            updated += 1

        db.commit()
        print(f"[sb] Updated {updated} team-season rows in team_season_stats")

    except Exception as exc:
        db.rollback()
        print(f"[sb] ERROR computing aggregates: {exc}")
    finally:
        db.close()


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════


def collect_statsbomb() -> None:
    """
    Download StatsBomb open data for all
    target competitions and store in database.
    """
    print("=" * 60)
    print("STATSBOMB OPEN DATA COLLECTION")
    print("=" * 60)

    db = SessionLocal()
    try:
        ensure_tables(db)
    finally:
        db.close()

    total_matches = 0
    total_processed = 0
    total_skipped = 0

    for comp_id, season_id, label in TARGET_COMPETITIONS:
        print(f"\n[sb] Competition: {label}")

        matches = get_matches(comp_id, season_id)
        if not matches:
            print(f"[sb] WARNING: no matches found for {label}")
            continue

        print(
            f"[sb] Found {len(matches)} matches — "
            f"downloading up to {MAX_MATCHES_PER_COMP}"
        )

        matches = matches[:MAX_MATCHES_PER_COMP]

        for match_info in matches:
            match_id = match_info.get("match_id")
            if not match_id:
                continue

            date_str = match_info.get("match_date", "")
            total_matches += 1

            # Download events
            events = get_events(match_id)
            if not events:
                total_skipped += 1
                continue

            # Process events into team stats
            processed = process_match_events(events, match_info)

            # Store in database
            db = SessionLocal()
            try:
                store_match_stats(
                    db, match_id, label, label, date_str, processed, match_info
                )
                db.commit()
                total_processed += 1

                home = processed["home_team"]
                away = processed["away_team"]
                hxg = processed["home_stats"]["xg"]
                axg = processed["away_stats"]["xg"]
                print(f"[sb]   {home} vs {away} | xG: {hxg:.2f} - {axg:.2f}")

            except Exception as exc:
                db.rollback()
                print(f"[sb] ERROR storing match {match_id}: {exc}")
                total_skipped += 1
            finally:
                db.close()

            # Small delay to be respectful
            time.sleep(0.5)

    # Compute team aggregates after all downloads
    compute_team_season_aggregates()

    print("\n" + "=" * 60)
    print("STATSBOMB COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Competitions : {len(TARGET_COMPETITIONS)}")
    print(f"Matches found: {total_matches}")
    print(f"Processed    : {total_processed}")
    print(f"Skipped      : {total_skipped}")


if __name__ == "__main__":
    collect_statsbomb()
