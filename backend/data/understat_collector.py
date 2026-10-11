"""
Understat xG data collector.

Fetches team and match level xG statistics
from Understat via soccerdata library.
No Chrome, no CAPTCHA — simple HTTP requests.

Covers 6 leagues:
  ENG-Premier League
  ESP-La Liga
  GER-Bundesliga
  ITA-Serie A
  FRA-Ligue 1
  NED-Eredivisie

Stores results in team_season_stats table
updating xG and PPDA fields.

Run from backend/ directory:
    python data/understat_collector.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import soccerdata as sd

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, TeamSeasonStats
from sqlalchemy import and_


# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

# Understat only covers these leagues
UNDERSTAT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

SEASONS = ["2023-2024", "2024-2025"]

# Understat uses different league name format
# Map our league names to Understat format
UNDERSTAT_LEAGUE_MAP = {
    "ENG-Premier League": "EPL",
    "ESP-La Liga": "La_liga",
    "GER-Bundesliga": "Bundesliga",
    "ITA-Serie A": "Serie_A",
    "FRA-Ligue 1": "Ligue_1",
    "NED-Eredivisie": "RFPL",
}

# Map Understat season format to our format
# Understat uses the year the season starts
UNDERSTAT_SEASON_MAP = {
    "2023-2024": 2023,
    "2024-2025": 2024,
}


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════


def _safe(val) -> float | None:
    """Convert value to float or None."""
    try:
        import math

        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _upsert_team(
    db,
    team: str,
    league: str,
    season: str,
    attrs: dict,
) -> None:
    """Insert or update one TeamSeasonStats row."""
    existing = (
        db.query(TeamSeasonStats)
        .filter(
            and_(
                TeamSeasonStats.team == team,
                TeamSeasonStats.league == league,
                TeamSeasonStats.season == season,
            )
        )
        .first()
    )
    if existing is None:
        existing = TeamSeasonStats(team=team, league=league, season=season)
        db.add(existing)
    for k, v in attrs.items():
        setattr(existing, k, v)


# ═══════════════════════════════════════════
# TEAM SEASON STATS
# ═══════════════════════════════════════════


def collect_team_xg_stats() -> int:
    """
    Fetch team season xG statistics from Understat.
    Returns total rows upserted.

    Key stats collected:
      xG   - expected goals scored
      xGA  - expected goals against
      npxG - non-penalty xG
      npxGA- non-penalty xGA
      ppda - passes per defensive action (pressing)
      deep - deep completions (progressive)
    """
    print("[understat] Collecting team xG stats...")
    total = 0

    for league_name in UNDERSTAT_LEAGUES:
        for season_label in SEASONS:
            season_year = UNDERSTAT_SEASON_MAP[season_label]
            print(f"[understat] {league_name} {season_label}...")

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    understat = sd.Understat(
                        leagues=[league_name],
                        seasons=[season_label],
                    )
                    df = understat.read_team_season_stats()

                if df is None or df.empty:
                    print(
                        f"[understat] WARNING: no data for "
                        f"{league_name} {season_label}"
                    )
                    continue

                print(
                    f"[understat] Got {len(df)} teams, "
                    f"columns: {list(df.columns)[:8]}..."
                )

                db = SessionLocal()
                try:
                    for idx, row in df.iterrows():
                        # Handle multi-index
                        if isinstance(idx, tuple):
                            team = idx[-1]
                        else:
                            team = str(idx)

                        attrs = {}

                        # xG stats — map column names
                        # Understat may use different column names
                        col_map = {
                            "xG": "sh_xg",
                            "xGA": "gk_ga",
                            "npxG": "sh_npxg",
                            "ppda_coef": "def_press_pct",
                            "oppda_coef": "def_press",
                            "deep": "ps_prgp",
                            "deep_allowed": "pos_prgr",
                            "scored": "sh_goals",
                            "missed": None,
                            "xpts": None,
                        }

                        for src_col, dst_attr in col_map.items():
                            if dst_attr is None:
                                continue
                            # Try exact match first
                            if src_col in row.index:
                                attrs[dst_attr] = _safe(row[src_col])
                            else:
                                # Try case-insensitive
                                matches = [
                                    c
                                    for c in row.index
                                    if c.lower() == src_col.lower()
                                ]
                                if matches:
                                    attrs[dst_attr] = _safe(row[matches[0]])

                        # Also try to get wins/draws/losses
                        for col in ["wins", "draws", "loses", "losses"]:
                            if col in row.index:
                                val = _safe(row[col])
                                if col == "wins":
                                    attrs["gk_w"] = val
                                elif col == "draws":
                                    attrs["gk_d"] = val
                                elif col in ("loses", "losses"):
                                    attrs["gk_l"] = val

                        if attrs:
                            _upsert_team(db, team, league_name, season_label, attrs)
                            total += 1

                    db.commit()
                    print(
                        f"[understat] Committed {len(df)} teams for "
                        f"{league_name} {season_label}"
                    )

                except Exception as exc:
                    db.rollback()
                    print(
                        f"[understat] ERROR commit "
                        f"{league_name} {season_label}: {exc}"
                    )
                finally:
                    db.close()

            except Exception as exc:
                print(f"[understat] ERROR {league_name} {season_label}: {exc}")
                continue

    print(f"[understat] Done. Total rows: {total}")
    return total


# ═══════════════════════════════════════════
# MATCH LEVEL xG
# ═══════════════════════════════════════════


def collect_match_xg() -> int:
    """
    Fetch match level xG from Understat.
    Stores in statsbomb_match_stats table
    (reuses same table since structure matches).
    Returns total matches processed.
    """
    print("[understat] Collecting match-level xG...")
    total = 0

    for league_name in UNDERSTAT_LEAGUES:
        for season_label in SEASONS:
            print(f"[understat] Match xG: {league_name} {season_label}...")

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    understat = sd.Understat(
                        leagues=[league_name],
                        seasons=[season_label],
                    )
                    df = understat.read_schedule()

                if df is None or df.empty:
                    continue

                # Filter to completed matches
                if "status" in df.columns:
                    df = df[df["status"] == "Finished"]

                print(
                    f"[understat] {len(df)} completed "
                    f"matches for {league_name} {season_label}"
                )

                db = SessionLocal()
                try:
                    from sqlalchemy import text

                    for idx, row in df.iterrows():
                        home_xg = _safe(
                            row.get("xG_h")
                            or row.get("home_xg")
                            or row.get("xg_home")
                        )
                        away_xg = _safe(
                            row.get("xG_a")
                            or row.get("away_xg")
                            or row.get("xg_away")
                        )

                        if home_xg is None and away_xg is None:
                            continue

                        home_team = row.get("home_team") or row.get("home") or ""
                        away_team = row.get("away_team") or row.get("away") or ""
                        date_str = str(row.get("date", ""))
                        home_goals = row.get("home_score") or row.get("score_h")
                        away_goals = row.get("away_score") or row.get("score_a")

                        # Use game index as match_id
                        # offset by 1M to avoid collision with StatsBomb
                        match_id = int(
                            str(
                                abs(
                                    hash(
                                        f"{league_name}{season_label}"
                                        f"{home_team}{away_team}{date_str}"
                                    )
                                )
                            )[:9]
                        ) + 1_000_000

                        db.execute(
                            text(
                                """
                            INSERT INTO statsbomb_match_stats (
                                match_id, competition, season,
                                date, home_team, away_team,
                                home_goals, away_goals,
                                home_xg, away_xg
                            ) VALUES (
                                :match_id, :competition, :season,
                                :date, :home_team, :away_team,
                                :home_goals, :away_goals,
                                :home_xg, :away_xg
                            )
                            ON CONFLICT (match_id) DO UPDATE SET
                                home_xg    = EXCLUDED.home_xg,
                                away_xg    = EXCLUDED.away_xg,
                                home_goals = EXCLUDED.home_goals,
                                away_goals = EXCLUDED.away_goals
                        """
                            ),
                            {
                                "match_id": match_id,
                                "competition": league_name,
                                "season": season_label,
                                "date": date_str[:10],
                                "home_team": str(home_team),
                                "away_team": str(away_team),
                                "home_goals": (
                                    int(home_goals) if home_goals else None
                                ),
                                "away_goals": (
                                    int(away_goals) if away_goals else None
                                ),
                                "home_xg": home_xg,
                                "away_xg": away_xg,
                            },
                        )
                        total += 1

                    db.commit()
                    print(
                        f"[understat] Stored match xG for "
                        f"{league_name} {season_label}"
                    )

                except Exception as exc:
                    db.rollback()
                    print(
                        f"[understat] ERROR storing match xG "
                        f"{league_name} {season_label}: {exc}"
                    )
                finally:
                    db.close()

            except Exception as exc:
                print(
                    f"[understat] ERROR reading schedule "
                    f"{league_name} {season_label}: {exc}"
                )
                continue

    print(f"[understat] Match xG done. Total: {total}")
    return total


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════


def collect_understat() -> None:
    """Run all Understat collectors."""
    print("=" * 60)
    print("UNDERSTAT xG DATA COLLECTION")
    print("=" * 60)

    print("\n--- TEAM SEASON xG STATS ---")
    team_total = collect_team_xg_stats()

    print("\n--- MATCH LEVEL xG ---")
    match_total = collect_match_xg()

    print("\n" + "=" * 60)
    print("UNDERSTAT COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Team-season rows : {team_total}")
    print(f"Match xG records : {match_total}")


if __name__ == "__main__":
    collect_understat()
