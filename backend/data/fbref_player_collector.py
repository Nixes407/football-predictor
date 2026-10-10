"""
FBref advanced player stats collector.

Scrapes player-level stats for every team
across 9 leagues and 2 seasons.
Stores results in the player_season_stats table.

Run from backend/ directory:
    python data/fbref_player_collector.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import soccerdata as sd

# Make backend/ importable
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, PlayerSeasonStats

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

LEAGUES = [
    "ENG-Premier League",
    "ENG-Championship",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "POR-Primeira Liga",
    "NED-Eredivisie",
    "USA-MLS",
]

SEASONS = ["2023-2024", "2024-2025"]
MLS_SEASONS = ["2024", "2025"]

# ═══════════════════════════════════════════
# COLUMN MAPPINGS
# ═══════════════════════════════════════════

STANDARD_MAP = {
    ("Playing Time", "MP"): "mp",
    ("Playing Time", "Starts"): "starts",
    ("Playing Time", "Min"): "minutes",
    ("Playing Time", "90s"): "minutes_per90",
    ("Performance", "Gls"): "goals",
    ("Performance", "Ast"): "assists",
    ("Performance", "G+A"): "g_plus_a",
    ("Performance", "G-PK"): "g_minus_pk",
    ("Performance", "PK"): "pk",
    ("Performance", "PKatt"): "pkatt",
    ("Performance", "CrdY"): "crdy",
    ("Performance", "CrdR"): "crdr",
    ("Expected", "xG"): "xg",
    ("Expected", "npxG"): "npxg",
    ("Expected", "xAG"): "xag",
    ("Expected", "npxG+xAG"): "npxg_plus_xag",
    ("Progression", "PrgC"): "prgc",
    ("Progression", "PrgP"): "prgp",
    ("Progression", "PrgR"): "prgr",
    ("Per 90 Minutes", "Gls"): "gls_per90",
    ("Per 90 Minutes", "Ast"): "ast_per90",
    ("Per 90 Minutes", "G+A"): "g_plus_a_per90",
    ("Per 90 Minutes", "G-PK"): "g_minus_pk_per90",
    ("Per 90 Minutes", "xG"): "xg_per90",
    ("Per 90 Minutes", "xAG"): "xag_per90",
    ("Per 90 Minutes", "xG+xAG"): "xg_plus_xag_per90",
    ("Per 90 Minutes", "npxG"): "npxg_per90",
    ("Per 90 Minutes", "npxG+xAG"): "npxg_plus_xag_per90",
}

SHOOTING_MAP = {
    ("Standard", "Sh"): "sh_shots",
    ("Standard", "SoT"): "sh_sot",
    ("Standard", "SoT%"): "sh_sot_pct",
    ("Standard", "Sh/90"): "sh_sh_per90",
    ("Standard", "SoT/90"): "sh_sot_per90",
    ("Standard", "G/Sh"): "sh_g_per_sh",
    ("Standard", "G/SoT"): "sh_g_per_sot",
    ("Standard", "Dist"): "sh_avg_dist",
    ("Expected", "xG"): "sh_xg",
    ("Expected", "npxG"): "sh_npxg",
    ("Expected", "G-xG"): "sh_g_minus_xg",
}

GSC_MAP = {
    ("SCA", "SCA"): "sca",
    ("SCA", "SCA90"): "sca90",
    ("GCA", "GCA"): "gca",
    ("GCA", "GCA90"): "gca90",
}

DEFENSE_MAP = {
    ("Tackles", "Tkl"): "tkl",
    ("Tackles", "TklW"): "tklw",
    ("Pressures", "Press"): "press",
    ("Pressures", "Succ"): "press_succ",
    ("Blocks", "Blocks"): "blocks",
    ("Int", "Int"): "int_",
    ("Tkl+Int", "Tkl+Int"): "tkl_plus_int",
    ("Clr", "Clr"): "clr",
    ("Err", "Err"): "err",
}

POSSESSION_MAP = {
    ("Touches", "Touches"): "touches",
    ("Touches", "Att 3rd"): "touch_att3rd",
    ("Touches", "Att Pen"): "touch_attpen",
    ("Carries", "Carries"): "carries",
    ("Carries", "PrgC"): "prgc_carries",
    ("Carries", "CPA"): "cpa",
    ("Receiving", "Rec"): "rec",
}

MISC_MAP = {
    ("Performance", "Fls"): "fls",
    ("Performance", "Fld"): "fld",
    ("Performance", "Off"): "off",
    ("Performance", "Crs"): "crs",
    ("Performance", "PKwon"): "pkwon",
    ("Performance", "PKcon"): "pkcon",
    ("Aerial Duels", "Won"): "aerial_won",
    ("Aerial Duels", "Lost"): "aerial_lost",
    ("Aerial Duels", "Won%"): "aerial_pct",
}

PLAYER_MAPS = {
    "standard": STANDARD_MAP,
    "shooting": SHOOTING_MAP,
    "goal_shot_creation": GSC_MAP,
    "defense": DEFENSE_MAP,
    "possession": POSSESSION_MAP,
    "misc": MISC_MAP,
}

# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    f = _safe_float(val)
    return None if f is None else int(f)


def _get_val(row, col_key: tuple):
    """Get a value from a row using a multi-index key."""
    try:
        if isinstance(row.index, pd.MultiIndex):
            if col_key in row.index:
                return row[col_key]
            matches = [c for c in row.index if c[-1] == col_key[-1]]
            if matches:
                return row[matches[0]]
        else:
            if col_key[-1] in row.index:
                return row[col_key[-1]]
    except Exception:
        pass
    return None


def _read_player_stat(
    fbref: sd.FBref,
    stat_type: str,
) -> pd.DataFrame | None:
    """Read one player stat table."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = fbref.read_player_season_stats(stat_type=stat_type)
        if df is None or df.empty:
            return None
        return df
    except Exception as exc:
        print(f"[player] WARNING: could not read {stat_type}: {exc}")
        return None


def _upsert_player(
    db,
    player: str,
    team: str,
    league: str,
    season: str,
    position: str | None,
    age: int | None,
    attrs: dict,
) -> None:
    """Insert or update one PlayerSeasonStats row."""
    from sqlalchemy import and_

    existing = (
        db.query(PlayerSeasonStats)
        .filter(
            and_(
                PlayerSeasonStats.player == player,
                PlayerSeasonStats.team == team,
                PlayerSeasonStats.league == league,
                PlayerSeasonStats.season == season,
            )
        )
        .first()
    )
    if existing is None:
        existing = PlayerSeasonStats(
            player=player,
            team=team,
            league=league,
            season=season,
        )
        db.add(existing)

    existing.position = position
    existing.age = age
    for attr, val in attrs.items():
        setattr(existing, attr, val)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════


def collect_player_stats() -> None:
    """
    Scrape all FBref player stats for all
    configured leagues and seasons.
    Upserts into player_season_stats table.
    """
    print("[player] Starting FBref player stats collection...")
    inserted = 0
    skipped = 0

    non_mls = [l for l in LEAGUES if l != "USA-MLS"]
    mls = ["USA-MLS"]

    league_season_pairs = (
        [(l, s) for l in non_mls for s in SEASONS]
        + [(l, s) for l in mls for s in MLS_SEASONS]
    )

    for league, season in league_season_pairs:
        print(f"[player] Processing {league} {season}...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fbref = sd.FBref(
                    leagues=[league],
                    seasons=[season],
                )

            # Read all player stat tables
            stat_dfs = {}
            for stat_type in PLAYER_MAPS:
                df = _read_player_stat(fbref, stat_type)
                stat_dfs[stat_type] = df
                shape = df.shape if df is not None else None
                print(f"[player]   {stat_type}: {shape}")

            # Use standard stats as the base
            standard_df = stat_dfs.get("standard")
            if standard_df is None or standard_df.empty:
                print(
                    f"[player] WARNING: no standard data for "
                    f"{league} {season} — skipping"
                )
                skipped += 1
                continue

            # Filter to this league/season
            try:
                idx = standard_df.index
                if hasattr(idx, "levels"):
                    league_season_df = standard_df.xs(
                        (league, season),
                        level=("league", "season"),
                    )
                else:
                    league_season_df = standard_df
            except KeyError:
                print(
                    f"[player] WARNING: no data for "
                    f"{league} {season} in standard table"
                )
                skipped += 1
                continue

            if league_season_df.empty:
                skipped += 1
                continue

            print(f"[player]   Found {len(league_season_df)} players")

            db = SessionLocal()
            try:
                for (team, player), row in league_season_df.iterrows():
                    attrs = {}

                    # Standard stats
                    for col_key, attr in STANDARD_MAP.items():
                        val = _get_val(row, col_key)
                        if val is not None:
                            if attr in ("mp", "starts"):
                                attrs[attr] = _safe_int(val)
                            else:
                                attrs[attr] = _safe_float(val)

                    # Other stat tables
                    for stat_type, col_map in PLAYER_MAPS.items():
                        if stat_type == "standard":
                            continue
                        df = stat_dfs.get(stat_type)
                        if df is None:
                            continue
                        try:
                            if hasattr(df.index, "levels"):
                                sub = df.xs(
                                    (league, season, team, player),
                                    level=(
                                        "league", "season",
                                        "team", "player"
                                    ),
                                )
                                player_row = (
                                    sub.iloc[0]
                                    if len(sub) > 0
                                    else None
                                )
                            else:
                                player_row = None
                        except (KeyError, TypeError):
                            player_row = None

                        if player_row is None:
                            continue

                        for col_key, attr in col_map.items():
                            val = _get_val(player_row, col_key)
                            if val is not None:
                                attrs[attr] = _safe_float(val)

                    # Get position and age
                    pos = None
                    age = None
                    try:
                        pos_col = _get_val(row, ("", "Pos"))
                        if pos_col is not None:
                            pos = str(pos_col)
                    except Exception:
                        pass
                    try:
                        age_col = _get_val(row, ("", "Age"))
                        if age_col is not None:
                            age = _safe_int(str(age_col).split("-")[0])
                    except Exception:
                        pass

                    if attrs:
                        _upsert_player(
                            db,
                            player,
                            team,
                            league,
                            season,
                            pos,
                            age,
                            attrs,
                        )
                        inserted += 1

                db.commit()
                print(
                    f"[player]   Committed players for "
                    f"{league} {season}"
                )

            except Exception as exc:
                db.rollback()
                print(
                    f"[player] ERROR: DB commit failed for "
                    f"{league} {season}: {exc}"
                )
            finally:
                db.close()

        except Exception as exc:
            print(f"[player] ERROR: {league} {season}: {exc}")
            skipped += 1
            continue

    print(f"[player] Done. inserted={inserted} skipped={skipped}")


if __name__ == "__main__":
    collect_player_stats()
