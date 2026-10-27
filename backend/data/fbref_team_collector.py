"""
FBref advanced team stats collector.

Scrapes all stat categories for every team
across 9 leagues and 2 seasons.
Stores results in the team_season_stats table.

Run from backend/ directory:
    python data/fbref_team_collector.py
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

from database import SessionLocal, TeamSeasonStats, upsert_team_season_stats

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

# MLS uses calendar year seasons
MLS_SEASONS = ["2024", "2025"]

STAT_TYPES = [
    "shooting",
    "passing",
    "goal_shot_creation",
    "defense",
    "possession",
    "misc",
    "keeper",
    "keeper_adv",
]

# ═══════════════════════════════════════════
# COLUMN MAPPINGS
# soccerdata FBref column → model attribute
# ═══════════════════════════════════════════

SHOOTING_MAP = {
    ("Performance", "Gls"): "sh_goals",
    ("Performance", "Sh"): "sh_shots",
    ("Performance", "SoT"): "sh_sot",
    ("Performance", "SoT%"): "sh_sot_pct",
    ("Performance", "Sh/90"): "sh_sh_per90",
    ("Performance", "SoT/90"): "sh_sot_per90",
    ("Performance", "G/Sh"): "sh_g_per_sh",
    ("Performance", "G/SoT"): "sh_g_per_sot",
    ("Performance", "Dist"): "sh_avg_dist",
    ("Expected", "xG"): "sh_xg",
    ("Expected", "npxG"): "sh_npxg",
    ("Expected", "npxG/Sh"): "sh_npxg_per_sh",
    ("Expected", "G-xG"): "sh_g_minus_xg",
    ("Expected", "np:G-xG"): "sh_npg_minus_npxg",
}

PASSING_MAP = {
    ("Total", "Cmp"): "ps_cmp",
    ("Total", "Att"): "ps_att",
    ("Total", "Cmp%"): "ps_cmp_pct",
    ("Total", "TotDist"): "ps_totdist",
    ("Total", "PrgDist"): "ps_prgdist",
    ("Short", "Cmp"): "ps_cmp_short",
    ("Short", "Att"): "ps_att_short",
    ("Short", "Cmp%"): "ps_cmp_pct_short",
    ("Medium", "Cmp"): "ps_cmp_medium",
    ("Medium", "Att"): "ps_att_medium",
    ("Medium", "Cmp%"): "ps_cmp_pct_medium",
    ("Long", "Cmp"): "ps_cmp_long",
    ("Long", "Att"): "ps_att_long",
    ("Long", "Cmp%"): "ps_cmp_pct_long",
    ("Ast", "Ast"): "ps_ast",
    ("Expected", "xAG"): "ps_xag",
    ("Expected", "xA"): "ps_xa",
    ("KP", "KP"): "ps_kp",
    ("1/3", "1/3"): "ps_final_third",
    ("PPA", "PPA"): "ps_ppa",
    ("CrsPA", "CrsPA"): "ps_crspa",
    ("Prog", "PrgP"): "ps_prgp",
}

GSC_MAP = {
    ("SCA", "SCA"): "gsc_sca",
    ("SCA", "SCA90"): "gsc_sca90",
    ("SCA Types", "PassLive"): "gsc_sca_passlive",
    ("SCA Types", "PassDead"): "gsc_sca_passdead",
    ("SCA Types", "TO"): "gsc_sca_to",
    ("SCA Types", "Sh"): "gsc_sca_sh",
    ("SCA Types", "Fld"): "gsc_sca_fld",
    ("SCA Types", "Def"): "gsc_sca_def",
    ("GCA", "GCA"): "gsc_gca",
    ("GCA", "GCA90"): "gsc_gca90",
    ("GCA Types", "PassLive"): "gsc_gca_passlive",
    ("GCA Types", "PassDead"): "gsc_gca_passdead",
    ("GCA Types", "TO"): "gsc_gca_to",
    ("GCA Types", "Sh"): "gsc_gca_sh",
    ("GCA Types", "Fld"): "gsc_gca_fld",
    ("GCA Types", "Def"): "gsc_gca_def",
}

DEFENSE_MAP = {
    ("Tackles", "Tkl"): "def_tkl",
    ("Tackles", "TklW"): "def_tklw",
    ("Tackles", "Def 3rd"): "def_tkl_def3rd",
    ("Tackles", "Mid 3rd"): "def_tkl_mid3rd",
    ("Tackles", "Att 3rd"): "def_tkl_att3rd",
    ("Challenges", "Tkl"): "def_drib_tkl",
    ("Challenges", "Att"): "def_drib_att",
    ("Challenges", "Tkl%"): "def_drib_tkl_pct",
    ("Pressures", "Press"): "def_press",
    ("Pressures", "Succ"): "def_press_succ",
    ("Pressures", "%"): "def_press_pct",
    ("Pressures", "Def 3rd"): "def_press_def3rd",
    ("Pressures", "Mid 3rd"): "def_press_mid3rd",
    ("Pressures", "Att 3rd"): "def_press_att3rd",
    ("Blocks", "Blocks"): "def_blocks",
    ("Blocks", "Sh"): "def_blk_sh",
    ("Blocks", "ShSv"): "def_blk_shsv",
    ("Blocks", "Pass"): "def_blk_pass",
    ("Int", "Int"): "def_int",
    ("Tkl+Int", "Tkl+Int"): "def_tkl_int",
    ("Clr", "Clr"): "def_clr",
    ("Err", "Err"): "def_err",
}

POSSESSION_MAP = {
    ("Poss", "Poss"): "pos_poss",
    ("Touches", "Touches"): "pos_touches",
    ("Touches", "Def Pen"): "pos_touch_defpen",
    ("Touches", "Def 3rd"): "pos_touch_def3rd",
    ("Touches", "Mid 3rd"): "pos_touch_mid3rd",
    ("Touches", "Att 3rd"): "pos_touch_att3rd",
    ("Touches", "Att Pen"): "pos_touch_attpen",
    ("Take-Ons", "Att"): "pos_to_att",
    ("Take-Ons", "Succ"): "pos_to_succ",
    ("Take-Ons", "Succ%"): "pos_to_succ_pct",
    ("Take-Ons", "Tkld"): "pos_to_tkld",
    ("Carries", "Carries"): "pos_carries",
    ("Carries", "TotDist"): "pos_carry_totdist",
    ("Carries", "PrgDist"): "pos_carry_prgdist",
    ("Carries", "PrgC"): "pos_prgc",
    ("Carries", "1/3"): "pos_carry_final3rd",
    ("Carries", "CPA"): "pos_cpa",
    ("Carries", "Mis"): "pos_mis",
    ("Carries", "Dis"): "pos_dis",
    ("Receiving", "Rec"): "pos_rec",
    ("Receiving", "PrgR"): "pos_prgr",
}

MISC_MAP = {
    ("Performance", "CrdY"): "misc_crdy",
    ("Performance", "CrdR"): "misc_crdr",
    ("Performance", "Fls"): "misc_fls",
    ("Performance", "Fld"): "misc_fld",
    ("Performance", "Off"): "misc_off",
    ("Performance", "Crs"): "misc_crs",
    ("Performance", "TklW"): "misc_tklw",
    ("Performance", "PKwon"): "misc_pkwon",
    ("Performance", "PKcon"): "misc_pkcon",
    ("Performance", "OG"): "misc_og",
    ("Performance", "Recov"): "misc_recov",
    ("Aerial Duels", "Won"): "misc_aerial_won",
    ("Aerial Duels", "Lost"): "misc_aerial_lost",
    ("Aerial Duels", "Won%"): "misc_aerial_pct",
}

KEEPER_MAP = {
    ("Performance", "GA"): "gk_ga",
    ("Performance", "GA90"): "gk_ga90",
    ("Performance", "SoTA"): "gk_sota",
    ("Performance", "Saves"): "gk_saves",
    ("Performance", "Save%"): "gk_save_pct",
    ("Performance", "W"): "gk_w",
    ("Performance", "D"): "gk_d",
    ("Performance", "L"): "gk_l",
    ("Performance", "CS"): "gk_cs",
    ("Performance", "CS%"): "gk_cs_pct",
    ("Penalty Kicks", "PKatt"): "gk_pka",
    ("Penalty Kicks", "PKA"): "gk_pksvd",
    ("Penalty Kicks", "Save%"): "gk_pksv_pct",
}

KEEPER_ADV_MAP = {
    ("Expected", "PSxG"): "gka_psxg",
    ("Expected", "PSxG/SoT"): "gka_psxg_per_sot",
    ("Expected", "PSxG+/-"): "gka_psxg_diff",
    ("Expected", "/90"): "gka_psxg_diff90",
    ("Launched", "Cmp"): "gka_launch_cmp",
    ("Launched", "Att"): "gka_launch_att",
    ("Launched", "Cmp%"): "gka_launch_pct",
    ("Passes", "Att"): "gka_pass_att",
    ("Passes", "Launch%"): "gka_pass_launch",
    ("Passes", "AvgLen"): "gka_pass_avglen",
    ("Goal Kicks", "Att"): "gka_gkick_att",
    ("Goal Kicks", "Launch%"): "gka_gkick_launch",
    ("Goal Kicks", "AvgLen"): "gka_gkick_avglen",
    ("Crosses", "Opp"): "gka_cross_opp",
    ("Crosses", "Stp"): "gka_cross_stp",
    ("Crosses", "Stp%"): "gka_cross_stp_pct",
    ("Sweeper", "#OPA"): "gka_opa",
    ("Sweeper", "#OPA/90"): "gka_opa90",
    ("Sweeper", "AvgDist"): "gka_avgdist",
}

ALL_MAPS = {
    "shooting": SHOOTING_MAP,
    "passing": PASSING_MAP,
    "goal_shot_creation": GSC_MAP,
    "defense": DEFENSE_MAP,
    "possession": POSSESSION_MAP,
    "misc": MISC_MAP,
    "keeper": KEEPER_MAP,
    "keeper_adv": KEEPER_ADV_MAP,
}

# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None if not possible."""
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _get_col(df: pd.DataFrame, key: tuple) -> pd.Series | None:
    """
    Safely get a column from a multi-index DataFrame.
    Returns None if column does not exist.
    """
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if key in df.columns:
                return df[key]
            # Try partial match on last level
            matches = [c for c in df.columns if c[-1] == key[-1]]
            if matches:
                return df[matches[0]]
        else:
            if key[-1] in df.columns:
                return df[key[-1]]
    except Exception:
        pass
    return None


def _read_stat(
    fbref: sd.FBref,
    stat_type: str,
    league: str,
    season: str,
) -> pd.DataFrame | None:
    """Read one stat table for one league/season."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = fbref.read_team_season_stats(stat_type=stat_type)
        if df is None or df.empty:
            return None
        # Filter to this league/season
        idx = df.index
        if hasattr(idx, "levels"):
            try:
                return df.xs((league, season), level=("league", "season"))
            except KeyError:
                return None
        return None
    except Exception as exc:
        print(f"[team] WARNING: {stat_type} {league} {season}: {exc}")
        return None


def _upsert_row(
    db,
    team: str,
    league: str,
    season: str,
    attrs: dict,
) -> None:
    """Insert or update one TeamSeasonStats row via the guarded write."""
    upsert_team_season_stats(db, team, league, season, attrs)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════


def collect_team_stats() -> None:
    """
    Scrape all FBref team stat categories for
    all configured leagues and seasons.
    Upserts into team_season_stats table.
    """
    print("[team] Starting FBref team stats collection...")
    inserted = 0
    skipped = 0

    non_mls = [l for l in LEAGUES if l != "USA-MLS"]
    mls = ["USA-MLS"]

    league_season_pairs = (
        [(l, s) for l in non_mls for s in SEASONS]
        + [(l, s) for l in mls for s in MLS_SEASONS]
    )

    for league, season in league_season_pairs:
        print(f"[team] Processing {league} {season}...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fbref = sd.FBref(
                    leagues=[league],
                    seasons=[season],
                )

            # Read all stat tables
            stat_dfs = {}
            for stat_type, col_map in ALL_MAPS.items():
                df = _read_stat(fbref, stat_type, league, season)
                stat_dfs[stat_type] = df
                shape = df.shape if df is not None else None
                print(f"[team]   {stat_type}: {shape}")

            # Get team list from shooting table
            shooting_df = stat_dfs.get("shooting")
            if shooting_df is None or shooting_df.empty:
                print(
                    f"[team] WARNING: no shooting data for "
                    f"{league} {season} — skipping"
                )
                skipped += 1
                continue

            teams = list(shooting_df.index.unique())
            print(f"[team]   Found {len(teams)} teams")

            db = SessionLocal()
            try:
                for team in teams:
                    attrs = {}
                    for stat_type, col_map in ALL_MAPS.items():
                        df = stat_dfs.get(stat_type)
                        if df is None:
                            continue
                        try:
                            team_row = df.loc[team] if team in df.index else None
                        except Exception:
                            team_row = None
                        if team_row is None:
                            continue

                        for col_key, attr in col_map.items():
                            col = _get_col(
                                team_row.to_frame().T
                                if isinstance(team_row, pd.Series)
                                else team_row,
                                col_key,
                            )
                            if col is not None:
                                val = col.iloc[0] if hasattr(col, "iloc") else col
                                attrs[attr] = _safe_float(val)

                    if attrs:
                        _upsert_row(db, team, league, season, attrs)
                        inserted += 1

                db.commit()
                print(
                    f"[team]   Committed {len(teams)} rows "
                    f"for {league} {season}"
                )
            except Exception as exc:
                db.rollback()
                print(
                    f"[team] ERROR: DB commit failed for "
                    f"{league} {season}: {exc}"
                )
            finally:
                db.close()

        except Exception as exc:
            print(f"[team] ERROR: {league} {season}: {exc}")
            skipped += 1
            continue

    print(f"[team] Done. inserted={inserted} skipped={skipped}")


if __name__ == "__main__":
    collect_team_stats()
