"""
StatsBomb -> FBref team-season stats aggregator for the Football Predictor backend.

Why this exists
---------------
``data/fbref_collector.py`` scraped FBref through headless Chrome. That path is
permanently broken: Chrome is not installable on Render's free tier and GitHub
Actions runners hit unsolvable Cloudflare CAPTCHAs. The ML model still reads its
46 ``home_fbref_*`` / ``away_fbref_*`` features (``features/feature_columns.py``)
from ``fbref_team_stats``, whose schema and column names must not change.

This script fills ``fbref_team_stats`` without a browser: it aggregates the
match-level rows already collected into ``statsbomb_match_stats`` up to one row
per (team, league, season) and upserts them under the same column names FBref
used. No new data source is introduced.

Run from the ``backend/`` directory so that ``database.py`` is importable::

    python data/statsbomb_aggregator.py            # aggregate + upsert
    python data/statsbomb_aggregator.py --dry-run  # compute only, no DB writes

Design guarantees
-----------------
- Covers ONLY the Big Five European leagues (the same scope FBref had):
  ``ENG-Premier League``, ``ESP-La Liga``, ``GER-Bundesliga``,
  ``ITA-Serie A``, ``FRA-Ligue 1``.
- Covers every season in ``data.collector.season_range()`` ("2021-2022" up to
  the current season). Rows outside that window are ignored, never deleted.
- Normalises the raw ``competition`` / ``season`` labels both writers leave in
  ``statsbomb_match_stats`` (StatsBomb open-data stores the full label such as
  ``"ESP-La Liga 2021-2022"`` in both columns; Understat stores clean
  ``"ENG-Premier League"`` / ``"2024-2025"`` values) back to the canonical
  league names and ``"YYYY-YYYY"`` season labels the model joins on.
- Column mapping is dynamic: each fbref field takes its value from the first
  candidate ``statsbomb_match_stats`` column that actually exists in the live
  table (see ``TEAM_FIELD_SOURCES``), so a source table that gains columns
  (fouls, cards, crosses, ...) is picked up without code changes. A field with
  no source column is written as 0 on insert — the same value the model has
  always seen for it — and existing values are left untouched on update.
- Never exits non-zero: partial data is fine. Every (league, season) group
  commits on its own; a failure there is logged and the run continues.

Column mapping (team-side; the away side mirrors ``home_*`` -> ``away_*``)::

    fbref_team_stats   source in statsbomb_match_stats
    -----------------  -------------------------------------------------
    games              count of matches the team played
    shots              home_shots
    shots_on_target    home_shots_ot (fallbacks: home_shots_on_target,
                       home_sot)
    shot_accuracy      computed: shots_on_target / shots * 100 (FBref SoT%)
    goals_per_shot     computed: goals_for / shots (FBref G/Sh)
    goals_per_sot      computed: goals_for / shots_on_target (FBref G/SoT)
    yellow_cards       home_yellow_cards (fallbacks: home_cards_yellow,
                       home_yellow, home_crdy) — none exist today -> 0
    red_cards          home_red_cards (fallbacks: home_cards_red, home_red,
                       home_crdr) — none exist today -> 0
    fouls_committed    home_fouls_committed (fallbacks: home_fouls, home_fl)
                       — none exist today -> 0
    fouls_drawn        home_fouls_drawn (fallback: home_fld) -> 0
    offsides           home_offsides (fallback: home_off) -> 0
    crosses            home_crosses (fallbacks: home_crs, home_cross) -> 0
    interceptions      home_interceptions (fallback: home_int) -> 0
    tackles_won        home_tackles_won (fallback: home_tklw) -> 0
    pens_won           home_pens_won (fallbacks: home_pk_won,
                       home_penalties_won) -> 0
    pens_conceded      home_pens_conceded (fallbacks: home_pk_conceded,
                       home_penalties_conceded) -> 0
    sota               opponent shots_on_target faced (from the opponent's
                       shots_ot column)
    goals_against      opponent goals scored
    saves              home_gk_saves (fallback: home_saves)
    save_pct           computed: saves / sota * 100 (FBref Save%)
    clean_sheets       count of matches with 0 goals against
    pk_faced           home_pk_faced (fallbacks: home_pens_faced, home_pk_att,
                       then the pens_conceded source) -> 0
    pk_saved           home_pk_saved (fallbacks: home_pens_saved,
                       home_penalty_saves) -> 0

``statsbomb_match_stats`` also carries xG, pressures and progressive carries;
those have no FBref counterpart and are intentionally unused here.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy import inspect as sa_inspect, text

# Make database.py importable when this script is run as a subprocess from any cwd
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from database import (  # noqa: E402
    FBrefTeamStats,
    SessionLocal,
    ensure_fbref_team_stats_columns,
    init_db,
)

# Seasons follow data.collector's dynamic range (first training season .. the
# current one) so the aggregator and the match collector never drift apart.
try:
    from data.collector import SEASONS  # noqa: F401
except Exception:  # pragma: no cover - standalone/fallback path
    from datetime import date as _date

    def _fallback_seasons() -> list[str]:
        _today = _date.today()
        _start = _today.year if _today.month >= 7 else _today.year - 1
        return [f"{y}-{y + 1}" for y in range(2021, _start + 1)]

    SEASONS = _fallback_seasons()

SOURCE_TABLE = "statsbomb_match_stats"

# Strictly Big Five leagues ONLY (same scope the FBref scrape had). Never add
# non-Big-Five leagues here.
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# fbref_team_stats field -> ordered candidate HOME-side source columns in
# statsbomb_match_stats. The first candidate that exists in the live table
# wins. The away side mirrors each candidate (home_* -> away_*).
TEAM_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "shots": ("home_shots",),
    "shots_on_target": ("home_shots_ot", "home_shots_on_target", "home_sot"),
    "yellow_cards": ("home_yellow_cards", "home_cards_yellow", "home_yellow", "home_crdy"),
    "red_cards": ("home_red_cards", "home_cards_red", "home_red", "home_crdr"),
    "fouls_committed": ("home_fouls_committed", "home_fouls", "home_fl"),
    "fouls_drawn": ("home_fouls_drawn", "home_fld"),
    "offsides": ("home_offsides", "home_off"),
    "crosses": ("home_crosses", "home_crs", "home_cross"),
    "interceptions": ("home_interceptions", "home_int"),
    "tackles_won": ("home_tackles_won", "home_tklw"),
    "pens_won": ("home_pens_won", "home_pk_won", "home_penalties_won"),
    "pens_conceded": (
        "home_pens_conceded",
        "home_pk_conceded",
        "home_penalties_conceded",
    ),
    "saves": ("home_gk_saves", "home_saves"),
    "pk_faced": (
        "home_pk_faced",
        "home_pens_faced",
        "home_pk_att",
        "home_pens_conceded",
        "home_pk_conceded",
    ),
    "pk_saved": ("home_pk_saved", "home_pens_saved", "home_penalty_saves"),
}

# fbref_team_stats field -> candidate HOME-side columns whose value is an
# OPPONENT stat (the team faces the opponent's shots on target).
OPPONENT_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "sota": ("home_shots_ot", "home_shots_on_target", "home_sot"),
}

# Counting stats summed over the season. Everything else the model reads is
# either computed below (rates, clean sheets) or has no source (stays 0).
SUM_FIELDS: tuple[str, ...] = tuple(TEAM_FIELD_SOURCES) + tuple(OPPONENT_FIELD_SOURCES)

#: FBref rate conventions: accuracy-style stats are percentages 0-100,
#: goals-per-shot style stats are decimals.
RATE_FIELDS: dict[str, tuple[str, ...]] = {
    # fbref field: (numerator, denominator, scale)
    "shot_accuracy": ("shots_on_target", "shots", 100.0),
    "goals_per_shot": ("goals_for", "shots", 1.0),
    "goals_per_sot": ("goals_for", "shots_on_target", 1.0),
    "save_pct": ("saves", "sota", 100.0),
}

#: Fields with no candidate source column in today's statsbomb_match_stats
#: schema. Computed at runtime from the live table's columns; kept as a
#: constant for reporting ("which FBref columns had no StatsBomb equivalent").
NO_SOURCE_NOTE_FIELDS: tuple[str, ...] = (
    "yellow_cards",
    "red_cards",
    "fouls_committed",
    "fouls_drawn",
    "offsides",
    "crosses",
    "interceptions",
    "tackles_won",
    "pens_won",
    "pens_conceded",
    "pk_faced",
    "pk_saved",
)

_SEASON_RANGE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})(?!\d)")
_SEASON_CODE_RE = re.compile(r"(\d{2})(\d{2})")


# ═══════════════════════════════════════════
# LABEL NORMALISATION
# ═══════════════════════════════════════════


def normalise_league(competition: Any) -> str | None:
    """
    Map a raw ``statsbomb_match_stats.competition`` label onto one of the Big
    Five canonical league names, or ``None`` when the row is out of scope.

    Both writers leave different shapes: ``"ENG-Premier League"`` (Understat)
    and ``"ESP-La Liga 2021-2022"`` (StatsBomb open-data labels). Non-Big-Five
    competitions (``"UEFA Champions League ..."``, ``"FIFA World Cup ..."``,
    ``"Women's Super League ..."``, ``"ENG-Championship"``, ``"AUT-Bundesliga"``,
    ...) are rejected so BSA/Belgian/other leagues can never leak in.
    """
    label = str(competition or "").strip()
    if not label:
        return None
    for league in LEAGUES:
        if label == league or label.startswith(league + " "):
            return league
    return None


def normalise_season(season_label: Any, date_str: Any = None) -> str | None:
    """
    Normalise a raw ``season`` label to canonical ``"YYYY-YYYY"``.

    Accepts ``"2024-2025"``, full labels like ``"ESP-La Liga 2021-2022"``,
    soccerdata 4-digit codes like ``"2425"``, and falls back to deriving the
    season from the match date (seasons start in July). Returns ``None`` when
    nothing can be resolved.
    """
    text = str(season_label or "").strip()
    match = _SEASON_RANGE_RE.search(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    if re.fullmatch(r"\d{4}", text):
        code = _SEASON_CODE_RE.fullmatch(text)
        if code and int(code.group(2)) == int(code.group(1)) + 1:
            return f"20{code.group(1)}-20{code.group(2)}"
        if text.startswith(("19", "20")):
            # Bare year: treat as the season that STARTED in that year's
            # summer only when the date fallback is unavailable.
            pass
    ts = pd.to_datetime(date_str, errors="coerce")
    if ts is not None and not pd.isna(ts):
        year = int(ts.year)
        start = year if ts.month >= 7 else year - 1
        return f"{start}-{start + 1}"
    if re.fullmatch(r"(19|20)\d{2}", text):
        start = int(text)
        return f"{start}-{start + 1}"
    return None


# ═══════════════════════════════════════════
# SOURCE LOADING
# ═══════════════════════════════════════════


def load_match_rows(db) -> pd.DataFrame:
    """
    Load every ``statsbomb_match_stats`` row into a DataFrame.

    Returns an empty frame when the table does not exist yet (nothing to
    aggregate is not an error). A dedicated session is used by the caller.
    """
    try:
        inspector = sa_inspect(db.get_bind())
        if SOURCE_TABLE not in inspector.get_table_names():
            print(
                f"[sb-agg] WARNING: table {SOURCE_TABLE} does not exist — "
                "nothing to aggregate."
            )
            return pd.DataFrame()
        result = db.execute(text(f"SELECT * FROM {SOURCE_TABLE}"))
        rows = result.fetchall()
        columns = list(result.keys())
    except Exception as exc:  # noqa: BLE001 - never abort the run
        print(f"[sb-agg] WARNING: could not read {SOURCE_TABLE}: {exc}")
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=columns)
    print(f"[sb-agg] Loaded {len(frame)} match rows from {SOURCE_TABLE}")
    return frame


def _to_away_column(home_column: str) -> str:
    return home_column.replace("home_", "away_", 1)


def _resolve_sources(
    field_sources: dict[str, tuple[str, ...]],
    available: set[str],
) -> dict[str, tuple[str | None, str | None]]:
    """Pick the (home, away) source column pair for each field, or (None, None)."""
    resolved: dict[str, tuple[str | None, str | None]] = {}
    for field, candidates in field_sources.items():
        home_col = next((c for c in candidates if c in available), None)
        away_col = _to_away_column(home_col) if home_col else None
        if away_col not in available:
            away_col = None
            home_col = None
        resolved[field] = (home_col, away_col)
    return resolved


# ═══════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════


def build_team_match_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Reshape match rows into one row per team appearance.

    Returns ``(team_match_rows, resolved_fields)`` where ``resolved_fields``
    names the fbref fields that had a usable source column in the input table.
    """
    if df.empty:
        return pd.DataFrame(), set()

    available = set(df.columns)
    team_sources = _resolve_sources(TEAM_FIELD_SOURCES, available)
    opp_sources = _resolve_sources(OPPONENT_FIELD_SOURCES, available)

    resolved_fields: set[str] = set()
    for field, (home_col, away_col) in team_sources.items():
        if home_col is not None:
            resolved_fields.add(field)
    for field, (home_col, away_col) in opp_sources.items():
        if home_col is not None:
            resolved_fields.add(field)

    league = df.get("competition", pd.Series(index=df.index, dtype=object)).map(
        normalise_league
    )
    season = [
        normalise_season(s, d)
        for s, d in zip(
            df.get("season", pd.Series(index=df.index, dtype=object)),
            df.get("date", pd.Series(index=df.index, dtype=object)),
        )
    ]
    season = pd.Series(season, index=df.index)

    games_mask = (
        league.notna()
        & season.notna()
        & season.isin(set(SEASONS))
        & df.get(
            "home_team", pd.Series(index=df.index, dtype=object)
        ).notna()
    )
    if not bool(games_mask.any()):
        return pd.DataFrame(), resolved_fields

    frames: list[pd.DataFrame] = []
    for side, other in (("home", "away"), ("away", "home")):
        team_col = f"{side}_team"
        if team_col not in available:
            continue
        block = pd.DataFrame(index=df.index)
        block["team"] = df[team_col]
        block["league"] = league
        block["season"] = season
        block["games"] = 1.0
        for field, (home_col, away_col) in team_sources.items():
            source = home_col if side == "home" else away_col
            block[field] = (
                pd.to_numeric(df[source], errors="coerce")
                if source is not None
                else float("nan")
            )
        for field, (home_col, away_col) in opp_sources.items():
            # Opponent stats: the home team FACES the away side's values and
            # vice versa (e.g. home team's sota == away_shots_ot).
            source = away_col if side == "home" else home_col
            block[field] = (
                pd.to_numeric(df[source], errors="coerce")
                if source is not None
                else float("nan")
            )
        goals_for = (
            pd.to_numeric(df[f"{side}_goals"], errors="coerce")
            if f"{side}_goals" in available
            else pd.Series(float("nan"), index=df.index)
        )
        goals_against = (
            pd.to_numeric(df[f"{other}_goals"], errors="coerce")
            if f"{other}_goals" in available
            else pd.Series(float("nan"), index=df.index)
        )
        block["goals_for"] = goals_for
        block["clean_sheets"] = (
            (goals_against == 0).astype(float).where(goals_against.notna())
        )
        block["goals_against"] = goals_against
        frames.append(block.loc[games_mask].copy())

    if not frames:
        return pd.DataFrame(), resolved_fields

    team_rows = pd.concat(frames, ignore_index=True)
    return team_rows, resolved_fields


def aggregate_team_seasons(
    team_rows: pd.DataFrame,
    resolved_fields: Iterable[str],
) -> pd.DataFrame:
    """
    Sum team-appearance rows to one row per (team, league, season).

    Counting stats become season totals; ratio stats follow FBref's
    conventions (percentages 0-100 for accuracy/save rate, decimals for
    goals-per-shot). Fields without a source column are filled with 0.0 so the
    model always gets a complete row.
    """
    resolved = set(resolved_fields)
    if team_rows.empty:
        return pd.DataFrame()

    value_cols = [
        "games",
        "goals_for",
        "goals_against",
        "clean_sheets",
        *[f for f in SUM_FIELDS if f in resolved],
    ]
    grouped = team_rows.groupby(["team", "league", "season"], sort=True)
    totals = grouped[value_cols].sum(min_count=1)
    totals["games"] = grouped["games"].sum().astype(int)

    for field in SUM_FIELDS:
        if field not in resolved:
            totals[field] = 0.0
    for field in ("goals_for", "goals_against", "clean_sheets"):
        totals[field] = totals[field].fillna(0.0)

    for field, (numerator, denominator, scale) in RATE_FIELDS.items():
        numerator_ok = numerator in resolved or numerator == "goals_for"
        if not numerator_ok or denominator not in resolved:
            totals[field] = 0.0
            continue
        num = totals[numerator]
        den = totals[denominator]
        ratio = (num / den.where(den != 0)) * scale
        totals[field] = ratio.fillna(0.0).astype(float)

    totals = totals.fillna(0.0)
    return totals.reset_index()


def writable_fields(resolved_fields: Iterable[str]) -> set[str]:
    """
    fbref fields that may be overwritten on update.

    Counting stats with a real source column and the fields derived purely
    from them (or from goals) are the aggregator's source of truth. A field
    with no source column is only written on INSERT (0.0) so existing values
    are never clobbered.
    """
    resolved = set(resolved_fields)
    writable = set(resolved)
    writable.update({"games", "goals_against", "clean_sheets"})
    for field, (numerator, denominator, _scale) in RATE_FIELDS.items():
        numerator_ok = numerator in resolved or numerator == "goals_for"
        if numerator_ok and denominator in resolved:
            writable.add(field)
    return writable


# ═══════════════════════════════════════════
# DATABASE UPSERT
# ═══════════════════════════════════════════


def _upsert_team_row(
    db,
    record: Any,
    writable: set[str],
    now_iso: str,
) -> bool:
    """
    Upsert one ``FBrefTeamStats`` row from an aggregate record.

    Returns ``True`` when a new row was inserted, ``False`` when an existing
    row was updated. Fields outside ``writable`` (no source column in the
    statsbomb table) are only written on insert — never clobber values a
    legacy FBref scrape may have stored.
    """
    existing = (
        db.query(FBrefTeamStats)
        .filter(
            FBrefTeamStats.team == record.team,
            FBrefTeamStats.league == record.league,
            FBrefTeamStats.season == record.season,
        )
        .first()
    )
    created = existing is None
    row = existing or FBrefTeamStats(team=record.team, league=record.league, season=record.season)
    row.season = str(record.season)

    row.games = int(record.games)

    for field in ("shots", "shots_on_target", "shot_accuracy", "goals_per_shot",
                  "goals_per_sot", "yellow_cards", "red_cards", "fouls_committed",
                  "fouls_drawn", "offsides", "crosses", "interceptions",
                  "tackles_won", "pens_won", "pens_conceded", "sota",
                  "goals_against", "saves", "save_pct", "clean_sheets",
                  "pk_faced", "pk_saved"):
        has_source = field in writable
        value = getattr(record, field, 0.0)
        value = float(value) if value is not None and value == value else 0.0
        if has_source or created:
            setattr(row, field, value)

    row.updated_at = now_iso
    if created:
        db.add(row)
    return created


def run_aggregation(dry_run: bool = False) -> dict:
    """
    Aggregate ``statsbomb_match_stats`` into ``fbref_team_stats``.

    Never raises: every (league, season) group commits independently and the
    summary reports whatever succeeded. Returns a summary dict.
    """
    summary: dict[str, Any] = {
        "inserted": 0,
        "updated": 0,
        "failed_groups": 0,
        "source_rows": 0,
        "in_scope_teams": 0,
        "resolved_fields": [],
        "no_source_fields": list(NO_SOURCE_NOTE_FIELDS),
        "per_league_season": {},
    }

    print("[sb-agg] Starting StatsBomb -> fbref_team_stats aggregation...")
    print(f"[sb-agg] Leagues ({len(LEAGUES)}): {LEAGUES}")
    print(f"[sb-agg] Seasons ({len(SEASONS)}): {SEASONS}")
    if dry_run:
        print("[sb-agg] DRY-RUN mode enabled — no database writes will be performed")

    try:
        init_db()
        ensure_fbref_team_stats_columns()
    except Exception as exc:  # noqa: BLE001 - keep going; upserts report failures
        print(f"[sb-agg] WARNING: could not initialise schema: {exc}")

    db = None
    try:
        db = SessionLocal()
        source = load_match_rows(db)
        summary["source_rows"] = int(len(source))
        team_rows, resolved_fields = build_team_match_rows(source)
        writable = writable_fields(resolved_fields)
        summary["resolved_fields"] = sorted(resolved_fields)
        all_fbref_fields = (
            set(SUM_FIELDS) | set(RATE_FIELDS) | {"clean_sheets", "games"}
        )
        summary["no_source_fields"] = sorted(all_fbref_fields - writable)
        aggregates = aggregate_team_seasons(team_rows, resolved_fields)
        summary["in_scope_teams"] = int(len(aggregates))

        if aggregates.empty:
            print(
                "[sb-agg] No in-scope (Big Five, season in range) rows found — "
                "fbref_team_stats left unchanged."
            )
            return summary

        print(
            f"[sb-agg] Aggregated {len(aggregates)} team-seasons; "
            f"fields with a StatsBomb source: {sorted(resolved_fields) or 'none'}"
        )
        print(f"[sb-agg] FBref fields with no source (stay 0): {summary['no_source_fields']}")

        now_iso = datetime.now(timezone.utc).isoformat()
        for (league, season), group in aggregates.groupby(["league", "season"], sort=True):
            key = f"{league} {season}"
            inserted = updated = 0
            if dry_run:
                sample = group.iloc[0]
                print(
                    f"[sb-agg] [dry-run] {key}: would upsert {len(group)} rows "
                    f"(e.g. {sample.team}: games={int(sample.games)} "
                    f"shots={sample.shots:.0f} "
                    f"shots_pg={sample.shots / max(int(sample.games), 1):.1f})"
                )
                summary["per_league_season"][key] = {
                    "teams": int(len(group)),
                    "inserted": 0,
                    "updated": 0,
                }
                summary["inserted"] += int(len(group))
                continue

            try:
                for record in group.itertuples(index=False):
                    if _upsert_team_row(db, record, writable, now_iso):
                        inserted += 1
                    else:
                        updated += 1
                db.commit()
            except Exception as exc:  # noqa: BLE001 - one group must not stop the rest
                db.rollback()
                summary["failed_groups"] += 1
                summary["per_league_season"][key] = {"error": str(exc)[:300]}
                print(f"[sb-agg] ERROR {key}: {exc}")
                traceback.print_exc()
                continue

            summary["inserted"] += inserted
            summary["updated"] += updated
            summary["per_league_season"][key] = {
                "teams": int(len(group)),
                "inserted": inserted,
                "updated": updated,
            }
            first = group.iloc[0]
            games = max(int(first.games), 1)
            print(
                f"[sb-agg] {key}: teams={len(group)} "
                f"inserted={inserted} updated={updated} | "
                f"e.g. {first.team}: games={int(first.games)} "
                f"shots={first.shots:.0f} (per-game {first.shots / games:.1f}) "
                f"sot={first.shots_on_target:.0f} saves={first.saves:.0f} "
                f"ga={first.goals_against:.0f} cs={first.clean_sheets:.0f}"
            )
    except Exception as exc:  # noqa: BLE001 - never abort the run
        summary["failed_groups"] += 1
        print(f"[sb-agg] ERROR aggregating {SOURCE_TABLE}: {exc}")
        traceback.print_exc()
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass

    total = summary["inserted"] + summary["updated"]
    print(
        f"[sb-agg] Summary: source_rows={summary['source_rows']} "
        f"team_seasons={summary['in_scope_teams']} "
        f"inserted={summary['inserted']} updated={summary['updated']} "
        f"failed_groups={summary['failed_groups']}"
    )
    print(f"[sb-agg] Aggregation complete ({total} rows written).")
    return summary


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Always returns 0 (partial data is fine)."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate statsbomb_match_stats into fbref_team_stats "
            "(Big Five leagues, season totals per team)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute aggregates and print them without writing to the database.",
    )
    args = parser.parse_args(argv)
    try:
        run_aggregation(dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - the script must exit 0 always
        print(f"[sb-agg] ERROR: aggregation encountered an error: {exc}")
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
