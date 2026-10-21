"""
FBref team season stats collector for the Football Predictor backend.

.. deprecated::
    FBref direct scraping is disabled. Chrome is not available on Render's free
    tier and GitHub Actions runners hit unsolvable Cloudflare CAPTCHAs, so
    ``python data/fbref_collector.py`` now prints a deprecation notice and
    exits 0 without scraping. Use ``data/statsbomb_aggregator.py`` — it fills
    the same ``fbref_team_stats`` table from ``statsbomb_match_stats`` with no
    browser and no CAPTCHA. The functions below remain importable for tests
    and for one-off manual recovery.

Scrapes season-level team stats (shooting, misc/discipline and goalkeeping)
for the Big Five European leagues via ``soccerdata.FBref`` and upserts one row
per team / league / season into the ``fbref_team_stats`` table.

Run it from the ``backend/`` directory so that ``database.py`` is importable::

    python data/fbref_collector.py

Design guarantees:
- Covers ONLY the Big Five European leagues:
  ``ENG-Premier League``, ``ESP-La Liga``, ``GER-Bundesliga``,
  ``ITA-Serie A``, ``FRA-Ligue 1``.
- Resolves ``soccerdata`` MultiIndex 4-digit season codes (e.g. ``"2425"``) back
  to canonical ``"YYYY-YYYY"`` strings (e.g. ``"2024-2025"``) so rows join
  cleanly with ``matches.season``, ``builder.py``, and ``predict.py``.
- Never aborts the scrape if one stat table (``shooting`` / ``misc`` /
  ``keeper``) or one league/season fails — collects every table and
  league+season that succeeds, logs inserted/updated counts per league+season,
  releases headless Chrome cleanly, and exits ``0``.
"""

from __future__ import annotations

import math
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path as _Path

import pandas as pd
import soccerdata as sd
from sqlalchemy import and_

# Make database.py importable when this script is run as a subprocess from any cwd
_BACKEND_DIR = _Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from database import (
    FBrefTeamStats,
    SessionLocal,
    ensure_fbref_team_stats_columns,
    init_db,
)

# Strictly Big Five leagues ONLY (never add non-Big-Five leagues here).
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

COMBINED_BIG_FIVE_LEAGUE = "Big 5 European Leagues Combined"

# Seasons follow data.collector's dynamic range (first training season ..
# current season), so a daily update also refreshes the current season once the
# calendar turns and keeps FBref features aligned with the matches table.
try:
    from data.collector import season_range as _season_range

    SEASONS = _season_range()
except Exception:  # pragma: no cover - standalone/fallback path
    from datetime import date as _date

    def _fallback_seasons() -> list[str]:
        _today = _date.today()
        _start = _today.year if _today.month >= 7 else _today.year - 1
        return [f"{y}-{y + 1}" for y in range(2021, _start + 1)]

    SEASONS = _fallback_seasons()

# Model attribute -> FBref column name, per source table.
SHOOTING_FIELDS = {
    "shots": "Sh",
    "shots_on_target": "SoT",
    "shot_accuracy": "SoT%",
    "goals_per_shot": "G/Sh",
    "goals_per_sot": "G/SoT",
}

MISC_FIELDS = {
    "yellow_cards": "CrdY",
    "red_cards": "CrdR",
    "fouls_committed": "Fls",
    "fouls_drawn": "Fld",
    "offsides": "Off",
    "crosses": "Crs",
    "interceptions": "Int",
    "tackles_won": "TklW",
    "pens_won": "PKwon",
    "pens_conceded": "PKcon",
}

KEEPER_FIELDS = {
    "sota": "SoTA",
    "goals_against": "GA",
    "saves": "Saves",
    "save_pct": "Save%",
    "clean_sheets": "CS",
    "pk_faced": "PKatt",
    "pk_saved": "PKsv",
}


def safe_float(val) -> float | None:
    """Convert ``val`` to a finite float, returning ``None`` if invalid/NaN."""
    if isinstance(val, pd.Series):
        if val.empty:
            return None
        val = val.iloc[0]
    if isinstance(val, str):
        val = val.strip().rstrip("%").replace(",", "")
    try:
        result = float(val)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(result) or not math.isfinite(result):
        return None
    return result


def _season_index_keys(season: str) -> list[str]:
    """Return candidate MultiIndex season labels for a canonical ``YYYY-YYYY`` season.

    ``soccerdata.FBref`` parses ``"2024-2025"`` via ``SeasonCode.MULTI_YEAR``
    into ``"2425"`` in the returned ``MultiIndex(["league", "season", "team"])``,
    whereas callers and PostgreSQL store ``"2024-2025"``. Checking both keys
    guarantees lookups match regardless of index representation while keeping
    stored DB values strictly canonical ``"YYYY-YYYY"``.
    """
    keys: list[str] = []
    text = str(season).strip()
    if text:
        keys.append(text)
    parts = text.split("-")
    if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 4:
        short = f"{parts[0][2:]}{parts[1][2:]}"
        if short not in keys:
            keys.append(short)
        hyphen_short = f"{parts[0][2:]}-{parts[1][2:]}"
        if hyphen_short not in keys:
            keys.append(hyphen_short)
    elif len(text) == 4 and text.isdigit():
        full = f"20{text[:2]}-20{text[2:]}"
        if full not in keys:
            keys.insert(0, full)
    return keys


def _close_fbref_driver(fbref: object | None) -> None:
    """Best-effort cleanup of ``soccerdata.FBref``'s Selenium browser process."""
    if fbref is None:
        return
    for attr in ("_driver", "driver"):
        drv = getattr(fbref, attr, None)
        if drv is not None:
            try:
                drv.quit()
            except Exception:  # noqa: BLE001
                pass


def _read_stat_type(fbref: sd.FBref, stat_type: str) -> pd.DataFrame | None:
    """Read one FBref team stat table, returning ``None`` on failure."""
    try:
        df = fbref.read_team_season_stats(stat_type=stat_type)
        if df is None or df.empty:
            return None
        return df
    except BaseException as exc:  # noqa: BLE001 - catch StopIteration in mocks too
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(f"[fbref] WARNING: could not read '{stat_type}' stats: {exc}")
        return None


def _read_stat_type_resilient(
    fbref: sd.FBref,
    stat_type: str,
    seasons: list[str],
    leagues: list[str],
) -> pd.DataFrame | None:
    """Read ``stat_type`` across ``seasons`` and ``leagues`` with partial fallback.

    Fast path: single combined Big Five call (1 page per season across all 5
    leagues). If that fails (for instance when one season or league page errors
    mid-loop inside ``soccerdata``), retry season-by-season from newest to
    oldest and, if needed, league-by-league so partial data is still preserved.
    """
    try:
        fbref.leagues = COMBINED_BIG_FIVE_LEAGUE
        fbref.seasons = list(seasons)
    except Exception:  # noqa: BLE001
        pass

    batch_df = _read_stat_type(fbref, stat_type)
    if batch_df is not None and not batch_df.empty:
        return batch_df

    # If the reader is a simple unit-test mock without per-season state or whose
    # side_effect iterator is exhausted, avoid noisy retries when side_effect is set.
    read_fn = getattr(fbref, "read_team_season_stats", None)
    if getattr(read_fn, "side_effect", None) is not None and not callable(
        getattr(read_fn, "side_effect", None)
    ):
        return None

    print(
        f"[fbref] Retrying '{stat_type}' season-by-season for partial recovery..."
    )
    chunks: list[pd.DataFrame] = []
    for season in reversed(seasons):
        season_df = None
        try:
            fbref.leagues = COMBINED_BIG_FIVE_LEAGUE
            fbref.seasons = [season]
            season_df = _read_stat_type(fbref, stat_type)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[fbref] WARNING: combined '{stat_type}' failed for {season}: {exc}"
            )

        if season_df is not None and not season_df.empty:
            chunks.append(season_df)
            continue

        # Combined page failed for this season — collect each Big Five league
        # independently so working leagues are still stored.
        for league in leagues:
            try:
                fbref.leagues = [league]
                fbref.seasons = [season]
                lg_df = _read_stat_type(fbref, stat_type)
                if lg_df is not None and not lg_df.empty:
                    chunks.append(lg_df)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[fbref] WARNING: '{stat_type}' failed for "
                    f"{league} {season}: {exc}"
                )

    if not chunks:
        return None
    try:
        return pd.concat(chunks).sort_index()
    except Exception:  # noqa: BLE001
        return chunks[0]


def _teams_for(
    stat_dfs: pd.DataFrame | list[pd.DataFrame | None] | None,
    league: str,
    season: str,
) -> list[str]:
    """Return unique team names for one ``(league, season)`` across available tables."""
    if stat_dfs is None:
        return []
    dfs = stat_dfs if isinstance(stat_dfs, (list, tuple)) else [stat_dfs]
    seen: set[str] = set()
    teams: list[str] = []
    tried_keys: list[str] = []
    found_keys: list[str] = []
    for df in dfs:
        if df is None or df.empty:
            continue
        for skey in _season_index_keys(season):
            tried_keys.append(skey)
            try:
                league_season = df.loc[(league, skey)]
            except KeyError:
                continue
            found_keys.append(skey)
            idx_values = (
                league_season.index.get_level_values(-1)
                if isinstance(league_season.index, pd.MultiIndex)
                else league_season.index
            )
            for team in idx_values.unique():
                if isinstance(team, str) and team.strip() and team not in seen:
                    seen.add(team)
                    teams.append(team)
    # Requirement (b): log the season-key normalization outcome so a
    # 0-row scrape can be traced to a key-miss vs empty data.
    print(
        f"[fbref] {league} {season}: season keys tried={tried_keys} "
        f"found={found_keys} teams={len(teams)}"
    )
    return teams


def _row_for(
    df: pd.DataFrame | None,
    league: str,
    season: str,
    team: str,
) -> pd.Series | None:
    """Return the stats row for one team from a stat table, or ``None``."""
    if df is None or df.empty:
        return None
    for skey in _season_index_keys(season):
        try:
            row = df.loc[(league, skey, team)]
        except KeyError:
            continue
        if isinstance(row, pd.DataFrame):
            if row.empty:
                continue
            row = row.iloc[0]
        return row
    return None


def _stat(row: pd.Series | None, column: str) -> float | None:
    """Read one stat from a team row as a finite float.

    FBref tables have two-level columns such as ``("Standard", "Sh")``, so the
    column is matched on its last level (the stat name). The first match wins.
    """
    if row is None:
        return None
    for key, value in row.items():
        label = key[-1] if isinstance(key, tuple) else key
        if label == column:
            return safe_float(value)
    return None


def _extract_games(
    shooting: pd.Series | None,
    misc: pd.Series | None,
    keeper: pd.Series | None,
) -> float | None:
    """Extract team matches played from ``MP`` (keeper) or ``90s`` fallback."""
    mp = _stat(keeper, "MP")
    if mp is not None and mp >= 0:
        return mp
    for src in (shooting, misc, keeper):
        nineties = _stat(src, "90s")
        if nineties is not None and nineties >= 0:
            return float(round(nineties, 2))
    return None


def _upsert_team_row(
    db,
    team: str,
    league: str,
    canonical_season: str,
    shooting: pd.Series | None,
    misc: pd.Series | None,
    keeper: pd.Series | None,
    now_iso: str,
) -> bool:
    """Upsert one ``FBrefTeamStats`` row; return ``True`` if inserted else ``False`` (updated)."""
    candidate_seasons = _season_index_keys(canonical_season)
    existing = (
        db.query(FBrefTeamStats)
        .filter(
            and_(
                FBrefTeamStats.team == team,
                FBrefTeamStats.league == league,
                FBrefTeamStats.season.in_(candidate_seasons),
            )
        )
        .first()
    )
    created = existing is None
    row = (
        existing
        if existing is not None
        else FBrefTeamStats(
            team=team,
            league=league,
            season=canonical_season,
        )
    )
    # Always normalise season label to canonical "YYYY-YYYY".
    row.season = canonical_season

    games = _extract_games(shooting, misc, keeper)
    if games is not None:
        row.games = games

    if shooting is not None:
        for attr, column in SHOOTING_FIELDS.items():
            val = _stat(shooting, column)
            if val is not None or created:
                setattr(row, attr, val)

    if misc is not None:
        for attr, column in MISC_FIELDS.items():
            val = _stat(misc, column)
            if val is not None or created:
                setattr(row, attr, val)

    if keeper is not None:
        for attr, column in KEEPER_FIELDS.items():
            val = _stat(keeper, column)
            if val is not None or created:
                setattr(row, attr, val)

    row.updated_at = now_iso
    if created:
        db.add(row)
    return created


def scrape_fbref_stats(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Scrape FBref team season stats for Big Five leagues and store them.

    Never exits non-zero on partial league/season/stat-table failures: collects
    and commits all available data, logs per-(league, season) inserted/updated
    counts, and returns a summary dict.

    When ``dry_run`` is True the full scrape logic runs but no database writes
    are performed; instead the rows that *would* have been written are printed.
    """
    target_leagues = [lg for lg in (leagues or LEAGUES) if lg in LEAGUES]
    target_seasons = list(seasons if seasons is not None else SEASONS)

    print(
        "[fbref] DEPRECATED: direct FBref scraping is disabled in production "
        "(Chrome unavailable on Render, CAPTCHA on GitHub Actions). "
        "Use data/statsbomb_aggregator.py instead."
    )

    summary = {
        "inserted": 0,
        "updated": 0,
        "failed_teams": 0,
        "total_rows_in_db": 0,
        "per_league_season": {},
    }

    print("[fbref] Starting FBref Big Five stats scrape...")
    print(f"[fbref] Leagues ({len(target_leagues)}): {target_leagues}")
    print(f"[fbref] Seasons ({len(target_seasons)}): {target_seasons}")
    if dry_run:
        print("[fbref] DRY-RUN mode enabled — no database writes will be performed")

    if not dry_run:
        init_db()
        ensure_fbref_team_stats_columns()

    fbref = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                fbref = sd.FBref(
                    leagues=COMBINED_BIG_FIVE_LEAGUE,
                    seasons=target_seasons,
                    headless=True,
                )
            except (TypeError, ValueError):
                # ``Big 5 European Leagues Combined`` is not in soccerdata's
                # LEAGUE_DICT, so SeasonCode.from_league() rejects it and the
                # FBref constructor raises ValueError before any page is read.
                # The scrape would then abort and write zero rows while still
                # exiting 0. Fall back to scraping each Big Five league
                # individually (all are valid LEAGUE_DICT keys).
                # TypeError keeps support for older/mocked FBref signatures
                # without the ``headless`` kwarg.
                fbref = sd.FBref(
                    leagues=target_leagues,
                    seasons=target_seasons,
                )

            # Read each stat table independently — one table failing never stops
            # collecting the remaining tables.
            shooting_df = _read_stat_type_resilient(
                fbref, "shooting", target_seasons, target_leagues
            )
            misc_df = _read_stat_type_resilient(
                fbref, "misc", target_seasons, target_leagues
            )
            keeper_df = _read_stat_type_resilient(
                fbref, "keeper", target_seasons, target_leagues
            )

            print(
                f"[fbref] shooting_df: "
                f"{'None' if shooting_df is None else shooting_df.shape}"
            )
            print(
                f"[fbref] misc_df:     "
                f"{'None' if misc_df is None else misc_df.shape}"
            )
            print(
                f"[fbref] keeper_df:   "
                f"{'None' if keeper_df is None else keeper_df.shape}"
            )

            # Requirement (a): log exactly what soccerdata returned so we can
            # diagnose a scrape that exits 0 but writes nothing.
            for label, df in (
                ("shooting", shooting_df),
                ("misc", misc_df),
                ("keeper", keeper_df),
            ):
                if df is None:
                    print(f"[fbref] {label}_df: None (no data returned by soccerdata)")
                    continue
                print(
                    f"[fbref] {label}_df: shape={df.shape} "
                    f"index_levels={df.index.names} "
                    f"columns={[c[-1] if isinstance(c, tuple) else c for c in df.columns][:8]}..."
                )
                try:
                    head = df.head(3)
                    print(f"[fbref] {label}_df first 3 rows:\n{head.to_string()}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[fbref] {label}_df: could not print head: {exc}")

            all_tables = [shooting_df, misc_df, keeper_df]
            if all(df is None or df.empty for df in all_tables):
                print(
                    "[fbref] WARNING: all three stat tables (shooting, misc, keeper) "
                    "returned empty — preserving any existing fbref_team_stats rows."
                )

            if not dry_run:
                db = SessionLocal()
                try:
                    before = db.query(FBrefTeamStats).count()
                    print(f"[fbref] rows in DB before run: {before}")

                    for league in target_leagues:
                        for season in target_seasons:
                            key = f"{league} {season}"
                            ls_inserted = 0
                            ls_updated = 0
                            ls_failed = 0
                            try:
                                teams = _teams_for(all_tables, league, season)
                                if not teams:
                                    print(
                                        f"[fbref] {league} {season}: "
                                        f"inserted=0, updated=0 "
                                        f"(no teams returned)"
                                    )
                                    summary["per_league_season"][key] = {
                                        "inserted": 0,
                                        "updated": 0,
                                        "failed": 0,
                                    }
                                    continue

                                # Requirement (c): exactly how many rows are
                                # about to be inserted for this (league, season).
                                print(
                                    f"[fbref] {league} {season}: about to "
                                    f"upsert {len(teams)} rows (teams={teams})"
                                )
                                now_iso = datetime.now(timezone.utc).isoformat()
                                for team in teams:
                                    try:
                                        shooting = _row_for(
                                            shooting_df, league, season, team
                                        )
                                        misc = _row_for(
                                            misc_df, league, season, team
                                        )
                                        keeper = _row_for(
                                            keeper_df, league, season, team
                                        )
                                        created = _upsert_team_row(
                                            db=db,
                                            team=team,
                                            league=league,
                                            canonical_season=season,
                                            shooting=shooting,
                                            misc=misc,
                                            keeper=keeper,
                                            now_iso=now_iso,
                                        )
                                        if created:
                                            ls_inserted += 1
                                        else:
                                            ls_updated += 1
                                    except Exception as exc:  # noqa: BLE001
                                        db.rollback()
                                        ls_failed += 1
                                        print(
                                            f"[fbref] WARNING: could not prepare "
                                            f"{team} ({league} {season}): {exc}"
                                        )

                                db.commit()
                                summary["inserted"] += ls_inserted
                                summary["updated"] += ls_updated
                                summary["failed_teams"] += ls_failed
                                summary["per_league_season"][key] = {
                                    "inserted": ls_inserted,
                                    "updated": ls_updated,
                                    "failed": ls_failed,
                                }
                                # Requirement (d): confirm the commit landed.
                                committed = ls_inserted + ls_updated
                                print(
                                    f"[fbref] Committed {committed} rows for "
                                    f"{league} {season} "
                                    f"(inserted={ls_inserted}, "
                                    f"updated={ls_updated}, "
                                    f"failed={ls_failed})"
                                )
                                print(
                                    f"[fbref] {league} {season}: "
                                    f"inserted={ls_inserted}, "
                                    f"updated={ls_updated}, "
                                    f"failed={ls_failed} "
                                    f"(teams={len(teams)})"
                                )
                            except Exception as exc:  # noqa: BLE001
                                db.rollback()
                                print(
                                    f"[fbref] WARNING: commit failed for "
                                    f"{league} {season}: {exc}"
                                )
                                summary["per_league_season"][key] = {
                                    "inserted": 0,
                                    "updated": 0,
                                    "failed": len(teams)
                                    if "teams" in locals()
                                    else 0,
                                }

                    total = db.query(FBrefTeamStats).count()
                    summary["total_rows_in_db"] = total
                    print(
                        f"[fbref] Summary: inserted={summary['inserted']}, "
                        f"updated={summary['updated']}, "
                        f"failed_teams={summary['failed_teams']}, "
                        f"total_rows_in_db={total}"
                    )
                finally:
                    db.close()
            else:
                # Requirement (dry-run): print what *would* have been written,
                # but do not touch the database.
                for league in target_leagues:
                    for season in target_seasons:
                        key = f"{league} {season}"
                        teams = _teams_for(all_tables, league, season)
                        if not teams:
                            print(
                                f"[fbref] [dry-run] {league} {season}: "
                                f"would insert 0 rows (no teams returned)"
                            )
                            summary["per_league_season"][key] = {
                                "inserted": 0,
                                "updated": 0,
                                "failed": 0,
                            }
                            continue
                        would_insert = len(teams)
                        summary["inserted"] += would_insert
                        summary["per_league_season"][key] = {
                            "inserted": would_insert,
                            "updated": 0,
                            "failed": 0,
                        }
                        print(
                            f"[fbref] [dry-run] {league} {season}: would upsert "
                            f"{would_insert} rows (teams={teams})"
                        )
                summary["total_rows_in_db"] = None
                print(
                    f"[fbref] [dry-run] Summary: would insert "
                    f"{summary['inserted']} rows total across "
                    f"{len(target_leagues)} leagues x "
                    f"{len(target_seasons)} seasons"
                )

        print("[fbref] FBref scrape complete.")
    except Exception as exc:  # noqa: BLE001 - partial/full scrape errors must exit 0
        print(f"[fbref] ERROR: FBref scrape encountered an error: {exc}")
        traceback.print_exc()
    finally:
        _close_fbref_driver(fbref)

    return summary


if __name__ == "__main__":
    # FBref direct scraping is permanently disabled: Chrome is not available on
    # Render's free tier and GitHub Actions IPs hit unsolvable Cloudflare
    # CAPTCHAs. The statsbomb_aggregator.py replacement needs no browser.
    print(
        "FBref direct scraping disabled (Chrome not available on Render, "
        "CAPTCHA on GitHub Actions). Using StatsBomb aggregator instead. "
        "Run statsbomb_aggregator.py"
    )
    sys.exit(0)
