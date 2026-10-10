"""
FBref team season stats collector for the Football Predictor backend.

Scrapes season-level team stats (shooting, misc/discipline and goalkeeping)
for the big five European leagues via ``soccerdata.FBref`` and upserts one row
per team / league / season into the ``fbref_team_stats`` table.

Run it from the ``backend/`` directory so that ``database.py`` is importable::

    python -m data.fbref_collector

A failure for one league, season or team is logged as a warning and the run
carries on. A failure of the whole scrape is printed with its traceback and
never raises out of :func:`scrape_fbref_stats`.
"""

import warnings
import traceback
from datetime import datetime, timezone

import pandas as pd
import soccerdata as sd
from database import SessionLocal, FBrefTeamStats
from sqlalchemy import and_

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]

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
    """Convert ``val`` to a float, returning ``None`` if that fails or it is NaN."""
    try:
        result = float(val)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(result):
        return None
    return result


def _read_stat_type(fbref: sd.FBref, stat_type: str) -> pd.DataFrame | None:
    """Read one FBref team stat table, or return ``None`` if it cannot be read."""
    try:
        return fbref.read_team_season_stats(stat_type=stat_type)
    except Exception as exc:  # noqa: BLE001 - one bad table must not stop the run
        print(f"[fbref] WARNING: could not read '{stat_type}' stats: {exc}")
        return None


def _teams_for(shooting_df: pd.DataFrame | None, league: str, season: str) -> list[str]:
    """Return the unique team names for one league+season, taken from the shooting table."""
    if shooting_df is None:
        return []
    try:
        league_season = shooting_df.loc[(league, season)]
    except KeyError:
        return []
    return [team for team in league_season.index.unique() if isinstance(team, str)]


def _row_for(df: pd.DataFrame | None, league: str, season: str, team: str) -> pd.Series | None:
    """Return the stats row for one team from a stat table, or ``None`` if it is missing."""
    if df is None:
        return None
    try:
        row = df.loc[(league, season, team)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):  # duplicate index entries: use the first
        if row.empty:
            return None
        row = row.iloc[0]
    return row


def _stat(row: pd.Series | None, column: str) -> float | None:
    """
    Read one stat from a team row as a float.

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


def scrape_fbref_stats() -> None:
    """Scrape FBref team season stats for every league and season and store them."""
    print("[fbref] Starting FBref stats scrape...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Step 1 - create the FBref instance.
            fbref = sd.FBref(
                leagues=LEAGUES,
                seasons=SEASONS,
            )

            # Step 2 - read the three stat tables.
            shooting_df = _read_stat_type(fbref, "shooting")
            misc_df = _read_stat_type(fbref, "misc")
            keeper_df = _read_stat_type(fbref, "keeper")

            # Step 3 - process and store.
            db = SessionLocal()
            try:
                for league in LEAGUES:
                    for season in SEASONS:
                        try:
                            teams = _teams_for(shooting_df, league, season)
                            if not teams:
                                print(f"[fbref] WARNING: no teams found for {league} {season}, skipping.")
                                continue

                            for team in teams:
                                try:
                                    shooting = _row_for(shooting_df, league, season, team)
                                    misc = _row_for(misc_df, league, season, team)
                                    keeper = _row_for(keeper_df, league, season, team)

                                    existing = (
                                        db.query(FBrefTeamStats)
                                        .filter(
                                            and_(
                                                FBrefTeamStats.team == team,
                                                FBrefTeamStats.league == league,
                                                FBrefTeamStats.season == season,
                                            )
                                        )
                                        .first()
                                    )
                                    row = existing if existing is not None else FBrefTeamStats(
                                        team=team,
                                        league=league,
                                        season=season,
                                    )

                                    for attr, column in SHOOTING_FIELDS.items():
                                        setattr(row, attr, _stat(shooting, column))
                                    for attr, column in MISC_FIELDS.items():
                                        setattr(row, attr, _stat(misc, column))
                                    for attr, column in KEEPER_FIELDS.items():
                                        setattr(row, attr, _stat(keeper, column))

                                    row.updated_at = datetime.now(timezone.utc).isoformat()

                                    db.add(row)
                                    db.commit()
                                    print(f"[fbref] Stored: {team} {league} {season}")
                                except Exception as exc:  # noqa: BLE001 - keep going
                                    db.rollback()
                                    print(f"[fbref] WARNING: could not store {team} {league} {season}: {exc}")
                        except Exception as exc:  # noqa: BLE001 - keep going
                            db.rollback()
                            print(f"[fbref] WARNING: skipping {league} {season}: {exc}")
            finally:
                db.close()

        print("[fbref] FBref scrape complete.")
    except Exception as exc:  # noqa: BLE001 - a total failure must never crash the process
        print(f"[fbref] ERROR: FBref scrape failed: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    scrape_fbref_stats()
