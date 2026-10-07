"""
Data collector for the Football Predictor backend.

Two collectors, both built on `soccerdata <https://soccerdata.readthedocs.io>`_:

``collect_match_history()``
    Pulls match results, shots, corners, cards, fouls and closing bookmaker odds
    from football-data.co.uk (via ``soccerdata.MatchHistory``) for the big five
    European leagues and stores them in the ``matches`` table.

``collect_elo_ratings()``
    Pulls the full ClubElo rating history for every team found in ``matches``
    (via ``soccerdata.ClubElo``) and stores it in the ``elo_ratings`` table.

Everything is written with SQLAlchemy Core / ORM expressions -- no raw SQL and
no ``pandas.DataFrame.to_sql()``. Writes are chunked upserts:

* ``matches``      -> ``INSERT ... ON CONFLICT (match_id) DO NOTHING`` (skip if seen)
* ``elo_ratings``  -> ``INSERT ... ON CONFLICT (team, date) DO UPDATE`` (refresh)

Run it from the ``backend/`` directory::

    python data/collector.py

Team names are read from football-data.co.uk. ClubElo does not know every one of
them, and mismatches are skipped with a warning rather than aborting the run.
If a few teams keep being skipped, add them to soccerdata's optional
``teamname_replacements.json`` config file and re-run.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# Make `database.py` importable no matter how this script is launched
# (e.g. `python data/collector.py` or `python -m data.collector`).
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# --------------------------------------------------------------------------- #
# Optional imports: report missing dependencies instead of crashing.
# --------------------------------------------------------------------------- #

_MISSING_DEPENDENCIES: list[str] = []

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on the environment
    pd = None  # type: ignore[assignment]
    _MISSING_DEPENDENCIES.append("pandas")

try:
    from soccerdata import ClubElo, MatchHistory
except ImportError:  # pragma: no cover - depends on the environment
    ClubElo = None  # type: ignore[assignment]
    MatchHistory = None  # type: ignore[assignment]
    _MISSING_DEPENDENCIES.append("soccerdata")

try:
    from sqlalchemy import func, select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from database import EloRating, Match, session_scope
except ImportError:  # pragma: no cover - depends on the environment
    _MISSING_DEPENDENCIES.append("SQLAlchemy/database")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LEAGUES: list[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

SEASONS: list[str] = [
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

# football-data.co.uk column -> matches table column. The scraped frame keeps
# the raw codes (FTHG, FTAG, ...), so they are mapped explicitly here.
MATCH_COLUMN_MAP: dict[str, str] = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_ot",
    "AST": "away_shots_ot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
}

# Odds are not available for every league/season and the available bookmaker
# columns change over time, so each price falls back through a preference list
# (market averages first, then Pinnacle closing, then Bet365/William Hill/max).
ODDS_COLUMNS: dict[str, Sequence[str]] = {
    "odds_home": ("AvgH", "BbAvH", "PSCH", "B365H", "WHH", "MaxH", "VCH", "1XBH"),
    "odds_draw": ("AvgD", "BbAvD", "PSCD", "B365D", "WHD", "MaxD", "VCD", "1XBD"),
    "odds_away": ("AvgA", "BbAvA", "PSCA", "B365A", "WHA", "MaxA", "VCA", "1XBA"),
}

VALID_RESULTS = {"H", "D", "A"}
CHUNK_SIZE = 500


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _check_dependencies() -> bool:
    """Print a clear message and return False when a dependency is missing."""
    if _MISSING_DEPENDENCIES:
        print(
            "[collector] ERROR: missing dependencies: "
            + ", ".join(_MISSING_DEPENDENCIES)
        )
        print("[collector] Install them with: pip install -r requirements.txt")
        return False
    return True


def _chunked(items: Sequence[Any], size: int = CHUNK_SIZE) -> Iterator[Sequence[Any]]:
    """Yield successive sub-sequences of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _is_missing(value: Any) -> bool:
    """True for None/NaN/NaT/empty-string values."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if pd is not None:
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False
    return False


def _column_lookup(df) -> dict[str, str]:
    """Map lowercase column names to the frame's actual column names."""
    return {str(column).strip().lower(): column for column in df.columns}


def _pick(row, lookup: dict[str, str], name: str) -> Any:
    """Return ``row[name]`` using a case-insensitive lookup, or None."""
    column = lookup.get(name.lower())
    if column is None:
        return None
    value = row.get(column) if hasattr(row, "get") else row[column]
    return None if _is_missing(value) else value


def _clean_int(value: Any) -> int | None:
    """Coerce a scraped value to int, returning None when unusable."""
    if _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clean_float(value: Any) -> float | None:
    """Coerce a scraped value to float, returning None when unusable."""
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_date(value: Any) -> date | None:
    """Coerce a scraped value to ``datetime.date``, returning None when unusable."""
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if pd is not None:
        try:
            timestamp = pd.to_datetime(value, errors="coerce")
            if not pd.isna(timestamp):
                return timestamp.date()
        except (TypeError, ValueError):
            return None
    return None


def _slug(text: Any) -> str:
    """Lowercase a value and replace spaces with underscores."""
    return str(text).strip().replace(" ", "_").lower()


def make_match_id(league: Any, match_date: Any, home_team: Any, away_team: Any) -> str:
    """
    Build the unique match identifier used by the ``matches`` table.

    Format: ``"{league}_{date}_{home_team}_{away_team}"`` with spaces replaced
    by underscores, lowercased -- e.g.
    ``"eng-premier_league_2024-08-16_man_city_chelsea"``.
    """
    day = match_date.isoformat() if hasattr(match_date, "isoformat") else str(match_date)
    return "_".join(
        [_slug(league), _slug(day), _slug(home_team), _slug(away_team)]
    )


def _derive_result(home_goals: int | None, away_goals: int | None) -> str | None:
    """Derive "H"/"D"/"A" from the score when the source did not provide it."""
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


# --------------------------------------------------------------------------- #
# Match history
# --------------------------------------------------------------------------- #


def _scrape_matches(league: str, season: str):
    """Scrape one league/season pair, returning a DataFrame or None on failure."""
    try:
        reader = MatchHistory(leagues=league, seasons=season)
        frame = reader.read_games()
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
        print(f"[matches] WARNING: could not scrape {league} {season}: {exc}")
        return None

    if frame is None or len(frame) == 0:
        print(f"[matches] WARNING: no rows returned for {league} {season}")
        return None
    return frame


def _build_match_rows(frame, league: str, season: str) -> list[dict[str, Any]]:
    """Convert a scraped frame into a list of ``matches`` row dicts."""
    flat = frame.reset_index()
    records = flat.to_dict(orient="records")
    lookup = _column_lookup(flat)
    rows: list[dict[str, Any]] = []

    for record in records:
        match_date = _clean_date(_pick(record, lookup, "date"))
        home_team = _pick(record, lookup, "home_team")
        away_team = _pick(record, lookup, "away_team")

        # Rows without a date or both teams cannot be identified -- skip them.
        if match_date is None or _is_missing(home_team) or _is_missing(away_team):
            continue

        home_team = str(home_team).strip()
        away_team = str(away_team).strip()

        row: dict[str, Any] = {
            "match_id": make_match_id(league, match_date, home_team, away_team),
            "league": league,
            "season": season,
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
        }

        # Stats: every entry defaults to None when its column is absent.
        for source_column, target_column in MATCH_COLUMN_MAP.items():
            value: Any = _pick(record, lookup, source_column)
            if target_column == "result":
                row[target_column] = (
                    str(value).strip().upper() if not _is_missing(value) else None
                )
            else:
                row[target_column] = _clean_int(value)

        if row["result"] not in VALID_RESULTS:
            row["result"] = _derive_result(row["home_goals"], row["away_goals"])

        for target_column, candidates in ODDS_COLUMNS.items():
            row[target_column] = None
            for candidate in candidates:
                odds = _clean_float(_pick(record, lookup, candidate))
                if odds is not None:
                    row[target_column] = odds
                    break

        rows.append(row)

    return rows


def _upsert_matches(rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """
    Insert match rows, skipping any ``match_id`` already in the table.

    Returns ``(inserted, skipped)`` where ``skipped`` counts both rows already
    present in the database and duplicates inside the batch.
    """
    all_rows = list(rows)
    if not all_rows:
        return 0, 0

    unique: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        unique.setdefault(row["match_id"], row)

    inserted = 0
    with session_scope() as session:
        for chunk in _chunked(list(unique.keys())):
            existing = set(
                session.execute(
                    select(Match.match_id).where(Match.match_id.in_(chunk))
                ).scalars()
            )
            fresh = [unique[match_id] for match_id in chunk if match_id not in existing]
            if fresh:
                statement = pg_insert(Match).values(fresh).on_conflict_do_nothing(
                    index_elements=["match_id"]
                )
                session.execute(statement)
                inserted += len(fresh)

    return inserted, len(all_rows) - inserted


def collect_match_history() -> dict[str, int]:
    """
    Collect every league/season combination into the ``matches`` table.

    Returns a dict with the ``inserted``, ``skipped`` and ``failed`` counters.
    """
    totals = {"inserted": 0, "skipped": 0, "failed": 0}
    if not _check_dependencies():
        return totals

    print("=" * 70)
    print("[matches] Collecting match history from football-data.co.uk")
    print(f"[matches] Leagues: {', '.join(LEAGUES)}")
    print(f"[matches] Seasons: {', '.join(SEASONS)}")
    print("=" * 70)

    for league, season in itertools.product(LEAGUES, SEASONS):
        frame = _scrape_matches(league, season)
        if frame is None:
            totals["failed"] += 1
            continue

        try:
            rows = _build_match_rows(frame, league, season)
        except Exception as exc:  # noqa: BLE001
            print(f"[matches] WARNING: could not parse {league} {season}: {exc}")
            totals["failed"] += 1
            continue

        try:
            inserted, skipped = _upsert_matches(rows)
        except Exception as exc:  # noqa: BLE001 - keep collecting other seasons
            print(f"[matches] WARNING: could not store {league} {season}: {exc}")
            totals["failed"] += 1
            continue

        totals["inserted"] += inserted
        totals["skipped"] += skipped
        print(
            f"[matches] {league} {season}: {len(rows)} scraped -> "
            f"{inserted} inserted, {skipped} skipped"
        )

    print(
        f"[matches] Done: {totals['inserted']} inserted, "
        f"{totals['skipped']} skipped, {totals['failed']} league-seasons failed"
    )
    return totals


# --------------------------------------------------------------------------- #
# Elo ratings
# --------------------------------------------------------------------------- #


def _teams_in_matches(session) -> list[str]:
    """Return every distinct team name appearing in the ``matches`` table."""
    home = select(Match.home_team)
    away = select(Match.away_team)
    teams = {
        row[0] for row in session.execute(home.union(away)) if not _is_missing(row[0])
    }
    return sorted(teams)


def _team_league_lookup(session) -> dict[str, str]:
    """Map each team name to the league it appears in most often."""
    counts: dict[str, Counter] = defaultdict(Counter)
    rows = session.execute(
        select(Match.home_team, Match.league).union(select(Match.away_team, Match.league))
    )
    for team, league in rows:
        if not _is_missing(team) and not _is_missing(league):
            counts[team][league] += 1
    return {team: counter.most_common(1)[0][0] for team, counter in counts.items()}


def _build_elo_rows(frame, team: str, league: str | None, start_date: date | None):
    """Convert a ClubElo history frame into ``elo_ratings`` row dicts."""
    if frame is None or len(frame) == 0 or "elo" not in frame.columns:
        return []

    rows: list[dict[str, Any]] = []
    for rating_date, elo in zip(frame.index, frame["elo"]):
        cleaned_date = _clean_date(rating_date)
        cleaned_elo = _clean_float(elo)
        if cleaned_date is None or cleaned_elo is None:
            continue
        if start_date is not None and cleaned_date < start_date:
            continue
        rows.append(
            {
                "team": team,
                "date": cleaned_date,
                "elo": cleaned_elo,
                "league": league,
            }
        )
    return rows


def _upsert_elo(rows: Iterable[dict[str, Any]]) -> int:
    """Insert or refresh Elo rows on the ``(team, date)`` unique key."""
    rows = list(rows)
    if not rows:
        return 0

    # A single INSERT cannot touch the same conflict key twice, so keep the last
    # value seen for each (team, date) pair.
    unique: dict[tuple[str, date], dict[str, Any]] = {}
    for row in rows:
        unique[(row["team"], row["date"])] = row
    payload = list(unique.values())

    with session_scope() as session:
        for chunk in _chunked(payload):
            statement = pg_insert(EloRating).values(list(chunk))
            statement = statement.on_conflict_do_update(
                index_elements=["team", "date"],
                set_={
                    "elo": statement.excluded.elo,
                    "league": statement.excluded.league,
                },
            )
            session.execute(statement)

    return len(payload)


def collect_elo_ratings(start_date: date | None = None) -> dict[str, int]:
    """
    Collect ClubElo history for every team in the ``matches`` table.

    ``start_date`` optionally limits the history that is stored (the full
    history is stored by default). Teams unknown to ClubElo are skipped with a
    warning instead of aborting the run.
    """
    totals = {"teams": 0, "records": 0, "skipped": 0, "failed": 0}
    if not _check_dependencies():
        return totals

    try:
        with session_scope() as session:
            teams = _teams_in_matches(session)
            leagues = _team_league_lookup(session)
    except Exception as exc:  # noqa: BLE001
        print(f"[elo] ERROR: could not read teams from the matches table: {exc}")
        return totals

    if not teams:
        print("[elo] WARNING: no teams found in the matches table -- run "
              "collect_match_history() first.")
        return totals

    print("=" * 70)
    print(f"[elo] Collecting ClubElo history for {len(teams)} teams")
    print("=" * 70)

    scraper = ClubElo()

    for index, team in enumerate(teams, start=1):
        try:
            frame = scraper.read_team_history(team)
        except ValueError:
            totals["skipped"] += 1
            print(f"[elo] [{index}/{len(teams)}] WARNING: ClubElo does not know "
                  f"'{team}' -- skipping")
            continue
        except Exception as exc:  # noqa: BLE001 - network/parsing issues
            totals["skipped"] += 1
            print(f"[elo] [{index}/{len(teams)}] WARNING: could not fetch "
                  f"'{team}': {exc} -- skipping")
            continue

        try:
            rows = _build_elo_rows(frame, team, leagues.get(team), start_date)
            stored = _upsert_elo(rows)
        except Exception as exc:  # noqa: BLE001
            totals["failed"] += 1
            print(f"[elo] [{index}/{len(teams)}] WARNING: could not store "
                  f"'{team}': {exc}")
            continue

        totals["teams"] += 1
        totals["records"] += stored
        print(f"[elo] [{index}/{len(teams)}] {team}: {stored} rows upserted")

    print(
        f"[elo] Done: {totals['teams']} teams, {totals['records']} rows upserted, "
        f"{totals['skipped']} skipped, {totals['failed']} failed"
    )
    return totals


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _print_summary() -> None:
    """Print how much data is now stored in the database."""
    print("=" * 70)
    print("[summary] Database contents")
    print("=" * 70)

    try:
        with session_scope() as session:
            match_count = session.execute(
                select(func.count()).select_from(Match)
            ).scalar_one()
            elo_count = session.execute(
                select(func.count()).select_from(EloRating)
            ).scalar_one()
            leagues = list(
                session.execute(
                    select(Match.league).distinct().order_by(Match.league)
                ).scalars()
            )
            first_date, last_date = session.execute(
                select(func.min(Match.date), func.max(Match.date))
            ).one()
    except Exception as exc:  # noqa: BLE001
        print(f"[summary] ERROR: could not read the database: {exc}")
        return

    date_range = (
        f"{first_date} to {last_date}"
        if first_date is not None and last_date is not None
        else "no matches stored"
    )
    print(f"  Total matches in DB    : {match_count}")
    print(f"  Total elo records in DB: {elo_count}")
    print(f"  Leagues covered        : {len(leagues)}")
    for league in leagues:
        print(f"      - {league}")
    print(f"  Date range covered     : {date_range}")
    print("=" * 70)


def run_collection() -> None:
    """Run the full data collection pipeline and print a final summary."""
    if not _check_dependencies():
        return

    print()
    print("#" * 70)
    print("# Football Predictor -- data collection")
    print("#" * 70)

    collect_match_history()
    collect_elo_ratings()

    _print_summary()


if __name__ == "__main__":
    from database import init_db

    init_db()
    run_collection()
