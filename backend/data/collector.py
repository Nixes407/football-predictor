"""
Data collector for the Football Predictor backend.

``collect_match_history()``
    Pulls match results, shots, corners, cards, fouls and closing bookmaker odds
    from football-data.co.uk (via ``soccerdata.MatchHistory``) for the Big Five
    plus the ENG-Championship, POR-Primeira Liga, NED-Eredivisie and
    TUR-Süper Lig division files, and for NOR-Eliteserien / AUT-Bundesliga from
    football-data.co.uk's all-time "extra leagues" CSVs (goals and odds only).
    All of it is stored in the ``matches`` table.

Elo ratings
    ``elo_ratings`` is filled by :mod:`data.elo`, which replays the stored
    ``matches`` through the standard Elo recursion (K=32, start 1500). That
    runs automatically at the end of ``run_collection()``, so collecting
    matches always leaves the ratings table populated — it is a pure function
    of results already in the database, with no extra network dependency.

    ``collect_elo_ratings()`` (below) is the older ClubElo scraper. It is kept
    as an *optional* enrichment behind ``COLLECT_CLUBELO=1``: clubelo.com
    resolves only part of this project's team set under its own spellings, and
    it used to swallow every miss into a ``skipped`` counter, so a run that
    stored zero ratings exited 0 and looked successful. See ``data/elo.py``.

Everything is written with SQLAlchemy Core / ORM expressions -- no raw SQL and
no ``pandas.DataFrame.to_sql()``. Writes are chunked upserts:

* ``matches``      -> ``INSERT ... ON CONFLICT (match_id) DO NOTHING`` (skip if seen)
* ``elo_ratings``  -> ``INSERT ... ON CONFLICT (team, date) DO UPDATE`` (refresh)

Run it from the ``backend/`` directory::

    python data/collector.py
    COLLECT_CLUBELO=1 python data/collector.py   # also scrape clubelo.com

Team names are read from football-data.co.uk. ClubElo does not know every one of
them, and mismatches are skipped with a warning rather than aborting the run.
If a few teams keep being skipped, add them to soccerdata's optional
``teamname_replacements.json`` config file and re-run.
"""

from __future__ import annotations

import itertools
import math
import os
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
    import io

    import requests
except ImportError:  # pragma: no cover - depends on the environment
    requests = None  # type: ignore[assignment]
    _MISSING_DEPENDENCIES.append("requests")

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
    "ENG-Championship",
    "POR-Primeira Liga",
    "NED-Eredivisie",
    "TUR-Süper Lig",
    "NOR-Eliteserien",
    "AUT-Bundesliga",
]

# ---------------------------------------------------------------------------
# League support beyond soccerdata's built-in list
# ---------------------------------------------------------------------------
#
# BUG FIX (league name mismatch / silent collection failure):
# soccerdata's ``MatchHistory`` only accepts the leagues present in
# ``soccerdata._config.LEAGUE_DICT`` (currently the Big Five + a few
# international tournaments). Constructing it with e.g.
# leagues="ENG-Championship" raises ``ValueError: Invalid league`` and — in
# this collector — every one of those failures was previously swallowed into
# the per-league-season "failed" counter, so simply adding a league to
# ``LEAGUES`` below would ingest ZERO rows for it.
#
# The leagues football-data.co.uk still publishes as regular per-season
# division files (``mmz4281/{season}/{code}.csv``) are registered into the
# soccerdata config at import time, which makes ``MatchHistory`` accept them.
EXTRA_MATCH_HISTORY_CODES: dict[str, str] = {
    "ENG-Championship": "E1",     # England Division 1 / Championship
    "POR-Primeira Liga": "P1",    # Liga Portugal
    "NED-Eredivisie": "N1",       # Eredivisie
    "TUR-Süper Lig": "T1",        # Süper Lig
}

# Leagues football-data.co.uk only publishes as ONE all-time CSV under
# ``https://www.football-data.co.uk/new/{code}.csv`` (columns: Country,
# League, Season, Date, Time, Home, Away, HG, AG, Res, ...). These carry
# goals and odds only — no shots, corners or cards — so those stat columns
# are stored as NULL and the feature builder falls back to its zero
# defaults. MatchHistory cannot read these files at all, hence the
# dedicated downloader below. Note: unknown codes under /new/ silently
# REDIRECT to another file, so every download is guarded by a Country check.
EXTRA_CSV_LEAGUES: dict[str, dict[str, Any]] = {
    "NOR-Eliteserien": {"code": "NOR", "country": "Norway", "calendar_year": True},
    "AUT-Bundesliga": {"code": "AUT", "country": "Austria", "calendar_year": False},
}

EXTRA_CSV_BASE_URL = "https://www.football-data.co.uk/new/{code}.csv"
EXTRA_CSV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}

#: Process-local cache: all-time CSV code -> downloaded frame (None = failed).
_EXTRA_CSV_CACHE: dict[str, Any] = {}

#: First season of the training window. The list itself is generated
#: dynamically up to the CURRENT season (see ``current_season``), so a
#: collector run in e.g. October 2026 also ingests 2025-2026 (finished)
#: and 2026-2027 (in progress) instead of silently stopping at a
#: hard-coded year. football-data.co.uk files for an in-progress season
#: grow daily, and soccerdata re-downloads incomplete seasons
#: (``_is_complete``) rather than serving stale cache.
FIRST_TRAINING_SEASON: int = 2021


def current_season(today: date | None = None) -> str:
    """
    Return the "YYYY-YYYY" label of the European season that ``today`` falls
    in: the season starting in July of the current year when the month is
    July or later, otherwise the season that started last July.
    """
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{start + 1}"


def season_range(
    first_year: int = FIRST_TRAINING_SEASON,
    today: date | None = None,
) -> list[str]:
    """
    All season labels from ``first_year`` up to and including the current one.

    "2021-2022" ... current_season(today). Built from one shared rule so the
    collector, the FBref collectors and the ops scripts never drift apart.
    """
    last_year = int(current_season(today)[:4])
    return [f"{y}-{y + 1}" for y in range(first_year, last_year + 1)]


SEASONS: list[str] = season_range()

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


def _register_extra_leagues_with_soccerdata() -> None:
    """
    Add the leagues from :data:`EXTRA_MATCH_HISTORY_CODES` to soccerdata's
    league config so ``MatchHistory`` accepts them.

    ``soccerdata._common`` imports the ``LEAGUE_DICT`` object once and looks
    leagues up in that same dict, so mutating it in place here is enough —
    no monkey-patching of class internals is required.
    """
    if MatchHistory is None or pd is None:  # pragma: no cover - env-dependent
        return
    try:
        from soccerdata import _config as soccerdata_config
    except Exception as exc:  # noqa: BLE001
        print(f"[matches] WARNING: cannot import soccerdata config: {exc}")
        return
    league_dict = getattr(soccerdata_config, "LEAGUE_DICT", None)
    if not isinstance(league_dict, dict):
        print("[matches] WARNING: soccerdata LEAGUE_DICT is not a plain dict; "
              "extra leagues may be rejected by MatchHistory")
        return
    for name, code in EXTRA_MATCH_HISTORY_CODES.items():
        existing = league_dict.get(name)
        if isinstance(existing, dict) and existing.get("MatchHistory"):
            continue
        league_dict[name] = {
            "MatchHistory": code,
            "season_start": "Aug",
            "season_end": "May",
        }
        print(f"[matches] Registered '{name}' (football-data.co.uk code "
              f"{code!r}) with soccerdata's league config")


def _season_start_year(season_label: object) -> int | None:
    """Return the starting year of a season label ("2021-2022" -> 2021)."""
    text = str(season_label).strip()
    head = text.split("-")[0].split("/")[0].strip()
    try:
        year = int(head)
    except ValueError:
        return None
    return year if 1900 < year < 2100 else None


def _download_extra_league_csv(code: str, expected_country: str):
    """
    Download one football-data.co.uk ``/new/{code}.csv`` all-time file.

    Cached per code for the lifetime of the process. Returns None (and caches
    the failure) on network, parse or country-guard errors. The guard exists
    because unknown codes under /new/ silently redirect to a different
    country's file — loading those rows would poison the table.
    """
    if code in _EXTRA_CSV_CACHE:
        return _EXTRA_CSV_CACHE[code]
    frame = None
    try:
        if requests is None:
            raise RuntimeError("requests is not installed")
        response = requests.get(
            EXTRA_CSV_BASE_URL.format(code=code),
            headers=EXTRA_CSV_HEADERS,
            timeout=60,
            allow_redirects=False,
        )
        response.raise_for_status()
        try:
            text = response.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = response.content.decode("latin-1")
        frame = pd.read_csv(io.StringIO(text), on_bad_lines="warn")
        frame.columns = [str(c).strip() for c in frame.columns]
        if "Country" in frame.columns:
            got = {str(c).strip() for c in frame["Country"].dropna().unique()}
            if got and got != {expected_country}:
                print(f"[matches] WARNING: {code}.csv serves Country={sorted(got)} "
                      f"(expected {expected_country!r}) — suspecting a redirect; "
                      f"skipping the file")
                frame = None
    except Exception as exc:  # noqa: BLE001
        print(f"[matches] WARNING: could not download {code}.csv: {exc}")
        frame = None
    _EXTRA_CSV_CACHE[code] = frame
    return frame


def _load_extra_matches(league: str, config: dict, season: str):
    """
    Return one league/season frame for an EXTRA_CSV_LEAGUES entry, shaped
    like a MatchHistory download so ``_build_match_rows`` can consume it.

    football-data's all-time CSVs use Home/Away/HG/AG/Res column names and a
    "Season" column; goals and odds only (no shots/corners/cards).
    """
    frame = _download_extra_league_csv(config["code"], config["country"])
    if frame is None or len(frame) == 0:
        print(f"[matches] WARNING: no rows available for {league} (all-time CSV)")
        return None

    out = frame.rename(columns={
        "Home": "home_team",
        "Away": "away_team",
        "HG": "FTHG",
        "AG": "FTAG",
        "Res": "FTR",
    })

    # dd/mm/yyyy in these files; dayfirst parsing keeps ambiguous dates sane.
    if "Date" in out.columns:
        out["date"] = pd.to_datetime(
            out["Date"], dayfirst=True, errors="coerce"
        )
    if "date" not in out.columns:
        print(f"[matches] WARNING: {league} all-time CSV has no Date column")
        return None

    target_year = _season_start_year(season)
    if target_year is None:
        print(f"[matches] WARNING: unparseable season label {season!r}")
        return None

    def _season_year_from_dates(dates) -> "pd.Series":
        if config.get("calendar_year"):
            return dates.dt.year
        # European Aug-May seasons: Jan-Jun dates belong to the previous label.
        return dates.dt.year - (dates.dt.month < 7).astype(int)

    if "Season" in out.columns:
        season_year = out["Season"].map(_season_start_year)
        # Rows with an unparsable/absent Season label fall back to the date.
        season_year = season_year.fillna(_season_year_from_dates(out["date"]))
    else:
        season_year = _season_year_from_dates(out["date"])
    out = out[season_year == target_year]

    out = out.dropna(subset=["date"])
    if len(out) == 0:
        print(f"[matches] WARNING: {league} {season}: no rows in the all-time CSV")
        return None
    return out



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

    _register_extra_leagues_with_soccerdata()

    for league, season in itertools.product(LEAGUES, SEASONS):
        extra_cfg = EXTRA_CSV_LEAGUES.get(league)
        if extra_cfg is not None:
            frame = _load_extra_matches(league, extra_cfg, season)
        else:
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
    """Insert or refresh Elo rows on the ``(team, date)`` unique key.

    Delegates to :func:`data.elo.upsert_elo_rows`, the one Elo writer both
    sources share. That helper is dialect-aware: this function used to build a
    ``postgresql`` INSERT unconditionally, which cannot compile on any other
    backend and left the whole write path untestable off Postgres. It also
    deduplicates ``(team, date)`` and refuses non-finite ratings, so a scraped
    NaN can never blank a stored value.
    """
    from data.elo import upsert_elo_rows

    return upsert_elo_rows(list(rows))


def collect_elo_ratings(start_date: date | None = None) -> dict[str, int]:
    """
    Collect ClubElo history for every team in the ``matches`` table.

    ``start_date`` optionally limits the history that is stored (the full
    history is stored by default). Teams unknown to ClubElo are skipped with a
    warning instead of aborting the run.

    OPTIONAL SOURCE — not part of the default pipeline any more. Ratings the
    model consumes are computed from stored match results by
    :func:`data.elo.update_elo_from_matches`; see that module for why. Two
    reasons this is opt-in rather than removed:

    * clubelo.com only resolves a subset of the 11-league team set under its
      own spellings, and every miss is counted as ``skipped`` — the function
      returns ``records: 0`` and exit code 0, so a run that stored nothing
      looked identical to a successful one;
    * its ratings live on a different scale (Liverpool ~2047) from the
      computed ones (a team starts at 1500). Merging both into ``elo_ratings``
      would make ``elo_difference`` compare numbers that are not comparable.

    Set ``COLLECT_CLUBELO=1`` to run it as a manual enrichment; it upserts
    through the same writer and never blanks an existing rating.
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
            elo_teams = session.execute(
                select(func.count(func.distinct(EloRating.team)))
            ).scalar_one()
            elo_min, elo_max = session.execute(
                select(func.min(EloRating.elo), func.max(EloRating.elo))
            ).one()
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
    print(f"  Teams with an elo rating: {elo_teams}")
    if elo_count:
        print(f"  Elo rating range       : {elo_min:.1f} to {elo_max:.1f}")
    print(f"  Leagues covered        : {len(leagues)}")
    for league in leagues:
        print(f"      - {league}")
    print(f"  Date range covered     : {date_range}")
    print("=" * 70)
    if match_count and not elo_count:
        # This is the exact state that shipped the 1500-fallback bug: matches
        # present, no ratings, pipeline reporting success. Say so loudly.
        print("[summary] ERROR: matches are stored but elo_ratings is EMPTY.")
        print("          Every prediction would use the 1500 default and")
        print("          elo_cache_entries would report 0. Run")
        print("          `python data/elo.py` to rebuild the ratings.")
        print("=" * 70)


def _clubelo_enabled() -> bool:
    """True when the optional ClubElo enrichment should also run."""
    return os.getenv("COLLECT_CLUBELO", "").strip().lower() in {"1", "true", "yes", "on"}


def run_collection() -> None:
    """Run the full data collection pipeline and print a final summary."""
    if not _check_dependencies():
        return

    print()
    print("#" * 70)
    print("# Football Predictor -- data collection")
    print("#" * 70)

    collect_match_history()

    # Elo is computed from the matches just collected, not scraped: the
    # ClubElo writer silently stored zero rows whenever api.clubelo.com was
    # unreachable or a football-data.co.uk spelling did not resolve, which left
    # elo_ratings empty and every prediction on the 1500 default. See
    # data/elo.py for the full diagnosis.
    try:
        from data.elo import update_elo_from_matches

        update_elo_from_matches()
    except Exception as exc:  # noqa: BLE001 - never abort collection on Elo
        print(f"[elo] ERROR: could not update Elo ratings from matches: {exc}")

    if _clubelo_enabled():
        collect_elo_ratings()
    else:
        print("[elo] Skipping the optional ClubElo scrape "
              "(set COLLECT_CLUBELO=1 to enable it).")

    _print_summary()


if __name__ == "__main__":
    from database import init_db

    init_db()
    run_collection()
