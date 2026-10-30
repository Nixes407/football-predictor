"""
Deterministic Elo ratings computed from the ``matches`` table.

Why this module exists
----------------------
``elo_ratings`` used to have exactly one writer:
``data.collector.collect_elo_ratings()``, which scrapes
``http://api.clubelo.com/<team>`` through ``soccerdata.ClubElo``. That writer
cannot keep this project's table populated:

* Every per-team failure — a connection error, an HTTP error, or a
  football-data.co.uk spelling that clubelo.com does not resolve in its URL —
  is swallowed into a ``skipped`` counter. ``collect_elo_ratings()`` then
  returns normally with ``records: 0`` and ``run_collection()`` exits 0.
* The ``[3/5] Run Match Collector`` workflow step is *also*
  ``continue-on-error: true``, so a run that stored zero ratings is
  indistinguishable from a successful one. Nothing ever failed loudly.
* clubelo.com ranks a subset of European clubs under its own spellings, so
  much of the 11-league team set is never resolvable at all: ``Bayern Munich``
  vs ``Bayern``, ``RB Leipzig`` vs ``Leipzig``, ``M'gladbach`` vs
  ``Gladbach``, ``Werder Bremen`` vs ``Bremen``, ``Atletico Madrid`` vs
  ``Atletico``, ``Athletic Bilbao`` vs ``Athletic``, ``Real Sociedad`` vs
  ``Sociedad``, ``Real Betis`` vs ``Betis``, ``Sporting CP`` vs ``Sporting``,
  ``AZ Alkmaar`` vs ``AZ``, ``St Etienne`` vs ``Saint-Etienne``,
  ``Sheffield Weds``, and most of the Championship, Primeira Liga,
  Eredivisie, Süper Lig, Eliteserien and Austrian clubs.

An empty ``elo_ratings`` is precisely what production reported: the read path
warmed 0 cache entries, every lookup returned ``None``, and
``features.builder.compute_elo_features()`` fell through to its hard-coded
1500 default — Arsenal vs Chelsea came out ``home_elo=1500``,
``away_elo=1500``, ``elo_difference=0``.

What this module does
---------------------
It replays every played match already stored in ``matches`` — data the daily
pipeline collects anyway — and applies the standard Elo recursion::

    expected = 1 / (1 + 10 ** ((opponent_rating - own_rating) / 400))
    new      = own + K * (actual - expected)

with ``K = 32``, ``actual`` = 1 / 0.5 / 0 for a win / draw / loss, and a
starting rating of 1500 for a team seen for the first time. No value is
invented and no new data source is introduced: the table is a pure function of
results already in the database, so the same ``matches`` content always
reproduces the same ratings.

Because Elo is path-dependent, the whole history is replayed on every run
rather than appended to. That is what makes the writer idempotent: running it
twice in a row produces byte-identical rows.

Semantics of a stored row
-------------------------
One row per team per match date, holding that team's rating *after* the match.
The feature builder reads ``date < match_date`` (see
``features.builder.get_pre_match_elo``), so a match is always rated against
the rating the team carried *into* it — no leakage. The live lookup takes the
most recent row, which is the same number for a fixture played today.

Rows are never written for a match whose outcome cannot be determined, and a
rating is never written as ``None``/``NaN``: existing non-null values are only
ever replaced by another real, finite rating.

For the original leagues, a replay overwrites a rating that came from
somewhere else (e.g. rows an old ClubElo run left behind), to keep ratings on
one scale. For newly ingested BRA-Série A, existing non-null ratings/league
values are preserved instead; this protects data already in the database but
means a later historical backfill cannot revise those stored Brazilian ratings.
"""

from __future__ import annotations

import math
import sys
from datetime import date as date_type
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# Make `database.py` importable no matter how this script is launched
# (``python data/elo.py`` or ``python -m data.elo``).
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from sqlalchemy import case, func, inspect, or_, select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from data.league_config import BRA_LEAGUE  # noqa: E402
from database import EloRating, Match, engine, session_scope  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Rating a team starts at before its first stored match.
DEFAULT_RATING = 1500.0

#: Elo K-factor: maximum rating points exchanged in one match.
K_FACTOR = 32.0

#: Home advantage, in Elo points, added to the home side before computing the
#: expectation. Deliberately 0.0: the model contract already carries separate
#: venue-form features (``home_form_*_5_home`` / ``away_form_*_5_away``), so
#: folding venue into the rating would double-count it. Kept as a named
#: constant so the assumption is explicit and reviewable.
HOME_ADVANTAGE = 0.0

#: Rows per INSERT statement.
CHUNK_SIZE = 500

#: The table this module writes and the read path queries.
ELO_TABLE = "elo_ratings"

#: Result codes understood in ``matches.result``.
_RESULT_SCORES = {"H": 1.0, "D": 0.5, "A": 0.0}


# --------------------------------------------------------------------------- #
# Small value helpers
# --------------------------------------------------------------------------- #


def _is_missing(value: Any) -> bool:
    """True for None/NaN/NaT/blank values."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _to_int(value: Any) -> int | None:
    """Coerce a stored goal count to int, or None when unusable."""
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


def _to_date(value: Any) -> date_type | None:
    """Coerce a stored match date to ``datetime.date``."""
    if _is_missing(value):
        return None
    if isinstance(value, date_type):
        # datetime is a subclass of date; normalise to a plain date.
        return value if type(value) is date_type else value.date()
    text = str(value).strip()
    try:
        return date_type.fromisoformat(text[:10])
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# The Elo formula
# --------------------------------------------------------------------------- #


def expected_score(own_rating: float, opponent_rating: float) -> float:
    """Elo win expectancy of ``own_rating`` against ``opponent_rating``.

    ``1 / (1 + 10 ** ((opponent - own) / 400))`` — 0.5 for equal ratings,
    approaching 1.0 as the own rating grows. Degenerate input yields 0.5
    rather than raising, so one bad row cannot abort a replay.
    """
    try:
        delta = (float(opponent_rating) - float(own_rating)) / 400.0
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(delta):
        return 0.5
    # CPython raises OverflowError (not inf) for ``10.0 ** 400``, so the gap is
    # clamped inside the representable range before the power is taken. +-300
    # is already a win expectancy of 1e-300 / 1-1e-300: far below the precision
    # a K=32 update can express, and no realistic rating pair gets near it.
    delta = max(-300.0, min(300.0, delta))
    try:
        return 1.0 / (1.0 + 10.0 ** delta)
    except OverflowError:  # pragma: no cover - the clamp makes this unreachable
        return 0.0 if delta > 0 else 1.0


def actual_score(
    result: Any,
    home_goals: Any = None,
    away_goals: Any = None,
) -> float | None:
    """Home-side outcome: 1.0 win, 0.5 draw, 0.0 loss, ``None`` if unknown.

    ``matches.result`` ("H"/"D"/"A") is authoritative when present. Rows that
    only carry goals — the football-data.co.uk ``/new/`` leagues store no
    result code — are derived from the scoreline instead, so they still
    contribute to the ratings.
    """
    if not _is_missing(result):
        code = str(result).strip().upper()[:1]
        if code in _RESULT_SCORES:
            return _RESULT_SCORES[code]

    home = _to_int(home_goals)
    away = _to_int(away_goals)
    if home is None or away is None:
        return None
    if home > away:
        return 1.0
    if home == away:
        return 0.5
    return 0.0


def _replay(matches: Iterable[Mapping[str, Any]]) -> tuple[list[dict], int, int]:
    """Replay ``matches``; return ``(rows, rated_matches, skipped_matches)``.

    The shared engine behind :func:`compute_elo_ratings` and
    :func:`update_elo_from_matches`. Counting rated matches here rather than
    deriving it from ``len(rows) // 2`` matters: two fixtures for the same team
    on the same date collapse into one ``(team, date)`` row, which made the
    "skipped" figure look like unrateable data when it was only a key
    collision.
    """
    ratings: dict[str, float] = {}
    rows: dict[tuple[str, date_type], dict[str, Any]] = {}
    rated = 0
    skipped = 0

    for match in matches:
        played_on = _to_date(match.get("date"))
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        if played_on is None or _is_missing(home_team) or _is_missing(away_team):
            skipped += 1
            continue
        home_team = str(home_team).strip()
        away_team = str(away_team).strip()
        if not home_team or not away_team or home_team == away_team:
            skipped += 1
            continue

        outcome = actual_score(
            match.get("result"), match.get("home_goals"), match.get("away_goals")
        )
        if outcome is None:
            skipped += 1
            continue

        home_before = ratings.get(home_team, DEFAULT_RATING)
        away_before = ratings.get(away_team, DEFAULT_RATING)

        home_expect = expected_score(
            home_before + HOME_ADVANTAGE, away_before
        )
        away_expect = 1.0 - home_expect

        home_after = home_before + K_FACTOR * (outcome - home_expect)
        away_after = away_before + K_FACTOR * ((1.0 - outcome) - away_expect)

        # A rating must stay finite and real: never persist NaN/inf.
        if not (math.isfinite(home_after) and math.isfinite(away_after)):
            skipped += 1
            continue

        ratings[home_team] = home_after
        ratings[away_team] = away_after
        rated += 1

        league = match.get("league")
        league = None if _is_missing(league) else str(league).strip()
        rows[(home_team, played_on)] = {
            "team": home_team,
            "date": played_on,
            "elo": round(home_after, 4),
            "league": league,
        }
        rows[(away_team, played_on)] = {
            "team": away_team,
            "date": played_on,
            "elo": round(away_after, 4),
            "league": league,
        }

    return list(rows.values()), rated, skipped


def compute_elo_ratings(matches: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Replay ``matches`` chronologically into ``elo_ratings`` row dicts.

    Pure function: no database, no clock, no global state. Each input mapping
    needs ``date``, ``home_team``, ``away_team`` and enough of
    ``result``/``home_goals``/``away_goals`` to decide the outcome; ``league``
    is optional and stored alongside the rating.

    The caller is responsible for chronological ordering — the result is
    order-dependent by definition. Returns one row per team per match date
    holding the post-match rating. A match with an undeterminable outcome is
    skipped entirely (never written as None or 0). Where two rows would share
    a ``(team, date)`` key, the later replayed value wins so the payload can
    be inserted in one statement.
    """
    return _replay(matches)[0]


def final_ratings(matches: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """``{team: current rating}`` after replaying ``matches``."""
    ratings: dict[str, float] = {}
    for row in compute_elo_ratings(matches):
        ratings[row["team"]] = row["elo"]
    return ratings


# --------------------------------------------------------------------------- #
# Database access
# --------------------------------------------------------------------------- #


def elo_table_exists(bind=None) -> bool:
    """True when the ``elo_ratings`` table is present in the database."""
    target = bind if bind is not None else engine
    if target is None:
        return False
    try:
        return ELO_TABLE in inspect(target).get_table_names()
    except Exception as exc:  # noqa: BLE001 - unreachable DB is a "no"
        print(f"[elo] WARNING: could not inspect the schema: {exc}")
        return False


def load_matches_for_elo(session) -> list[dict[str, Any]]:
    """Every played match, oldest first, in the shape :func:`compute_elo_ratings` wants.

    Ordered by ``(date, id)`` so a replay is deterministic even when several
    matches share a date.
    """
    rows = session.execute(
        select(
            Match.date,
            Match.home_team,
            Match.away_team,
            Match.home_goals,
            Match.away_goals,
            Match.result,
            Match.league,
        ).order_by(Match.date.asc(), Match.id.asc())
    ).all()
    return [
        {
            "date": row[0],
            "home_team": row[1],
            "away_team": row[2],
            "home_goals": row[3],
            "away_goals": row[4],
            "result": row[5],
            "league": row[6],
        }
        for row in rows
    ]


def _chunked(items: Sequence[Any], size: int = CHUNK_SIZE) -> Iterator[Sequence[Any]]:
    """Yield successive sub-sequences of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def upsert_elo_rows(rows: Iterable[Mapping[str, Any]]) -> int:
    """Insert or refresh rows on the ``(team, date)`` unique key.

    Dialect-aware on purpose: the previous writer used
    ``sqlalchemy.dialects.postgresql.insert`` unconditionally, which cannot
    compile against anything else and made the whole write path impossible to
    unit-test. PostgreSQL gets a set-based ``ON CONFLICT DO UPDATE``; other
    backends fall back to an equivalent per-row merge.

    Only finite, non-null ratings are written, so this can never blank an
    existing value. Brazilian conflicts additionally preserve existing non-null
    values; the original leagues retain their established refresh behavior.
    Returns the number of rows sent to the database.
    """
    payload: dict[tuple[str, date_type], dict[str, Any]] = {}
    for row in rows:
        team = row.get("team")
        played_on = _to_date(row.get("date"))
        if _is_missing(team) or played_on is None:
            continue
        try:
            rating = float(row.get("elo"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rating):
            continue
        payload[(str(team), played_on)] = {
            "team": str(team),
            "date": played_on,
            "elo": rating,
            "league": row.get("league"),
        }

    if not payload:
        return 0

    values = list(payload.values())
    is_postgres = engine is not None and engine.dialect.name == "postgresql"

    with session_scope() as session:
        if is_postgres:
            for chunk in _chunked(values):
                statement = pg_insert(EloRating).values(list(chunk))
                brazil_conflict = or_(
                    EloRating.league == BRA_LEAGUE,
                    statement.excluded.league == BRA_LEAGUE,
                )
                statement = statement.on_conflict_do_update(
                    index_elements=["team", "date"],
                    set_={
                        "elo": case(
                            (brazil_conflict, func.coalesce(
                                EloRating.elo, statement.excluded.elo)),
                            else_=statement.excluded.elo,
                        ),
                        "league": case(
                            (brazil_conflict, func.coalesce(
                                EloRating.league, statement.excluded.league)),
                            else_=statement.excluded.league,
                        ),
                    },
                )
                session.execute(statement)
        else:
            for row in values:
                existing = (
                    session.query(EloRating)
                    .filter(
                        EloRating.team == row["team"],
                        EloRating.date == row["date"],
                    )
                    .one_or_none()
                )
                if existing is None:
                    session.add(EloRating(**row))
                elif existing.league == BRA_LEAGUE or row["league"] == BRA_LEAGUE:
                    if existing.elo is None:
                        existing.elo = row["elo"]
                    if existing.league is None:
                        existing.league = row["league"]
                else:
                    existing.elo = row["elo"]
                    existing.league = row["league"]

    return len(values)


def update_elo_from_matches(verbose: bool = True) -> dict[str, int]:
    """Recompute ``elo_ratings`` from every match in the database.

    Returns ``{"matches", "rated_matches", "skipped_matches", "teams",
    "records"}``. Raises nothing on an empty ``matches`` table — it reports
    zeros and lets the caller decide whether that is fatal (the CLI and the
    workflow step do; ``run_collection()`` does not).
    """
    totals = {
        "matches": 0,
        "rated_matches": 0,
        "skipped_matches": 0,
        "teams": 0,
        "records": 0,
    }

    with session_scope() as session:
        matches = load_matches_for_elo(session)
    totals["matches"] = len(matches)

    if not matches:
        if verbose:
            print("[elo] WARNING: the matches table is empty -- run "
                  "collect_match_history() first.")
        return totals

    rows, rated, skipped = _replay(matches)
    totals["teams"] = len({row["team"] for row in rows})
    totals["rated_matches"] = rated
    totals["skipped_matches"] = skipped

    if verbose:
        print("=" * 70)
        print(f"[elo] Replaying {totals['matches']} matches "
              f"(K={K_FACTOR:g}, start={DEFAULT_RATING:g})")
        print("=" * 70)

    totals["records"] = upsert_elo_rows(rows)

    if verbose:
        print(f"[elo] Done: {totals['records']} rows upserted for "
              f"{totals['teams']} teams "
              f"({totals['rated_matches']} rated matches, "
              f"{totals['skipped_matches']} without a determinable result)")
        if totals["records"] == 0:
            print("[elo] ERROR: no ratings were written. Every match in the "
                  "table lacks a result, or the write failed -- elo_ratings "
                  "will stay empty and every prediction will use the 1500 "
                  "default.")

    return totals


def elo_summary() -> dict[str, Any]:
    """Cheap health report for the CLI, the workflow log and diagnostics.

    ``table_exists`` distinguishes "the table was never created" from "the
    table exists but holds no ratings" — the two look identical from the API
    and were impossible to tell apart in production.
    """
    summary: dict[str, Any] = {
        "table_exists": elo_table_exists(),
        "rows": 0,
        "teams": 0,
        "min": None,
        "max": None,
        "avg": None,
        "latest_date": None,
        "error": None,
    }
    if not summary["table_exists"]:
        summary["error"] = f"table {ELO_TABLE!r} does not exist"
        return summary

    try:
        with session_scope() as session:
            row = session.execute(
                select(
                    func.count(),
                    func.count(func.distinct(EloRating.team)),
                    func.min(EloRating.elo),
                    func.max(EloRating.elo),
                    func.avg(EloRating.elo),
                    func.max(EloRating.date),
                ).select_from(EloRating)
            ).one()
    except Exception as exc:  # noqa: BLE001 - report, do not crash the API
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    summary["rows"] = int(row[0] or 0)
    summary["teams"] = int(row[1] or 0)
    summary["min"] = float(row[2]) if row[2] is not None else None
    summary["max"] = float(row[3]) if row[3] is not None else None
    summary["avg"] = round(float(row[4]), 2) if row[4] is not None else None
    summary["latest_date"] = str(row[5]) if row[5] is not None else None
    return summary


def rating_for(team: str) -> float | None:
    """Latest stored rating for ``team``, or None. Read-only convenience."""
    if not elo_table_exists():
        return None
    try:
        with session_scope() as session:
            row = session.execute(
                select(EloRating.elo)
                .where(EloRating.team == team)
                .order_by(EloRating.date.desc())
                .limit(1)
            ).fetchone()
    except Exception as exc:  # noqa: BLE001 - a miss is not a crash
        print(f"[elo] WARNING: could not read the rating for {team}: {exc}")
        return None
    return float(row[0]) if row and row[0] is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python data/elo.py``.

    Exits 1 when no rating could be written, so the workflow step turns red
    instead of reporting success over an empty table — the failure mode that
    let this bug ship silently.
    """
    from database import init_db

    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0

    if engine is None:
        print("[elo] ERROR: no database engine -- is DATABASE_URL set?")
        return 1

    init_db()
    totals = update_elo_from_matches()

    summary = elo_summary()
    print("-" * 70)
    print(f"[elo] {ELO_TABLE}: exists={summary['table_exists']} "
          f"rows={summary['rows']} teams={summary['teams']} "
          f"min={summary['min']} max={summary['max']} avg={summary['avg']} "
          f"latest={summary['latest_date']}")
    if summary["error"]:
        print(f"[elo] ERROR: {summary['error']}")

    if totals["records"] == 0 or summary["rows"] == 0:
        print("[elo] FAILED: elo_ratings is still empty -- every prediction "
              "would fall back to the 1500 default.")
        return 1

    print(f"[elo] OK: {summary['teams']} teams rated "
          f"({summary['rows']} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
