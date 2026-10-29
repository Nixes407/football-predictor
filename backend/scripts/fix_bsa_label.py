#!/usr/bin/env python3
"""One-time repair: relabel mislabelled Brazilian fixtures in ``daily_matches``.

Why this exists
---------------
football-data.org serves the Brazilian top flight under the competition code
``BSA``.  That code was absent from ``LEAGUE_MAP`` in ``backend/data/fixtures.py``
while the fixtures targeted below were fetched, so the daily fixture job stored
them under another competition's label: the rows still say
``BEL-Jupiler Pro League``.

Adding ``"BSA": "BRA-Série A"`` to ``LEAGUE_MAP`` only fixes *newly fetched*
fixtures — rows already committed to ``daily_matches`` keep the stale label,
which is why this repair has to be applied once to the live database::

    UPDATE daily_matches
       SET league = 'BRA-Série A'
     WHERE league ILIKE '%jupiler%'

No competition fetched by ``backend/data/fixtures.py`` is Belgian — the module
speaks football-data.org's PL/ELC/PD/BL1/SA/FL1/PPL/DED/BSA/TR1/NOR/AUT/CL plus
FotMob's USA-MLS — so every Jupiler-labelled row in ``daily_matches`` is one of
these mislabelled Brazilian fixtures.  The match is a case-insensitive
substring rather than an equality test so a stray spelling ("Jupiler Pro
League") is repaired by the same run.

Usage
-----
::

    cd backend
    python scripts/fix_bsa_label.py             # apply the repair
    python scripts/fix_bsa_label.py --dry-run   # report, change nothing

``DATABASE_URL`` has to point at the target database; ``backend/.env`` is loaded
automatically (see ``backend/database.py``).  The script is idempotent: a second
run reports ``updated 0``.

The script is idempotent: a second run reports ``updated 0``.  It is a manual
repair tool only — no workflow or API route calls it any more (the one-off
``fix_bsa_label.yml`` workflow and the ``GET /predict/admin/fix-bsa-label``
route were both retired once the stored rows had been corrected).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Running this file as a script puts backend/scripts on sys.path — not backend/ —
# so ``import database`` would fail without this.  Harmless when the module is
# imported from the API instead (backend/ is already there).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

#: Label the mislabelled rows carry today.
STALE_LEAGUE = "BEL-Jupiler Pro League"

#: Label they should carry.
TARGET_LEAGUE = "BRA-Série A"

#: Case-insensitive substring identifying the stale rows.
STALE_LEAGUE_PATTERN = "%jupiler%"


def describe_row(row: Any) -> dict:
    """Return the fields worth showing for one ``daily_matches`` row."""
    return {
        "id": row.id,
        "external_id": row.external_id,
        "match_date": row.match_date,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "league": row.league,
        "season": row.season,
    }


def find_stale_rows(session) -> list:
    """Every ``daily_matches`` row still carrying a Belgian label."""
    from database import DailyMatch

    return (
        session.query(DailyMatch)
        .filter(DailyMatch.league.ilike(STALE_LEAGUE_PATTERN))
        .order_by(DailyMatch.match_date, DailyMatch.id)
        .all()
    )


def apply_bsa_label_fix(session, dry_run: bool = False) -> dict:
    """Repoint every Belgian-labelled daily fixture at ``BRA-Série A``.

    Takes an open SQLAlchemy session so the API endpoint and this script share
    one implementation; the caller owns the transaction when ``dry_run`` is
    used, this function commits when it writes.  Returns a report dict:

        {"stale_league", "target_league", "matched_count", "matched_rows",
         "updated", "remaining_stale", "dry_run"}
    """
    from database import DailyMatch

    rows = find_stale_rows(session)
    affected = [describe_row(row) for row in rows]

    updated = 0
    if rows and not dry_run:
        updated = (
            session.query(DailyMatch)
            .filter(DailyMatch.league.ilike(STALE_LEAGUE_PATTERN))
            .update(
                {DailyMatch.league: TARGET_LEAGUE},
                synchronize_session=False,
            )
        )
        session.commit()

    session.expire_all()
    remaining = (
        session.query(DailyMatch)
        .filter(DailyMatch.league.ilike(STALE_LEAGUE_PATTERN))
        .count()
    )

    return {
        "stale_league": STALE_LEAGUE,
        "target_league": TARGET_LEAGUE,
        "pattern": STALE_LEAGUE_PATTERN,
        "matched_count": len(affected),
        "matched_rows": affected,
        "updated": updated,
        "remaining_stale": remaining,
        "dry_run": bool(dry_run),
    }


def _print_report(result: dict) -> None:
    """Human-readable summary of :func:`apply_bsa_label_fix`."""
    matched = result["matched_count"]
    print(
        f"[fix-bsa] {matched} row(s) labelled {result['stale_league']!r} "
        f"matched (pattern {result['pattern']!r})"
    )
    for row in result["matched_rows"]:
        print(
            f"[fix-bsa]   id={row['id']} {row['match_date'] or '?'} "
            f"{row['home_team']} vs {row['away_team']} "
            f"(season {row['season'] or '?'}, external_id "
            f"{row['external_id']})"
        )

    if result["dry_run"]:
        print(
            f"[fix-bsa] DRY RUN — would set league = "
            f"{result['target_league']!r} on {matched} row(s), "
            f"nothing written"
        )
        return

    print(
        f"[fix-bsa] Updated {result['updated']} daily_matches row(s) to "
        f"{result['target_league']!r}"
    )
    if result["remaining_stale"]:
        print(
            f"[fix-bsa] WARNING: {result['remaining_stale']} row(s) still "
            f"labelled {result['stale_league']!r}"
        )
    else:
        print(
            f"[fix-bsa] Verified: no {result['stale_league']!r} rows remain "
            f"in daily_matches"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-time repair: relabel daily_matches rows stored as "
            f"{STALE_LEAGUE!r} as {TARGET_LEAGUE!r} instead."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the rows that would change without writing anything",
    )
    args = parser.parse_args(argv)

    try:
        from database import SessionLocal, engine, get_database_url
    except Exception as exc:  # noqa: BLE001 - report the import error clearly
        print(f"[fix-bsa] ERROR: could not import the database layer: {exc}")
        return 2

    if engine is None:
        print(
            "[fix-bsa] ERROR: no database engine is configured. Set "
            "DATABASE_URL (or backend/.env) and try again."
        )
        return 2

    # Never print the credentials, only enough to tell deployments apart.
    url = get_database_url()
    print(f"[fix-bsa] Target database host: {url.split('@')[-1] or 'unknown'}")

    session = SessionLocal()
    try:
        result = apply_bsa_label_fix(session, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - a one-off script reports and exits
        session.rollback()
        print(f"[fix-bsa] ERROR: the repair failed: {exc}")
        return 1
    finally:
        session.close()

    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
