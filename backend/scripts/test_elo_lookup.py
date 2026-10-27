"""
Verify that the Elo read path actually resolves ratings from the database.

This is the acceptance check for the ``elo_cache_entries: 0`` /
``home_elo: 1500`` bug: it drives the *same* functions the live API drives —
``features.builder.load_latest_elo_ratings`` (the cache-load query),
``features.builder.get_latest_elo_before`` (the per-team lookup) and
``routers.predict.warm_elo_cache`` / ``diagnostics`` (what production reports)
— and prints what each of them returns.

It distinguishes the four states that all look like "ELO is broken" from the
outside:

* ``elo_ratings`` does not exist          -> table missing
* table exists, 0 rows                    -> nothing was ever written
* table has rows, Arsenal resolves        -> healthy
* table has rows, Arsenal does not        -> name-format mismatch

Usage (from ``backend/``, with ``DATABASE_URL`` set)::

    python scripts/test_elo_lookup.py
    python scripts/test_elo_lookup.py --update     # rebuild ratings first
    python scripts/test_elo_lookup.py --json

``--update`` runs :func:`data.elo.update_elo_from_matches` first, so the script
can both repair and verify in one call. Exit code is 0 on PASS, 1 on FAIL, 2
when there is no database to check at all.

Everything here is read-only unless ``--update`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

#: The fixture from the bug report: both sides reported 1500 / difference 0.
PROBE_TEAMS = ("Arsenal", "Chelsea")

#: The hard-coded fallback in ``builder.compute_elo_features``.
DEFAULT_RATING = 1500.0


def _fmt(value) -> str:
    """Human-readable rating, keeping None visibly distinct from a number."""
    if value is None:
        return "None (no rating found -> 1500 default)"
    return f"{float(value):.2f}"


def run(update: bool = False) -> dict:
    """Execute every check and return a machine-readable report."""
    from data import elo as elo_writer
    from features import builder

    report: dict = {
        "database_url_configured": None,
        "table_exists": False,
        "table_rows": 0,
        "table_teams": 0,
        "rating_min": None,
        "rating_max": None,
        "rating_avg": None,
        "latest_rating_date": None,
        "update": None,
        "cache_load_rows": 0,
        "elo_cache_entries": 0,
        "elo_lookup": None,
        "elo_cache_error": None,
        "lookups": {},
        "fixture": {},
        "pass": False,
        "reason": "",
    }

    import database

    report["database_url_configured"] = database.engine is not None
    if database.engine is None:
        report["reason"] = (
            "no database engine — set DATABASE_URL (or backend/.env) to a "
            "reachable PostgreSQL instance"
        )
        return report

    # -- Optional repair step ------------------------------------------- #
    if update:
        database.init_db()
        report["update"] = elo_writer.update_elo_from_matches()

    # -- 1. Table state -------------------------------------------------- #
    summary = elo_writer.elo_summary()
    report["table_exists"] = summary["table_exists"]
    report["table_rows"] = summary["rows"]
    report["table_teams"] = summary["teams"]
    report["rating_min"] = summary["min"]
    report["rating_max"] = summary["max"]
    report["rating_avg"] = summary["avg"]
    report["latest_rating_date"] = summary["latest_date"]

    if not summary["table_exists"]:
        report["reason"] = (
            "the elo_ratings table does not exist — no ELO has ever been "
            "written to this database"
        )
        return report
    if summary["rows"] == 0:
        report["reason"] = (
            "elo_ratings exists but is EMPTY — the writer ran and stored 0 "
            "rows, so every lookup falls back to 1500"
        )
        return report

    # -- 2. The cache-load query ---------------------------------------- #
    ratings = builder.load_latest_elo_ratings()
    report["cache_load_rows"] = len(ratings)

    # -- 3. What the API's own warm cache + diagnostics report ----------- #
    try:
        from routers import predict

        predict.warm_elo_cache(force=True)
        report["elo_cache_entries"] = len(predict._elo_latest_cache)
        report["elo_lookup"] = (
            "database"
            if predict._elo_cache_error is None
            and predict._elo_lookup_failures < predict._ELO_MAX_FAILURES
            else "features.csv_fallback"
        )
        report["elo_cache_error"] = predict._elo_cache_error
    except Exception as exc:  # noqa: BLE001 - the API layer is optional here
        report["elo_cache_error"] = f"{type(exc).__name__}: {exc}"

    # -- 4. Per-team lookups -------------------------------------------- #
    import pandas as pd

    cutoff = pd.Timestamp.today().normalize()
    for team in PROBE_TEAMS:
        report["lookups"][team] = builder.get_latest_elo_before(team, cutoff)

    # -- 5. The exact feature triple from the bug report ----------------- #
    def _lookup(team, before_date):
        try:
            from routers import predict

            return predict._elo_lookup(team, before_date)
        except Exception:  # noqa: BLE001 - fall back to the builder directly
            return builder.get_latest_elo_before(team, before_date)

    report["fixture"] = builder.compute_elo_features(
        pd.DataFrame(), PROBE_TEAMS[0], PROBE_TEAMS[1], cutoff,
        elo_lookup=_lookup,
    )

    # -- Verdict --------------------------------------------------------- #
    home = report["fixture"].get("home_elo")
    away = report["fixture"].get("away_elo")
    resolved = [
        team for team in PROBE_TEAMS
        if report["lookups"].get(team) is not None
        and float(report["lookups"][team]) != DEFAULT_RATING
    ]
    report["pass"] = bool(
        len(resolved) == len(PROBE_TEAMS)
        and home is not None and float(home) != DEFAULT_RATING
        and away is not None and float(away) != DEFAULT_RATING
        and report["elo_cache_entries"] > 0
    )
    if not report["pass"]:
        missing = [t for t in PROBE_TEAMS if t not in resolved]
        report["reason"] = (
            f"{', '.join(missing)} did not resolve to a real rating "
            f"(cache entries: {report['elo_cache_entries']})"
        )
    return report


def print_report(report: dict) -> None:
    """Pretty-print :func:`run`'s report to stdout."""
    line = "-" * 70
    print(line)
    print("ELO LOOKUP VERIFICATION")
    print(line)
    print(f"  DATABASE_URL configured    : {report['database_url_configured']}")
    print(f"  elo_ratings table exists   : {report['table_exists']}")
    print(f"  total rows                 : {report['table_rows']}")
    print(f"  distinct teams rated       : {report['table_teams']}")
    print(f"  rating range               : "
          f"{report['rating_min']} .. {report['rating_max']} "
          f"(avg {report['rating_avg']})")
    print(f"  latest rating date         : {report['latest_rating_date']}")
    if report["update"] is not None:
        print(f"  --update totals            : {report['update']}")
    print(line)
    print("  CACHE LOAD QUERY (builder.load_latest_elo_ratings)")
    print(f"    rows returned            : {report['cache_load_rows']}")
    print(f"    elo_cache_entries        : {report['elo_cache_entries']}")
    print(f"    elo_lookup               : {report['elo_lookup']}")
    print(f"    elo_cache_error          : {report['elo_cache_error']}")
    print(line)
    print("  PER-TEAM LOOKUPS (builder.get_latest_elo_before)")
    for team, value in report["lookups"].items():
        print(f"    {team:<12} -> {_fmt(value)}")
    print(line)
    print("  FEATURE TRIPLE (builder.compute_elo_features, Arsenal vs Chelsea)")
    for key in ("home_elo", "away_elo", "elo_difference"):
        print(f"    {key:<14}: {report['fixture'].get(key)}")
    print(line)
    if report["reason"]:
        print(f"  reason: {report['reason']}")
    print(f"  {'PASS' if report['pass'] else 'FAIL'}")
    print(line)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Elo ratings resolve from the database."
    )
    parser.add_argument(
        "--update", action="store_true",
        help="recompute elo_ratings from the matches table before checking",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON only"
    )
    args = parser.parse_args(argv)

    report = run(update=args.update)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    if report["database_url_configured"] is False:
        return 2
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
