#!/usr/bin/env python3
"""Data-availability report for the expanded training leagues.

Answers the two questions that must be checked before (re)training on the
wider league set:

1. How many COMPLETED matches does the ``matches`` table hold per league and
   season (default: 2021-2022 .. 2024-2025) for every league in the trainer's
   ``TRAINING_LEAGUES``?
2. Does ``fbref_team_stats`` contain rows for those leagues?  Missing FBref
   coverage is NOT fatal: the feature builder zero-fills the FBref columns,
   so those leagues train on ELO + rolling-results features only.

The script reads whatever ``DATABASE_URL`` points at (``backend/.env`` is
loaded automatically by ``backend/database.py``), so it works against the
local Postgres and the hosted Supabase instance alike.  A connection failure
is reported and turns into exit code 2 — it never traces.

Usage
-----
::

    cd backend
    python scripts/analyze_training_leagues.py
    python scripts/analyze_training_leagues.py --json report.json
    python scripts/analyze_training_leagues.py \
        --seasons 2021-2022 2022-2023 --leagues "ENG-Championship" "NED-Eredivisie"
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Running this file as a script puts backend/scripts on sys.path — not
# backend/ — so `import database` would fail without this (same trick as
# scripts/fix_bsa_label.py).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]

#: Fallback if backend/models/trainer.py cannot be imported (its xgboost /
#: supabase dependencies are optional for this report). Keep in sync with
#: trainer.TRAINING_LEAGUES.
FALLBACK_TRAINING_LEAGUES = [
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


def _training_leagues() -> list[str]:
    try:
        from models.trainer import TRAINING_LEAGUES  # type: ignore

        return list(TRAINING_LEAGUES)
    except Exception:
        return list(FALLBACK_TRAINING_LEAGUES)


def _norm(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(ascii_only.casefold().split())


def _matches_database() -> Any:
    from sqlalchemy import text  # local import: fail soft when not installed

    from database import engine  # noqa: WPS433 - path set above

    if engine is None:
        raise RuntimeError("database.engine is not configured")
    return engine, text


def run_report(
    leagues: list[str],
    seasons: list[str],
) -> dict[str, Any]:
    """Query availability; returns a JSON-serialisable report dict."""
    engine, text = _matches_database()

    report: dict[str, Any] = {
        "database_ok": True,
        "seasons": seasons,
        "leagues": leagues,
        "matches": {},
        "fbref": {},
    }

    season_placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    season_params = {f"s{i}": s for i, s in enumerate(seasons)}

    with engine.connect() as conn:
        # 1) completed matches per league/season. "Completed" = a score and a
        # valid 1X2 result are stored. Comparison is case/accent-insensitive
        # so a DB that stores e.g. "TUR-Super Lig" is still matched.
        rows = conn.execute(
            text(
                f"""
                SELECT league, season, COUNT(*) AS n
                FROM matches
                WHERE home_goals IS NOT NULL
                  AND away_goals IS NOT NULL
                  AND result IN ('H', 'D', 'A')
                  AND season IN ({season_placeholders})
                GROUP BY league, season
                ORDER BY league, season
                """
            ),
            season_params,
        ).fetchall()

        by_league_norm: dict[str, dict[str, int]] = {}
        actual_names: dict[str, str] = {}
        for league, season, n in rows:
            key = _norm(league)
            by_league_norm.setdefault(key, {})[str(season)] = int(n)
            actual_names.setdefault(key, str(league))

        wanted = {_norm(l): l for l in leagues}
        for key, per_season in sorted(by_league_norm.items()):
            tag = wanted.get(key)
            if tag is None:
                continue
            report["matches"][tag] = {
                "stored_name": actual_names[key],
                "seasons": per_season,
                "total": sum(per_season.values()),
            }
        missing = [l for l in leagues if _norm(l) not in by_league_norm]
        report["leagues_with_zero_completed_matches"] = missing

        # leagues present in matches but outside the training set
        extras = sorted(set(by_league_norm) - set(wanted))
        report["other_leagues_in_matches"] = [actual_names[e] for e in extras]

        # 2) fbref_team_stats coverage (per stored league string).
        try:
            fb_rows = conn.execute(
                text(
                    """
                    SELECT league, COUNT(*) AS n, MAX(updated_at) AS latest
                    FROM fbref_team_stats
                    GROUP BY league
                    ORDER BY league
                    """
                )
            ).fetchall()
            for league, n, latest in fb_rows:
                report["fbref"][str(league)] = {
                    "rows": int(n),
                    "latest_updated_at": str(latest) if latest is not None else None,
                }
            report["fbref_coverage_error"] = None
        except Exception as exc:  # table may not exist on a fresh DB
            report["fbref_coverage_error"] = str(exc)

        fbref_by_norm = {_norm(k): v for k, v in report["fbref"].items()}
        report["leagues_without_fbref"] = [
            l for l in leagues if _norm(l) not in fbref_by_norm
        ]

    return report


def print_report(report: dict[str, Any]) -> None:
    print("=" * 74)
    print("TRAINING LEAGUE DATA AVAILABILITY")
    print("=" * 74)
    print(f"seasons: {', '.join(report['seasons'])}")
    print()
    print("completed matches in `matches` (by league x season)")
    header = f"{'league':<24}" + "".join(f"{s:>12}" for s in report["seasons"]) + f"{'total':>10}"
    print(header)
    print("-" * len(header))
    for league in report["leagues"]:
        stats = report["matches"].get(league)
        if stats is None:
            print(f"{league:<24}" + "".join(f"{'0':>12}" for _ in report["seasons"]) + f"{0:>10}")
            continue
        per = stats["seasons"]
        cells = "".join(f"{per.get(s, 0):>12}" for s in report["seasons"])
        print(f"{league:<24}{cells}{stats['total']:>10}")
    zero = report.get("leagues_with_zero_completed_matches") or []
    if zero:
        print()
        print(f"WARNING — leagues with NO completed matches: {zero}")
        print("  (collector.py could not ingest them, or they need backfilling)")
    other = report.get("other_leagues_in_matches") or []
    if other:
        print(f"\nother leagues stored in `matches` (excluded from training): {other}")

    print()
    print("fbref_team_stats coverage")
    if report.get("fbref_coverage_error"):
        print(f"  ERROR: {report['fbref_coverage_error']}")
    elif not report["fbref"]:
        print("  (table is empty — every league trains without FBref features)")
    else:
        for league, n in sorted(report["fbref"].items()):
            print(f"  {league:<28} rows={n['rows']:<6} latest={n['latest_updated_at']}")
    no_fb = report.get("leagues_without_fbref") or []
    if no_fb:
        print()
        print("leagues WITHOUT fbref_team_stats rows (acceptable: they fall back "
              "to ELO + rolling results):")
        for league in no_fb:
            print(f"  - {league}")
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--leagues", nargs="*", default=None,
                        help="defaults to trainer.TRAINING_LEAGUES")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="also write the machine-readable report")
    args = parser.parse_args(argv)

    leagues = args.leagues or _training_leagues()
    try:
        report = run_report(leagues, args.seasons)
    except Exception as exc:  # noqa: BLE001 - never trace at operators
        print(f"[analyze] ERROR: could not read the database: {exc}")
        print("[analyze] Set DATABASE_URL (or backend/.env) to a reachable "
              "PostgreSQL instance and re-run.")
        return 2

    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[analyze] JSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
