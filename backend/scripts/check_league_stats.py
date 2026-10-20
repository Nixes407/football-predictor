"""
Report what the formula engine actually aggregates per league and season.

``league_teams`` in a prediction is the number of clubs the engine calibrated
against (see ``models.formula_engine.compute_league_baselines``). A 20-club
league must report 20; anything else means ``team_season_stats`` holds rows
that are not that league's clubs for that season (another competition's rows,
a stale campaign, empty shells, provider duplicates that do not collapse).

This script runs the engine's own loader against the live database, so what it
prints is exactly what a prediction would use.

Usage (from ``backend/``, with ``DATABASE_URL`` set)::

    python scripts/check_league_stats.py
    python scripts/check_league_stats.py --league "ENG-Premier League"
    python scripts/check_league_stats.py --league "ENG-Premier League" \
        --season 2026-2027 --show-teams
    python scripts/check_league_stats.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from models import formula_engine as fe  # noqa: E402


def _leagues_to_check(requested: list[str] | None) -> list[str]:
    """Configured leagues, or the ones stored in team_season_stats."""
    if requested:
        return requested

    db = fe._new_session()
    if db is None:
        print("[check] ERROR: no database session — is DATABASE_URL set?")
        return []
    try:
        from sqlalchemy import func

        from database import TeamSeasonStats

        rows = (
            db.query(TeamSeasonStats.league)
            .group_by(TeamSeasonStats.league)
            .order_by(func.lower(TeamSeasonStats.league))
            .all()
        )
        return [row[0] for row in rows if row[0]]
    finally:
        fe._close_session(db)


def _row_count(league: str, season: str) -> int:
    """Every stored row for the key, before any filtering."""
    db = fe._new_session()
    if db is None:
        return 0
    try:
        from sqlalchemy import and_

        from database import TeamSeasonStats

        return (
            db.query(TeamSeasonStats)
            .filter(and_(
                TeamSeasonStats.league == league,
                TeamSeasonStats.season == season,
            ))
            .count()
        )
    finally:
        fe._close_session(db)


def report(league: str, season: str | None, show_teams: bool) -> dict:
    fe.clear_cache()
    rows, season_used = fe._league_season_rows(league, season)
    stored = _row_count(league, season_used) if season_used else 0

    entry = {
        "league": league,
        "season": season_used,
        "teams_counted": len(rows),
        "rows_stored": stored,
        "plausible": len(rows) <= fe.MAX_PLAUSIBLE_LEAGUE_TEAMS,
    }
    if show_teams:
        entry["teams"] = sorted(str(row.team) for row in rows)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league", action="append", dest="leagues",
        help="league to check (repeatable). Default: every league stored.",
    )
    parser.add_argument(
        "--season", default=None,
        help="season label. Default: every season stored for the league.",
    )
    parser.add_argument("--show-teams", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    results = []
    for league in _leagues_to_check(args.leagues):
        if args.season:
            seasons = [args.season]
        else:
            seasons = fe.league_seasons(league) or [None]
        for season in seasons:
            results.append(report(league, season, args.show_teams))

    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        for entry in results:
            flag = "ok " if entry["plausible"] else "BAD"
            print(
                f"[{flag}] {entry['league']} {entry['season']}: "
                f"{entry['teams_counted']} teams counted "
                f"({entry['rows_stored']} rows stored)"
            )
            if entry.get("teams") is not None:
                for team in entry["teams"]:
                    print(f"        - {team}")

    return 1 if any(not entry["plausible"] for entry in results) else 0


if __name__ == "__main__":
    sys.exit(main())
