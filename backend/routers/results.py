"""Endpoints for finished football match results and prediction accuracy.

Finished results are fetched from football-data.org, stored in the
``match_results`` table (graded against the pre-match predictions saved in
``daily_matches``) and served from the database.
"""

import os
import traceback
from datetime import date, datetime, timedelta, timezone

import requests
from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from data.fixtures import LEAGUE_MAP, derive_season, map_team_name
from database import SessionLocal, MatchResult, DailyMatch


router = APIRouter()

# Inverse of LEAGUE_MAP, e.g. "ENG-Premier League" -> "PL".
LEAGUE_CODE_MAP = {league: code for code, league in LEAGUE_MAP.items()}


def fetch_results(target_date: str) -> list[dict]:
    """Fetch and normalize finished match results for ``target_date`` (UTC)."""
    results = []
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print(
            "[results] WARNING: FOOTBALL_DATA_API_KEY is not set; "
            "skipping results fetch."
        )
        return results

    headers = {"X-Auth-Token": api_key}
    for code in LEAGUE_CODE_MAP.values():
        url = f"https://api.football-data.org/v4/competitions/{code}/matches"
        try:
            response = requests.get(
                url,
                params={"dateFrom": target_date, "dateTo": target_date},
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            matches = response.json().get("matches", [])
            if not isinstance(matches, list):
                raise ValueError("response 'matches' value is not a list")
        except Exception as exc:
            print(f"[results] WARNING: failed to fetch league {code}: {exc}")
            continue

        for match in matches:
            try:
                if match["status"] != "FINISHED":
                    continue

                utc_date = match["utcDate"]
                season_info = match.get("season")
                season_start = (
                    season_info.get("startDate")
                    if isinstance(season_info, dict) else None
                )
                full_time = match["score"]["fullTime"]
                home_score = full_time["home"]
                away_score = full_time["away"]
                if home_score is None or away_score is None:
                    raise ValueError("finished match has no full-time score")

                if home_score > away_score:
                    outcome = "H"
                elif away_score > home_score:
                    outcome = "A"
                else:
                    outcome = "D"

                results.append(
                    {
                        "external_id": str(match["id"]),
                        "match_date": target_date,
                        "kickoff_time": utc_date,
                        "home_team": map_team_name(match["homeTeam"]["name"]),
                        "away_team": map_team_name(match["awayTeam"]["name"]),
                        "league": LEAGUE_MAP[code],
                        "season": derive_season(utc_date, LEAGUE_MAP[code], season_start),
                        "status": "finished",
                        "home_score": home_score,
                        "away_score": away_score,
                        "result": outcome,
                        "home_scorers": "",
                        "away_scorers": "",
                        "home_yellows": None,
                        "away_yellows": None,
                        "home_reds": None,
                        "away_reds": None,
                    }
                )
            except Exception as exc:
                print(
                    f"[results] WARNING: could not parse a match in league "
                    f"{code}: {exc}"
                )

    return results


def _prediction_correct(outcome, home_win, draw, away_win):
    """Grade a stored prediction: did its most likely outcome happen?

    The prediction is correct when the probability of the actual outcome
    (``"H"``, ``"D"`` or ``"A"``) equals the highest of the three, so a tie for
    the top spot counts as correct. Returns ``None`` when the match cannot be
    graded (unknown outcome or no stored prediction), which keeps it out of the
    accuracy stats instead of counting it as a miss.
    """
    if outcome not in ("H", "D", "A") or None in (home_win, draw, away_win):
        return None
    probabilities = {"H": home_win, "D": draw, "A": away_win}
    return probabilities[outcome] == max(probabilities.values())


def store_results(results: list[dict], db: Session) -> None:
    """Insert or update finished results and grade their stored predictions."""
    for item in results:
        home_team = item.get("home_team", "Unknown")
        away_team = item.get("away_team", "Unknown")
        try:
            external_id = item.get("external_id")
            if external_id is None:
                raise ValueError("result has no external_id")
            external_id = str(external_id)

            row = (
                db.query(MatchResult)
                .filter(MatchResult.external_id == external_id)
                .first()
            )
            if row is None:
                row = MatchResult(
                    external_id=external_id,
                    match_date=item.get("match_date"),
                    kickoff_time=item.get("kickoff_time"),
                    home_team=item.get("home_team"),
                    away_team=item.get("away_team"),
                    league=item.get("league"),
                    season=item.get("season"),
                    status=item.get("status"),
                    home_score=item.get("home_score"),
                    away_score=item.get("away_score"),
                    result=item.get("result"),
                    home_scorers=item.get("home_scorers"),
                    away_scorers=item.get("away_scorers"),
                    home_yellows=item.get("home_yellows"),
                    away_yellows=item.get("away_yellows"),
                    home_reds=item.get("home_reds"),
                    away_reds=item.get("away_reds"),
                )
            else:
                row.status = item.get("status")
                row.home_score = item.get("home_score")
                row.away_score = item.get("away_score")
                row.result = item.get("result")

            # Carry over the pre-match prediction for the same fixture, if any.
            daily = (
                db.query(DailyMatch)
                .filter(DailyMatch.external_id == external_id)
                .first()
            )
            if daily is not None:
                row.pred_home_win = daily.pred_home_win
                row.pred_draw = daily.pred_draw
                row.pred_away_win = daily.pred_away_win
            row.pred_correct = _prediction_correct(
                row.result,
                row.pred_home_win,
                row.pred_draw,
                row.pred_away_win,
            )

            row.fetched_at = datetime.now(timezone.utc).isoformat()
            db.add(row)
            db.commit()
        except Exception as exc:
            db.rollback()
            print(
                f"[results] WARNING: could not store {home_team} vs "
                f"{away_team}: {exc}"
            )
            traceback.print_exc()
            continue


def row_to_dict(row: MatchResult) -> dict:
    """Serialize every column of a MatchResult row."""
    return {
        column.name: getattr(row, column.name) for column in row.__table__.columns
    }


def _stored_results(target_date: str, db: Session) -> list[MatchResult]:
    """Return the stored results for a date in kickoff order."""
    return (
        db.query(MatchResult)
        .filter(MatchResult.match_date == target_date)
        .order_by(MatchResult.kickoff_time, MatchResult.id)
        .all()
    )


def _get_or_fetch_results(target_date: str, db: Session) -> list[dict]:
    """Return stored results for a date, fetching and storing them if needed."""
    rows = _stored_results(target_date, db)
    if not rows:
        store_results(fetch_results(target_date), db)
        rows = _stored_results(target_date, db)
    return [row_to_dict(row) for row in rows]


def _close_session(db) -> None:
    if db is not None:
        try:
            db.close()
        except Exception as exc:
            print(f"[results] WARNING: could not close DB session: {exc}")


def _results_for_date(target_date: str) -> list[dict]:
    """Load the results for ``target_date``; any error is logged and gives []."""
    db = None
    try:
        parsed_date = date.fromisoformat(target_date)
        if parsed_date.isoformat() != target_date:
            raise ValueError("date must be in YYYY-MM-DD format")
        db = SessionLocal()
        return _get_or_fetch_results(target_date, db)
    except Exception as exc:
        print(f"[results] WARNING: could not load results for {target_date}: {exc}")
        traceback.print_exc()
        return []
    finally:
        _close_session(db)


@router.get("/date")
def get_results_for_date(
    target_date: str = Query(
        ..., alias="date", description="Date in YYYY-MM-DD format"
    ),
) -> list[dict]:
    """Return finished results for the requested date, fetching them if not cached."""
    return _results_for_date(target_date)


@router.get("/today")
def get_today_results() -> list[dict]:
    """Return today's UTC results, fetching them if they are not cached."""
    return _results_for_date(datetime.now(timezone.utc).date().isoformat())


@router.get("/yesterday")
def get_yesterday_results() -> list[dict]:
    """Return yesterday's UTC results, fetching them if they are not cached."""
    # Derived from the same UTC clock as /today so the two can never overlap.
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    return _results_for_date(yesterday)


def _accuracy_pct(correct: int, total: int) -> float:
    return round(correct / total * 100, 1) if total else 0.0


def _accuracy_summary(graded: list) -> dict:
    """Build the accuracy response from ``(league, pred_correct)`` pairs."""
    by_league = {}
    for league, is_correct in graded:
        stats = by_league.setdefault(league or "Unknown", {"total": 0, "correct": 0})
        stats["total"] += 1
        stats["correct"] += 1 if is_correct else 0

    total = len(graded)
    correct = sum(1 for _, is_correct in graded if is_correct)
    return {
        "total_predicted": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy_pct": _accuracy_pct(correct, total),
        "by_league": {
            league: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy_pct": _accuracy_pct(stats["correct"], stats["total"]),
            }
            for league, stats in sorted(by_league.items())
        },
    }


@router.get("/accuracy")
def get_accuracy() -> dict:
    """Return overall and per-league accuracy of the graded predictions."""
    db = None
    try:
        db = SessionLocal()
        graded = (
            db.query(MatchResult.league, MatchResult.pred_correct)
            .filter(MatchResult.pred_correct.is_not(None))
            .all()
        )
        return _accuracy_summary(graded)
    except Exception as exc:
        print(f"[results] WARNING: could not compute prediction accuracy: {exc}")
        traceback.print_exc()
        return _accuracy_summary([])
    finally:
        _close_session(db)
