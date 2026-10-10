"""
Database layer for the Football Predictor backend.

Defines the SQLAlchemy engine, the declarative ``Base`` and the ORM models that
back the PostgreSQL schema:

    matches      -- one row per played match, plus closing bookmaker odds
    elo_ratings  -- ClubElo rating history, one row per team per day

The connection string is read from the ``DATABASE_URL`` environment variable
(see ``backend/.env.example``). Call :func:`init_db` to create any tables that
do not exist yet, e.g. from a script or from the FastAPI startup hook.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    inspect,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Load backend/.env regardless of the current working directory. Real
# environment variables always win: load_dotenv() does not override them.
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# Used only when DATABASE_URL is not configured, so that importing this module
# can never fail. Matches the value shipped in .env.example.
DEFAULT_DATABASE_URL = "postgresql://user:password@localhost:5432/football_predictor"


def get_database_url() -> str:
    """Return the PostgreSQL connection string to use."""
    return os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


def get_engine_url() -> str:
    """
    Return the connection string with an explicit DBAPI driver.

    ``DATABASE_URL`` uses the plain ``postgresql://`` scheme, whose default
    driver changed in SQLAlchemy 2.1 (it now resolves to psycopg3, while this
    project ships ``psycopg2-binary``). Pinning the driver to whichever of the
    two is actually installed keeps the documented URL working on both
    SQLAlchemy 2.0 and 2.1+.
    """
    url = get_database_url()
    if not url.startswith("postgresql://"):
        return url

    for driver, module in (("psycopg2", "psycopg2"), ("psycopg", "psycopg")):
        try:
            if importlib.util.find_spec(module) is not None:
                return url.replace("postgresql://", f"postgresql+{driver}://", 1)
        except (ImportError, ValueError):
            continue
    return url  # no driver installed: let SQLAlchemy report it


# --------------------------------------------------------------------------- #
# Engine / session
# --------------------------------------------------------------------------- #

try:
    engine = create_engine(
        get_engine_url(),
        pool_pre_ping=True,  # transparently discard stale connections
        future=True,
    )
except Exception as exc:  # noqa: BLE001 - importing this module must never fail
    engine = None  # type: ignore[assignment]
    print(f"[database] ERROR: could not create the database engine: {exc}")
    print("[database] Install the drivers with: pip install -r requirements.txt")

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

# Declarative base that all models below inherit from.
Base = declarative_base()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope: commit on success, roll back on error."""
    if engine is None:
        raise RuntimeError(
            "No database engine is configured. Check DATABASE_URL and that the "
            "database dependencies are installed."
        )
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class Match(Base):
    """A single played football match with its result, stats and odds."""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)  # SERIAL
    match_id = Column(Text, unique=True, nullable=False)
    league = Column(Text, nullable=False)
    season = Column(Text, nullable=False)
    date = Column(Date, nullable=False)
    home_team = Column(Text, nullable=False)
    away_team = Column(Text, nullable=False)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    result = Column(Text)  # "H", "D" or "A"
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_ot = Column(Integer)
    away_shots_ot = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellow = Column(Integer)
    away_yellow = Column(Integer)
    home_red = Column(Integer)
    away_red = Column(Integer)
    odds_home = Column(Float)
    odds_draw = Column(Float)
    odds_away = Column(Float)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<Match {self.match_id} {self.home_team} "
            f"{self.home_goals}-{self.away_goals} {self.away_team}>"
        )


class EloRating(Base):
    """A ClubElo rating for one team on one date."""

    __tablename__ = "elo_ratings"
    __table_args__ = (
        UniqueConstraint("team", "date", name="uq_elo_ratings_team_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)  # SERIAL
    team = Column(Text, nullable=False)
    date = Column(Date, nullable=False)
    elo = Column(Float, nullable=False)
    league = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<EloRating {self.team} {self.date} {self.elo}>"


class DailyMatch(Base):
    """A daily fixture and its pre-match predictions."""

    __tablename__ = "daily_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)  # SERIAL
    external_id = Column(String, unique=True, nullable=False)
    match_date = Column(String)  # YYYY-MM-DD
    kickoff_time = Column(String)  # ISO UTC datetime
    home_team = Column(String)
    away_team = Column(String)
    league = Column(String)
    season = Column(String)
    status = Column(String)  # scheduled, in_play or finished
    pred_home_win = Column(Float, nullable=True)
    pred_draw = Column(Float, nullable=True)
    pred_away_win = Column(Float, nullable=True)
    pred_home_goals = Column(Float, nullable=True)
    pred_away_goals = Column(Float, nullable=True)
    pred_home_corners = Column(Float, nullable=True)
    pred_away_corners = Column(Float, nullable=True)
    prediction_computed_at = Column(String, nullable=True)


class FBrefTeamStats(Base):
    """Season-level team stats (shooting, discipline, goalkeeping) from FBref."""

    __tablename__ = "fbref_team_stats"
    __table_args__ = (
        UniqueConstraint("team", "league", "season", name="uq_fbref_team_stats_team_league_season"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)  # SERIAL
    team = Column(String, nullable=False)
    league = Column(String, nullable=False)
    season = Column(String, nullable=False)
    games = Column(Integer, nullable=True)

    # Shooting
    shots = Column(Float, nullable=True)
    shots_on_target = Column(Float, nullable=True)
    shot_accuracy = Column(Float, nullable=True)
    goals_per_shot = Column(Float, nullable=True)
    goals_per_sot = Column(Float, nullable=True)

    # Discipline
    yellow_cards = Column(Float, nullable=True)
    red_cards = Column(Float, nullable=True)
    fouls_committed = Column(Float, nullable=True)
    fouls_drawn = Column(Float, nullable=True)
    offsides = Column(Float, nullable=True)
    crosses = Column(Float, nullable=True)
    interceptions = Column(Float, nullable=True)
    tackles_won = Column(Float, nullable=True)
    pens_won = Column(Float, nullable=True)
    pens_conceded = Column(Float, nullable=True)

    # Goalkeeping
    sota = Column(Float, nullable=True)
    goals_against = Column(Float, nullable=True)
    saves = Column(Float, nullable=True)
    save_pct = Column(Float, nullable=True)
    clean_sheets = Column(Float, nullable=True)
    pk_faced = Column(Float, nullable=True)
    pk_saved = Column(Float, nullable=True)

    updated_at = Column(String, nullable=True)  # ISO UTC datetime

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<FBrefTeamStats {self.team} {self.league} {self.season}>"


# --------------------------------------------------------------------------- #
# Schema management
# --------------------------------------------------------------------------- #


def init_db(bind=None) -> bool:
    """
    Create every table that does not exist yet.

    Idempotent: ``Base.metadata.create_all`` skips tables that are already
    present. Returns ``True`` on success and ``False`` if the database could
    not be reached, so callers (e.g. the API startup hook) never crash.
    """
    target = engine if bind is None else bind

    if target is None:
        print("[database] ERROR: no database engine configured, skipping table creation.")
        return False

    try:
        Base.metadata.create_all(bind=target)
        tables = sorted(inspect(target).get_table_names())
        print(f"[database] Tables ready: {', '.join(tables) or 'none'}")
        return True
    except Exception as exc:  # noqa: BLE001 - report any driver/connection error
        print(f"[database] ERROR: could not create tables: {exc}")
        print("[database] Check that PostgreSQL is running and that DATABASE_URL is correct.")
        return False


__all__ = [
    "Base",
    "DailyMatch",
    "EloRating",
    "FBrefTeamStats",
    "Match",
    "SessionLocal",
    "engine",
    "get_database_url",
    "get_engine_url",
    "init_db",
    "session_scope",
]
