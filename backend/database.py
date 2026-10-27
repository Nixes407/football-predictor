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
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
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

logger = logging.getLogger(__name__)

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


class StatsBombMatchStats(Base):
    """
    Match-level stats written by ``data/statsbomb_collector.py`` and
    ``data/understat_collector.py``.

    The authoritative DDL is created by ``data/statsbomb_collector.py:
    ensure_tables`` (``CREATE TABLE IF NOT EXISTS statsbomb_match_stats``);
    this model mirrors it so ``Base.metadata.create_all`` can create the table
    on fresh test/local databases and so ``data/statsbomb_aggregator.py`` can
    document the columns it reads. The live production table may carry extra
    columns — the aggregator discovers columns at runtime and maps whichever
    are present.
    """

    __tablename__ = "statsbomb_match_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)  # SERIAL
    match_id = Column(Integer, unique=True, nullable=False)
    competition = Column(Text, nullable=True)
    season = Column(Text, nullable=True)
    date = Column(Text, nullable=True)  # YYYY-MM-DD
    home_team = Column(Text, nullable=True)
    away_team = Column(Text, nullable=True)
    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)
    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)
    home_shots = Column(Integer, nullable=True)
    away_shots = Column(Integer, nullable=True)
    home_shots_ot = Column(Integer, nullable=True)
    away_shots_ot = Column(Integer, nullable=True)
    home_pressures = Column(Integer, nullable=True)
    away_pressures = Column(Integer, nullable=True)
    home_prog_carr = Column(Integer, nullable=True)
    away_prog_carr = Column(Integer, nullable=True)
    home_gk_saves = Column(Integer, nullable=True)
    away_gk_saves = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<StatsBombMatchStats {self.match_id} {self.home_team} vs {self.away_team}>"


class MatchResult(Base):
    """A match result with scorers, cards and the prediction graded against it."""

    __tablename__ = "match_results"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_match_results_external_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)  # SERIAL
    external_id = Column(String, nullable=False)
    match_date = Column(String)  # YYYY-MM-DD
    kickoff_time = Column(String)  # ISO UTC datetime
    home_team = Column(String)
    away_team = Column(String)
    league = Column(String)
    season = Column(String)
    status = Column(String)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    result = Column(String, nullable=True)  # "H", "D" or "A"
    home_scorers = Column(String, nullable=True)
    away_scorers = Column(String, nullable=True)
    home_yellows = Column(Integer, nullable=True)
    away_yellows = Column(Integer, nullable=True)
    home_reds = Column(Integer, nullable=True)
    away_reds = Column(Integer, nullable=True)
    pred_home_win = Column(Float, nullable=True)
    pred_draw = Column(Float, nullable=True)
    pred_away_win = Column(Float, nullable=True)
    pred_correct = Column(Boolean, nullable=True)
    fetched_at = Column(String, nullable=True)


class TeamSeasonStats(Base):
    """
    One row per team / league / season.
    Stores all FBref advanced team statistics
    across all stat categories.
    Updated daily by the GitHub Actions workflow.
    """

    __tablename__ = "team_season_stats"
    __table_args__ = (
        UniqueConstraint(
            "team", "league", "season",
            name="uq_team_season_stats"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    team = Column(Text, nullable=False)
    league = Column(Text, nullable=False)
    season = Column(Text, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # --- SHOOTING ---
    sh_goals = Column(Float)  # Goals
    sh_shots = Column(Float)  # Shots total
    sh_sot = Column(Float)  # Shots on target
    sh_sot_pct = Column(Float)  # Shots on target %
    sh_sh_per90 = Column(Float)  # Shots per 90
    sh_sot_per90 = Column(Float)  # SoT per 90
    sh_g_per_sh = Column(Float)  # Goals per shot
    sh_g_per_sot = Column(Float)  # Goals per SoT
    sh_avg_dist = Column(Float)  # Avg shot distance
    sh_xg = Column(Float)  # xG
    sh_npxg = Column(Float)  # Non-penalty xG
    sh_npxg_per_sh = Column(Float)  # npxG per shot
    sh_g_minus_xg = Column(Float)  # Goals minus xG
    sh_npg_minus_npxg = Column(Float)  # np:G minus npxG

    # --- PASSING ---
    ps_cmp = Column(Float)  # Passes completed
    ps_att = Column(Float)  # Passes attempted
    ps_cmp_pct = Column(Float)  # Pass completion %
    ps_totdist = Column(Float)  # Total pass distance
    ps_prgdist = Column(Float)  # Progressive pass distance
    ps_cmp_short = Column(Float)  # Short passes completed
    ps_att_short = Column(Float)  # Short passes attempted
    ps_cmp_pct_short = Column(Float)  # Short pass completion %
    ps_cmp_medium = Column(Float)  # Medium passes completed
    ps_att_medium = Column(Float)  # Medium passes attempted
    ps_cmp_pct_medium = Column(Float)  # Medium pass completion %
    ps_cmp_long = Column(Float)  # Long passes completed
    ps_att_long = Column(Float)  # Long passes attempted
    ps_cmp_pct_long = Column(Float)  # Long pass completion %
    ps_ast = Column(Float)  # Assists
    ps_xag = Column(Float)  # xAG
    ps_xa = Column(Float)  # xA
    ps_kp = Column(Float)  # Key passes
    ps_final_third = Column(Float)  # Passes into final third
    ps_ppa = Column(Float)  # Passes into penalty area
    ps_crspa = Column(Float)  # Crosses into penalty area
    ps_prgp = Column(Float)  # Progressive passes

    # --- GOAL & SHOT CREATION ---
    gsc_sca = Column(Float)  # Shot-creating actions
    gsc_sca90 = Column(Float)  # SCA per 90
    gsc_sca_passlive = Column(Float)  # SCA from live passes
    gsc_sca_passdead = Column(Float)  # SCA from dead passes
    gsc_sca_to = Column(Float)  # SCA from take-ons
    gsc_sca_sh = Column(Float)  # SCA from shots
    gsc_sca_fld = Column(Float)  # SCA from fouls drawn
    gsc_sca_def = Column(Float)  # SCA from defensive actions
    gsc_gca = Column(Float)  # Goal-creating actions
    gsc_gca90 = Column(Float)  # GCA per 90
    gsc_gca_passlive = Column(Float)  # GCA from live passes
    gsc_gca_passdead = Column(Float)  # GCA from dead passes
    gsc_gca_to = Column(Float)  # GCA from take-ons
    gsc_gca_sh = Column(Float)  # GCA from shots
    gsc_gca_fld = Column(Float)  # GCA from fouls drawn
    gsc_gca_def = Column(Float)  # GCA from defensive actions

    # --- DEFENSIVE ACTIONS ---
    def_tkl = Column(Float)  # Tackles
    def_tklw = Column(Float)  # Tackles won
    def_tkl_def3rd = Column(Float)  # Tackles def third
    def_tkl_mid3rd = Column(Float)  # Tackles mid third
    def_tkl_att3rd = Column(Float)  # Tackles att third
    def_drib_tkl = Column(Float)  # Dribblers tackled
    def_drib_att = Column(Float)  # Dribbler tackle attempts
    def_drib_tkl_pct = Column(Float)  # Dribbler tackle %
    def_press = Column(Float)  # Pressures
    def_press_succ = Column(Float)  # Successful pressures
    def_press_pct = Column(Float)  # Pressure success %
    def_press_def3rd = Column(Float)  # Pressures def third
    def_press_mid3rd = Column(Float)  # Pressures mid third
    def_press_att3rd = Column(Float)  # Pressures att third
    def_blocks = Column(Float)  # Blocks
    def_blk_sh = Column(Float)  # Shots blocked
    def_blk_shsv = Column(Float)  # Shots on target blocked
    def_blk_pass = Column(Float)  # Passes blocked
    def_int = Column(Float)  # Interceptions
    def_tkl_int = Column(Float)  # Tackles + interceptions
    def_clr = Column(Float)  # Clearances
    def_err = Column(Float)  # Errors leading to shot

    # --- POSSESSION ---
    pos_poss = Column(Float)  # Possession %
    pos_touches = Column(Float)  # Touches
    pos_touch_defpen = Column(Float)  # Touches def pen area
    pos_touch_def3rd = Column(Float)  # Touches def third
    pos_touch_mid3rd = Column(Float)  # Touches mid third
    pos_touch_att3rd = Column(Float)  # Touches att third
    pos_touch_attpen = Column(Float)  # Touches att pen area
    pos_to_att = Column(Float)  # Take-ons attempted
    pos_to_succ = Column(Float)  # Take-ons successful
    pos_to_succ_pct = Column(Float)  # Take-on success %
    pos_to_tkld = Column(Float)  # Times tackled during take-on
    pos_carries = Column(Float)  # Carries
    pos_carry_totdist = Column(Float)  # Carry total distance
    pos_carry_prgdist = Column(Float)  # Carry progressive distance
    pos_prgc = Column(Float)  # Progressive carries
    pos_carry_final3rd = Column(Float)  # Carries into final third
    pos_cpa = Column(Float)  # Carries into pen area
    pos_mis = Column(Float)  # Miscontrols
    pos_dis = Column(Float)  # Dispossessed
    pos_rec = Column(Float)  # Passes received
    pos_prgr = Column(Float)  # Progressive passes received

    # --- MISCELLANEOUS ---
    misc_crdy = Column(Float)  # Yellow cards
    misc_crdr = Column(Float)  # Red cards
    misc_fls = Column(Float)  # Fouls committed
    misc_fld = Column(Float)  # Fouls drawn
    misc_off = Column(Float)  # Offsides
    misc_crs = Column(Float)  # Crosses
    misc_corners = Column(Float)  # Corners taken (season total)
    misc_tklw = Column(Float)  # Tackles won
    misc_pkwon = Column(Float)  # Penalties won
    misc_pkcon = Column(Float)  # Penalties conceded
    misc_og = Column(Float)  # Own goals
    misc_recov = Column(Float)  # Ball recoveries
    misc_aerial_won = Column(Float)  # Aerial duels won
    misc_aerial_lost = Column(Float)  # Aerial duels lost
    misc_aerial_pct = Column(Float)  # Aerial duel win %

    # --- GOALKEEPING ---
    gk_ga = Column(Float)  # Goals against
    gk_ga90 = Column(Float)  # GA per 90
    gk_sota = Column(Float)  # Shots on target against
    gk_saves = Column(Float)  # Saves
    gk_save_pct = Column(Float)  # Save %
    gk_w = Column(Float)  # Wins
    gk_d = Column(Float)  # Draws
    gk_l = Column(Float)  # Losses
    gk_cs = Column(Float)  # Clean sheets
    gk_cs_pct = Column(Float)  # Clean sheet %
    gk_pka = Column(Float)  # Penalties faced
    gk_pksvd = Column(Float)  # Penalties saved
    gk_pksv_pct = Column(Float)  # Penalty save %

    # --- ADVANCED GOALKEEPING ---
    gka_psxg = Column(Float)  # Post-shot xG
    gka_psxg_per_sot = Column(Float)  # PSxG per SoT
    gka_psxg_diff = Column(Float)  # PSxG+/- (GK quality)
    gka_psxg_diff90 = Column(Float)  # PSxG+/- per 90
    gka_launch_cmp = Column(Float)  # Launch passes completed
    gka_launch_att = Column(Float)  # Launch passes attempted
    gka_launch_pct = Column(Float)  # Launch pass completion %
    gka_pass_att = Column(Float)  # Pass attempts (GK)
    gka_pass_launch = Column(Float)  # Launch % (passes)
    gka_pass_avglen = Column(Float)  # Avg pass length
    gka_gkick_att = Column(Float)  # Goal kick attempts
    gka_gkick_launch = Column(Float)  # Goal kick launch %
    gka_gkick_avglen = Column(Float)  # Avg goal kick length
    gka_cross_opp = Column(Float)  # Crosses faced
    gka_cross_stp = Column(Float)  # Crosses stopped
    gka_cross_stp_pct = Column(Float)  # Cross stop %
    gka_opa = Column(Float)  # Sweeper actions
    gka_opa90 = Column(Float)  # Sweeper actions per 90
    gka_avgdist = Column(Float)  # Avg distance from goal

    def __repr__(self):
        return f"<TeamSeasonStats {self.team} {self.league} {self.season}>"


class PlayerSeasonStats(Base):
    """
    One row per player / team / league / season.
    Stores FBref advanced player statistics.
    Updated daily by the GitHub Actions workflow.
    """

    __tablename__ = "player_season_stats"
    __table_args__ = (
        UniqueConstraint(
            "player", "team", "league", "season",
            name="uq_player_season_stats"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    player = Column(Text, nullable=False)
    team = Column(Text, nullable=False)
    league = Column(Text, nullable=False)
    season = Column(Text, nullable=False)
    position = Column(Text)
    age = Column(Integer)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # --- STANDARD ---
    mp = Column(Integer)  # Matches played
    starts = Column(Integer)  # Starts
    minutes = Column(Float)  # Minutes played
    minutes_per90 = Column(Float)  # 90s played
    goals = Column(Float)  # Goals
    assists = Column(Float)  # Assists
    g_plus_a = Column(Float)  # Goals + assists
    g_minus_pk = Column(Float)  # Non-penalty goals
    pk = Column(Float)  # Penalties scored
    pkatt = Column(Float)  # Penalties attempted
    crdy = Column(Float)  # Yellow cards
    crdr = Column(Float)  # Red cards
    xg = Column(Float)  # xG
    npxg = Column(Float)  # Non-penalty xG
    xag = Column(Float)  # xAG
    npxg_plus_xag = Column(Float)  # npxG + xAG
    prgc = Column(Float)  # Progressive carries
    prgp = Column(Float)  # Progressive passes
    prgr = Column(Float)  # Progressive passes received

    # --- PER 90 ---
    gls_per90 = Column(Float)  # Goals per 90
    ast_per90 = Column(Float)  # Assists per 90
    g_plus_a_per90 = Column(Float)  # G+A per 90
    g_minus_pk_per90 = Column(Float)  # Non-PK goals per 90
    xg_per90 = Column(Float)  # xG per 90
    xag_per90 = Column(Float)  # xAG per 90
    xg_plus_xag_per90 = Column(Float)  # xG+xAG per 90
    npxg_per90 = Column(Float)  # npxG per 90
    npxg_plus_xag_per90 = Column(Float)  # npxG+xAG per 90

    # --- SHOOTING ---
    sh_shots = Column(Float)  # Shots
    sh_sot = Column(Float)  # Shots on target
    sh_sot_pct = Column(Float)  # SoT %
    sh_sh_per90 = Column(Float)  # Shots per 90
    sh_sot_per90 = Column(Float)  # SoT per 90
    sh_g_per_sh = Column(Float)  # Goals per shot
    sh_g_per_sot = Column(Float)  # Goals per SoT
    sh_avg_dist = Column(Float)  # Avg shot distance
    sh_xg = Column(Float)  # xG
    sh_npxg = Column(Float)  # npxG
    sh_g_minus_xg = Column(Float)  # Goals minus xG

    # --- GOAL & SHOT CREATION ---
    sca = Column(Float)  # Shot-creating actions
    sca90 = Column(Float)  # SCA per 90
    gca = Column(Float)  # Goal-creating actions
    gca90 = Column(Float)  # GCA per 90

    # --- DEFENSIVE ---
    tkl = Column(Float)  # Tackles
    tklw = Column(Float)  # Tackles won
    press = Column(Float)  # Pressures
    press_succ = Column(Float)  # Successful pressures
    blocks = Column(Float)  # Blocks
    int_ = Column(Float)  # Interceptions
    tkl_plus_int = Column(Float)  # Tackles + interceptions
    clr = Column(Float)  # Clearances
    err = Column(Float)  # Errors

    # --- POSSESSION ---
    touches = Column(Float)  # Touches
    touch_att3rd = Column(Float)  # Touches att third
    touch_attpen = Column(Float)  # Touches att pen area
    carries = Column(Float)  # Carries
    prgc_carries = Column(Float)  # Progressive carries
    cpa = Column(Float)  # Carries into pen area
    rec = Column(Float)  # Passes received

    # --- MISCELLANEOUS ---
    fls = Column(Float)  # Fouls committed
    fld = Column(Float)  # Fouls drawn
    off = Column(Float)  # Offsides
    crs = Column(Float)  # Crosses
    pkwon = Column(Float)  # Penalties won
    pkcon = Column(Float)  # Penalties conceded
    aerial_won = Column(Float)  # Aerial duels won
    aerial_lost = Column(Float)  # Aerial duels lost
    aerial_pct = Column(Float)  # Aerial win %

    def __repr__(self):
        return (
            f"<PlayerSeasonStats {self.player} "
            f"{self.team} {self.season}>"
        )


# --------------------------------------------------------------------------- #
# League guard / one-time cleanup
# --------------------------------------------------------------------------- #

#: Leagues that must never reach ``team_season_stats``. The Belgian Jupiler
#: Pro League was never a supported competition: a bad BSA -> BEL fixture
#: mapping wrote rows under that label and they were never purged.
BLOCKED_LEAGUES = {"BEL-Jupiler Pro League"}


def is_blocked_league(league: object) -> bool:
    """True when ``league`` is a blocked label (or any Jupiler spelling)."""
    if not isinstance(league, str):
        return False
    cleaned = " ".join(league.split()).casefold()
    if cleaned in {name.casefold() for name in BLOCKED_LEAGUES}:
        return True
    # Catch stale variants such as "Jupiler Pro League" / "BEL-Jupiler".
    return "jupiler" in cleaned


def purge_bel_jupiler_rows(bind=None) -> int:
    """One-time cleanup: remove any BEL-Jupiler Pro League rows from team_season_stats.

    Safe to run on every start: after the first clean run it deletes 0 rows and
    becomes a no-op. Only ever deletes the blocked league; other rows and all
    stat columns are untouched.
    """
    target = engine if bind is None else bind
    if target is None:
        return 0
    try:
        if "team_season_stats" not in inspect(target).get_table_names():
            return 0
        with target.begin() as connection:
            result = connection.exec_driver_sql(
                "DELETE FROM team_season_stats "
                "WHERE league = 'BEL-Jupiler Pro League'"
            )
            deleted = int(result.rowcount or 0)
        if deleted > 0:
            logger.info(
                "Purged %d BEL-Jupiler Pro League rows from team_season_stats",
                deleted,
            )
        return deleted
    except Exception as exc:  # noqa: BLE001 - startup must not fail on cleanup
        logger.warning("Could not purge BEL-Jupiler Pro League rows: %s", exc)
        return 0


def upsert_team_season_stats(
    db,
    team: str,
    league: str,
    season: str,
    attrs: dict,
) -> bool:
    """Guarded upsert: the single write choke point for ``team_season_stats``.

    Every collector routes its team rows through here. A blocked league is
    rejected *before* any session/query/cursor work, so no Belgian Jupiler row
    can be created or updated. ``None`` values in ``attrs`` never clear an
    existing non-null column (a feed that lacks a stat must not erase the one
    another collector stored). Returns True when a row was written.
    """
    if is_blocked_league(league):
        logger.warning("Blocked attempt to upsert %s into team_season_stats", league)
        return False

    existing = (
        db.query(TeamSeasonStats)
        .filter(
            TeamSeasonStats.team == team,
            TeamSeasonStats.league == league,
            TeamSeasonStats.season == season,
        )
        .first()
    )
    if existing is None:
        existing = TeamSeasonStats(team=team, league=league, season=season)
        db.add(existing)

    columns = TeamSeasonStats.__table__.columns
    for key, value in (attrs or {}).items():
        if value is None:
            continue
        if key not in columns:
            # SQLAlchemy allows arbitrary Python attributes; silently keeping
            # one would lose the value on commit.
            logger.warning("Ignoring unknown TeamSeasonStats field %r", key)
            continue
        setattr(existing, key, value)
    return True


# --------------------------------------------------------------------------- #
# Schema management
# --------------------------------------------------------------------------- #


def ensure_fbref_team_stats_columns(bind=None) -> bool:
    """Add any missing FBrefTeamStats columns to an existing table safely.

    ``create_all`` creates new tables with every mapped column, whereas an
    already-deployed ``fbref_team_stats`` table only receives newer columns via
    additive ``ALTER TABLE ... ADD COLUMN``. Existing columns and stored data
    are never dropped or altered.
    """
    target = engine if bind is None else bind
    if target is None:
        print("[database] ERROR: no database engine configured, cannot ensure fbref_team_stats columns.")
        return False
    try:
        inspector = inspect(target)
        if "fbref_team_stats" not in inspector.get_table_names():
            return True
        existing_cols = {
            column["name"] for column in inspector.get_columns("fbref_team_stats")
        }
        dialect = target.dialect.name
        added: list[str] = []
        for col in FBrefTeamStats.__table__.columns:
            if col.name == "id" or col.name in existing_cols:
                continue
            is_string = isinstance(col.type, String)
            if dialect == "postgresql":
                sql_type = "VARCHAR" if is_string else "DOUBLE PRECISION"
                statement = (
                    f"ALTER TABLE fbref_team_stats "
                    f"ADD COLUMN IF NOT EXISTS {col.name} {sql_type}"
                )
            else:
                sql_type = "VARCHAR" if is_string else "FLOAT"
                statement = (
                    f"ALTER TABLE fbref_team_stats "
                    f"ADD COLUMN {col.name} {sql_type}"
                )
            with target.begin() as connection:
                connection.exec_driver_sql(statement)
            added.append(col.name)
        if added:
            print(f"[database] Added fbref_team_stats columns: {', '.join(added)}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[database] ERROR: could not ensure fbref_team_stats columns: {exc}")
        return False


def ensure_team_corners_column(bind=None) -> bool:
    """Add the season-total corners column to an existing table safely.

    ``create_all`` cannot evolve existing tables. Inspection makes this
    idempotent on every dialect; PostgreSQL additionally uses IF NOT EXISTS
    to remain race-safe when API startup and the daily collector overlap.
    """
    target = engine if bind is None else bind
    if target is None:
        print("[database] ERROR: no database engine configured, cannot ensure misc_corners.")
        return False
    try:
        inspector = inspect(target)
        if "team_season_stats" not in inspector.get_table_names():
            print("[database] misc_corners will be created with team_season_stats")
            return True
        columns = {column["name"] for column in inspector.get_columns("team_season_stats")}
        if "misc_corners" in columns:
            print("[database] team_season_stats.misc_corners already present")
            return True
        dialect = target.dialect.name
        statement = (
            "ALTER TABLE team_season_stats ADD COLUMN IF NOT EXISTS "
            "misc_corners DOUBLE PRECISION"
            if dialect == "postgresql"
            else "ALTER TABLE team_season_stats ADD COLUMN misc_corners FLOAT"
        )
        with target.begin() as connection:
            connection.exec_driver_sql(statement)
        print("[database] Added team_season_stats.misc_corners")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[database] ERROR: could not ensure misc_corners: {exc}")
        return False


def cleanup_non_team_rows(bind=None) -> int:
    """Delete leaked player leaderboard rows and empty shells from team_season_stats.

    FotMob's legacy ``teamOrPlayer="team"`` fallback on
    ``/api/data/leagueseasondeepstats`` returned player leaderboards for bare
    stat names (``total_scoring_att``, ``accurate_cross``, ...), inserting
    hundreds of player rows per league/season into ``team_season_stats``.
    Those rows carry neither league-standings columns (``gk_ga``, ``gk_w``,
    ``gk_d``, ``gk_l``), team possession (``pos_poss``), nor team-only
    FBref/Understat/StatsBomb columns. Deleting them at startup restores
    ``team_season_stats`` to actual clubs only.
    """
    target = engine if bind is None else bind
    if target is None:
        return 0
    try:
        inspector = inspect(target)
        if "team_season_stats" not in inspector.get_table_names():
            return 0
        statement = """
            DELETE FROM team_season_stats
            WHERE pos_poss IS NULL
              AND gk_ga IS NULL
              AND gk_ga90 IS NULL
              AND gk_w IS NULL
              AND gk_d IS NULL
              AND gk_l IS NULL
              AND sh_npxg IS NULL
              AND sh_sh_per90 IS NULL
              AND sh_sot_per90 IS NULL
              AND ps_prgp IS NULL
              AND pos_prgc IS NULL
              AND gsc_gca IS NULL
              AND gsc_gca90 IS NULL
              AND def_press IS NULL
              AND def_tkl_int IS NULL
              AND pos_touch_att3rd IS NULL
        """
        with target.begin() as connection:
            result = connection.exec_driver_sql(statement)
            deleted = int(result.rowcount or 0)
        if deleted > 0:
            print(f"[database] Removed {deleted} non-team/player rows from team_season_stats")
        return deleted
    except Exception as exc:  # noqa: BLE001
        print(f"[database] WARNING: could not clean non-team rows: {exc}")
        return 0


def init_db(bind=None) -> bool:
    """Create tables and safely apply the small additive schema evolution."""
    target = engine if bind is None else bind

    if target is None:
        print("[database] ERROR: no database engine configured, skipping table creation.")
        return False

    try:
        Base.metadata.create_all(bind=target)
        if not ensure_fbref_team_stats_columns(target):
            return False
        if not ensure_team_corners_column(target):
            return False
        cleanup_non_team_rows(target)
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
    "MatchResult",
    "PlayerSeasonStats",
    "SessionLocal",
    "StatsBombMatchStats",
    "TeamSeasonStats",
    "BLOCKED_LEAGUES",
    "cleanup_non_team_rows",
    "is_blocked_league",
    "purge_bel_jupiler_rows",
    "upsert_team_season_stats",
    "engine",
    "ensure_fbref_team_stats_columns",
    "ensure_team_corners_column",
    "get_database_url",
    "get_engine_url",
    "init_db",
    "session_scope",
]
