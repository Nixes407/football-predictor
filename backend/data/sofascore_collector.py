"""
Sofascore advanced stats collector — PRIMARY data source.

Collects team stats and player stats from Sofascore via soccerdata.

soccerdata's Sofascore reader only covers leagues, seasons, tables and
schedules — it exposes no stats readers of its own — so this module
resolves league/season IDs through it and fetches the Sofascore
statistics API with its cached session.

Coverage: 9 leagues, seasons 2025-2026 and 2026-2027 (latest season
only for players). UEFA Champions League is not stored here:
team_season_stats and player_season_stats are domestic-only, matching
api_football_collector ("UCL skipped: not stored").

League averages are derived from these rows by models.formula_engine
(load_league_averages) rather than collected separately.

Fallback sources (API-Football, Understat) fill the same tables when
this source fails — see api_football_collector.py and
understat_collector.py.

Run from backend/ directory:
    python data/sofascore_collector.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import pandas as pd
import soccerdata as sd

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, TeamSeasonStats, PlayerSeasonStats
from sqlalchemy import and_


# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

LEAGUES = [
    "ENG-Premier League",
    "ENG-Championship",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "POR-Primeira Liga",
    "NED-Eredivisie",
    "USA-MLS",
]


SEASONS     = ["2025-2026", "2026-2027"]
MLS_SEASONS = ["2025", "2026"]

# Sofascore API base — the same host soccerdata talks to.
SOFASCORE_API = "https://api.sofascore.com/api/v1/"

# soccerdata's league registry only knows the Big 5 leagues plus a few
# internationals and rejects everything else in its constructor. These
# entries register the remaining leagues with their Sofascore tournament
# names — what a ~/.soccerdata/config/league_dict.json would do, but
# self-contained so the collector runs without extra environment setup.
_SOFASCORE_LEAGUE_PATCH = {
    "ENG-Championship": {
        "Sofascore": "Championship",
        "season_start": "Aug",
        "season_end": "May",
    },
    "POR-Primeira Liga": {
        "Sofascore": "Primeira Liga",
        "season_start": "Aug",
        "season_end": "May",
    },
    "NED-Eredivisie": {
        "Sofascore": "Eredivisie",
        "season_start": "Aug",
        "season_end": "May",
    },
    "USA-MLS": {
        "Sofascore": "Major League Soccer",
        "season_start": "Mar",
        "season_end": "Nov",
        "season_code": "single-year",
    },
}


def _register_leagues() -> None:
    """Merge the missing league entries into soccerdata's registry."""
    try:
        from soccerdata._config import LEAGUE_DICT
    except Exception as exc:  # pragma: no cover - soccerdata layout change
        print(f"[sofascore] WARNING: cannot patch league dict: {exc}")
        return

    for key, entry in _SOFASCORE_LEAGUE_PATCH.items():
        current = LEAGUE_DICT.get(key)
        if current is None:
            LEAGUE_DICT[key] = dict(entry)
        elif "Sofascore" not in current:
            current.update(entry)

    # _all_leagues caches a filtered view — drop it so the patch takes
    # effect even if another collector already ran in this process.
    sofa = getattr(sd, "Sofascore", None)
    if sofa is not None:
        try:
            delattr(sofa, "_all_leagues_dict")
        except AttributeError:
            pass


_register_leagues()


try:
    # Apply soccerdata's own team-name translation (normally empty) so
    # standings names match the schedule names read below.
    from soccerdata._config import TEAMNAME_REPLACEMENTS
except Exception:  # pragma: no cover - soccerdata layout change
    TEAMNAME_REPLACEMENTS = {}


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

def _safe(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _get_seasons(league: str) -> list[str]:
    return MLS_SEASONS if league == "USA-MLS" else SEASONS


def _upsert_team(db, team, league, season, attrs):
    """Upsert team stats. Never overwrites with None."""
    existing = (
        db.query(TeamSeasonStats)
        .filter(and_(
            TeamSeasonStats.team   == team,
            TeamSeasonStats.league == league,
            TeamSeasonStats.season == season,
        )).first()
    )
    if existing is None:
        existing = TeamSeasonStats(
            team=team, league=league, season=season)
        db.add(existing)
    for k, v in attrs.items():
        if v is not None:
            setattr(existing, k, v)


def _upsert_player(db, player, team, league, season, attrs):
    """Upsert player stats. Never overwrites with None."""
    existing = (
        db.query(PlayerSeasonStats)
        .filter(and_(
            PlayerSeasonStats.player == player,
            PlayerSeasonStats.team   == team,
            PlayerSeasonStats.league == league,
            PlayerSeasonStats.season == season,
        )).first()
    )
    if existing is None:
        existing = PlayerSeasonStats(
            player=player, team=team,
            league=league, season=season)
        db.add(existing)
    for k, v in attrs.items():
        if v is not None:
            setattr(existing, k, v)


def _get_scraper(league: str, season: str):
    """Create Sofascore scraper with warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sd.Sofascore(
            leagues=[league],
            seasons=[season],
        )


def _find_col(df: pd.DataFrame,
              candidates: list[str]) -> str | None:
    """Find first matching column name."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _series_sum(group: pd.DataFrame,
                candidates: list[str]) -> float | None:
    """Sum the first matching column across all rows."""
    col = _find_col(group, candidates)
    if col is None:
        return None
    s = pd.to_numeric(group[col], errors="coerce")
    total = s.sum()
    return float(total) if pd.notna(total) else None


def _series_mean(group: pd.DataFrame,
                 candidates: list[str]) -> float | None:
    """Average the first matching column across all rows."""
    col = _find_col(group, candidates)
    if col is None:
        return None
    s = pd.to_numeric(group[col], errors="coerce")
    mean = s.mean()
    return float(mean) if pd.notna(mean) else None


def _group_games(group: pd.DataFrame) -> int:
    """Games covered by a team group.

    Season-aggregate rows carry a games/matches column; match-level
    sources produce one row per game, so the row count is the fallback.
    """
    col = _find_col(group, ["games", "matches", "matches_played", "MP"])
    if col is not None:
        vals = pd.to_numeric(group[col], errors="coerce").dropna()
        vals = vals[vals > 0]
        if len(vals):
            return int(round(float(vals.mean())))
    return len(group)


# ═══════════════════════════════════════════
# SOFASCORE API READERS
# ═══════════════════════════════════════════
#
# sd.Sofascore has no read_team_match_stats / read_player_season_stats
# (those exist on the FBref/Understat readers), so team season totals
# and per-player match stats are read from the Sofascore statistics API
# and returned as DataFrames in the column layout the extractors below
# expect.

def _api_get(scraper, path: str, cache_name: str,
             max_age: int | None = None) -> dict:
    """GET one Sofascore API path through soccerdata's cached session."""
    filepath = scraper.data_dir / "matches" / cache_name
    filepath.parent.mkdir(parents=True, exist_ok=True)
    reader = scraper.get(SOFASCORE_API + path, filepath, max_age=max_age)
    return json.load(reader)


def _resolve_season_ids(scraper):
    """(league_id, season_id, league_key, season_key) for the selection."""
    try:
        seasons = scraper.read_seasons()
    except Exception as exc:
        print(f"[sofascore] WARNING: cannot resolve seasons: {exc}")
        return None
    if seasons is None or seasons.empty:
        print(f"[sofascore] WARNING: no season entry for "
              f"{scraper.leagues} {scraper.seasons}")
        return None
    row = seasons.reset_index().iloc[0]
    return (
        int(row["league_id"]),
        int(row["season_id"]),
        row["league"],
        row["season"],
    )


def _max_age_for(scraper, league_key, season_key) -> int | None:
    """Cache completed seasons forever, refresh current ones daily."""
    try:
        complete = scraper._is_complete(league_key, season_key)
    except Exception:
        complete = False
    return None if complete else 1


# Sofa labels positions G/D/M/F; the prediction engine matches the
# FBref codes (formula_engine._position_groups), like
# api_football_collector does.
_POSITION_MAP = {
    "G": "GK", "D": "DF", "M": "MF", "F": "FW",
    "GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW",
}


def _norm_position(value) -> str | None:
    """FBref-style position code for a Sofascore position label."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    return _POSITION_MAP.get(raw.upper(), raw)


def _read_team_season_stats(scraper) -> pd.DataFrame:
    """
    Fetch team season totals from the Sofascore statistics API.

    Returns one row per team with the column names _extract_team_attrs
    looks for (xg, shots, shots_on_target, ...). Empty when the league
    or season cannot be resolved.
    """
    resolved = _resolve_season_ids(scraper)
    if resolved is None:
        return pd.DataFrame()
    league_id, season_id, league_key, season_key = resolved
    max_age = _max_age_for(scraper, league_key, season_key)

    teams: list[tuple[int, str]] = []
    try:
        standings = _api_get(
            scraper,
            f"unique-tournament/{league_id}/season/{season_id}"
            f"/standings/total",
            f"standings_{league_id}_{season_id}.json",
            max_age=max_age,
        )
        for group in standings.get("standings") or []:
            for row in group.get("rows") or []:
                team = row.get("team") or {}
                tid, name = team.get("id"), team.get("name")
                if tid and name:
                    teams.append((int(tid), str(name)))
    except Exception as exc:
        print(f"[sofascore] WARNING: no standings for "
              f"{league_key} {season_key}: {exc}")
        return pd.DataFrame()

    rows = []
    for team_id, name in teams:
        try:
            payload = _api_get(
                scraper,
                f"team/{team_id}/unique-tournament/{league_id}"
                f"/season/{season_id}/statistics/overall",
                f"team_stats_{team_id}_{league_id}_{season_id}.json",
                max_age=max_age,
            )
        except Exception as exc:
            print(f"[sofascore] WARNING: no stats for {name}: {exc}")
            continue
        stats = (payload or {}).get("statistics") or {}
        if not stats:
            continue
        rows.append({
            "team": TEAMNAME_REPLACEMENTS.get(name, name),
            "games": _safe(stats.get("matches")),
            "xg": _safe(stats.get("expectedGoals")),
            "shots": _safe(stats.get("shots")),
            "shots_on_target": _safe(stats.get("shotsOnTarget")),
            "goals": _safe(stats.get("goalsScored")),
            "possession": _safe(stats.get("averageBallPossession")),
            "goals_conceded": _safe(stats.get("goalsConceded")),
            "saves": _safe(stats.get("saves")),
            "xa": _safe(stats.get("expectedAssists")),
            "big_chances_created": _safe(stats.get("bigChancesCreated")),
            "tackles": _safe(stats.get("tackles")),
            "interceptions": _safe(stats.get("interceptions")),
            "fouls": _safe(stats.get("fouls")),
            "yellow_cards": _safe(stats.get("yellowCards")),
            "passes": _safe(stats.get("totalPasses")),
        })
    return pd.DataFrame(rows)


def _accumulate_player(rec: dict, stats: dict,
                       field: str, *candidates: str) -> None:
    """Add the first available Sofascore stat to a per-player total."""
    for key in candidates:
        if key in stats:
            value = _safe(stats[key])
            if value is not None:
                rec[field] = (rec[field] or 0.0) + value
            return


def _read_player_season_stats(scraper) -> pd.DataFrame:
    """
    Aggregate player season stats from per-match lineup data.

    Returns one row per player/team with the column names
    _extract_player_attrs looks for. The Sofascore lineups endpoint
    carries each player's match statistics (minutesPlayed, goals,
    goalAssist, totalShots, ...).
    """
    resolved = _resolve_season_ids(scraper)
    if resolved is None:
        return pd.DataFrame()
    _, _, league_key, season_key = resolved
    max_age = _max_age_for(scraper, league_key, season_key)

    try:
        schedule = scraper.read_schedule().reset_index()
    except Exception as exc:
        print(f"[sofascore] WARNING: no schedule for "
              f"{league_key} {season_key}: {exc}")
        return pd.DataFrame()
    if schedule.empty:
        return pd.DataFrame()

    players: dict[tuple[str, str], dict] = {}
    for _, match in schedule.iterrows():
        game_id = match.get("game_id")
        if game_id is None or (isinstance(game_id, float)
                               and math.isnan(game_id)):
            continue
        # Unplayed fixtures have no lineups yet — skip instead of
        # burning a 404 per future match.
        if pd.isna(match.get("home_score")):
            continue
        try:
            lineups = _api_get(
                scraper,
                f"event/{int(game_id)}/lineups",
                f"lineups_{int(game_id)}.json",
                max_age=max_age,
            )
        except Exception as exc:
            print(f"[sofascore] WARNING: no lineups for "
                  f"game {game_id}: {exc}")
            continue

        for side, team in (("home", match.get("home_team")),
                           ("away", match.get("away_team"))):
            if not isinstance(team, str) or not team.strip():
                continue
            for entry in (lineups.get(side) or {}).get("players") or []:
                stats = entry.get("statistics") or {}
                if not stats:
                    continue  # unused substitute
                person = entry.get("player") or {}
                name = person.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                rec = players.setdefault((name, team), {
                    "position": None,
                    "appearances": 0,
                    "minutes": None,
                    "goals": None,
                    "assists": None,
                    "xg": None,
                    "xa": None,
                    "shots": None,
                    "shots_on_target": None,
                    "tackles": None,
                    "interceptions": None,
                    "yellow_cards": None,
                    "red_cards": None,
                })
                rec["appearances"] += 1
                if rec["position"] is None:
                    rec["position"] = _norm_position(
                        entry.get("position")
                        or person.get("position"))
                _accumulate_player(rec, stats,
                                   "minutes", "minutesPlayed",
                                   "minutes")
                _accumulate_player(rec, stats,
                                   "goals", "goals")
                _accumulate_player(rec, stats,
                                   "assists", "goalAssist",
                                   "assists")
                _accumulate_player(rec, stats,
                                   "xg", "expectedGoals", "xg")
                _accumulate_player(rec, stats,
                                   "xa", "expectedAssists", "xa")
                _accumulate_player(rec, stats,
                                   "shots", "totalShots", "shots")
                _accumulate_player(rec, stats,
                                   "shots_on_target", "shotsOnGoal",
                                   "shotsOnTarget")
                _accumulate_player(rec, stats,
                                   "tackles", "tackles")
                _accumulate_player(rec, stats,
                                   "interceptions", "interceptions")
                _accumulate_player(rec, stats,
                                   "yellow_cards", "yellowCards")
                _accumulate_player(rec, stats,
                                   "red_cards", "redCards")

    rows = []
    for (name, team), rec in players.items():
        row = {"player": name, "team": team}
        row.update(rec)
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════
# TEAM STATS
# ═══════════════════════════════════════════

def _extract_team_attrs(
    group: pd.DataFrame,
    games: int,
) -> dict:
    """
    Extract all team stats from the rows of one team.

    Works on match-level rows (aggregated here) and on single
    season-aggregate rows alike: totals are summed, percentages
    averaged, then per-game values are derived.
    """
    attrs = {}

    # xG
    xg = _series_sum(group, ["xg", "xG",
        "expected_goals", "team_xg"])
    if xg is not None:
        attrs["sh_xg"] = round(xg, 3)

    # Shots
    shots = _series_sum(group, ["shots",
        "total_shots", "shots_total",
        "shots_attempted"])
    if shots is not None:
        attrs["sh_shots"] = round(shots, 1)
        if games > 0:
            attrs["sh_sh_per90"] = round(
                shots / games, 2)

    # Shots on target
    sot = _series_sum(group, ["shots_on_target",
        "on_target", "shotsOnTarget",
        "shots_on_goal"])
    if sot is not None:
        attrs["sh_sot"] = round(sot, 1)
        if games > 0:
            attrs["sh_sot_per90"] = round(
                sot / games, 2)

    # Goals scored
    goals = _series_sum(group, ["goals",
        "goals_scored", "score"])
    if goals is not None:
        attrs["sh_goals"] = round(goals, 1)
        if shots and shots > 0:
            attrs["sh_g_per_sh"] = round(
                goals / shots, 3)

    # Possession (average %)
    poss = _series_mean(group, ["possession",
        "ball_possession", "ballPossession"])
    if poss is not None:
        attrs["pos_poss"] = round(poss, 1)

    # Goals against
    ga = _series_sum(group, ["goals_conceded",
        "goals_against", "opponent_score"])
    if ga is not None:
        attrs["gk_ga"] = round(ga, 1)
        if games > 0:
            attrs["gk_ga90"] = round(
                ga / games, 2)

    # Saves
    saves = _series_sum(group, ["saves",
        "goalkeeper_saves", "keeper_saves"])
    if saves is not None:
        attrs["gk_saves"] = round(saves, 1)
        if ga is not None:
            sot_against = ga + saves
            if sot_against > 0:
                attrs["gk_save_pct"] = round(
                    saves / sot_against * 100, 1)

    # NOTE: corners are deliberately not stored. team_season_stats has
    # no corners column — pos_cpa is FBref "Carries into pen area" (see
    # fbref_team_collector) and the corner markets are computed from the
    # matches table in features/builder.

    # xA — stored in ps_xa (xA). ps_xag holds FBref's different xAG
    # metric; see fbref_team_collector's column map.
    xa = _series_sum(group, ["xa", "xA",
        "expected_assists"])
    if xa is not None:
        attrs["ps_xa"] = round(xa, 3)

    # Big chances created (proxy for GCA)
    bcc = _series_sum(group, ["big_chances_created",
        "bigChancesCreated", "big_chances"])
    if bcc is not None:
        attrs["gsc_gca"] = round(bcc, 1)
        if games > 0:
            attrs["gsc_gca90"] = round(
                bcc / games, 2)

    # Tackles
    tackles = _series_sum(group, ["tackles",
        "tackles_won", "tacklesWon"])
    if tackles is not None:
        attrs["def_tkl"] = round(tackles, 1)

    # Interceptions
    ints = _series_sum(group, ["interceptions",
        "interceptions_won"])
    if ints is not None:
        attrs["def_int"] = round(ints, 1)
        if tackles is not None:
            attrs["def_tkl_int"] = round(
                tackles + ints, 1)

    # Fouls
    fouls = _series_sum(group, ["fouls",
        "fouls_committed"])
    if fouls is not None:
        attrs["misc_fls"] = round(fouls, 1)

    # Yellow cards
    yellows = _series_sum(group, ["yellow_cards",
        "yellowCards"])
    if yellows is not None:
        attrs["misc_crdy"] = round(yellows, 1)

    # Passes
    passes = _series_sum(group, ["passes",
        "total_passes", "accurate_passes"])
    if passes is not None:
        attrs["ps_att"] = round(passes, 1)

    return attrs


def collect_team_stats() -> int:
    """
    Fetch team season stats from Sofascore.
    Returns total rows upserted.
    """
    print("[sofascore] Collecting team stats...")
    total = 0

    for league in LEAGUES:
        for season in _get_seasons(league):
            print(f"[sofascore] Team: "
                  f"{league} {season}...")
            try:
                scraper = _get_scraper(league, season)
                df = _read_team_season_stats(scraper)

                if df is None or df.empty:
                    print(f"[sofascore] WARNING: "
                          f"no team data for "
                          f"{league} {season}")
                    continue

                # Reset index if multi-index
                if isinstance(df.index,
                              pd.MultiIndex):
                    df = df.reset_index()

                # Find team column
                team_col = _find_col(df, [
                    "team", "Team", "team_name"])
                if team_col is None:
                    print(f"[sofascore] WARNING: "
                          f"no team column. "
                          f"Cols: {list(df.columns)}")
                    continue

                db = SessionLocal()
                try:
                    for team_name, group in (
                            df.groupby(team_col)):
                        if not isinstance(
                                team_name, str):
                            continue
                        games = _group_games(group)
                        attrs = _extract_team_attrs(
                            group, games)
                        if attrs:
                            _upsert_team(
                                db, team_name,
                                league, season,
                                attrs)
                            total += 1

                    db.commit()
                    print(f"[sofascore] "
                          f"{league} {season}: "
                          f"committed")

                except Exception as exc:
                    db.rollback()
                    print(f"[sofascore] ERROR "
                          f"commit {league} "
                          f"{season}: {exc}")
                    raise
                finally:
                    db.close()

            except Exception as exc:
                print(f"[sofascore] ERROR "
                      f"{league} {season}: {exc}")
                continue

    print(f"[sofascore] Team stats done. "
          f"Total: {total}")
    return total


# ═══════════════════════════════════════════
# PLAYER STATS
# ═══════════════════════════════════════════

def _extract_player_attrs(row: pd.Series) -> dict:
    """Extract player stats from one row."""
    attrs = {}

    def get(*candidates):
        for c in candidates:
            if c in row.index:
                v = _safe(row[c])
                if v is not None:
                    return v
        return None

    # Goals
    goals = get("goals", "Goals", "goals_scored")
    if goals is not None:
        attrs["goals"] = goals

    # Assists
    assists = get("assists", "Assists")
    if assists is not None:
        attrs["assists"] = assists
    if goals is not None and assists is not None:
        attrs["g_plus_a"] = goals + assists

    # xG
    xg = get("xg", "xG", "expected_goals",
             "player_xg")
    if xg is not None:
        attrs["xg"] = xg

    # xA — xag is the only xA-family column on
    # PlayerSeasonStats.
    xa = get("xa", "xA", "expected_assists")
    if xa is not None:
        attrs["xag"] = xa

    # Minutes
    mins = get("minutes_played", "minutesPlayed",
               "minutes", "time_played")
    if mins is not None:
        attrs["minutes"] = mins
        per90 = mins / 90 if mins > 0 else 1
        attrs["minutes_per90"] = round(per90, 2)
        if goals is not None:
            attrs["gls_per90"] = round(
                goals / per90, 3)
        if assists is not None:
            attrs["ast_per90"] = round(
                assists / per90, 3)

    # Appearances
    apps = get("appearances", "matches_played",
               "games_played")
    if apps is not None:
        attrs["mp"] = int(apps)

    # Shots
    shots = get("shots", "total_shots",
                "shots_total")
    if shots is not None:
        attrs["sh_shots"] = shots
        if goals is not None and shots > 0:
            attrs["sh_g_per_sh"] = round(
                goals / shots, 3)

    # Shots on target
    sot = get("shots_on_target", "on_target",
              "shotsOnTarget")
    if sot is not None:
        attrs["sh_sot"] = sot
        if shots and shots > 0:
            attrs["sh_sot_pct"] = round(
                sot / shots * 100, 1)

    # Tackles
    tackles = get("tackles", "tackles_won")
    if tackles is not None:
        attrs["tkl"] = tackles

    # Interceptions
    ints = get("interceptions")
    if ints is not None:
        attrs["int_"] = ints
        if tackles is not None:
            attrs["tkl_plus_int"] = tackles + ints

    # Yellow cards
    yellows = get("yellow_cards", "yellowCards")
    if yellows is not None:
        attrs["crdy"] = yellows

    # Red cards
    reds = get("red_cards", "redCards")
    if reds is not None:
        attrs["crdr"] = reds

    # NOTE: the Sofascore rating (1-10) is deliberately not stored.
    # PlayerSeasonStats has no rating column and parking it in
    # sh_g_per_sh (goals per shot) would corrupt that metric.

    return attrs


def collect_player_stats() -> int:
    """
    Fetch player season stats from Sofascore.
    Returns total rows upserted.
    """
    print("[sofascore] Collecting player stats...")
    total = 0

    for league in LEAGUES:
        # Latest season only for players
        # to save time
        season = _get_seasons(league)[-1]
        print(f"[sofascore] Players: "
              f"{league} {season}...")
        try:
            scraper = _get_scraper(league, season)
            df = _read_player_season_stats(scraper)

            if df is None or df.empty:
                print(f"[sofascore] WARNING: "
                      f"no player data for "
                      f"{league} {season}")
                continue

            # Reset multi-index if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            print(f"[sofascore] Got {len(df)} "
                  f"player rows")
            print(f"[sofascore] Columns: "
                  f"{list(df.columns)[:8]}...")

            # Find key columns
            player_col = _find_col(df, [
                "player", "player_name", "Player",
                "name"])
            team_col   = _find_col(df, [
                "team", "Team", "team_name",
                "club"])
            pos_col    = _find_col(df, [
                "position", "pos", "Position"])

            if player_col is None:
                print(f"[sofascore] WARNING: "
                      f"no player column. "
                      f"Cols: {list(df.columns)}")
                continue

            db = SessionLocal()
            try:
                for _, row in df.iterrows():
                    player_name = row.get(
                        player_col)
                    team_name   = row.get(
                        team_col, "Unknown")
                    position    = row.get(pos_col)

                    if not isinstance(
                            player_name, str):
                        continue
                    if not player_name.strip():
                        continue
                    if not isinstance(team_name, str) \
                            or not team_name.strip():
                        team_name = "Unknown"

                    attrs = _extract_player_attrs(
                        row)
                    if position is not None \
                            and pd.notna(position) \
                            and str(position).strip():
                        attrs["position"] = str(
                            position)

                    if attrs:
                        _upsert_player(
                            db,
                            player_name,
                            str(team_name),
                            league,
                            season,
                            attrs,
                        )
                        total += 1

                db.commit()
                print(f"[sofascore] {league} "
                      f"{season}: "
                      f"{total} players so far")

            except Exception as exc:
                db.rollback()
                print(f"[sofascore] ERROR "
                      f"players {league}: {exc}")
                raise
            finally:
                db.close()

        except Exception as exc:
            print(f"[sofascore] ERROR players "
                  f"{league} {season}: {exc}")
            continue

    print(f"[sofascore] Player stats done. "
          f"Total: {total}")
    return total


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def collect_sofascore() -> None:
    """
    Run all Sofascore collectors.
    Primary data source for all advanced stats.
    """
    print("=" * 60)
    print("SOFASCORE — PRIMARY STATS COLLECTION")
    print("=" * 60)

    print("\n--- TEAM STATS ---")
    team_total = collect_team_stats()

    print("\n--- PLAYER STATS ---")
    player_total = collect_player_stats()

    print("\n" + "=" * 60)
    print("SOFASCORE COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Team rows updated  : {team_total}")
    print(f"Player rows updated: {player_total}")


if __name__ == "__main__":
    collect_sofascore()
