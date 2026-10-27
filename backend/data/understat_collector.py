"""
Understat xG data collector.

Fetches team and match level xG statistics
from Understat via soccerdata library.
No Chrome, no CAPTCHA — simple HTTP requests.

Covers 6 leagues:
  ENG-Premier League
  ESP-La Liga
  GER-Bundesliga
  ITA-Serie A
  FRA-Ligue 1
  NED-Eredivisie

Stores results in team_season_stats table
updating xG and PPDA fields.

Run from backend/ directory:
    python data/understat_collector.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import pandas as pd
import soccerdata as sd

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database import SessionLocal, TeamSeasonStats, upsert_team_season_stats
from sqlalchemy import and_


# ═══════════════════════════════════════════
# TEAM NAME MAPPING
# ═══════════════════════════════════════════

UNDERSTAT_TO_API_TEAM = {
    # English
    "Manchester City":      "Manchester City",
    "Manchester United":    "Manchester United",
    "Tottenham":            "Tottenham Hotspur",
    "Wolves":               "Wolverhampton Wanderers",
    "Newcastle":            "Newcastle United",
    "West Ham":             "West Ham United",
    "Nott'M Forest":        "Nottingham Forest",
    "Sheffield Utd":        "Sheffield United",
    "Leeds":                "Leeds United",
    "Leicester":            "Leicester City",
    "Luton":                "Luton Town",
    "Ipswich":              "Ipswich Town",
    # Spanish
    "Atletico Madrid":      "Atletico Madrid",
    "Athletic Club":        "Athletic Club",
    "Alaves":               "Deportivo Alaves",
    "Espanol":              "Espanyol",
    "Betis":                "Real Betis",
    # German
    "Greuther Furth":       "SpVgg Greuther Furth",
    "Bayern Munich":        "Bayern Munchen",
    "Leverkusen":           "Bayer Leverkusen",
    "M'gladbach":           "Borussia M.Gladbach",
    "Dortmund":             "Borussia Dortmund",
    "Mainz":                "1. FSV Mainz 05",
    "Koln":                 "FC Koln",
    "Hoffenheim":           "TSG Hoffenheim",
    # Italian
    "Inter":                "Inter",
    "Milan":                "AC Milan",
    "Verona":               "Hellas Verona",
    "SPAL":                 "SPAL",
    # French
    "Paris Saint-Germain":  "Paris Saint Germain",
    "Saint-Etienne":        "Saint Etienne",
    "Marseille":            "Olympique de Marseille",
    "Lyon":                 "Olympique Lyonnais",
    "Nice":                 "OGC Nice",
    "Lens":                 "RC Lens",
    "Rennes":               "Stade Rennais FC",
    "Brest":                "Stade Brestois 29",
    "Nantes":               "FC Nantes",
    "Strasbourg":           "RC Strasbourg",
    "Metz":                 "FC Metz",
    "Reims":                "Stade de Reims",
    "Lorient":              "FC Lorient",
    "Montpellier":          "Montpellier HSC",
    "Bordeaux":             "Girondins de Bordeaux",
    "Guingamp":             "En Avant Guingamp",
    "Dijon":                "Dijon FCO",
    "Amiens":               "Amiens SC",
    "Toulouse":             "Toulouse FC",
    "Clermont":             "Clermont Foot",
    "Auxerre":              "AJ Auxerre",
    "Angers":               "Angers SCO",
    "Le Havre":             "Le Havre AC",
}


def normalise_understat_team(name: str) -> str:
    """
    Map Understat team name to API-Football
    team name so xG data merges into the
    correct existing row.
    """
    if not isinstance(name, str):
        return name
    return UNDERSTAT_TO_API_TEAM.get(name, name)


# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

# Understat only covers these leagues
UNDERSTAT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Understat lags roughly one season behind the live calendar, so
# 2025-2026 is the most recent season it publishes.
SEASONS = ["2024-2025", "2025-2026"]

# Understat uses different league name format
# Map our league names to Understat format.
# Reference only: soccerdata accepts our own names (the same
# keys as UNDERSTAT_LEAGUES), so the collectors pass them through
# unchanged. Understat has no Eredivisie data, so there is no
# entry for it ("RFPL" is the Russian Premier League).
UNDERSTAT_LEAGUE_MAP = {
    "ENG-Premier League": "EPL",
    "ESP-La Liga": "La_liga",
    "GER-Bundesliga": "Bundesliga",
    "ITA-Serie A": "Serie_A",
    "FRA-Ligue 1": "Ligue_1",
}

# Map Understat season format to our format
# Understat uses the year the season starts
UNDERSTAT_SEASON_MAP = {
    "2024-2025": 2024,
    "2025-2026": 2025,
}


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════


def _safe(val) -> float | None:
    """Convert value to float or None."""
    try:
        import math

        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _pick(row, *keys):
    """
    First of ``keys`` that the row actually carries a value for.

    ``row.get(a) or row.get(b)`` cannot be used on these frames: it throws
    away real zeros (a 0-0 score line, a 0.0 xG) and returns NaN rather
    than None for a column that exists but is empty.
    """
    index = getattr(row, "index", ())
    for key in keys:
        if key not in index:
            continue
        val = row.get(key)
        if val is None:
            continue
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass
        return val
    return None


def _only_completed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop fixtures that have not been played yet.

    ``read_schedule()`` lists the whole season, and an unplayed fixture
    carries placeholder goals and xG of 0 — writing those stored 0-0
    "matches" for games that never happened. The old code filtered on a
    "status" column, which the Understat reader does not return.
    """
    for flag in ("status", "is_result", "has_data"):
        if flag not in df.columns:
            continue
        series = df[flag]
        if flag == "status":
            mask = series.astype(str).str.lower().isin(
                ("finished", "ft", "played", "aet", "pen")
            )
        else:
            mask = series.fillna(False).astype(bool)
        return df[mask]
    return df


def _understat_match_id(
    competition: str,
    season: str,
    home_team: str,
    away_team: str,
    date_str: str,
) -> int:
    """
    Stable id for one match, derived from the fixture itself.

    The old id came from Python's ``hash()``, which is salted per process:
    every run produced different ids, the ``ON CONFLICT (match_id)`` clause
    never fired, and each run inserted another copy of all 3504 matches.
    md5 is stable, and the value is kept in the 1,000,000+ block this
    collector reserves so it cannot collide with the StatsBomb ids
    (INTEGER column, so also below 2**31).
    """
    key = f"{competition}{season}{home_team}{away_team}{date_str}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1_000_000_000 + 1_000_000


# Club-type words that do not identify a club ("AS Monaco" and "Monaco",
# "FC Koln" and "Koln"). Compared names have these dropped, so the
# Understat spelling reaches the API-Football row even when the mapping
# table has no entry for it.
_CLUB_TOKENS = {
    "ac", "afc", "as", "asc", "bsc", "ca", "cd", "cf", "cfa", "club",
    "efc", "fc", "fco", "fcs", "fsv", "hsc", "ogc", "osc", "rc", "rcd",
    "sc", "sd", "ss", "ssc", "sv", "tsg", "ud", "us", "vfb", "vfl",
}


# Spellings of the same word used by different providers, folded to one
# form so "Bayern Munich" and "Bayern Munchen" are the same club even when
# the mapping table has no entry for that pair.
_TOKEN_SYNONYMS = {
    "muenchen":         "munich",
    "munchen":          "munich",
    "koeln":            "koln",
    "moenchengladbach": "gladbach",
    "monchengladbach":  "gladbach",
}


def _fold_team_name(name: str) -> str:
    """Lower-case, accent- and punctuation-free form of a club name."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(
        _TOKEN_SYNONYMS.get(tok, tok) for tok in text.split()
    )


def _team_name_tokens(name: str) -> frozenset:
    """Identifying words of a club name, club-type words removed."""
    return frozenset(
        tok
        for tok in _fold_team_name(name).split()
        if len(tok) > 1 and tok not in _CLUB_TOKENS
    )


def _team_index(db, league: str, season: str, cache: dict) -> dict:
    """
    Folded name -> stored names for one league and season.

    Read once per collector run: the resolver is called for every team and
    every match, and the row set only changes through this module's own
    upserts.
    """
    key = (league, season)
    if key not in cache:
        rows = (
            db.query(TeamSeasonStats.team)
            .filter(
                and_(
                    TeamSeasonStats.league == league,
                    TeamSeasonStats.season == season,
                )
            )
            .all()
        )
        index: dict[str, list[str]] = {}
        for (stored,) in rows:
            if not stored:
                continue
            index.setdefault(_fold_team_name(stored), []).append(stored)
        cache[key] = index
    return cache[key]


def _resolve_understat_team(
    db,
    name,
    league: str,
    season: str,
    cache: dict | None = None,
) -> str:
    """
    Name to store an Understat row under.

    Applies :func:`normalise_understat_team`, then matches the result
    against the names already stored for that league and season, so
    "Tottenham" updates the "Tottenham Hotspur" row instead of inserting a
    second club. Matching folds case, accents and punctuation and ignores
    club-type words; a token-subset match is only accepted when it is
    unique, otherwise the mapped name is returned and a row is inserted.
    """
    mapped = normalise_understat_team(name)
    if not isinstance(mapped, str) or not mapped.strip():
        return mapped

    index = _team_index(db, league, season, cache if cache is not None else {})
    if not index:
        return mapped

    exact = index.get(_fold_team_name(mapped))
    if exact and len(exact) == 1:
        return exact[0]

    candidate = _team_name_tokens(mapped)
    if not candidate:
        return mapped

    scored: list[tuple[int, str]] = []
    for fold, names in index.items():
        stored = _team_name_tokens(fold)
        if not stored:
            continue
        if stored <= candidate or candidate <= stored:
            for stored_name in names:
                scored.append((len(stored ^ candidate), stored_name))

    if not scored:
        return mapped

    scored.sort(key=lambda item: (item[0], item[1]))
    best = scored[0][0]
    winners = {stored_name for score, stored_name in scored if score == best}
    if len(winners) == 1:
        return next(iter(winners))
    # Several stored names are equally close: if they are the same club
    # written differently (e.g. "Koln" and "FC Koln"), take one of them;
    # otherwise the name is genuinely ambiguous and is left as it is.
    if all(_fold_team_name(w) == _fold_team_name(mapped) for w in winners):
        return sorted(winners)[0]
    return mapped


def _report_aliases(
    aliases: dict,
    league: str,
    season: str,
    limit: int = 8,
) -> None:
    """Print the Understat names that were stored under a different name."""
    if not aliases:
        return
    items = sorted(aliases.items())
    shown = ", ".join(f"{src} -> {dst}" for src, dst in items[:limit])
    extra = "" if len(items) <= limit else f" (+{len(items) - limit} more)"
    print(f"[understat] aliases {league} {season}: {shown}{extra}")


def _upsert_team(
    db,
    team: str,
    league: str,
    season: str,
    attrs: dict,
) -> None:
    """Insert or update one TeamSeasonStats row via the guarded write."""
    upsert_team_season_stats(db, team, league, season, attrs)


# ═══════════════════════════════════════════
# TEAM SEASON STATS
# ═══════════════════════════════════════════


def _aggregate_team_season_rows(df: pd.DataFrame) -> list[tuple[str, dict]]:
    """
    Sum the per-match team rows of ``read_team_match_stats()`` into one
    season row per club: ``(understat_team_name, attrs)``.

    ``read_team_match_stats()`` is the reader the installed soccerdata
    actually exposes; there is no team-season endpoint (checked on
    soccerdata 1.7 / 1.8 / 1.9), so the season totals are summed from each
    club's match history: xG and npxG for, goals scored, and the W-D-L
    record from the match points.
    """
    totals: dict[str, dict] = {}

    for _, row in df.iterrows():
        for side in ("home", "away"):
            team = _pick(row, f"{side}_team")
            if not team:
                continue

            bucket = totals.setdefault(
                str(team),
                {"xg": 0.0, "npxg": 0.0, "goals": 0.0,
                 "w": 0.0, "d": 0.0, "l": 0.0},
            )

            xg = _safe(_pick(row, f"{side}_xg"))
            npxg = _safe(_pick(row, f"{side}_np_xg"))
            goals = _safe(_pick(row, f"{side}_goals"))
            points = _safe(_pick(row, f"{side}_points"))

            if xg is not None:
                bucket["xg"] += xg
            if npxg is not None:
                bucket["npxg"] += npxg
            if goals is not None:
                bucket["goals"] += goals
            if points is not None:
                if points >= 3:
                    bucket["w"] += 1
                elif points >= 1:
                    bucket["d"] += 1
                else:
                    bucket["l"] += 1

    rows: list[tuple[str, dict]] = []
    for team, bucket in totals.items():
        attrs: dict = {}
        if bucket["xg"]:
            attrs["sh_xg"] = round(bucket["xg"], 3)
        if bucket["npxg"]:
            attrs["sh_npxg"] = round(bucket["npxg"], 3)
        if bucket["goals"]:
            attrs["sh_goals"] = round(bucket["goals"], 3)
        if bucket["w"] or bucket["d"] or bucket["l"]:
            attrs["gk_w"] = bucket["w"]
            attrs["gk_d"] = bucket["d"]
            attrs["gk_l"] = bucket["l"]
        if attrs:
            rows.append((team, attrs))
    return rows


def collect_team_xg_stats() -> int:
    """
    Fetch team season xG statistics from Understat.
    Returns total rows upserted.

    Stats written (all season totals, matching the FBref columns of the
    same name):
      xG    - expected goals scored      -> sh_xg
      npxG  - non-penalty xG             -> sh_npxg
      goals - goals scored               -> sh_goals
      W-D-L - league record from points  -> gk_w / gk_d / gk_l

    xGA is deliberately *not* written to gk_ga: there is no xGA column in
    team_season_stats, and mixing expected goals against into the real
    GA / GA90 pair makes the engine's games-played estimate disagree with
    itself (GA / GA90 no longer describes the same set of games). PPDA and
    deep completions are not written either: their only candidate columns
    (def_press_pct / def_press / ps_prgp / pos_prgr) hold FBref
    percentages and counts, so a PPDA coefficient stored there would drag
    every Understat-updated club towards the top of the defence rating.
    """
    print("[understat] Collecting team xG stats...")
    total = 0

    for league_name in UNDERSTAT_LEAGUES:
        for season_label in SEASONS:
            print(f"[understat] {league_name} {season_label}...")

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    understat = sd.Understat(
                        leagues=[league_name],
                        seasons=[season_label],
                    )
                    df = understat.read_team_match_stats()

                if df is None or df.empty:
                    print(
                        f"[understat] WARNING: no data for "
                        f"{league_name} {season_label}"
                    )
                    continue

                team_rows = _aggregate_team_season_rows(df)

                print(
                    f"[understat] Got {len(team_rows)} teams from "
                    f"{len(df)} matches for "
                    f"{league_name} {season_label}"
                )

                db = SessionLocal()
                try:
                    cache: dict = {}
                    aliases: dict = {}
                    written = 0

                    for team_name, attrs in team_rows:
                        if not attrs:
                            continue
                        # Understat spelling -> the name already stored for
                        # this league and season (API-Football / FBref), so
                        # the xG lands on that row instead of a new club.
                        team = _resolve_understat_team(
                            db, team_name, league_name, season_label, cache
                        )
                        if team != team_name:
                            aliases[team_name] = team
                        _upsert_team(db, team, league_name, season_label, attrs)
                        written += 1

                    db.commit()
                    _report_aliases(aliases, league_name, season_label)
                    print(
                        f"[understat] Committed {written} teams for "
                        f"{league_name} {season_label}"
                    )
                    total += written

                except Exception as exc:
                    db.rollback()
                    print(
                        f"[understat] ERROR commit "
                        f"{league_name} {season_label}: {exc}"
                    )
                finally:
                    db.close()

            except Exception as exc:
                print(f"[understat] ERROR {league_name} {season_label}: {exc}")
                continue

    print(f"[understat] Done. Total rows: {total}")
    return total


# ═══════════════════════════════════════════
# MATCH LEVEL xG
# ═══════════════════════════════════════════


def collect_match_xg() -> int:
    """
    Fetch match level xG from Understat.
    Stores in statsbomb_match_stats table
    (reuses same table since structure matches).
    Returns total matches processed.
    """
    print("[understat] Collecting match-level xG...")
    total = 0

    for league_name in UNDERSTAT_LEAGUES:
        for season_label in SEASONS:
            print(f"[understat] Match xG: {league_name} {season_label}...")

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    understat = sd.Understat(
                        leagues=[league_name],
                        seasons=[season_label],
                    )
                    df = understat.read_schedule()

                if df is None or df.empty:
                    continue

                # read_schedule() lists every fixture of the season,
                # including the ones not played yet (goals and xG are
                # placeholders there), so only completed matches are kept.
                df = _only_completed(df)

                print(
                    f"[understat] {len(df)} completed "
                    f"matches for {league_name} {season_label}"
                )

                db = SessionLocal()
                try:
                    from sqlalchemy import text

                    cache: dict = {}
                    aliases: dict = {}

                    for idx, row in df.iterrows():
                        home_xg = _safe(
                            _pick(row, "xG_h", "home_xg", "xg_home")
                        )
                        away_xg = _safe(
                            _pick(row, "xG_a", "away_xg", "xg_away")
                        )

                        if home_xg is None and away_xg is None:
                            continue

                        home_raw = normalise_understat_team(
                            str(_pick(row, "home_team", "home") or "")
                        )
                        away_raw = normalise_understat_team(
                            str(_pick(row, "away_team", "away") or "")
                        )
                        date_str = str(_pick(row, "date", "datetime") or "")
                        # The schedule carries its goals in "home_goals" /
                        # "away_goals"; the old column names never existed,
                        # so every match was stored with NULL goals.
                        home_goals = _safe(
                            _pick(row, "home_goals", "home_score", "score_h")
                        )
                        away_goals = _safe(
                            _pick(row, "away_goals", "away_score", "score_a")
                        )

                        # Same mapping the team xG rows use, so a head to
                        # head lookup matches the names the engine queries
                        # with (API-Football spellings).
                        home_team = _resolve_understat_team(
                            db, home_raw, league_name, season_label, cache
                        )
                        away_team = _resolve_understat_team(
                            db, away_raw, league_name, season_label, cache
                        )
                        if home_team != home_raw:
                            aliases[home_raw] = home_team
                        if away_team != away_raw:
                            aliases[away_raw] = away_team

                        match_id = _understat_match_id(
                            league_name, season_label,
                            home_team, away_team, date_str,
                        )

                        db.execute(
                            text(
                                """
                            INSERT INTO statsbomb_match_stats (
                                match_id, competition, season,
                                date, home_team, away_team,
                                home_goals, away_goals,
                                home_xg, away_xg
                            ) VALUES (
                                :match_id, :competition, :season,
                                :date, :home_team, :away_team,
                                :home_goals, :away_goals,
                                :home_xg, :away_xg
                            )
                            ON CONFLICT (match_id) DO UPDATE SET
                                home_xg    = EXCLUDED.home_xg,
                                away_xg    = EXCLUDED.away_xg,
                                home_goals = EXCLUDED.home_goals,
                                away_goals = EXCLUDED.away_goals
                        """
                            ),
                            {
                                "match_id": match_id,
                                "competition": league_name,
                                "season": season_label,
                                "date": date_str[:10],
                                "home_team": str(home_team),
                                "away_team": str(away_team),
                                "home_goals": (
                                    int(home_goals)
                                    if home_goals is not None else None
                                ),
                                "away_goals": (
                                    int(away_goals)
                                    if away_goals is not None else None
                                ),
                                "home_xg": home_xg,
                                "away_xg": away_xg,
                            },
                        )
                        total += 1

                    db.commit()
                    _report_aliases(aliases, league_name, season_label)
                    print(
                        f"[understat] Stored match xG for "
                        f"{league_name} {season_label}"
                    )

                except Exception as exc:
                    db.rollback()
                    print(
                        f"[understat] ERROR storing match xG "
                        f"{league_name} {season_label}: {exc}"
                    )
                finally:
                    db.close()

            except Exception as exc:
                print(
                    f"[understat] ERROR reading schedule "
                    f"{league_name} {season_label}: {exc}"
                )
                continue

    print(f"[understat] Match xG done. Total: {total}")
    return total


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════


def collect_understat() -> None:
    """Run all Understat collectors."""
    print("=" * 60)
    print("UNDERSTAT xG DATA COLLECTION")
    print("=" * 60)

    print("\n--- TEAM SEASON xG STATS ---")
    team_total = collect_team_xg_stats()

    print("\n--- MATCH LEVEL xG ---")
    match_total = collect_match_xg()

    print("\n" + "=" * 60)
    print("UNDERSTAT COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Team-season rows : {team_total}")
    print(f"Match xG records : {match_total}")


if __name__ == "__main__":
    collect_understat()
