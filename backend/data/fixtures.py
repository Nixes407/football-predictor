"""Fetch and normalize daily football fixtures from football-data.org.

football-data.org does not cover USA-MLS, so that one league is appended from
FotMob's date-based match listing (``fotmob_collector.fetch_fixtures_for_date``).
The MLS source is optional by design: if the FotMob collector cannot be
imported, or answers with nothing, the leagues above are returned untouched.
"""

import os
import requests
from datetime import date, datetime, timezone

# USA-MLS comes from FotMob. API-Football is gone: that account is suspended
# and answers with errors, so nothing in this module may reach for it. The
# collector is imported defensively because it pulls in the database layer,
# and a fixture feed must not depend on that being importable.
try:
    from .fotmob_collector import fetch_fixtures_for_date
except Exception:
    try:
        from data.fotmob_collector import fetch_fixtures_for_date
    except Exception:
        try:
            from fotmob_collector import fetch_fixtures_for_date
        except Exception:  # pragma: no cover - optional dependency
            fetch_fixtures_for_date = None


# The one league this module sources outside football-data.org.
MLS_LEAGUE = "USA-MLS"


# football-data.org competition code -> internal league name.
# The codes are football-data.org's own (e.g. "CL" = UEFA Champions League),
# not API-Football IDs, which this module no longer speaks at all.
LEAGUE_MAP = {
    "PL":  "ENG-Premier League",
    "ELC": "ENG-Championship",
    "PD":  "ESP-La Liga",
    "BL1": "GER-Bundesliga",
    "SA":  "ITA-Serie A",
    "FL1": "FRA-Ligue 1",
    "PPL": "POR-Primeira Liga",
    "DED": "NED-Eredivisie",
    "BSA": "BRA-Série A",
    "TR1": "TUR-Süper Lig",
    "NOR": "NOR-Eliteserien",
    "AUT": "AUT-Bundesliga",
    "CL":  "UEFA-Champions League",
}

TEAM_NAME_MAP = {
    # Premier League
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle United",
    "Aston Villa FC": "Aston Villa",
    "West Ham United FC": "West Ham",
    "Brighton & Hove Albion FC": "Brighton",
    "Fulham FC": "Fulham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Everton FC": "Everton",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal Palace",
    "Nottingham Forest FC": "Nottingham Forest",
    "AFC Bournemouth": "Bournemouth",
    "Luton Town FC": "Luton",
    "Sheffield United FC": "Sheffield United",
    "Burnley FC": "Burnley",
    "Leicester City FC": "Leicester",
    "Ipswich Town FC": "Ipswich",
    "Southampton FC": "Southampton",
    "Leeds United FC": "Leeds United",
    "Sunderland AFC": "Sunderland",
    "West Bromwich Albion FC": "West Brom",
    "Watford FC": "Watford",
    "Norwich City FC": "Norwich",
    # La Liga
    "FC Barcelona": "Barcelona",
    "Real Madrid CF": "Real Madrid",
    "Club Atlético de Madrid": "Atletico Madrid",
    "Sevilla FC": "Sevilla",
    "Real Betis Balompié": "Betis",
    "Villarreal CF": "Villarreal",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Athletic Club": "Athletic Club",
    "RC Celta de Vigo": "Celta Vigo",
    "Getafe CF": "Getafe",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "RCD Mallorca": "Mallorca",
    "Girona FC": "Girona",
    "UD Almería": "Almeria",
    "Cádiz CF": "Cadiz",
    "Deportivo Alavés": "Alaves",
    "UD Las Palmas": "Las Palmas",
    "Valencia CF": "Valencia",
    "CA Osasuna": "Osasuna",
    "Leganés": "Leganes",
    "Elche CF": "Elche",
    "Granada CF": "Granada",
    "RCD Espanyol de Barcelona": "Espanyol",
    "SD Eibar": "Eibar",
    "SD Huesca": "Huesca",
    "Real Valladolid CF": "Valladolid",
    # Promoted clubs: football-data.org spellings -> the names the stats
    # collectors store (FotMob/FBref spellings), so the fixture, the
    # formula engine and the H2H history all agree on one name.
    "RC Deportivo La Coruña": "Deportivo A Coruña",
    "Levante UD": "Levante",
    # Bundesliga
    "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Frankfurt",
    "VfL Wolfsburg": "Wolfsburg",
    "SC Freiburg": "Freiburg",
    "Union Berlin": "Union Berlin",
    "Borussia Mönchengladbach": "Gladbach",
    "VfB Stuttgart": "Stuttgart",
    "FC Augsburg": "Augsburg",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "1. FSV Mainz 05": "Mainz",
    "SV Darmstadt 98": "Darmstadt",
    "VfL Bochum 1848": "Bochum",
    "1. FC Köln": "Köln",
    "1. FC Heidenheim 1846": "Heidenheim",
    "SV Werder Bremen": "Werder Bremen",
    "FC St. Pauli 1910": "St. Pauli",
    "Holstein Kiel": "Kiel",
    "1. FC Union Berlin": "Union Berlin",
    "SC Paderborn 07": "Paderborn",
    "SV 07 Elversberg": "Elversberg",
    "Hamburger SV": "Hamburg",
    "SpVgg Greuther Fürth": "Greuther Furth",
    "Arminia Bielefeld": "Arminia",
    "Hertha BSC": "Hertha",
    "FC Schalke 04": "Schalke",
    "Eintracht Braunschweig": "Braunschweig",
    # Serie A
    "Juventus FC": "Juventus",
    "AC Milan": "Milan",
    "FC Internazionale Milano": "Inter",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Atalanta BC": "Atalanta",
    "ACF Fiorentina": "Fiorentina",
    "Bologna FC 1909": "Bologna",
    "Torino FC": "Torino",
    "Udinese Calcio": "Udinese",
    "Hellas Verona FC": "Verona",
    "US Sassuolo Calcio": "Sassuolo",
    "Empoli FC": "Empoli",
    "US Lecce": "Lecce",
    "Genoa CFC": "Genoa",
    "Frosinone Calcio": "Frosinone",
    "Cagliari Calcio": "Cagliari",
    "AC Monza": "Monza",
    "Venezia FC": "Venezia",
    "Como 1907": "Como",
    "Parma Calcio 1913": "Parma",
    "US Salernitana 1919": "Salernitana",
    "Spezia Calcio": "Spezia",
    "US Cremonese": "Cremonese",
    "Benevento Calcio": "Benevento",
    # Ligue 1
    "Paris Saint-Germain FC": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco",
    "LOSC Lille": "Lille",
    "Stade Rennais FC 1901": "Rennes",
    "RC Lens": "Lens",
    "OGC Nice": "Nice",
    "Montpellier HSC": "Montpellier",
    "Stade de Reims": "Reims",
    "RC Strasbourg Alsace": "Strasbourg",
    "Nantes": "Nantes",
    "Toulouse FC": "Toulouse",
    "FC Lorient": "Lorient",
    "Le Havre AC": "Le Havre",
    "FC Metz": "Metz",
    "Clermont Foot 63": "Clermont",
    "Stade Brestois 29": "Brest",
    "AJ Auxerre": "Auxerre",
    "Saint-Étienne": "Saint-Etienne",
    "Angers SCO": "Angers",
    "Lille OSC": "Lille",
    "Paris FC": "Paris FC",
    "Le Mans FC": "Le Mans",
    "FC Nantes": "Nantes",
    "AS Saint-Étienne": "Saint-Etienne",
    "Stade Malherbe Caen": "Caen",
    "Troyes AC": "Troyes",
    "Nîmes Olympique": "Nimes",
    "Dijon FCO": "Dijon",
}


def map_team_name(name: str) -> str:
    """Return the internal name for a known team, or ``name`` unchanged."""
    return TEAM_NAME_MAP.get(name, name)


def derive_season(utc_date_str: str) -> str:
    """Derive the football season, which starts in July, from a UTC date."""
    match_date = date.fromisoformat(utc_date_str[:10])
    if match_date.month >= 7:
        return f"{match_date.year}-{match_date.year + 1}"
    return f"{match_date.year - 1}-{match_date.year}"


def _collect_mls_fixtures(target_date: str) -> list[dict]:
    """USA-MLS fixtures for ``target_date`` from FotMob, in this file's shape.

    Returns ``[]`` (never raises) when the collector is unavailable or the day
    has no MLS fixtures, so the football-data.org leagues are unaffected.
    """
    if fetch_fixtures_for_date is None:
        print(
            "[fixtures] WARNING: fotmob_collector.fetch_fixtures_for_date is "
            "not importable; skipping USA-MLS fixtures."
        )
        return []

    try:
        collected = fetch_fixtures_for_date(target_date) or []
    except Exception as exc:
        print(
            f"[fixtures] WARNING: failed to fetch USA-MLS fixtures from "
            f"FotMob for {target_date}: {exc}"
        )
        return []

    fixtures: list[dict] = []
    for item in collected:
        try:
            if not isinstance(item, dict) or item.get("league") != MLS_LEAGUE:
                continue
            home_team = item.get("home_team")
            away_team = item.get("away_team")
            external_id = item.get("external_id")
            if not home_team or not away_team or external_id in (None, ""):
                continue
            kickoff = str(item.get("kickoff_time") or "") or None
            day = str(item.get("match_date") or "")[:10] or target_date
            # FotMob already returns the calendar-year season ("2026"). Do not
            # re-derive it with derive_season(): MLS rows in team_season_stats
            # are keyed by year, and a July-based "2025-2026" label would find
            # nothing and silently lose every MLS prediction.
            season = str(item.get("season") or "") or (
                (kickoff or day)[:4] if (kickoff or day)[:4].isdigit() else None
            )
            fixtures.append(
                {
                    "external_id": str(external_id),
                    "match_date": day,
                    "kickoff_time": kickoff,
                    # FotMob spellings are kept verbatim: TEAM_NAME_MAP exists
                    # to translate football-data.org names, and the stats the
                    # predictor joins against were written by the FotMob
                    # collector using these same spellings.
                    "home_team": home_team,
                    "away_team": away_team,
                    "league": MLS_LEAGUE,
                    "season": season,
                    "status": str(item.get("status") or "").lower() or None,
                }
            )
        except Exception as exc:
            print(
                f"[fixtures] WARNING: could not parse a FotMob USA-MLS "
                f"fixture for {target_date}: {exc}"
            )
            continue

    if not fixtures:
        print(
            f"[fixtures] WARNING: no USA-MLS fixtures available for "
            f"{target_date} from FotMob (expected outside the MLS season)."
        )
    return fixtures


def fetch_fixtures(target_date: str = None) -> list[dict]:
    """Fetch and normalize fixtures for ``target_date`` (UTC), if available."""
    fixtures = []
    if target_date is None:
        utc_today = datetime.now(timezone.utc).date()
        target_date = date(utc_today.year, utc_today.month, utc_today.day).isoformat()

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print(
            "[fixtures] WARNING: FOOTBALL_DATA_API_KEY is not set; "
            "skipping fixture fetch."
        )
    else:
        headers = {"X-Auth-Token": api_key}
        for code, league_name in LEAGUE_MAP.items():
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
                print(f"[fixtures] WARNING: failed to fetch league {code}: {exc}")
                continue

            for match in matches:
                try:
                    utc_date = match["utcDate"]
                    fixtures.append(
                        {
                            "external_id": str(match["id"]),
                            "match_date": target_date,
                            "kickoff_time": utc_date,
                            "home_team": map_team_name(match["homeTeam"]["name"]),
                            "away_team": map_team_name(match["awayTeam"]["name"]),
                            "league": league_name,
                            "season": derive_season(utc_date),
                            "status": match["status"].lower(),
                        }
                    )
                except Exception as exc:
                    print(
                        f"[fixtures] WARNING: could not parse a match in league "
                        f"{code}: {exc}"
                    )

    # Supplement football-data.org with USA-MLS fixtures from FotMob.
    # football-data.org does not cover MLS. Keep this optional so an
    # unavailable collector or an empty day never affects the leagues above.
    fixtures.extend(_collect_mls_fixtures(target_date))

    return fixtures
