"""Fetch and normalize daily football fixtures from football-data.org."""

import os
import requests
from datetime import date, datetime, timezone

try:
    from . import api_football_collector as _af_collector
    from .api_football_collector import collect_fixtures_and_results
except Exception:
    try:
        from data import api_football_collector as _af_collector
        from data.api_football_collector import collect_fixtures_and_results
    except Exception:
        try:
            import api_football_collector as _af_collector
            from api_football_collector import collect_fixtures_and_results
        except Exception:  # pragma: no cover - optional dependency
            _af_collector = None
            collect_fixtures_and_results = None

_default_collect_fixtures_and_results = collect_fixtures_and_results


# football-data.org competition code -> internal league name.
# The codes are football-data.org's own (e.g. "CL" = UEFA Champions League),
# not API-Football IDs: API-Football's ID 2 would be requested here as
# /v4/competitions/2/matches, which does not exist.
LEAGUE_MAP = {
    "PL":  "ENG-Premier League",
    "ELC": "ENG-Championship",
    "PD":  "ESP-La Liga",
    "BL1": "GER-Bundesliga",
    "SA":  "ITA-Serie A",
    "FL1": "FRA-Ligue 1",
    "PPL": "POR-Primeira Liga",
    "DED": "NED-Eredivisie",
    "BSA": "BEL-Jupiler Pro League",
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


def _collect_af_fixtures_for_date(target_date: str) -> list[dict]:
    """Fetch API-Football fixtures for ``target_date`` (YYYY-MM-DD).

    ``api_football_collector.collect_fixtures_and_results()`` only fetches
    ``date.today()`` into its ``fixtures`` list (and queries ``yesterday``
    with ``status="FT"`` into ``results`` without kickoff times). When
    ``target_date`` is any other date — such as yesterday's UTC date for the
    late USA-MLS spillover — we query API-Football's ``fixtures`` endpoint
    for ``target_date`` directly so all fixtures for that date are returned.
    """
    api_football_key = os.environ.get("API_FOOTBALL_KEY", "")
    if _af_collector is not None and api_football_key:
        # Sync module-level API_KEY in case .env was loaded after import.
        _af_collector.API_KEY = api_football_key

    if collect_fixtures_and_results is not None and (
        collect_fixtures_and_results is not _default_collect_fixtures_and_results
        or target_date == date.today().isoformat()
    ):
        af_result = collect_fixtures_and_results() or {}
        items = list(af_result.get("fixtures", []) or [])
        if (
            collect_fixtures_and_results is not _default_collect_fixtures_and_results
            or any(
                str(item.get("kickoff") or "")[:10] == target_date
                for item in items
            )
        ):
            return items

    if _af_collector is None or not hasattr(_af_collector, "_get"):
        return []

    data = _af_collector._get("fixtures", {"date": target_date})
    if not data:
        return []

    af_league_map = getattr(_af_collector, "LEAGUE_MAP", None) or {
        "USA-MLS": 253,
    }
    items: list[dict] = []
    for f in data.get("response", []) or []:
        try:
            lid = (f.get("league") or {}).get("id")
            ln = next((k for k, v in af_league_map.items() if v == lid), None)
            if not ln:
                continue
            fixture_info = f.get("fixture") or {}
            teams_info = f.get("teams") or {}
            home_info = teams_info.get("home") or {}
            away_info = teams_info.get("away") or {}
            status_info = fixture_info.get("status") or {}
            items.append(
                {
                    "fixture_id": fixture_info.get("id"),
                    "league": ln,
                    "home_team": home_info.get("name"),
                    "away_team": away_info.get("name"),
                    "kickoff": fixture_info.get("date"),
                    "status": status_info.get("short"),
                    "home_id": home_info.get("id"),
                    "away_id": away_info.get("id"),
                }
            )
        except Exception as exc:
            print(
                f"[fixtures] WARNING: could not parse an API-Football "
                f"fixture for {target_date}: {exc}"
            )
    return items


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

    # Supplement football-data.org with USA-MLS fixtures from API-Football.
    # football-data.org does not cover MLS, but api_football_collector already
    # maps and fetches it. Keep this optional so a missing key or a failed
    # request never affects the leagues above.
    if collect_fixtures_and_results is not None or _af_collector is not None:
        if not os.environ.get("API_FOOTBALL_KEY"):
            print(
                "[fixtures] WARNING: API_FOOTBALL_KEY is not set; "
                "skipping USA-MLS fixture fetch."
            )
        else:
            try:
                af_fixtures = _collect_af_fixtures_for_date(target_date)
                for item in af_fixtures:
                    if item.get("league") != "USA-MLS":
                        continue
                    kickoff = item.get("kickoff")
                    if kickoff is None:
                        continue
                    kickoff_str = str(kickoff)
                    if kickoff_str[:10] != target_date:
                        continue
                    fixtures.append(
                        {
                            "external_id": str(item.get("fixture_id")),
                            "match_date": target_date,
                            "kickoff_time": kickoff_str,
                            "home_team": item.get("home_team"),
                            "away_team": item.get("away_team"),
                            "league": item.get("league"),
                            "season": derive_season(kickoff_str),
                            "status": str(item.get("status") or "").lower(),
                        }
                    )
            except Exception as exc:
                print(
                    f"[fixtures] WARNING: failed to fetch USA-MLS fixtures "
                    f"from API-Football: {exc}"
                )

    return fixtures
