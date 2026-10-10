"""Fetch and normalize daily football fixtures from football-data.org."""

import os
import requests
from datetime import date, datetime, timezone


LEAGUE_MAP = {
    "PL": "ENG-Premier League",
    "PD": "ESP-La Liga",
    "BL1": "GER-Bundesliga",
    "SA": "ITA-Serie A",
    "FL1": "FRA-Ligue 1",
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


def fetch_fixtures(target_date: str = None) -> list[dict]:
    """Fetch and normalize fixtures for ``target_date`` (UTC), if available."""
    fixtures = []
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print(
            "[fixtures] WARNING: FOOTBALL_DATA_API_KEY is not set; "
            "skipping fixture fetch."
        )
        return fixtures

    if target_date is None:
        utc_today = datetime.now(timezone.utc).date()
        target_date = date(utc_today.year, utc_today.month, utc_today.day).isoformat()

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

    return fixtures
