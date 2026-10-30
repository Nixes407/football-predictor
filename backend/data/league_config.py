"""League-specific training/season configuration shared by backend modules.

The match-history pipeline retains the existing cross-year labels for all
other leagues, including leagues whose other providers use different labels.
"""

from __future__ import annotations

BRA_LEAGUE = "BRA-Série A"

# Only Brazil is switched to calendar-year labels in the training pipeline.
CALENDAR_YEAR_LEAGUES = frozenset({BRA_LEAGUE})

# Applied only to the newly added Brazilian league; existing leagues keep
# their previous training behavior regardless of row count.
MIN_ROWS_PER_LEAGUE = 10


def get_season_string(league: str | None, season: str | None) -> str | None:
    """Normalize a Brazilian season without changing other leagues' labels."""
    if league in CALENDAR_YEAR_LEAGUES and isinstance(season, str):
        return season.split("-", 1)[0].split("/", 1)[0]
    return season
