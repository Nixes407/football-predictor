export const BRAZIL_LEAGUE = 'BRA-Série A';

// Only normalise known Brazilian aliases; in particular, Italy's Serie A is
// a different competition. API requests always use the accented canonical key.
export function canonicalLeague(league) {
  if (typeof league !== 'string') return league;
  const key = league
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[\s-]+/g, '');
  return ['bsa', 'braseriea', 'brazilseriea', 'brazilianseriea', 'brasilseriea']
    .includes(key) ? BRAZIL_LEAGUE : league;
}

export const LEAGUE_FLAGS = {
  'ENG-Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  'ESP-La Liga': '🇪🇸',
  'GER-Bundesliga': '🇩🇪',
  'ITA-Serie A': '🇮🇹',
  'FRA-Ligue 1': '🇫🇷',
  [BRAZIL_LEAGUE]: '🇧🇷',
};

const LEAGUE_NAMES = {
  'ENG-Premier League': 'Premier League',
  'ESP-La Liga': 'La Liga',
  'GER-Bundesliga': 'Bundesliga',
  'ITA-Serie A': 'Serie A',
  'FRA-Ligue 1': 'Ligue 1',
  [BRAZIL_LEAGUE]: 'Brazilian Série A',
};

export function leagueFlag(league) {
  return LEAGUE_FLAGS[canonicalLeague(league)] || '';
}

export function leagueName(league) {
  return LEAGUE_NAMES[canonicalLeague(league)] || league || 'Unknown';
}
