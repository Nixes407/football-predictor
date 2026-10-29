import { useState, useEffect } from 'react';
import PredictionResult from './PredictionResult';
import { getApiBase } from '../lib/apiBase';
import { BRAZIL_LEAGUE, canonicalLeague, leagueFlag, leagueName } from '../lib/leagues';

const SEASONS = ['2024-2025', '2025-2026', '2026-2027'];

const selectStyle = {
  width: '100%',
  padding: '0.65rem 0.9rem',
  borderRadius: '10px',
  background: 'var(--navy-light)',
  color: 'var(--text)',
  border: '1px solid var(--border)',
  outline: 'none',
  fontFamily: 'inherit',
  fontSize: '0.95rem',
};

export default function CustomTab() {
  const [leagues, setLeagues] = useState([BRAZIL_LEAGUE]);
  const [selectedLeague, setSelectedLeague] = useState('');
  const [teams, setTeams] = useState([]);
  const [teamsStatus, setTeamsStatus] = useState('idle');
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [season, setSeason] = useState('2024-2025');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  // The encoder-backed API list need not include leagues supported by the
  // formula engine, so keep the canonical Brazilian league available too.
  useEffect(() => {
    let cancelled = false;
    fetch(`${getApiBase()}/predict/leagues`)
      .then((res) => {
        if (!res.ok) throw new Error('Could not load leagues');
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        const names = Array.isArray(data.leagues) ? data.leagues : [];
        const canonical = names.filter((name) => typeof name === 'string')
          .map(canonicalLeague);
        setLeagues([...new Set([...canonical, BRAZIL_LEAGUE])].sort());
      })
      .catch(() => {
        if (!cancelled) setError('Could not load leagues — is the API awake?');
      });
    return () => { cancelled = true; };
  }, []);

  // Load teams when the league changes; ignore late responses from a previous
  // selection so they cannot replace the newly selected league's teams.
  useEffect(() => {
    let cancelled = false;
    setTeams([]);
    setHomeTeam('');
    setAwayTeam('');
    setPrediction(null);
    if (!selectedLeague) {
      setTeamsStatus('idle');
      return () => { cancelled = true; };
    }
    setTeamsStatus('loading');
    fetch(
      `${getApiBase()}/predict/teams?league=${encodeURIComponent(selectedLeague)}`
    )
      .then((res) => {
        if (!res.ok) throw new Error('Could not load teams');
        return res.json();
      })
      .then((data) => {
        if (!cancelled) {
          setTeams(Array.isArray(data.teams) ? data.teams : []);
          setTeamsStatus('ready');
        }
      })
      .catch(() => {
        if (!cancelled) {
          setTeamsStatus('error');
          setError('Could not load teams for this league.');
        }
      });
    return () => { cancelled = true; };
  }, [selectedLeague]);

  const manualTeams = selectedLeague === BRAZIL_LEAGUE &&
    (teamsStatus === 'ready' || teamsStatus === 'error') && teams.length === 0;
  const currentYear = new Date().getUTCFullYear();
  const seasonOptions = selectedLeague === BRAZIL_LEAGUE
    ? [currentYear, currentYear - 1, currentYear - 2].map(String)
    : SEASONS;

  function handleLeagueChange(event) {
    const nextLeague = event.target.value;
    if (nextLeague === BRAZIL_LEAGUE && selectedLeague !== BRAZIL_LEAGUE) {
      setSeason(String(new Date().getUTCFullYear()));
    } else if (selectedLeague === BRAZIL_LEAGUE && nextLeague !== BRAZIL_LEAGUE) {
      setSeason('2024-2025');
    }
    setError(null);
    setSelectedLeague(nextLeague);
  }

  const canSubmit = !loading && selectedLeague && homeTeam.trim() &&
    awayTeam.trim() && homeTeam.trim() !== awayTeam.trim() && season;

  async function handleSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const res = await fetch(`${getApiBase()}/predict/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          home_team: homeTeam.trim(),
          away_team: awayTeam.trim(),
          league: selectedLeague,
          season,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Server responded with ${res.status}`);
      }
      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-1">🔮 Custom Prediction</h2>
      <p className="mb-6" style={{ color: 'var(--text-muted)' }}>
        Select any matchup to get an instant prediction
      </p>

      <form className="glass p-6 space-y-5" onSubmit={handleSubmit}>
        {/* League */}
        <div>
          <label
            htmlFor="custom-league"
            className="block text-sm font-semibold mb-2"
            style={{ color: 'var(--text-muted)' }}
          >
            League
          </label>
          <select
            id="custom-league"
            style={selectStyle}
            value={selectedLeague}
            onChange={handleLeagueChange}
            onFocus={(e) => (e.target.style.borderColor = 'var(--blue)')}
            onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
          >
            <option value="">Select a league...</option>
            {leagues.map((l) => (
              <option key={l} value={l}>
                {leagueFlag(l)} {leagueName(l)}
              </option>
            ))}
          </select>
        </div>

        {/* Teams */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label
              htmlFor="custom-home-team"
              className="block text-sm font-semibold mb-2"
              style={{ color: 'var(--text-muted)' }}
            >
              Home Team
            </label>
            {manualTeams ? (
              <input
                id="custom-home-team"
                style={selectStyle}
                value={homeTeam}
                onChange={(e) => setHomeTeam(e.target.value)}
                placeholder="Enter home team"
              />
            ) : (
              <select
                id="custom-home-team"
                style={{
                  ...selectStyle,
                  opacity: selectedLeague ? 1 : 0.5,
                }}
                value={homeTeam}
                onChange={(e) => setHomeTeam(e.target.value)}
                disabled={!selectedLeague || teamsStatus === 'loading'}
                onFocus={(e) => (e.target.style.borderColor = 'var(--blue)')}
                onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
              >
                <option value="">Select home team...</option>
                {teams.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            )}
          </div>
          <div>
            <label
              htmlFor="custom-away-team"
              className="block text-sm font-semibold mb-2"
              style={{ color: 'var(--text-muted)' }}
            >
              Away Team
            </label>
            {manualTeams ? (
              <input
                id="custom-away-team"
                style={selectStyle}
                value={awayTeam}
                onChange={(e) => setAwayTeam(e.target.value)}
                placeholder="Enter away team"
              />
            ) : (
              <select
                id="custom-away-team"
                style={{
                  ...selectStyle,
                  opacity: selectedLeague ? 1 : 0.5,
                }}
                value={awayTeam}
                onChange={(e) => setAwayTeam(e.target.value)}
                disabled={!selectedLeague || teamsStatus === 'loading'}
                onFocus={(e) => (e.target.style.borderColor = 'var(--blue)')}
                onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
              >
                <option value="">Select away team...</option>
                {teams
                  .filter((t) => t !== homeTeam)
                  .map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
              </select>
            )}
          </div>
        </div>
        {manualTeams && (
          <p className="text-sm text-gray-400">
            No team list is available for Brazilian Série A. Enter team names
            exactly as recognised by the prediction API.
          </p>
        )}

        {/* Season */}
        <div>
          <label
            htmlFor="custom-season"
            className="block text-sm font-semibold mb-2"
            style={{ color: 'var(--text-muted)' }}
          >
            Season
          </label>
          <select
            id="custom-season"
            style={selectStyle}
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            onFocus={(e) => (e.target.style.borderColor = 'var(--blue)')}
            onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
          >
            {seasonOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        {/* Submit */}
        <button
          type="submit"
          className="btn-neon w-full py-3 text-base"
          disabled={!canSubmit}
          style={!canSubmit ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
        >
          {loading ? '⏳ Predicting...' : '🔮 Get Prediction'}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div
          className="glass p-4 mt-4 text-center text-sm"
          style={{
            border: '1px solid rgba(239,68,68,0.4)',
            background: 'rgba(239,68,68,0.08)',
            color: 'var(--red)',
          }}
        >
          ⚠️ {error}
        </div>
      )}

      <PredictionResult data={prediction} />
    </div>
  );
}
