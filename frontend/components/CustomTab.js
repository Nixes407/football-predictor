import { useState, useEffect } from 'react';
import PredictionResult from './PredictionResult';
import { getApiBase } from '../lib/apiBase';

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
  const [leagues, setLeagues] = useState([]);
  const [selectedLeague, setSelectedLeague] = useState('');
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [season, setSeason] = useState('2024-2025');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  // Load leagues on mount
  useEffect(() => {
    fetch(`${getApiBase()}/predict/leagues`)
      .then((res) => res.json())
      .then((data) => setLeagues(data.leagues || []))
      .catch(() => setError('Could not load leagues — is the API awake?'));
  }, []);

  // Load teams when league changes
  useEffect(() => {
    if (!selectedLeague) {
      setTeams([]);
      return;
    }
    setHomeTeam('');
    setAwayTeam('');
    setPrediction(null);
    fetch(
      `${getApiBase()}/predict/teams?league=${encodeURIComponent(selectedLeague)}`
    )
      .then((res) => res.json())
      .then((data) => setTeams(data.teams || []))
      .catch(() => setError('Could not load teams for this league.'));
  }, [selectedLeague]);

  const canSubmit =
    !loading && selectedLeague && homeTeam && awayTeam && season;

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
          home_team: homeTeam,
          away_team: awayTeam,
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
            className="block text-sm font-semibold mb-2"
            style={{ color: 'var(--text-muted)' }}
          >
            League
          </label>
          <select
            style={selectStyle}
            value={selectedLeague}
            onChange={(e) => setSelectedLeague(e.target.value)}
            onFocus={(e) => (e.target.style.borderColor = 'var(--blue)')}
            onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
          >
            <option value="">Select a league...</option>
            {leagues.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </div>

        {/* Teams */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label
              className="block text-sm font-semibold mb-2"
              style={{ color: 'var(--text-muted)' }}
            >
              Home Team
            </label>
            <select
              style={{
                ...selectStyle,
                opacity: selectedLeague ? 1 : 0.5,
              }}
              value={homeTeam}
              onChange={(e) => setHomeTeam(e.target.value)}
              disabled={!selectedLeague}
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
          </div>
          <div>
            <label
              className="block text-sm font-semibold mb-2"
              style={{ color: 'var(--text-muted)' }}
            >
              Away Team
            </label>
            <select
              style={{
                ...selectStyle,
                opacity: selectedLeague ? 1 : 0.5,
              }}
              value={awayTeam}
              onChange={(e) => setAwayTeam(e.target.value)}
              disabled={!selectedLeague}
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
          </div>
        </div>

        {/* Season */}
        <div>
          <label
            className="block text-sm font-semibold mb-2"
            style={{ color: 'var(--text-muted)' }}
          >
            Season
          </label>
          <select
            style={selectStyle}
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            onFocus={(e) => (e.target.style.borderColor = 'var(--blue)')}
            onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
          >
            {SEASONS.map((s) => (
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
