import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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

function getConfidence(homeWin, draw, awayWin) {
  const maxProb = Math.max(homeWin, draw, awayWin);
  if (maxProb >= 0.65) return { label: '🔥 HIGH', className: 'badge-high' };
  if (maxProb >= 0.5) return { label: '⚡ MEDIUM', className: 'badge-medium' };
  return { label: '❄️ LOW', className: 'badge-low' };
}

function ProbBar({ pct, gradient, highlight }) {
  return (
    <div
      className="w-full rounded-full overflow-hidden"
      style={{ background: 'rgba(148,163,184,0.12)', height: 6 }}
    >
      <motion.div
        className={`prob-bar ${highlight ? 'glow-blue' : ''}`}
        style={{ background: gradient }}
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
      />
    </div>
  );
}

export default function CustomTab() {
  const [leagues, setLeagues] = useState([]);
  const [selectedLeague, setSelectedLeague] = useState('');
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [season, setSeason] = useState('2024-2025');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
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
    setResult(null);
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
    setResult(null);
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
      setResult(data);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }

  // Derived values for the result panel
  let panel = null;
  if (result) {
    const homeWin = result.result?.home_win ?? 0;
    const draw = result.result?.draw ?? 0;
    const awayWin = result.result?.away_win ?? 0;
    const homeGoals = result.goals?.expected_home ?? 0;
    const awayGoals = result.goals?.expected_away ?? 0;
    const homeCorners = result.corners?.expected_home ?? 0;
    const awayCorners = result.corners?.expected_away ?? 0;
    const maxProb = Math.max(homeWin, draw, awayWin);
    const totalGoals = homeGoals + awayGoals;
    const totalCorners = homeCorners + awayCorners;
    const btts = homeGoals >= 1.0 && awayGoals >= 1.0;
    const conf = getConfidence(homeWin, draw, awayWin);

    const outcomes = [
      {
        key: 'home',
        name: result.home_team,
        prob: homeWin,
        gradient: 'linear-gradient(90deg, var(--blue), var(--green))',
      },
      {
        key: 'draw',
        name: 'Draw',
        prob: draw,
        gradient: 'linear-gradient(90deg, var(--gold), #fbbf24)',
      },
      {
        key: 'away',
        name: result.away_team,
        prob: awayWin,
        gradient: 'linear-gradient(90deg, var(--purple), var(--blue))',
      },
    ];

    panel = (
      <motion.div
        key="result"
        className="glass p-6 mt-6"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -16 }}
        transition={{ duration: 0.3 }}
      >
        {/* Heading + confidence */}
        <div className="flex items-start justify-between gap-3 mb-5">
          <h3 className="text-xl md:text-2xl font-bold gradient-text">
            {result.home_team} vs {result.away_team}
          </h3>
          <span
            className={`${conf.className} text-xs font-bold px-3 py-1 rounded-full whitespace-nowrap`}
          >
            {conf.label}
          </span>
        </div>

        {/* Match Result */}
        <div className="mb-5">
          <h4 className="text-sm font-bold mb-3">⚽ Match Result</h4>
          <div className="grid grid-cols-3 gap-3">
            {outcomes.map((o) => {
              const highlight = o.prob === maxProb && maxProb > 0;
              return (
                <div
                  key={o.key}
                  className={`rounded-xl p-3 text-center ${
                    highlight ? 'gradient-border' : ''
                  }`}
                  style={{
                    background: highlight
                      ? 'rgba(0,212,255,0.06)'
                      : 'rgba(148,163,184,0.04)',
                  }}
                >
                  <div
                    className="text-xs mb-1 truncate"
                    style={{ color: 'var(--text-muted)' }}
                    title={o.name}
                  >
                    {o.name}
                  </div>
                  <div
                    className={`text-2xl font-bold mb-2 ${
                      highlight ? 'gradient-text' : ''
                    }`}
                  >
                    {(o.prob * 100).toFixed(1)}%
                  </div>
                  <ProbBar
                    pct={o.prob * 100}
                    gradient={o.gradient}
                    highlight={highlight}
                  />
                </div>
              );
            })}
          </div>
        </div>

        {/* Goals & Corners */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-5">
          <div
            className="rounded-xl p-4"
            style={{ background: 'rgba(148,163,184,0.04)' }}
          >
            <h4 className="text-sm font-bold mb-3">⚽ Expected Goals</h4>
            <div className="flex items-center justify-around">
              <div className="text-center">
                <div
                  className="text-3xl font-bold"
                  style={{ color: 'var(--green)' }}
                >
                  {homeGoals.toFixed(2)}
                </div>
                <div
                  className="text-xs mt-1"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Home
                </div>
              </div>
              <div className="text-center">
                <div
                  className="text-3xl font-bold"
                  style={{ color: 'var(--blue)' }}
                >
                  {awayGoals.toFixed(2)}
                </div>
                <div
                  className="text-xs mt-1"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Away
                </div>
              </div>
            </div>
          </div>
          <div
            className="rounded-xl p-4"
            style={{ background: 'rgba(148,163,184,0.04)' }}
          >
            <h4 className="text-sm font-bold mb-3">📐 Expected Corners</h4>
            <div className="flex items-center justify-around">
              <div className="text-center">
                <div
                  className="text-3xl font-bold"
                  style={{ color: 'var(--green)' }}
                >
                  {homeCorners.toFixed(2)}
                </div>
                <div
                  className="text-xs mt-1"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Home
                </div>
              </div>
              <div className="text-center">
                <div
                  className="text-3xl font-bold"
                  style={{ color: 'var(--blue)' }}
                >
                  {awayCorners.toFixed(2)}
                </div>
                <div
                  className="text-xs mt-1"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Away
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Markets */}
        <div className="flex flex-wrap gap-2">
          <span
            className={`px-4 py-1.5 rounded-full text-sm font-semibold ${
              totalGoals >= 2.5 ? 'badge-high' : 'badge-low'
            }`}
          >
            {totalGoals >= 2.5 ? 'Over 2.5 ✓' : 'Under 2.5'}
          </span>
          <span
            className={`px-4 py-1.5 rounded-full text-sm font-semibold ${
              btts ? 'badge-high' : 'badge-low'
            }`}
          >
            {btts ? 'BTTS Yes ✓' : 'BTTS No'}
          </span>
          <span
            className={`px-4 py-1.5 rounded-full text-sm font-semibold ${
              totalCorners >= 9.5 ? 'badge-high' : 'badge-low'
            }`}
          >
            {totalCorners >= 9.5 ? 'Over 9.5 Corners ✓' : 'Under 9.5 Corners'}
          </span>
        </div>
      </motion.div>
    );
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

      {/* Result */}
      <AnimatePresence>{panel}</AnimatePresence>
    </div>
  );
}
