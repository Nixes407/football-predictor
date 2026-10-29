import { useEffect, useState } from 'react';
import { getApiBase } from '../lib/apiBase';
import { canonicalLeague, leagueFlag, leagueName } from '../lib/leagues';

function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-GB', {
    weekday: 'short', day: 'numeric',
    month: 'short', year: 'numeric'
  });
}

function ResultBadge({ correct }) {
  if (correct === null || correct === undefined)
    return (
      <span className="px-2 py-0.5 rounded-full text-xs
        bg-gray-700 text-gray-400">
        ❓ No Data
      </span>
    );
  return correct ? (
    <span className="px-2 py-0.5 rounded-full text-xs
      bg-green-900 text-green-400">
      ✅ Predicted
    </span>
  ) : (
    <span className="px-2 py-0.5 rounded-full text-xs
      bg-red-900 text-red-400">
      ❌ Missed
    </span>
  );
}

function MatchCard({ match }) {
  const homeGoals = match.home_score ?? match.home_goals ?? '?';
  const awayGoals = match.away_score ?? match.away_goals ?? '?';
  const actual    = match.result;

  const probs = {
    H: match.pred_home_win,
    D: match.pred_draw,
    A: match.pred_away_win,
  };

  return (
    <div className="glass-card rounded-xl p-4 mb-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-400">
          {leagueFlag(match.league)} {leagueName(match.league)}
        </span>
        <ResultBadge correct={match.pred_correct} />
      </div>

      <div className="flex items-center justify-between my-2">
        <span className="text-sm font-semibold text-white
          flex-1 text-right">
          {match.home_team}
        </span>
        <span className="mx-3 text-xl font-bold gradient-text">
          {homeGoals} – {awayGoals}
        </span>
        <span className="text-sm font-semibold text-white
          flex-1 text-left">
          {match.away_team}
        </span>
      </div>

      {probs.H !== undefined && (
        <div className="flex gap-2 mt-2 text-xs">
          {[
            { label: 'H', key: 'H', prob: probs.H },
            { label: 'D', key: 'D', prob: probs.D },
            { label: 'A', key: 'A', prob: probs.A },
          ].map(({ label, key, prob }) => (
            <div
              key={key}
              className={`flex-1 text-center py-1 rounded-lg
                ${actual === key
                  ? 'bg-green-800 text-green-300 font-bold'
                  : 'bg-gray-800 text-gray-400'}`}
            >
              {label} {prob !== null
                ? `${(prob * 100).toFixed(1)}%`
                : '—'}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function HistoryTab() {
  const [mode, setMode]       = useState('today');
  const [customDate, setCustomDate] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  async function fetchResults(endpoint) {
    setLoading(true);
    setError(null);
    try {
      const res  = await fetch(`${getApiBase()}${endpoint}`);
      const data = await res.json();
      setResults(data.results || data || []);
    } catch {
      setError('Could not load results — is the API running?');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (mode === 'today')     fetchResults('/results/today');
    if (mode === 'yesterday') fetchResults('/results/yesterday');
  }, [mode]);

  function handleCustomDate() {
    if (!customDate) return;
    fetchResults(`/results/date?date=${customDate}`);
  }

  const byLeague = results.reduce((acc, m) => {
    const k = canonicalLeague(m.league) || 'Unknown';
    if (!acc[k]) acc[k] = [];
    acc[k].push(m);
    return acc;
  }, {});

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold gradient-text mb-4">
        Match History
      </h2>

      {/* Date controls */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {['today', 'yesterday'].map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-xl text-sm font-medium
              transition-all
              ${mode === m
                ? 'bg-blue-600 text-white'
                : 'glass-card text-gray-400 hover:text-white'}`}
          >
            {m === 'today' ? 'Today' : 'Yesterday'}
          </button>
        ))}
        <div className="flex gap-2 flex-1">
          <input
            type="date"
            value={customDate}
            onChange={e => {
              setCustomDate(e.target.value);
              setMode('custom');
            }}
            className="flex-1 px-3 py-2 rounded-xl text-sm
              bg-gray-800 text-white border border-gray-700
              focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={handleCustomDate}
            className="px-4 py-2 rounded-xl text-sm
              bg-blue-600 text-white hover:bg-blue-700"
          >
            📅 Go
          </button>
        </div>
      </div>

      {loading && (
        <p className="text-center text-gray-400 py-8">
          Loading results...
        </p>
      )}
      {error && (
        <p className="text-center text-red-400 py-4">{error}</p>
      )}

      {!loading && !error && results.length === 0 && (
        <p className="text-center text-gray-400 py-8">
          No results found for this date.
        </p>
      )}

      {Object.entries(byLeague).map(([league, matches]) => (
        <div key={league} className="mb-6">
          <h3 className="text-sm font-semibold text-gray-400
            uppercase tracking-wider mb-3">
            {leagueFlag(league)} {leagueName(league)}
          </h3>
          {matches.map((m, i) => (
            <MatchCard key={i} match={m} />
          ))}
        </div>
      ))}
    </div>
  );
}
