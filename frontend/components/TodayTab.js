import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { getApiBase } from '../lib/apiBase';

const LEAGUE_FLAGS = {
  'ENG-Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  'ESP-La Liga': '🇪🇸',
  'GER-Bundesliga': '🇩🇪',
  'ITA-Serie A': '🇮🇹',
  'FRA-Ligue 1': '🇫🇷',
};

const LEAGUE_NAMES = {
  'ENG-Premier League': 'Premier League',
  'ESP-La Liga': 'La Liga',
  'GER-Bundesliga': 'Bundesliga',
  'ITA-Serie A': 'Serie A',
  'FRA-Ligue 1': 'Ligue 1',
};

function getConfidence(homeWin, draw, awayWin) {
  const maxProb = Math.max(homeWin, draw, awayWin);
  if (maxProb > 0.65) return { label: '🔥 HIGH', className: 'badge-high' };
  if (maxProb > 0.5) return { label: '⚡ MEDIUM', className: 'badge-medium' };
  return { label: '❄️ LOW', className: 'badge-low' };
}

function isValidProbability(value) {
  return (
    value !== null &&
    value !== undefined &&
    value !== '' &&
    Number.isFinite(Number(value))
  );
}

function formatProbability(value) {
  if (!isValidProbability(value)) return '—';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatKickoff(kickoffTime) {
  if (!kickoffTime) return '--:--';
  const d = new Date(kickoffTime);
  if (isNaN(d.getTime())) return '--:--';
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addToAcca(match, market, probability, league) {
  try {
    const existing = JSON.parse(
      localStorage.getItem('accumulator') || '[]'
    );
    const newSel = {
      match:       `${match.home_team} vs ${match.away_team}`,
      market,
      probability: parseFloat((probability * 100).toFixed(1)),
      league:      league || match.league || null,
      addedAt:     new Date().toISOString(),
    };
    localStorage.setItem(
      'accumulator',
      JSON.stringify([...existing, newSel])
    );
    window.dispatchEvent(new Event('storage'));
    return true;
  } catch { return false; }
}

function AccaButtons({ match, prediction }) {
  const [toast, setToast] = useState('');

  function handleAdd(market, prob) {
    const ok = addToAcca(match, market, prob, match.league);
    if (ok) {
      setToast(`✅ ${market} added!`);
      setTimeout(() => setToast(''), 2000);
    }
  }

  const buttons = [
    {
      label: `Home Win ${formatProbability(prediction.result?.home_win)}`,
      market: 'Home Win',
      prob:   prediction.result?.home_win,
    },
    {
      label: `Draw ${formatProbability(prediction.result?.draw)}`,
      market: 'Draw',
      prob:   prediction.result?.draw,
    },
    {
      label: `Away Win ${formatProbability(prediction.result?.away_win)}`,
      market: 'Away Win',
      prob:   prediction.result?.away_win,
    },
  ];

  return (
    <div className="mt-3">
      {toast && (
        <p className="text-xs text-green-400 text-center mb-2">
          {toast}
        </p>
      )}
      <div className="flex gap-2">
        {buttons.map(({ label, market, prob }) => (
          <button
            key={market}
            onClick={() => handleAdd(market, prob)}
            className="flex-1 py-1.5 rounded-lg text-xs
              font-medium bg-gray-800 hover:bg-blue-900
              text-gray-300 hover:text-white
              border border-white/10 hover:border-blue-500/50
              transition-all"
          >
            + {label}
          </button>
        ))}
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  if (status === 'in_play') {
    return (
      <span className="pulse-glow text-xs font-bold px-3 py-1 rounded-full badge-high">
        🔴 LIVE
      </span>
    );
  }
  if (status === 'finished') {
    return (
      <span className="text-xs font-bold px-3 py-1 rounded-full badge-low">
        ✅ FT
      </span>
    );
  }
  return (
    <span
      className="text-xs font-bold px-3 py-1 rounded-full"
      style={{
        background: 'rgba(0,212,255,0.12)',
        color: 'var(--blue)',
        border: '1px solid rgba(0,212,255,0.3)',
      }}
    >
      🕐 Upcoming
    </span>
  );
}

function PredictionPanel({ fixture }) {
  const homeWin = Number(fixture.pred_home_win ?? 0);
  const draw = Number(fixture.pred_draw ?? 0);
  const awayWin = Number(fixture.pred_away_win ?? 0);
  const maxProb = Math.max(homeWin, draw, awayWin);

  const prediction = {
    result: { home_win: homeWin, draw, away_win: awayWin },
  };

  const outcomes = [
    { key: 'home', label: 'Home Win', prob: homeWin },
    { key: 'draw', label: 'Draw', prob: draw },
    { key: 'away', label: 'Away Win', prob: awayWin },
  ];

  const quickMarkets = [
    { label: 'Over 2.5', probability: fixture.pred_over_2_5 },
    { label: 'BTTS', probability: fixture.pred_btts },
    {
      label: 'Corners Over 9.5',
      probability: fixture.pred_corners_over_9_5,
    },
  ];

  return (
    <div
      className="mt-4 pt-4 space-y-4"
      style={{ borderTop: '1px solid var(--border)' }}
    >
      <section className="glass-card rounded-2xl p-4 sm:p-5">
        <h4 className="text-sm font-bold uppercase tracking-wider mb-3 text-gray-400">
          Result Probabilities
        </h4>
        <div className="grid grid-cols-3 gap-2 sm:gap-3">
          {outcomes.map((outcome) => {
            const highlight = outcome.prob === maxProb;
            return (
              <div
                key={outcome.key}
                className={`min-w-0 rounded-xl border p-3 text-center transition-all ${
                  highlight
                    ? 'border-green-500/50 bg-green-800 text-green-300 font-bold'
                    : 'border-white/10 bg-gray-800 text-gray-300'
                }`}
              >
                <div className="text-xs mb-1 text-gray-400">
                  {outcome.label}
                </div>
                <div className="text-xl font-bold">
                  {formatProbability(outcome.prob)}
                </div>
              </div>
            );
          })}
        </div>
        <AccaButtons match={fixture} prediction={prediction} />
      </section>

      <section className="glass-card rounded-2xl p-4 sm:p-5">
        <h4 className="text-sm font-bold uppercase tracking-wider mb-3 text-gray-400">
          Quick Markets
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {quickMarkets.map((market) => (
            <div
              key={market.label}
              className="rounded-xl border border-white/10 bg-gray-800/80 p-3"
            >
              <div className="text-xs text-gray-400">{market.label}</div>
              <div className="text-lg font-bold text-white mt-1">
                {formatProbability(market.probability)}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default function TodayTab() {
  const [fixtures, setFixtures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch(`${getApiBase()}/fixtures/today`);
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        const data = await res.json();
        if (!cancelled) {
          setFixtures(Array.isArray(data) ? data : []);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Failed to load fixtures');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const toggleExpanded = (id) =>
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));

  const todayLabel = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  // Group fixtures by league, keeping a stable league order.
  const grouped = {};
  for (const fx of fixtures) {
    const league = fx.league || 'Other';
    if (!grouped[league]) grouped[league] = [];
    grouped[league].push(fx);
  }
  const leagueOrder = [
    ...Object.keys(LEAGUE_FLAGS).filter((l) => grouped[l]),
    ...Object.keys(grouped).filter((l) => !LEAGUE_FLAGS[l]),
  ];

  return (
    <div>
      {/* Header row */}
      <div className="flex items-end justify-between flex-wrap gap-2 mb-6">
        <h2 className="text-2xl font-bold">📅 Today&apos;s Matches</h2>
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {todayLabel}
        </span>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <div
            className="w-10 h-10 rounded-full animate-spin"
            style={{
              border: '3px solid rgba(0,212,255,0.2)',
              borderTopColor: 'var(--blue)',
            }}
          />
          <p style={{ color: 'var(--text-muted)' }}>Loading fixtures...</p>
        </div>
      )}

      {/* Error state */}
      {!loading && error && (
        <div
          className="glass p-6 text-center"
          style={{
            border: '1px solid rgba(239,68,68,0.4)',
            background: 'rgba(239,68,68,0.08)',
          }}
        >
          <p className="font-semibold" style={{ color: 'var(--red)' }}>
            ⚠️ Could not load fixtures
          </p>
          <p className="text-sm mt-2" style={{ color: 'var(--text-muted)' }}>
            {error}
          </p>
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && fixtures.length === 0 && (
        <div className="flex justify-center py-12">
          <div className="glass px-10 py-12 text-center max-w-lg">
            <div className="text-5xl mb-4">😴</div>
            <p className="text-lg font-semibold mb-2">
              No matches today across tracked leagues
            </p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Premier League · La Liga · Bundesliga · Serie A · Ligue 1
            </p>
          </div>
        </div>
      )}

      {/* Fixtures grouped by league */}
      {!loading &&
        !error &&
        leagueOrder.map((league) => (
          <section key={league} className="mb-8">
            {/* League header */}
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <span>{LEAGUE_FLAGS[league] || '🌍'}</span>
                <span className="gradient-text">
                  {LEAGUE_NAMES[league] || league}
                </span>
              </h3>
              <span className="badge-low text-xs font-semibold px-3 py-1 rounded-full">
                {grouped[league].length}{' '}
                {grouped[league].length === 1 ? 'match' : 'matches'}
              </span>
            </div>

            {/* Match cards */}
            <div className="space-y-4">
              {grouped[league].map((fx, fixtureIndex) => {
                const fixtureKey =
                  fx.id ??
                  `${league}:${fx.home_team}:${fx.away_team}:${
                    fx.kickoff ?? fx.kickoff_time ?? fixtureIndex
                  }`;
                // The API flags every fixture with `has_prediction` (true only
                // when all three result probabilities are stored). Fixtures
                // without one are still listed — they just show a "not
                // available" message instead of the prediction panel. The
                // local probability check guards against a stale/incorrect
                // flag so the panel can never render NaN values.
                const hasValidProbs = [
                  fx.pred_home_win,
                  fx.pred_draw,
                  fx.pred_away_win,
                ].every(isValidProbability);
                const hasPrediction =
                  fx.has_prediction !== false && hasValidProbs;
                // Confidence badge is derived only when a prediction exists.
                const conf = hasPrediction
                  ? getConfidence(
                      Number(fx.pred_home_win),
                      Number(fx.pred_draw),
                      Number(fx.pred_away_win)
                    )
                  : null;
                return (
                  <motion.div
                    key={fixtureKey}
                    className="glass p-5"
                    whileHover={{ y: -2 }}
                  >
                    {/* Top row */}
                    <div className="flex items-center justify-between mb-4">
                      <span
                        className="text-sm font-semibold"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        {formatKickoff(fx.kickoff ?? fx.kickoff_time)}
                      </span>
                      <StatusBadge status={fx.status} />
                      {/* Confidence badge — only shown when a prediction exists */}
                      {conf && (
                        <span
                          className={`${conf.className} text-xs font-bold px-3 py-1 rounded-full`}
                        >
                          {conf.label}
                        </span>
                      )}
                    </div>

                    {/* Teams row */}
                    <div className="grid grid-cols-3 items-center gap-2 mb-4">
                      <div className="text-right">
                        <div className="text-lg md:text-xl font-bold leading-tight">
                          {fx.home_team}
                        </div>
                        <div
                          className="text-xs mt-1"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          HOME
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold gradient-text">VS</div>
                        {hasPrediction && (
                          <div className="flex justify-center gap-2 text-xs font-semibold mt-1">
                            <span style={{ color: 'var(--green)' }}>
                              {(fx.pred_home_win * 100).toFixed(0)}%
                            </span>
                            <span style={{ color: 'var(--text-muted)' }}>
                              {(fx.pred_draw * 100).toFixed(0)}%
                            </span>
                            <span style={{ color: 'var(--blue)' }}>
                              {(fx.pred_away_win * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                      </div>
                      <div className="text-left">
                        <div className="text-lg md:text-xl font-bold leading-tight">
                          {fx.away_team}
                        </div>
                        <div
                          className="text-xs mt-1"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          AWAY
                        </div>
                      </div>
                    </div>

                    {hasPrediction ? (
                      <>
                        {/* Get prediction button */}
                        <button
                          className="btn-neon w-full py-2.5 text-sm"
                          onClick={() => toggleExpanded(fixtureKey)}
                        >
                          {expanded[fixtureKey] ? '🔼 Hide Prediction' : '🔮 Get Prediction'}
                        </button>

                        {/* Prediction panel */}
                        <AnimatePresence initial={false}>
                          {expanded[fixtureKey] && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.3 }}
                              style={{ overflow: 'hidden' }}
                            >
                              <PredictionPanel fixture={fx} />
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </>
                    ) : (
                      <div className="mt-3 text-xs text-gray-500 text-center py-2">
                        Prediction not available for this match
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </section>
        ))}
    </div>
  );
}
