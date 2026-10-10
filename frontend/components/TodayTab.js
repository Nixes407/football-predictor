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
  if (maxProb >= 0.65) return { label: '🔥 HIGH', className: 'badge-high' };
  if (maxProb >= 0.5) return { label: '⚡ MEDIUM', className: 'badge-medium' };
  return { label: '❄️ LOW', className: 'badge-low' };
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
      label: `H ${(prediction.result?.home_win * 100).toFixed(1)}%`,
      market: 'Home Win',
      prob:   prediction.result?.home_win,
    },
    {
      label: `D ${(prediction.result?.draw * 100).toFixed(1)}%`,
      market: 'Draw',
      prob:   prediction.result?.draw,
    },
    {
      label: `A ${(prediction.result?.away_win * 100).toFixed(1)}%`,
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

function PredictionPanel({ fixture }) {
  const homeWin = fixture.pred_home_win ?? 0;
  const draw = fixture.pred_draw ?? 0;
  const awayWin = fixture.pred_away_win ?? 0;
  const homeGoals = fixture.pred_home_goals ?? 0;
  const awayGoals = fixture.pred_away_goals ?? 0;
  const homeCorners = fixture.pred_home_corners ?? 0;
  const awayCorners = fixture.pred_away_corners ?? 0;

  const prediction = {
    result: { home_win: homeWin, draw, away_win: awayWin },
  };

  const maxProb = Math.max(homeWin, draw, awayWin);
  const totalGoals = homeGoals + awayGoals;
  const totalCorners = homeCorners + awayCorners;
  const btts = homeGoals >= 1.0 && awayGoals >= 1.0;

  const outcomes = [
    {
      key: 'home',
      name: fixture.home_team,
      sub: 'Home Win',
      prob: homeWin,
      gradient: 'linear-gradient(90deg, var(--blue), var(--green))',
    },
    {
      key: 'draw',
      name: 'Draw',
      sub: 'Draw',
      prob: draw,
      gradient: 'linear-gradient(90deg, var(--gold), #fbbf24)',
    },
    {
      key: 'away',
      name: fixture.away_team,
      sub: 'Away Win',
      prob: awayWin,
      gradient: 'linear-gradient(90deg, var(--purple), var(--blue))',
    },
  ];

  return (
    <div
      className="mt-4 pt-4 space-y-5"
      style={{ borderTop: '1px solid var(--border)' }}
    >
      {/* ── Section 1: Match Result ── */}
      <div>
        <h4 className="text-sm font-bold mb-3" style={{ color: 'var(--text)' }}>
          ⚽ Match Result
        </h4>
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

      {prediction && (
        <AccaButtons match={fixture} prediction={prediction} />
      )}

      {/* ── Section 2: Goals & Corners ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div
          className="rounded-xl p-4"
          style={{ background: 'rgba(148,163,184,0.04)' }}
        >
          <h4 className="text-sm font-bold mb-3">⚽ Expected Goals</h4>
          <div className="flex items-center justify-around">
            <div className="text-center">
              <div className="text-3xl font-bold" style={{ color: 'var(--green)' }}>
                {homeGoals.toFixed(2)}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                Home
              </div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold" style={{ color: 'var(--blue)' }}>
                {awayGoals.toFixed(2)}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
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
              <div className="text-3xl font-bold" style={{ color: 'var(--green)' }}>
                {homeCorners.toFixed(2)}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                Home
              </div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold" style={{ color: 'var(--blue)' }}>
                {awayCorners.toFixed(2)}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                Away
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Section 3: Quick Markets ── */}
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
              {grouped[league].map((fx) => {
                const hasPreds =
                  fx.pred_home_win != null &&
                  fx.pred_draw != null &&
                  fx.pred_away_win != null;
                const conf = hasPreds
                  ? getConfidence(fx.pred_home_win, fx.pred_draw, fx.pred_away_win)
                  : null;
                return (
                  <motion.div
                    key={fx.id}
                    className="glass p-5"
                    whileHover={{ y: -2 }}
                  >
                    {/* Top row */}
                    <div className="flex items-center justify-between mb-4">
                      <span
                        className="text-sm font-semibold"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        {formatKickoff(fx.kickoff_time)}
                      </span>
                      <StatusBadge status={fx.status} />
                      {conf ? (
                        <span
                          className={`${conf.className} text-xs font-bold px-3 py-1 rounded-full`}
                        >
                          {conf.label}
                        </span>
                      ) : (
                        <span className="badge-low text-xs font-bold px-3 py-1 rounded-full">
                          ❔ N/A
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
                        {hasPreds && (
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

                    {/* Get prediction button */}
                    <button
                      className="btn-neon w-full py-2.5 text-sm"
                      onClick={() => toggleExpanded(fx.id)}
                    >
                      {expanded[fx.id] ? '🔼 Hide Prediction' : '🔮 Get Prediction'}
                    </button>

                    {/* Prediction panel */}
                    <AnimatePresence initial={false}>
                      {expanded[fx.id] && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.3 }}
                          style={{ overflow: 'hidden' }}
                        >
                          {hasPreds ? (
                            <PredictionPanel fixture={fx} />
                          ) : (
                            <p
                              className="mt-4 pt-4 text-sm text-center"
                              style={{
                                color: 'var(--text-muted)',
                                borderTop: '1px solid var(--border)',
                              }}
                            >
                              Prediction not available for this match.
                            </p>
                          )}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })}
            </div>
          </section>
        ))}
    </div>
  );
}
