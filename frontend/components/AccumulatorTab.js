'use client';

import { useEffect, useState } from 'react';

const MARKETS = [
  'Home Win', 'Draw', 'Away Win',
  'Over 2.5', 'Under 2.5',
  'BTTS Yes', 'BTTS No',
];

function getAcca() {
  try {
    return JSON.parse(localStorage.getItem('accumulator') || '[]');
  } catch { return []; }
}

function saveAcca(selections) {
  localStorage.setItem('accumulator', JSON.stringify(selections));
}

function CombinedProbBar({ prob }) {
  const pct = prob * 100;
  const color = pct > 25
    ? 'bg-gradient-to-r from-green-500 to-green-400'
    : pct > 10
      ? 'bg-gradient-to-r from-yellow-500 to-yellow-400'
      : 'bg-gradient-to-r from-red-600 to-red-500';

  return (
    <div className="mt-4">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">Combined probability</span>
        <span className="font-bold gradient-text">
          {pct.toFixed(2)}%
        </span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-4">
        <div
          className={`h-4 rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>Implied odds</span>
        <span>{(1 / Math.max(prob, 0.001)).toFixed(2)}x</span>
      </div>
    </div>
  );
}

export default function AccumulatorTab() {
  const [selections, setSelections] = useState([]);
  const [showForm, setShowForm]     = useState(false);
  const [form, setForm] = useState({
    match: '', market: MARKETS[0], probability: '', league: ''
  });
  const [formError, setFormError] = useState('');

  useEffect(() => {
    setSelections(getAcca());
    function onStorage(e) {
      if (e.key === 'accumulator') setSelections(getAcca());
    }
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  // Keep legacy unsupported selections in storage, but don't show or include
  // them in the combined probability. Preserve their original indices so
  // removing a visible pick never removes a different stored selection.
  const visibleSelections = selections
    .map((selection, index) => ({ selection, index }))
    .filter(({ selection }) => MARKETS.includes(selection?.market));
  const combined = visibleSelections.reduce(
    (acc, { selection }) => acc * (selection.probability / 100), 1
  );

  function removeSelection(idx) {
    const updated = selections.filter((_, i) => i !== idx);
    saveAcca(updated);
    setSelections(updated);
  }

  function clearAll() {
    const retained = selections.filter((selection) => !MARKETS.includes(selection?.market));
    saveAcca(retained);
    setSelections(retained);
  }

  function handleAdd() {
    setFormError('');
    const prob = parseFloat(form.probability);
    if (!form.match.trim()) {
      setFormError('Match name is required.'); return;
    }
    if (isNaN(prob) || prob <= 0 || prob > 100) {
      setFormError('Probability must be between 0 and 100.'); return;
    }
    const newSel = {
      match:       form.match.trim(),
      market:      form.market,
      probability: prob,
      league:      form.league.trim() || null,
      addedAt:     new Date().toISOString(),
    };
    const updated = [...selections, newSel];
    saveAcca(updated);
    setSelections(updated);
    setForm({ match: '', market: MARKETS[0], probability: '', league: '' });
    setShowForm(false);
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold gradient-text">
          Accumulator Builder
        </h2>
        {visibleSelections.length > 0 && (
          <button
            onClick={clearAll}
            className="text-sm text-red-400 hover:text-red-300
              transition-colors"
          >
            Clear All
          </button>
        )}
      </div>

      {/* Selections list */}
      {visibleSelections.length === 0 ? (
        <div className="glass-card rounded-2xl p-8 text-center
          text-gray-400 mb-6">
          <p className="text-4xl mb-3">🎯</p>
          <p className="font-medium">No selections yet.</p>
          <p className="text-sm mt-1">
            Add picks from the Today tab or use the form below.
          </p>
        </div>
      ) : (
        <div className="glass-card rounded-2xl p-4 mb-6">
          {visibleSelections.map(({ selection: s, index: i }) => (
            <div
              key={i}
              className="flex items-center justify-between
                py-3 border-b border-white/5 last:border-0"
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {s.match}
                </p>
                <p className="text-xs text-gray-400 mt-0.5">
                  {s.market}
                  {s.league ? ` · ${s.league}` : ''}
                </p>
              </div>
              <div className="flex items-center gap-3 ml-3">
                <span className="text-sm font-bold text-green-400">
                  {s.probability.toFixed(1)}%
                </span>
                <button
                  onClick={() => removeSelection(i)}
                  className="text-gray-500 hover:text-red-400
                    transition-colors text-lg leading-none"
                  aria-label="Remove"
                >
                  ×
                </button>
              </div>
            </div>
          ))}

          {/* Combined probability bar */}
          <CombinedProbBar prob={combined} />
        </div>
      )}

      {/* Add manually button */}
      <button
        onClick={() => setShowForm(v => !v)}
        className="w-full py-3 rounded-xl text-sm font-medium
          glass-card text-gray-300 hover:text-white
          border border-white/10 hover:border-blue-500/50
          transition-all mb-4"
      >
        {showForm ? '▲ Hide Form' : '＋ Add Selection Manually'}
      </button>

      {/* Manual form */}
      {showForm && (
        <div className="glass-card rounded-2xl p-5 mb-6
          space-y-4">
          <div>
            <label className="text-xs text-gray-400 mb-1 block">
              Match *
            </label>
            <input
              type="text"
              placeholder="e.g. Arsenal vs Chelsea"
              value={form.match}
              onChange={e => setForm(f => ({
                ...f, match: e.target.value
              }))}
              className="w-full px-3 py-2 rounded-xl text-sm
                bg-gray-800 text-white border border-gray-700
                focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-1 block">
              Market *
            </label>
            <select
              value={form.market}
              onChange={e => setForm(f => ({
                ...f, market: e.target.value
              }))}
              className="w-full px-3 py-2 rounded-xl text-sm
                bg-gray-800 text-white border border-gray-700
                focus:outline-none focus:border-blue-500"
            >
              {MARKETS.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-1 block">
              Probability % *
            </label>
            <input
              type="number"
              min="0.1"
              max="100"
              step="0.1"
              placeholder="e.g. 65.5"
              value={form.probability}
              onChange={e => setForm(f => ({
                ...f, probability: e.target.value
              }))}
              className="w-full px-3 py-2 rounded-xl text-sm
                bg-gray-800 text-white border border-gray-700
                focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-1 block">
              League (optional)
            </label>
            <input
              type="text"
              placeholder="e.g. ENG-Premier League"
              value={form.league}
              onChange={e => setForm(f => ({
                ...f, league: e.target.value
              }))}
              className="w-full px-3 py-2 rounded-xl text-sm
                bg-gray-800 text-white border border-gray-700
                focus:outline-none focus:border-blue-500"
            />
          </div>
          {formError && (
            <p className="text-red-400 text-xs">{formError}</p>
          )}
          <button
            onClick={handleAdd}
            className="w-full py-2.5 rounded-xl text-sm
              font-semibold bg-blue-600 hover:bg-blue-700
              text-white transition-colors"
          >
            Add to Accumulator
          </button>
        </div>
      )}

      {/* Responsible gambling */}
      <div className="glass-card rounded-xl p-4 text-xs
        text-gray-500 text-center">
        ⚠️ This tool is for informational purposes only.
        Please gamble responsibly. Never bet more than
        you can afford to lose.
        <a
          href="https://www.begambleaware.org"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-400 hover:underline ml-1"
        >
          BeGambleAware.org
        </a>
      </div>
    </div>
  );
}
