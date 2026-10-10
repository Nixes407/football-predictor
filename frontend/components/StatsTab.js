import { useEffect, useState } from 'react';
import { getApiBase } from '../lib/apiBase';

const LEAGUE_FLAGS = {
  'ENG-Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  'ESP-La Liga':        '🇪🇸',
  'GER-Bundesliga':     '🇩🇪',
  'ITA-Serie A':        '🇮🇹',
  'FRA-Ligue 1':        '🇫🇷',
};

function AccuracyBar({ pct, color }) {
  return (
    <div className="w-full bg-gray-800 rounded-full h-2 mt-1">
      <div
        className={`h-2 rounded-full transition-all duration-700
          ${color}`}
        style={{ width: `${Math.min(pct, 100)}%` }}
      />
    </div>
  );
}

function barColor(pct) {
  if (pct >= 60) return 'bg-green-500';
  if (pct >= 50) return 'bg-yellow-500';
  return 'bg-red-500';
}

export default function StatsTab() {
  const [stats, setStats]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const res  = await fetch(`${getApiBase()}/results/accuracy`);
        const data = await res.json();
        setStats(data);
      } catch {
        setError('Could not load stats — is the API running?');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return (
    <p className="text-center text-gray-400 py-8">
      Loading stats...
    </p>
  );
  if (error) return (
    <p className="text-center text-red-400 py-8">{error}</p>
  );
  if (!stats) return null;

  const overallPct = stats.accuracy_pct ?? 0;

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold gradient-text mb-6">
        Prediction Accuracy
      </h2>

      {/* Overall stats */}
      <div className="glass-card rounded-2xl p-6 mb-6">
        <h3 className="text-sm font-semibold text-gray-400
          uppercase tracking-wider mb-4">
          Overall Performance
        </h3>
        <div className="grid grid-cols-3 gap-4 mb-4">
          {[
            { label: 'Predicted', value: stats.total_predicted ?? 0 },
            { label: 'Correct',   value: stats.correct ?? 0 },
            { label: 'Accuracy',  value: `${overallPct.toFixed(1)}%` },
          ].map(({ label, value }) => (
            <div key={label} className="text-center">
              <div className="text-2xl font-bold gradient-text">
                {value}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {label}
              </div>
            </div>
          ))}
        </div>
        <AccuracyBar
          pct={overallPct}
          color="bg-gradient-to-r from-blue-500 to-green-500"
        />
        <p className="text-xs text-gray-500 mt-2 text-right">
          {overallPct.toFixed(1)}% correct
        </p>
      </div>

      {/* By league */}
      {stats.by_league && (
        <div className="glass-card rounded-2xl p-6">
          <h3 className="text-sm font-semibold text-gray-400
            uppercase tracking-wider mb-4">
            By League
          </h3>
          <div className="space-y-4">
            {Object.entries(stats.by_league).map(
              ([league, info]) => {
                const pct = info.accuracy_pct ?? 0;
                return (
                  <div key={league}>
                    <div className="flex justify-between
                      items-center mb-1">
                      <span className="text-sm text-white">
                        {LEAGUE_FLAGS[league] || ''} {league}
                      </span>
                      <span className={`text-sm font-bold
                        ${pct >= 60
                          ? 'text-green-400'
                          : pct >= 50
                            ? 'text-yellow-400'
                            : 'text-red-400'}`}>
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                    <AccuracyBar
                      pct={pct}
                      color={barColor(pct)}
                    />
                    <p className="text-xs text-gray-500 mt-0.5">
                      {info.correct ?? 0} / {info.total ?? 0}
                    </p>
                  </div>
                );
              }
            )}
          </div>
        </div>
      )}
    </div>
  );
}
