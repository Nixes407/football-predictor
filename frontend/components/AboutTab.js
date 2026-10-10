const STEPS = [
  {
    emoji: '1️⃣',
    title: 'Data Collection',
    text: '6,776 matches across 5 top European leagues, 4 seasons of historical data',
  },
  {
    emoji: '2️⃣',
    title: 'Feature Engineering',
    text: '62 features per match including form, ELO ratings, head-to-head records and league position',
  },
  {
    emoji: '3️⃣',
    title: 'Machine Learning',
    text: '5 specialist models trained: Match Result (1X2), Home Goals, Away Goals, Home Corners, Away Corners',
  },
  {
    emoji: '4️⃣',
    title: 'Daily Predictions',
    text: 'Every day at midnight UTC, fixtures are fetched and predictions computed automatically for all 5 leagues',
  },
];

const LEAGUES = [
  { flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', name: 'Premier League', country: 'England' },
  { flag: '🇪🇸', name: 'La Liga', country: 'Spain' },
  { flag: '🇩🇪', name: 'Bundesliga', country: 'Germany' },
  { flag: '🇮🇹', name: 'Serie A', country: 'Italy' },
  { flag: '🇫🇷', name: 'Ligue 1', country: 'France' },
];

const MARKETS = [
  '🏆 Match Result (1X2)',
  '⚽ Total Goals',
  '📐 Total Corners',
  '🔄 Both Teams To Score',
  '📈 Over/Under 2.5 Goals',
  '🔲 Over/Under 9.5 Corners',
];

export default function AboutTab() {
  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* ── Card 1: How It Works ── */}
      <div className="glass p-6">
        <h3 className="text-xl font-bold mb-5">⚙️ How It Works</h3>
        <ol className="space-y-4">
          {STEPS.map((step) => (
            <li key={step.title} className="flex gap-4">
              <span className="text-2xl leading-none mt-0.5">{step.emoji}</span>
              <div>
                <div className="font-bold">{step.title}</div>
                <div
                  className="text-sm mt-0.5"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {step.text}
                </div>
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* ── Card 2: Leagues Covered ── */}
      <div className="glass p-6">
        <h3 className="text-xl font-bold mb-5">🌍 Leagues Covered</h3>
        <div className="space-y-3">
          {LEAGUES.map((league) => (
            <div
              key={league.name}
              className="flex items-center gap-3 pb-3"
              style={{ borderBottom: '1px solid var(--border)' }}
            >
              <span className="text-2xl">{league.flag}</span>
              <span className="font-semibold">{league.name}</span>
              <span
                className="ml-auto text-sm"
                style={{ color: 'var(--text-muted)' }}
              >
                {league.country}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Card 3: Prediction Markets ── */}
      <div className="glass p-6">
        <h3 className="text-xl font-bold mb-5">📊 Prediction Markets</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {MARKETS.map((market) => (
            <div
              key={market}
              className="rounded-xl px-4 py-3 font-semibold text-sm"
              style={{
                background: 'rgba(148,163,184,0.05)',
                border: '1px solid var(--border)',
              }}
            >
              {market}
            </div>
          ))}
        </div>
      </div>

      {/* ── Card 4: Disclaimer ── */}
      <div className="glass p-5">
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          For entertainment purposes only. Predictions are based on historical
          data and statistical models. Past performance does not guarantee
          future results.
        </p>
      </div>
    </div>
  );
}
