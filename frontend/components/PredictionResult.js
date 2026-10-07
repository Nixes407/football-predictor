/**
 * PredictionResult — renders the full /predict/match response.
 *
 * Props:
 *   data  the API response object:
 *         { home_team, away_team, league, season,
 *           result:  { home_win, draw, away_win },
 *           goals:   { expected_home, expected_away, over_1_5, over_2_5, over_3_5, btts },
 *           corners: { expected_home, expected_away, over_8_5, over_10_5 } }
 */

/** "0.5812" -> "58.1%" */
function asPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

/** "1.4213" -> "1.42" */
function asDecimal(value) {
  return Number(value).toFixed(2);
}

function ResultBox({ label, value, highlight }) {
  return (
    <div
      className={`rounded-lg p-4 text-center ${
        highlight ? 'bg-green-100 ring-1 ring-green-300' : 'bg-gray-50'
      }`}
    >
      <div className="text-xs font-medium uppercase tracking-wide text-gray-500">
        {label}
      </div>
      <div className="mt-1 text-2xl font-bold text-gray-900">{value}</div>
    </div>
  );
}

function StatRow({ label, value }) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-sm text-gray-600">{label}</span>
      <span className="text-sm font-semibold text-gray-900">{value}</span>
    </div>
  );
}

export default function PredictionResult({ data }) {
  if (!data) return null;

  const { result, goals, corners } = data;

  // Highest-probability outcome gets the highlighted box.
  const outcomes = [
    { key: 'home', label: 'Home Win', value: result.home_win },
    { key: 'draw', label: 'Draw', value: result.draw },
    { key: 'away', label: 'Away Win', value: result.away_win },
  ];
  const highest = Math.max(...outcomes.map((o) => Number(o.value)));

  return (
    <div className="mt-8 space-y-6">
      {/* SECTION 1 — MATCH HEADER */}
      <div className="text-center">
        <h2 className="text-xl font-bold text-gray-900">
          {data.home_team} vs {data.away_team}
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          {data.league} | {data.season}
        </p>
      </div>

      {/* SECTION 2 — MATCH RESULT */}
      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Match Result
        </h3>
        <div className="grid grid-cols-3 gap-3">
          {outcomes.map((outcome) => (
            <ResultBox
              key={outcome.key}
              label={outcome.label}
              value={asPercent(outcome.value)}
              highlight={Number(outcome.value) === highest}
            />
          ))}
        </div>
      </div>

      {/* SECTION 3 — GOALS */}
      <div className="rounded-lg bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Goals
        </h3>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <StatRow
              label="Expected Home Goals"
              value={asDecimal(goals.expected_home)}
            />
            <StatRow
              label="Expected Away Goals"
              value={asDecimal(goals.expected_away)}
            />
          </div>
          <div>
            <StatRow label="Over 1.5" value={asPercent(goals.over_1_5)} />
            <StatRow label="Over 2.5" value={asPercent(goals.over_2_5)} />
            <StatRow label="Over 3.5" value={asPercent(goals.over_3_5)} />
            <StatRow label="BTTS" value={asPercent(goals.btts)} />
          </div>
        </div>
      </div>

      {/* SECTION 4 — CORNERS */}
      <div className="rounded-lg bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Corners
        </h3>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <StatRow
              label="Expected Home Corners"
              value={asDecimal(corners.expected_home)}
            />
            <StatRow
              label="Expected Away Corners"
              value={asDecimal(corners.expected_away)}
            />
          </div>
          <div>
            <StatRow label="Over 8.5" value={asPercent(corners.over_8_5)} />
            <StatRow label="Over 10.5" value={asPercent(corners.over_10_5)} />
          </div>
        </div>
      </div>
    </div>
  );
}
