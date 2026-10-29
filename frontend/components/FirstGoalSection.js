/**
 * @param {{homeTeam: string, awayTeam: string,
 *   firstGoal?: {home: number, away: number, no_goal: number}}} props
 */
export default function FirstGoalSection({ homeTeam, awayTeam, firstGoal }) {
  const outcomes = [
    { key: 'home', label: homeTeam, probability: firstGoal?.home },
    { key: 'away', label: awayTeam, probability: firstGoal?.away },
    { key: 'no_goal', label: 'No Goal (0-0)', probability: firstGoal?.no_goal },
  ];
  const allAvailable = outcomes.every(({ probability }) =>
    typeof probability === 'number' && Number.isFinite(probability)
  );
  const highest = allAvailable
    ? Math.max(...outcomes.map(({ probability }) => probability))
    : null;

  return (
    <section className="glass-card rounded-2xl p-5">
      <h3 className="text-sm font-semibold text-gray-400
        uppercase tracking-wider mb-3">
        Team to Score First
      </h3>
      <div className="space-y-2">
        {outcomes.map(({ key, label, probability }) => (
          <div
            key={key}
            className={`flex justify-between items-center text-sm py-3 px-3 rounded-xl ${
              highest !== null && probability === highest
                ? 'bg-green-800 text-green-300 font-bold'
                : 'bg-gray-800 text-gray-400'
            }`}
          >
            <span>{label}</span>
            <span>{typeof probability === 'number' && Number.isFinite(probability)
              ? `${(probability * 100).toFixed(1)}%`
              : '—'}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
