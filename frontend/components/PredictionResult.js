import FirstGoalSection from './FirstGoalSection';
import { leagueFlag, leagueName } from '../lib/leagues';

/**
 * @typedef {Object} PredictionResponse
 * @property {string} home_team
 * @property {string} away_team
 * @property {{home: number, away: number, no_goal: number}} first_goal
 */

/** @param {{data: (PredictionResponse & Record<string, any>) | null}} props */
export default function PredictionResult({ data }) {
  if (!data) return null;

  const result   = data.result   || {};
  const goals    = data.goals    || {};
  const ratings  = data.ratings  || {};
  const homeR    = ratings.home  || {};
  const awayR    = ratings.away  || {};
  const h2h      = ratings.head_to_head || {};
  const topLines = data.top_scorelines  || [];
  const engine   = data.engine || "ml_models";
  const isFormula= engine.includes("formula");

  const hw = result.home_win || 0;
  const d  = result.draw     || 0;
  const aw = result.away_win || 0;
  const maxProb = Math.max(hw, d, aw);

  function pct(val) {
    return val != null
      ? (val * 100).toFixed(1) + "%"
      : "—";
  }

  function prob(val) {
    return val != null
      ? (val * 100).toFixed(1) + "%"
      : "—";
  }

  function dec(val, places = 2) {
    return val != null ? Number(val).toFixed(places) : "—";
  }

  function boxClass(val) {
    const base = "flex-1 text-center py-3 rounded-xl transition-all";
    return val === maxProb
      ? `${base} bg-green-800 text-green-300 font-bold`
      : `${base} bg-gray-800 text-gray-400`;
  }

  return (
    <div className="space-y-4 mt-6">

      {/* Header */}
      <div className="glass-card rounded-2xl p-5 text-center">
        <h2 className="text-xl font-bold text-white mb-1">
          {data.home_team} vs {data.away_team}
        </h2>
        <p className="text-sm text-gray-400">
          {leagueFlag(data.league)} {leagueName(data.league)} | {data.season}
        </p>
        {isFormula && data.predicted_result && (
          <div className="mt-2 flex justify-center gap-3
            items-center">
            <span className="text-xs px-2 py-1 rounded-full
              bg-blue-900 text-blue-300">
              Formula Engine
            </span>
            <span className="text-xs text-gray-400">
              Confidence: {pct(data.confidence)}
            </span>
          </div>
        )}
      </div>

      {/* 1X2 Result */}
      <div className="glass-card rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-gray-400
          uppercase tracking-wider mb-3">
          Match Result
        </h3>
        <div className="flex gap-3">
          <div className={boxClass(hw)}>
            <div className="text-xs text-gray-400 mb-1">
              Home Win
            </div>
            <div className="text-lg font-bold">{pct(hw)}</div>
          </div>
          <div className={boxClass(d)}>
            <div className="text-xs text-gray-400 mb-1">
              Draw
            </div>
            <div className="text-lg font-bold">{pct(d)}</div>
          </div>
          <div className={boxClass(aw)}>
            <div className="text-xs text-gray-400 mb-1">
              Away Win
            </div>
            <div className="text-lg font-bold">{pct(aw)}</div>
          </div>
        </div>
      </div>

      {/* Goals */}
      <div className="glass-card rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-gray-400
          uppercase tracking-wider mb-3">
          Goals
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">
                Exp. Home Goals
              </span>
              <span className="text-white font-medium">
                {dec(goals.expected_home)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">
                Exp. Away Goals
              </span>
              <span className="text-white font-medium">
                {dec(goals.expected_away)}
              </span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Over 1.5</span>
              <span className="text-white font-medium">
                {prob(goals.over_1_5)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Over 2.5</span>
              <span className="text-white font-medium">
                {prob(goals.over_2_5)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Over 3.5</span>
              <span className="text-white font-medium">
                {prob(goals.over_3_5)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">BTTS</span>
              <span className="text-white font-medium">
                {prob(goals.btts)}
              </span>
            </div>
          </div>
        </div>
      </div>

      <FirstGoalSection
        homeTeam={data.home_team}
        awayTeam={data.away_team}
        firstGoal={data.first_goal}
      />

      {/* Top Scorelines — formula engine only */}
      {topLines.length > 0 && (
        <div className="glass-card rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-gray-400
            uppercase tracking-wider mb-3">
            Most Likely Scorelines
          </h3>
          <div className="space-y-2">
            {topLines.slice(0, 5).map((s, i) => (
              <div key={i}
                className="flex justify-between items-center">
                <span className="text-white font-mono
                  font-bold text-lg">
                  {s.score}
                </span>
                <div className="flex items-center gap-3">
                  <div className="w-32 bg-gray-800
                    rounded-full h-2">
                    <div
                      className="h-2 rounded-full
                        bg-gradient-to-r
                        from-blue-500 to-green-500"
                      style={{
                        width: `${Math.min(
                          s.probability * 400, 100)}%`
                      }}
                    />
                  </div>
                  <span className="text-gray-400 text-sm
                    w-12 text-right">
                    {(s.probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Team Ratings — formula engine only */}
      {isFormula && (homeR.attack_rel || awayR.attack_rel) && (
        <div className="glass-card rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-gray-400
            uppercase tracking-wider mb-3">
            Team Ratings
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-500 mb-2">
                {data.home_team}
              </p>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Attack</span>
                  <span className={
                    (homeR.attack_rel || 1) >= 1
                      ? "text-green-400"
                      : "text-red-400"
                  }>
                    {dec(homeR.attack_rel)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Defence</span>
                  <span className={
                    (homeR.defence_rel || 1) <= 1
                      ? "text-green-400"
                      : "text-red-400"
                  }>
                    {dec(homeR.defence_rel)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Form</span>
                  <span className="text-white">
                    {dec(homeR.form)}
                  </span>
                </div>
              </div>
            </div>
            <div>
              <p className="text-xs text-gray-500 mb-2">
                {data.away_team}
              </p>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Attack</span>
                  <span className={
                    (awayR.attack_rel || 1) >= 1
                      ? "text-green-400"
                      : "text-red-400"
                  }>
                    {dec(awayR.attack_rel)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Defence</span>
                  <span className={
                    (awayR.defence_rel || 1) <= 1
                      ? "text-green-400"
                      : "text-red-400"
                  }>
                    {dec(awayR.defence_rel)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Form</span>
                  <span className="text-white">
                    {dec(awayR.form)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* H2H */}
          {h2h.meetings > 0 && (
            <div className="mt-4 pt-4 border-t
              border-white/5">
              <p className="text-xs text-gray-500 mb-2">
                Head to Head
                ({h2h.meetings} meetings)
              </p>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">
                  Home xG avg
                </span>
                <span className="text-white">
                  {dec(h2h.home_xg)}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">
                  Away xG avg
                </span>
                <span className="text-white">
                  {dec(h2h.away_xg)}
                </span>
              </div>
            </div>
          )}
        </div>
      )}

    </div>
  );
}
