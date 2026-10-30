// The daily fixtures API stores expected goals but not every goal market.
// Derive the missing over probabilities from the same Poisson goal rates.
export function probabilityOverGoals(homeRate, awayRate, maxGoals) {
  const rates = [homeRate, awayRate];
  if (rates.some((rate) =>
    rate === null || rate === undefined || rate === '' ||
    !Number.isFinite(Number(rate)) || Number(rate) < 0
  )) return null;

  const total = Number(homeRate) + Number(awayRate);
  if (!Number.isFinite(total)) return null;
  let term = Math.exp(-total);
  let cumulative = term;
  for (let goals = 1; goals <= maxGoals; goals += 1) {
    term *= total / goals;
    cumulative += term;
  }
  return Math.max(0, Math.min(1, 1 - cumulative));
}
