import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import TodayTab from '../components/TodayTab';
import FirstGoalSection from '../components/FirstGoalSection';
import AboutTab from '../components/AboutTab';
import AccumulatorTab from '../components/AccumulatorTab';
import HistoryTab from '../components/HistoryTab';
import StatsTab from '../components/StatsTab';
import { canonicalLeague, leagueFlag, leagueName } from '../lib/leagues';
import { probabilityOverGoals } from '../lib/predictionMarkets';

const response = (data) => ({ ok: true, json: async () => data });
const fixture = {
  id: 7, league: 'BRA-Série A', home_team: 'Flamengo', away_team: 'Palmeiras',
  status: 'scheduled', has_prediction: true,
  pred_home_win: 0.5, pred_draw: 0.3, pred_away_win: 0.2,
  pred_home_goals: 2, pred_away_goals: 1,
  pred_over_2_5: 0.5768, pred_btts: 0.5466,
  first_goal: { home: 0.6335, away: 0.3167, no_goal: 0.0498 },
  pred_corners_over_9_5: 0.54,
};

describe('league labels and Poisson goal markets', () => {
  it('canonicalises Brazilian aliases without confusing Italy or other leagues', () => {
    for (const alias of ['BSA', 'BRA-Serie A', 'BRA-Série A', 'Brazil Serie A', 'brasil série a']) {
      expect(canonicalLeague(alias)).toBe('BRA-Série A');
      expect(leagueName(alias)).toBe('Brazilian Série A');
      expect(leagueFlag(alias)).toBe('🇧🇷');
    }
    expect(canonicalLeague('ITA-Serie A')).toBe('ITA-Serie A');
    expect(leagueFlag('ITA-Serie A')).toBe('🇮🇹');
    expect(leagueName('Unknown-League')).toBe('Unknown-League');
  });

  it('derives over probabilities from available rates without inventing missing values', () => {
    expect(probabilityOverGoals(2, 1, 1)).toBeCloseTo(0.80085, 5);
    expect(probabilityOverGoals(2, 1, 3)).toBeCloseTo(0.35277, 5);
    expect(probabilityOverGoals(0, 0, 1)).toBe(0);
    expect(probabilityOverGoals(0, 1, 1)).toBeCloseTo(0.26424, 5);
    for (const invalid of [null, undefined, '', -1, NaN, Infinity]) {
      expect(probabilityOverGoals(invalid, 1, 1)).toBeNull();
    }
  });
});

describe('daily fixture predictions', () => {
  it('shows Brazilian fixtures and the new markets, but never a corners card', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response([fixture])));
    const user = userEvent.setup();
    render(<TodayTab />);
    const league = await screen.findByRole('heading', { name: /Brazilian Série A/ });
    expect(league.textContent).toContain('🇧🇷');
    await user.click(screen.getByRole('button', { name: /Get Prediction/ }));

    const goals = screen.getByRole('heading', { name: 'Goals' }).closest('section');
    expect(within(goals).getByText('80.1%')).toBeTruthy();
    expect(within(goals).getByText('57.7%')).toBeTruthy();
    expect(within(goals).getByText('35.3%')).toBeTruthy();
    expect(within(goals).getByText('54.7%')).toBeTruthy();
    const text = goals.textContent;
    expect(text.indexOf('Over 1.5')).toBeLessThan(text.indexOf('Over 2.5'));
    expect(text.indexOf('Over 2.5')).toBeLessThan(text.indexOf('Over 3.5'));
    expect(text.indexOf('Over 3.5')).toBeLessThan(text.indexOf('BTTS'));

    const firstGoal = screen.getByRole('heading', { name: 'Team to Score First' }).closest('section');
    expect(within(firstGoal).getByText('Flamengo').parentElement.className).toContain('bg-green-800');
    expect(within(firstGoal).getByText('63.3%')).toBeTruthy();
    expect(within(firstGoal).getByText('Palmeiras')).toBeTruthy();
    expect(within(firstGoal).getByText('31.7%')).toBeTruthy();
    expect(within(firstGoal).getByText('No Goal (0-0)')).toBeTruthy();
    expect(within(firstGoal).getByText('5.0%')).toBeTruthy();
    expect(screen.queryByText(/corners/i)).toBeNull();
  });

  it('keeps missing predictions unavailable rather than fabricating probabilities', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response([{
      ...fixture, has_prediction: false, pred_home_win: null, first_goal: undefined,
    }])));
    render(<TodayTab />);
    await screen.findByText('Prediction not available for this match');
    expect(screen.queryByRole('button', { name: /Get Prediction/ })).toBeNull();
    expect(screen.queryByText('Team to Score First')).toBeNull();
  });

  it('shows unavailable goal markets when a predicted fixture has no goal rates', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response([{
      ...fixture, pred_home_goals: null, pred_away_goals: null, first_goal: undefined,
    }])));
    const user = userEvent.setup();
    render(<TodayTab />);
    await screen.findByRole('heading', { name: /Brazilian Série A/ });
    await user.click(screen.getByRole('button', { name: /Get Prediction/ }));
    const goals = screen.getByRole('heading', { name: 'Goals' }).closest('section');
    expect(within(goals).getAllByText('—')).toHaveLength(2);
    const firstGoal = screen.getByRole('heading', { name: 'Team to Score First' }).closest('section');
    expect(within(firstGoal).getAllByText('—')).toHaveLength(3);
    expect(firstGoal.querySelectorAll('.bg-green-800')).toHaveLength(0);
  });
});

describe('first-goal display and other frontend areas', () => {
  it('highlights ties and no-goal winners, and leaves missing values unhighlighted', () => {
    const { rerender, container } = render(<FirstGoalSection
      homeTeam="Arsenal" awayTeam="Chelsea"
      firstGoal={{ home: 0.45, away: 0.45, no_goal: 0.1 }}
    />);
    expect(container.querySelectorAll('.bg-green-800')).toHaveLength(2);
    rerender(<FirstGoalSection
      homeTeam="Arsenal" awayTeam="Chelsea"
      firstGoal={{ home: 0, away: 0, no_goal: 1 }}
    />);
    expect(screen.getByText('No Goal (0-0)').parentElement.className).toContain('bg-green-800');
    expect(screen.getByText('100.0%')).toBeTruthy();
    rerender(<FirstGoalSection homeTeam="Arsenal" awayTeam="Chelsea" />);
    expect(container.querySelectorAll('.bg-green-800')).toHaveLength(0);
    expect(screen.getAllByText('—')).toHaveLength(3);
  });

  it('removes corner markets from About and the accumulator selector', async () => {
    const { unmount } = render(<AboutTab />);
    expect(screen.getByText('Brazilian Série A')).toBeTruthy();
    expect(screen.getByText(/Team to Score First/)).toBeTruthy();
    expect(screen.queryByText(/corners/i)).toBeNull();
    unmount();
    const user = userEvent.setup();
    render(<AccumulatorTab />);
    await user.click(screen.getByRole('button', { name: /Add Selection Manually/ }));
    expect(screen.queryByRole('option', { name: /corners/i })).toBeNull();
  });

  it('uses the Brazilian label and flag in history and accuracy', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response({ results: [{
      league: 'BRA-Série A', home_team: 'Flamengo', away_team: 'Palmeiras',
      home_score: 1, away_score: 0, result: 'H', pred_correct: true,
    }] })));
    const { unmount } = render(<HistoryTab />);
    const historyLeague = await screen.findByRole('heading', { name: /Brazilian Série A/ });
    expect(historyLeague.textContent).toContain('🇧🇷');
    unmount();

    vi.stubGlobal('fetch', vi.fn(async () => response({
      accuracy_pct: 50, total_predicted: 2, correct: 1,
      by_league: { BSA: { accuracy_pct: 50 } },
    })));
    render(<StatsTab />);
    expect((await screen.findByText(/Brazilian Série A/)).textContent).toContain('🇧🇷');
  });

  it('hides retired saved markets without including them or deleting their stored data', async () => {
    const retired = { match: 'Old match', market: 'Over 9.5 Corners', probability: 60 };
    const active = { match: 'Arsenal vs Chelsea', market: 'Home Win', probability: 50 };
    localStorage.setItem('accumulator', JSON.stringify([retired, active]));
    const user = userEvent.setup();
    render(<AccumulatorTab />);
    expect(await screen.findByText('Arsenal vs Chelsea')).toBeTruthy();
    expect(screen.queryByText(/corners/i)).toBeNull();
    expect(screen.getByText('50.00%')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Remove' }));
    expect(JSON.parse(localStorage.getItem('accumulator'))).toEqual([retired]);
    expect(screen.getByText('No selections yet.')).toBeTruthy();
  });
});
