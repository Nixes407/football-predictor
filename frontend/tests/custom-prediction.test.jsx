import React from 'react';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import CustomTab from '../components/CustomTab';
import { BRAZIL_LEAGUE } from '../lib/leagues';

const response = (data) => ({ ok: true, json: async () => data });
const sample = {
  home_team: 'Flamengo',
  away_team: 'Palmeiras',
  league: BRAZIL_LEAGUE,
  season: '2026',
  result: { home_win: 0.50, draw: 0.25, away_win: 0.25 },
  goals: {
    expected_home: 2.0, expected_away: 1.0,
    over_1_5: 0.833, over_2_5: 0.5768, over_3_5: 0.3528, btts: 0.5466,
  },
  first_goal: { home: 0.5234, away: 0.3891, no_goal: 0.0875 },
  corners: { expected_home: 6, over_9_5: 0.54 },
};

describe('Brazilian custom prediction', () => {
  it('deduplicates aliases, sends the accented league, and renders the new markets', async () => {
    const fetchMock = vi.fn(async (url) => {
      if (url.endsWith('/predict/leagues')) {
        return response({ leagues: ['ENG-Premier League', 'BSA', 'BRA-Serie A', BRAZIL_LEAGUE] });
      }
      if (url.includes('/predict/teams?')) return response({ teams: ['Flamengo', 'Palmeiras'] });
      if (url.endsWith('/predict/match')) return response(sample);
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<CustomTab />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    await waitFor(() => expect(screen.getAllByRole('option', { name: /Brazilian Série A/ })).toHaveLength(1));
    const option = screen.getByRole('option', { name: /Brazilian Série A/ });
    expect(option.value).toBe(BRAZIL_LEAGUE);
    expect(option.textContent).toContain('🇧🇷');
    await user.selectOptions(screen.getByLabelText('League'), BRAZIL_LEAGUE);
    await screen.findAllByRole('option', { name: 'Flamengo' });
    const teamRequest = fetchMock.mock.calls.find(([url]) => url.includes('/predict/teams?'))[0];
    expect(new URL(teamRequest).searchParams.get('league')).toBe(BRAZIL_LEAGUE);
    expect(screen.getByLabelText('Season').value).toBe(String(new Date().getUTCFullYear()));

    await user.selectOptions(screen.getByLabelText('Home Team'), 'Flamengo');
    await user.selectOptions(screen.getByLabelText('Away Team'), 'Palmeiras');
    await user.click(screen.getByRole('button', { name: /Get Prediction/ }));
    await screen.findByRole('heading', { name: 'Team to Score First' });

    const post = fetchMock.mock.calls.find(([url]) => url.endsWith('/predict/match'));
    expect(JSON.parse(post[1].body)).toEqual({
      home_team: 'Flamengo', away_team: 'Palmeiras',
      league: BRAZIL_LEAGUE, season: String(new Date().getUTCFullYear()),
    });
    const goals = screen.getByRole('heading', { name: 'Goals' }).parentElement;
    expect(within(goals).getByText('83.3%')).toBeTruthy();
    const text = goals.textContent;
    expect(text.indexOf('Over 1.5')).toBeLessThan(text.indexOf('Over 2.5'));
    expect(text.indexOf('Over 2.5')).toBeLessThan(text.indexOf('Over 3.5'));
    expect(text.indexOf('Over 3.5')).toBeLessThan(text.indexOf('BTTS'));

    const firstGoal = screen.getByRole('heading', { name: 'Team to Score First' }).closest('section');
    expect(within(firstGoal).getByText('Flamengo').parentElement.className).toContain('bg-green-800');
    expect(within(firstGoal).getByText('52.3%')).toBeTruthy();
    expect(within(firstGoal).getByText('Palmeiras')).toBeTruthy();
    expect(within(firstGoal).getByText('38.9%')).toBeTruthy();
    expect(within(firstGoal).getByText('No Goal (0-0)')).toBeTruthy();
    expect(within(firstGoal).getByText('8.8%')).toBeTruthy();
    expect(screen.queryByText(/corners/i)).toBeNull();
  });

  it('allows manually entered Brazilian teams when the encoder-backed list is empty', async () => {
    const fetchMock = vi.fn(async (url) => {
      if (url.endsWith('/predict/leagues')) return response({ leagues: ['ENG-Premier League'] });
      if (url.includes('/predict/teams?')) return response({ teams: [] });
      if (url.endsWith('/predict/match')) return response(sample);
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<CustomTab />);
    await user.selectOptions(screen.getByLabelText('League'), BRAZIL_LEAGUE);
    const home = await screen.findByRole('textbox', { name: 'Home Team' });
    const away = screen.getByRole('textbox', { name: 'Away Team' });
    expect(screen.getByText(/No team list is available/)).toBeTruthy();
    await user.type(home, 'Flamengo');
    await user.type(away, 'Flamengo');
    expect(screen.getByRole('button', { name: /Get Prediction/ }).disabled).toBe(true);
    await user.clear(away);
    await user.type(away, 'Palmeiras');
    await user.click(screen.getByRole('button', { name: /Get Prediction/ }));
    await screen.findByRole('heading', { name: 'Team to Score First' });
    const post = fetchMock.mock.calls.find(([url]) => url.endsWith('/predict/match'));
    expect(JSON.parse(post[1].body).league).toBe('BRA-Série A');
  });

  it('does not let a stale team response replace the latest league selection', async () => {
    let resolveBrazil;
    const brazilResponse = new Promise((resolve) => { resolveBrazil = resolve; });
    const fetchMock = vi.fn((url) => {
      if (url.endsWith('/predict/leagues')) {
        return Promise.resolve(response({ leagues: ['ENG-Premier League'] }));
      }
      if (url.includes('/predict/teams?')) {
        return new URL(url).searchParams.get('league') === BRAZIL_LEAGUE
          ? brazilResponse
          : Promise.resolve(response({ teams: ['Arsenal', 'Chelsea'] }));
      }
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<CustomTab />);
    await screen.findByRole('option', { name: /Premier League/ });
    await user.selectOptions(screen.getByLabelText('League'), BRAZIL_LEAGUE);
    await user.selectOptions(screen.getByLabelText('League'), 'ENG-Premier League');
    await screen.findAllByRole('option', { name: 'Arsenal' });
    await act(async () => {
      resolveBrazil(response({ teams: ['Flamengo', 'Palmeiras'] }));
      await Promise.resolve();
    });
    expect(screen.queryByRole('option', { name: 'Flamengo' })).toBeNull();
    expect(screen.getAllByRole('option', { name: 'Arsenal' })).toHaveLength(2);
  });
});
