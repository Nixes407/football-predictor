import { useEffect, useState } from 'react';
import Head from 'next/head';

import { apiUrl } from '../lib/apiBase';
import PredictionResult from '../components/PredictionResult';
import TeamSelector from '../components/TeamSelector';

export default function Home() {
  const [leagues, setLeagues] = useState([]);
  const [selectedLeague, setSelectedLeague] = useState('');
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [season, setSeason] = useState('2024-2025');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load the leagues once on mount.
  useEffect(() => {
    let cancelled = false;

    async function fetchLeagues() {
      try {
        const res = await fetch(apiUrl('/predict/leagues'));
        const data = await res.json();
        if (!cancelled) setLeagues(data.leagues || []);
      } catch (err) {
        if (!cancelled) setError('Could not reach the API — is it running?');
      }
    }

    fetchLeagues();
    return () => {
      cancelled = true;
    };
  }, []);

  // Reload the team list whenever the league changes.
  async function handleLeagueChange(league) {
    setSelectedLeague(league);
    setHomeTeam('');
    setAwayTeam('');
    setPrediction(null);
    setError(null);

    if (!league) {
      setTeams([]);
      return;
    }

    try {
      const res = await fetch(
        apiUrl(`/predict/teams?league=${encodeURIComponent(league)}`)
      );
      const data = await res.json();
      setTeams(data.teams || []);
    } catch (err) {
      setTeams([]);
      setError('Could not load teams — is the API running?');
    }
  }

  async function handlePredict() {
    if (homeTeam === awayTeam) {
      setError('Home and away teams must be different');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(apiUrl('/predict/match'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          league: selectedLeague,
          season,
        }),
      });

      if (res.status === 503) {
        setError('Models not loaded yet — run the data pipeline first');
        setPrediction(null);
      } else if (!res.ok) {
        setError('Prediction failed — is the API running?');
        setPrediction(null);
      } else {
        const data = await res.json();
        setPrediction(data);
      }
    } catch (err) {
      setError('Prediction failed — is the API running?');
      setPrediction(null);
    }

    setLoading(false);
  }

  const canPredict =
    Boolean(selectedLeague) && Boolean(homeTeam) && Boolean(awayTeam);

  return (
    <>
      <Head>
        <title>Football Match Predictor</title>
        <meta
          name="description"
          content="Predict the outcome of football matches."
        />
      </Head>

      <div className="min-h-screen bg-gray-100">
        <main className="mx-auto max-w-2xl p-6">
          <h1 className="mb-6 text-center text-3xl font-bold text-gray-900">
            ⚽ Football Match Predictor
          </h1>

          <div className="rounded-xl bg-white p-6 shadow-md">
            <label className="mb-1 block text-sm font-medium text-gray-700">
              League
            </label>
            <select
              className="mb-4 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-gray-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              value={selectedLeague}
              onChange={(event) => handleLeagueChange(event.target.value)}
            >
              <option value="">Select league...</option>
              {leagues.map((league) => (
                <option key={league} value={league}>
                  {league}
                </option>
              ))}
            </select>

            <div className="mb-4">
              <label className="mb-1 block text-sm font-medium text-gray-700">
                Season
              </label>
              <input
                type="text"
                className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-gray-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                value={season}
                onChange={(event) => setSeason(event.target.value)}
                placeholder="2024-2025"
              />
            </div>

            <TeamSelector
              label="Home Team"
              teams={teams}
              value={homeTeam}
              onChange={setHomeTeam}
              exclude={awayTeam}
            />

            <TeamSelector
              label="Away Team"
              teams={teams}
              value={awayTeam}
              onChange={setAwayTeam}
              exclude={homeTeam}
            />

            <button
              type="button"
              onClick={handlePredict}
              disabled={loading || !canPredict}
              className="mt-2 w-full rounded-md bg-blue-600 px-4 py-2 font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-gray-300 disabled:text-gray-500"
            >
              {loading ? 'Predicting...' : 'Get Prediction'}
            </button>

            {error && (
              <p className="mt-4 rounded-md bg-red-50 p-3 text-center text-sm text-red-600">
                {error}
              </p>
            )}

            {prediction && <PredictionResult data={prediction} />}
          </div>
        </main>
      </div>
    </>
  );
}
