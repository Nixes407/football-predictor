/**
 * apiBase — the single source of truth for the backend API origin.
 *
 * The browser talks to the FastAPI backend directly (cross-origin), so there is
 * no proxy layer to configure: `vercel.json` ships no `routes`, and
 * `next.config.js` ships no `/api/*` rewrite. Every request URL is built here
 * from `NEXT_PUBLIC_API_URL`, which Vercel inlines into the client bundle at
 * build time.
 *
 * Set it in Vercel (Project → Settings → Environment Variables), e.g.
 *   NEXT_PUBLIC_API_URL=https://football-predictor-kyk5.onrender.com
 *
 * The backend must allow the frontend origin through CORS — see
 * `backend/.env.example` and `docs/deploy.md`.
 */

/** Fallback used only for local development, when no env var is set. */
const DEFAULT_API_URL = 'http://localhost:8000';

/**
 * Normalise a raw base URL: trim whitespace and strip any trailing slashes so
 * that `${base}/predict/match` can never produce a `//predict` double slash.
 *
 * @param {string | undefined} raw value of NEXT_PUBLIC_API_URL
 * @returns {string} origin without a trailing slash
 */
function normalizeBaseUrl(raw) {
  const value = (raw || '').trim();
  if (!value) return DEFAULT_API_URL;
  return value.replace(/\/+$/, '');
}

/** Backend origin, e.g. `https://football-predictor-kyk5.onrender.com`. */
export const API_BASE = normalizeBaseUrl(process.env.NEXT_PUBLIC_API_URL);

/**
 * Build an absolute API URL. A leading slash on `path` is optional, and a
 * query string may be included.
 *
 * @param {string} [path] e.g. `/predict/leagues` or `predict/teams?league=EPL`
 * @returns {string} absolute URL against API_BASE
 *
 * @example
 *   apiUrl('/predict/leagues');                 // https://api/predict/leagues
 *   apiUrl('predict/teams?league=EPL');         // https://api/predict/teams?league=EPL
 *   apiUrl();                                   // https://api
 */
export function apiUrl(path = '') {
  const clean = String(path).replace(/^\/+/, '');
  return clean ? `${API_BASE}/${clean}` : API_BASE;
}

export default apiUrl;
