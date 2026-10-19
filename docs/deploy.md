# Deployment Guide

Frontend on **Vercel** (Next.js), backend on **Render** (FastAPI). The browser
calls the backend **directly** — there is no proxy layer.

```
browser ──► Vercel (frontend/)                 next.config.js: no /api rewrite
   └──────────► Render  https://.../predict/*  vercel.json: no routes
```

---

## 1. Backend on Render

| Setting        | Value                                                   |
| -------------- | ------------------------------------------------------- |
| Root Directory | `backend`                                               |
| Build Command  | `pip install -r requirements.txt`                       |
| Start Command  | `uvicorn main:app --host 0.0.0.0 --port $PORT`          |

### Required environment variables

| Variable               | Required | Example                                                                | Purpose                                                     |
| ---------------------- | -------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| `DATABASE_URL`         | yes      | `postgresql://user:pass@host:5432/football_predictor`                  | PostgreSQL connection string                                 |
| `ALLOWED_ORIGINS`      | yes      | `https://your-frontend.vercel.app`                                     | Comma-separated CORS allowlist of frontend origins           |
| `FRONTEND_URL`         | optional | `https://your-frontend.vercel.app`                                     | Single-origin shortcut, merged into `ALLOWED_ORIGINS`        |
| `ALLOWED_ORIGIN_REGEX` | optional | `https://.*\.vercel\.app`                                              | Regex for hosts you cannot enumerate (Vercel previews)       |
| `SUPABASE_URL`         | yes      | `https://<project>.supabase.co`                                        | Supabase project URL (model bucket download)                 |
| `SUPABASE_SERVICE_ROLE_KEY` | yes | `eyJ...` (service_role JWT)                                           | Storage key for the `models` bucket. `SUPABASE_KEY` is accepted as a legacy fallback |
| `PIPELINE_SECRET`      | optional | `python -c "import secrets;print(secrets.token_urlsafe(32))"`           | Secret for `POST /predict/admin/reload-models`; empty = 403  |

CORS notes:

- List **full origins** — scheme plus host, no path, no trailing slash.
  `main.py` strips trailing slashes defensively, but browsers never send one.
- Set `ALLOWED_ORIGINS` **after** you know the Vercel domain; a preview-only
  domain will not match production traffic.
- The API enables credentialed CORS, so `"*"` is rejected by the CORS spec.
  If `"*"` appears in the allowlist, `main.py` downgrades to anonymous CORS
  rather than breaking the preflight.

### Where the data pipeline runs

**Not on Render.** Since the training set passed ~15,000 rows, builder.py and
trainer.py no longer fit in the 512 MB free tier (the service was OOM-killed
mid-pipeline). All collection and training runs in the daily GitHub Actions
workflow (`.github/workflows/daily_fbref_update.yml`, 7 GB runner):

```
collectors -> fbref -> builder.py -> trainer.py -> Supabase Storage "models"
                                                        |
                          Render: POST /predict/admin/reload-models
```

The API answers `503` for predictions until that reload has loaded a full set
of artifacts. It does this automatically at startup (background thread); to
refresh on demand:

```bash
curl -X POST "https://your-backend.onrender.com/predict/admin/reload-models?secret=$PIPELINE_SECRET"
# {"status":"reloaded","features_rows":15000,"models_loaded":[...]}

curl "https://your-backend.onrender.com/predict/admin/status"
```

The legacy `POST /predict/admin/run-pipeline` and
`POST /predict/admin/run-pipeline-fast` endpoints now return **410 Gone** — they
are stubs so nothing can restart training on Render by accident.

To train locally instead, run from `backend/` (then upload manually with the
trainer, which does it by default):

```bash
python data/collector.py
python features/builder.py
python models/trainer.py
```

### Smoke test

```bash
curl https://your-backend.onrender.com/
# {"status":"Football Predictor API is running"}
```

---

## 2. Frontend on Vercel

| Setting        | Value        |
| -------------- | ------------ |
| Root Directory | **`frontend`** |
| Framework      | auto-detected (Next.js) |

The **Root Directory must be `frontend`** — the repo root has no
`package.json`, so a root-directory build fails. `vercel.json` is intentionally
just `{ "version": 2 }`; Vercel auto-detects Next.js from the root directory.

### Required environment variable

| Variable              | Example                                             | Purpose                        |
| --------------------- | --------------------------------------------------- | ------------------------------ |
| `NEXT_PUBLIC_API_URL` | `https://football-predictor-kyk5.onrender.com`      | Backend origin, no trailing slash |

Set it under **Project → Settings → Environment Variables**, for all
environments you deploy, then **redeploy** — `NEXT_PUBLIC_*` values are inlined
into the client bundle at **build** time, so a change made after a build has no
effect on that deployment.

`frontend/lib/apiBase.js` normalises the value (trims it, strips trailing
slashes) and exposes `apiUrl(path)`, so every call is
`${NEXT_PUBLIC_API_URL}/predict/...` with no risk of a `//predict` double
slash. If the variable is unset it falls back to `http://localhost:8000`,
which is why a missing value shows up as "Could not reach the API" rather than
a build error.

---

## 3. `vercel.json` no longer proxies `/api`

The previous `routes`/`builds` block rewrote `/api/(.*)` to the backend and
conflicted with Vercel's automatic Next.js detection. It is gone, and nothing
in `frontend/` calls `/api/*` — `pages/index.js` requests `/predict/leagues`,
`/predict/teams` and `/predict/match` against the backend origin directly.

Consequences worth knowing:

- **The backend must send CORS headers.** Previously the rewrite made requests
  same-origin, so CORS never applied. Now it does — see §1.
- There is no build-time indirection: the backend URL is baked into the bundle,
  so changing it requires a redeploy.
- To add a same-origin path later, prefer a Next.js rewrite in
  `next.config.js` over `vercel.json` routes, and point it at the same
  `NEXT_PUBLIC_API_URL`.

---

## 4. Local development

```bash
# terminal 1 — backend
cd backend
pip install -r requirements.txt
cp .env.example .env            # edit credentials
uvicorn main:app --reload       # http://localhost:8000/docs

# terminal 2 — frontend
cd frontend
npm install
cp .env.local.example .env.local
npm run dev                     # http://localhost:3000
```

Point local development at the deployed backend instead by setting
`NEXT_PUBLIC_API_URL=https://your-backend.onrender.com` in
`frontend/.env.local` and adding `http://localhost:3000` to the backend's
`ALLOWED_ORIGINS`.

---

## 5. Troubleshooting

| Symptom | Likely cause |
| ------- | ------------ |
| Browser: `blocked by CORS policy: No 'Access-Control-Allow-Origin'` | Frontend origin missing from the backend's `ALLOWED_ORIGINS` |
| Browser: `... has been blocked by CORS policy: ... credentials mode` | `"*"` in the allowlist with credentials; list explicit origins |
| `net::ERR_NAME_NOT_RESOLVED` / requests to `localhost:8000` in production | `NEXT_PUBLIC_API_URL` not set on Vercel, or set after the build |
| Requests to `https://backend//predict/leagues` (double slash) | Fixed by `apiBase.js`; check any hand-rolled URL building |
| `503 Models not loaded` | Pipeline not run yet — see §1 |
| Vercel build: `no package.json found` | Root Directory is not set to `frontend` |
