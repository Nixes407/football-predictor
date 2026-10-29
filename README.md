# Football Match Predictor

A full-stack web app that predicts the outcome of football matches (home win /
draw / away win) from historical match data.

> **Status: Phase 1 — project scaffold.**
> The folder structure, dependencies and a bootable API/web shell are in place.
> There is **no prediction logic yet**: the router, data collector, feature
> builder and model trainer are all placeholders.

---

## Tech stack

| Layer      | Technology                                  |
| ---------- | ------------------------------------------- |
| Frontend   | Next.js (React) + Tailwind CSS              |
| Backend    | Python + FastAPI (served by Uvicorn)        |
| Database   | PostgreSQL                                  |
| ML toolkit | pandas, NumPy, SciPy, scikit-learn, joblib  |
| Data       | soccerdata                                  |
| Deploy     | Vercel (frontend)                           |

---

## Project structure

```
football-predictor/
├── frontend/
│   ├── pages/
│   │   ├── _app.js          # loads global (Tailwind) CSS
│   │   └── index.js         # landing page, calls the API
│   ├── components/          # (empty — React components go here)
│   ├── lib/
│   │   └── apiBase.js       # NEXT_PUBLIC_API_URL -> absolute API URLs
│   ├── styles/
│   │   └── globals.css      # Tailwind entry point
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── routers/
│   │   └── predict.py       # placeholder prediction router
│   ├── data/
│   │   └── collector.py     # placeholder data collection
│   ├── features/
│   │   └── builder.py       # placeholder feature engineering
│   ├── models/
│   │   └── trainer.py       # placeholder model training
│   ├── requirements.txt
│   └── .env.example
│
├── docs/
│   └── deploy.md            # Vercel + Render setup, env vars, CORS
├── vercel.json
├── .gitignore
└── README.md
```

---

## Running the backend (FastAPI + Uvicorn)

Requires **Python 3.10+** and a reachable PostgreSQL instance.

```bash
cd backend

# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env             # then edit .env with your real credentials

# 4. Start the API with auto-reload
uvicorn main:app --reload
```

The API is then available at:

- Status check — <http://localhost:8000/> → `{"status": "Football Predictor API is running"}`
- Interactive docs — <http://localhost:8000/docs>

To use a different host/port:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Running the frontend (Next.js)

Requires **Node.js 18.18+** (20 LTS recommended).

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Start the dev server
npm run dev
```

The app is then available at <http://localhost:3000>.

Other scripts:

```bash
npm run build     # production build
npm run start     # serve the production build
```

---

## Environment variables

`backend/.env` (created from `backend/.env.example`, never committed):

| Variable               | Description                                             | Example                                                    |
| ---------------------- | ------------------------------------------------------- | ---------------------------------------------------------- |
| `DATABASE_URL`         | PostgreSQL connection string                             | `postgresql://user:password@localhost:5432/football_predictor` |
| `ALLOWED_ORIGINS`      | Comma-separated CORS allowlist of frontend origins       | `http://localhost:3000,https://your-frontend.vercel.app`   |
| `FRONTEND_URL`         | Single-origin shortcut, merged into `ALLOWED_ORIGINS`    | `https://your-frontend.vercel.app`                          |
| `ALLOWED_ORIGIN_REGEX` | Optional regex for unenumerable origins (Vercel previews)| `https://.*\.vercel\.app`                                   |
| `SUPABASE_URL`         | Supabase project URL (model bucket)                      | `https://your-project.supabase.co`                          |
| `SUPABASE_SERVICE_ROLE_KEY` | Service-role key for the `models` bucket            | (Supabase -> Settings -> API)                               |
| `SUPABASE_KEY`         | Legacy alias for the key above (still accepted)          | (same service_role key)                                     |
| `PIPELINE_SECRET`      | Secret for `POST /predict/admin/reload-models`           | (generate with `secrets.token_urlsafe(32)`)                 |

`frontend/.env.local` (created from `frontend/.env.local.example`; see
`frontend/.env.example` for the deployed value):

| Variable              | Description                            | Example                                          |
| --------------------- | -------------------------------------- | ------------------------------------------------ |
| `NEXT_PUBLIC_API_URL` | Backend origin, no trailing slash      | `http://localhost:8000`                           |

The browser calls the backend **directly**, so `NEXT_PUBLIC_API_URL` on the
frontend and the matching origin in the backend's `ALLOWED_ORIGINS` are both
required. Details in [`docs/deploy.md`](docs/deploy.md).

---

## Deploying

- **Frontend → Vercel.** Import the repository and set the project's **Root
  Directory** to `frontend`. Vercel auto-detects Next.js, and `vercel.json`
  holds nothing but `{ "version": 2 }`. Set `NEXT_PUBLIC_API_URL` to the
  deployed backend origin.
- **Backend → Render / Railway / Fly.** FastAPI runs as a long-lived ASGI
  server. Set `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` and
  `ALLOWED_ORIGINS` containing the Vercel origin, since requests are now
  cross-origin.
- **Data collection + training → GitHub Actions.** The daily workflow
  (`.github/workflows/daily_fbref_update.yml`) runs the collectors, builder and
  trainer on a 7 GB runner, uploads the artifacts to Supabase Storage and then
  calls `POST /predict/admin/reload-models` on Render. The API itself never
  trains: the legacy `run-pipeline` / `run-pipeline-fast` routes were removed,
  and startup only downloads the pre-built artifacts.
- **No proxy layer.** `vercel.json` no longer rewrites `/api/*`; see
  [`docs/deploy.md`](docs/deploy.md) for the full setup and troubleshooting.

---

## Roadmap

- [x] **Phase 1** — project scaffold (this commit)
- [ ] **Phase 2** — data collection and PostgreSQL schema
- [ ] **Phase 3** — feature engineering
- [ ] **Phase 4** — model training and evaluation
- [ ] **Phase 5** — prediction endpoint wired to the UI
- [ ] 
