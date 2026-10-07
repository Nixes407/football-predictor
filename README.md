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
│   │   └── index.js         # placeholder landing page
│   ├── components/          # (empty — React components go here)
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

| Variable      | Description                                 | Example                                                          |
| ------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| `DATABASE_URL`| PostgreSQL connection string                | `postgresql://user:password@localhost:5432/football_predictor`    |

The API also reads an optional `ALLOWED_ORIGINS` (comma-separated) to control
CORS; it defaults to `http://localhost:3000`.

---

## Deploying

- **Frontend → Vercel.** Import the repository and set the project's **Root
  Directory** to `frontend`. Vercel auto-detects Next.js.
- **Backend.** FastAPI runs as a long-lived ASGI server, so host it on a service
  such as Render, Railway or Fly.io, then point the frontend at its URL via an
  environment variable.

---

## Roadmap

- [x] **Phase 1** — project scaffold (this commit)
- [ ] **Phase 2** — data collection and PostgreSQL schema
- [ ] **Phase 3** — feature engineering
- [ ] **Phase 4** — model training and evaluation
- [ ] **Phase 5** — prediction endpoint wired to the UI
