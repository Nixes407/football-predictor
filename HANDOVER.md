# Football Predictor — Project Handover

## What We Are Building

A full-stack football match prediction web app

that uses advanced stats from FotMob to power

a statistical formula engine predicting match

outcomes across 13+ leagues including UCL.

## Live URLs

- Frontend: https://football-predictor-gamma.vercel.app

- Backend: https://football-predictor-kyk5.onrender.com

- GitHub: https://github.com/Nixes407/football-predictor

## Stack

- Frontend: Next.js + Tailwind CSS → Vercel

- Backend: Python FastAPI → Render (free tier)

- Database: PostgreSQL → Supabase (free tier)

- Models: Supabase Storage bucket "models"

## How the Prediction Engine Works

1. FotMob collector fetches team stats daily

   via GitHub Actions (3am UTC)

2. Stats stored in team_season_stats table

3. Formula engine reads stats and computes:

   - Attack/defence ratings from xG, shots, possession

   - Weighted season blend (current vs previous season)

   - UCL matches: 30% UCL stats + 70% domestic

   - Home advantage (1.15x), form, H2H modifiers

   - Poisson distribution → all markets

4. Falls back to ML models if no stats found

## Leagues Covered

FotMob stats + fixtures for all:

- ENG-Premier League, ENG-Championship

- ESP-La Liga, GER-Bundesliga, ITA-Serie A

- FRA-Ligue 1, POR-Primeira Liga, NED-Eredivisie

- BEL-Jupiler Pro League, TUR-Süper Lig

- NOR-Eliteserien, AUT-Bundesliga

- UEFA-Champions League, USA-MLS

## Key Files

- `backend/models/formula_engine.py` — prediction engine

- `backend/data/fotmob_collector.py` — FotMob stats

- `backend/data/fixtures.py` — fixture fetching

- `backend/routers/fixtures.py` — fixture serving + UCL logic

- `backend/routers/predict.py` — prediction endpoints

- `backend/features/feature_columns.py` — THE feature lists (62/13/14/13 after the 46 all-zero FBref columns were removed), shared by trainer and API; never redefine them elsewhere

- `backend/features/inference.py` — builds a live fixture's feature row with the builder's own functions (see `docs/feature-parity-audit.md`)

- `frontend/components/TodayTab.js` — today tab

- `frontend/components/PredictionResult.js` — full results panel

## Data Sources

| Source | Purpose | Status |

|--------|---------|--------|

| FotMob (direct HTTP) | Primary team stats | ✅ Working |

| football-data.org | Daily fixtures | ✅ Working |

| StatsBomb Open Data | H2H historical xG | ✅ Working |

| Understat | Backup xG (5 leagues) | ✅ Backup only |

| API-Football | Fixtures only (free plan blocks 2025+ seasons) | ⚠️ Limited |

| Sofascore | ABANDONED — Cloudflare blocks all cloud IPs | ❌ Do not retry |

## Critical Git Rule

Every Arena agent commit MUST use:

```

git config user.email "ardenshop12@gmail.com"

git config user.name "Nixes407"

```

Otherwise Vercel blocks deployment.

## Environment Variables

**Render (backend):**

- DATABASE_URL — Supabase pooler URL

- SUPABASE_URL — https://fuhrgcwkvbtattrvijdp.supabase.co

- SUPABASE_SERVICE_ROLE_KEY — service_role key
  (SUPABASE_KEY is still accepted as a legacy fallback)

- PIPELINE_SECRET — pipeline trigger secret

- FRONTEND_URL — https://football-predictor-gamma.vercel.app

- API_FOOTBALL_KEY — api-football.com key

**Vercel (frontend):**

- NEXT_PUBLIC_API_URL — https://football-predictor-kyk5.onrender.com

**GitHub Secrets (Settings → Secrets and variables → Actions):**

- DATABASE_URL, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY),
  PIPELINE_SECRET, RENDER_BACKEND, API_FOOTBALL_KEY

## Pipeline Architecture (since the OOM fix)

The Render free tier has 512 MB RAM. Once the training set grew from ~6,776 to
15,000+ rows (PRs #78/#80/#81), builder.py was OOM-killed mid-run and Render
restarted the service. Training therefore lives in GitHub Actions now:

- `.github/workflows/daily_fbref_update.yml` (3:00 AM UTC): StatsBomb →
  FotMob (+ Understat backup) → Match collector → FBref collector →
  builder.py → trainer.py → upload to Supabase Storage bucket "models".
- Render is notified with `POST /predict/admin/reload-models`, which only
  re-downloads the artifacts into memory.
- `POST /predict/admin/run-pipeline` and `.../run-pipeline-fast` return
  **410 Gone** (stubs; they can no longer spawn builder/trainer on Render).
- On startup the API reloads models from Supabase Storage in a background
  thread; it never auto-starts training.

## Admin Endpoints

- `GET /predict/admin/status` — pipeline + model status

- `GET /predict/admin/fotmob-team-names` — all FotMob team names by league

- `GET /predict/admin/new-tables-check` — DB snapshot

- `GET /predict/admin/diagnostics` — formula engine status

- `DELETE /predict/admin/clear-fixtures-cache?secret=X` — force fixture refresh

- `POST /predict/admin/reload-models?secret=X` — re-download the pre-built models + features.csv from Supabase Storage (~seconds)

- `POST /predict/admin/run-pipeline` / `run-pipeline-fast` — **410 Gone**; training moved to GitHub Actions

- `GET /predict/admin/fix-bsa-label?secret=X` — one-off repair of stale `BEL-Jupiler Pro League` fixture labels; add `&dry_run=true` to preview. GET so it can be triggered from a phone browser

## Current Priorities (in order)

### PRIORITY 1 — COMPLETED ✅

UCL predictions now working for all coverable teams.

Fixed: away team was being looked up in home team's

domestic league instead of their own. Also fixed

163 UCL team name mappings.

### PRIORITY 2 — ADD FOTMOB PLAYER STATS — COMPLETED ✅

`fotmob_collector.py` now collects both TEAM and PLAYER stats via
FotMob's `leagueseasondeepstats` hub (type=teams + type=players).
Player stats are merged across goals/assists/rating/expected_goals/
expected_assists and upserted into `player_season_stats`
(unique on player, team, league, season).

Implemented 2026-10-14: DEEP_PLAYER_QUERIES, get_player_stats(),
_upsert_player(), collect_player_stats(), collect_fotmob() now runs
teams then players. Team failure exits 1; player zero-rows with team
success also exits 1 so broken player endpoints are noticed.
Verified via mocked SQLite smoke tests (imports, parsers, merges,
upserts, exit codes), then verified live in production: the latest collect
committed about 577 team rows and 14,759 player rows with exit 0;
goal_assist and teamId→name resolution are working.

Fixes:

- `top_scorer_goals: 0` in predictions

- `squad_xg_total: 0` in predictions

- Player impact modifier becomes real
  (formula_engine.load_squad_players / load_key_players / _summarise_squad)

FotMob player stats endpoint:

```

GET /api/data/leagueseasondeepstats

params: id={league_id}&amp;season={sid}&amp;stat=goals&amp;type=players

Also fetch: assists, rating, expected_goals

```

Store in `player_season_stats` table (already exists).

### PRIORITY 3 — FIX CORNERS PREDICTION — COMPLETED ✅

FotMob `corner_taken_team` is stored as season totals in the new
`misc_corners` column. Accurate crosses, opponent-box touches, and
best-effort total shots are mapped alongside it. Corner O/U markets now
prefer each team's real per-game corner rate and retain the pressure proxy
only where real corners are missing.

### PRIORITY 4 — RETRAIN ML BACKUP MODELS

Current ML models are retrained daily by GitHub Actions.

To retrain on demand, run the workflow manually:

GitHub → Actions → "Daily Data Collection + Model Retrain" → Run workflow

The workflow rebuilds features, retrains all 9 models, uploads them to
Supabase Storage and calls `POST /predict/admin/reload-models` on Render.
Do **not** run builder/trainer on Render — it OOMs the 512 MB instance.

### PRIORITY 5 — FRONTEND IMPROVEMENTS

- Show which engine produced each prediction

  (formula_v1 badge already exists on Custom tab,

  add it to Today tab too)

- Show team form (last 5 W/D/L) on prediction panel

- Mobile UI improvements

- Show UCL blend info when applicable

### PRIORITY 6 — SHAREABLE PREDICTION CARDS

Screenshot-ready match prediction cards that users

can share on social media. One card per match showing

the key markets in a clean visual layout.

### PRIORITY 7 — MODEL SPECIALISATION

Original concept from the Termux project:

Multiple models compete per market.

Each model used only where it proved best.

E.g. one model for Home Win, another for BTTS.

Implement after formula engine is fully stable.

## Known Issues (non-blocking)

1. API-Football free plan blocks season 2025+

   Only fixtures/results endpoints work

   Do not attempt to fix — needs paid plan upgrade

2. Player stats row-count issue is fixed (Priority 2; latest live collect

   wrote 14,759 player rows).

3. ELO ratings show 1500 for all teams

   ClubElo name mismatch — not a blocker

4. Some Championship/new league teams may still

   show "Prediction not available" if FotMob

   name doesn't match fixture name

   Use /predict/admin/fotmob-team-names to debug

5. NOR-Eliteserien and AUT-Bundesliga

   FotMob IDs recently corrected (59 and 38)

   Verify data is correct in next workflow run

## Standing Rule for All Agent Work

If you discover any issues, bugs, missing functions,

broken imports, or inconsistencies — fix them

immediately as part of the same task.

Do not stop and report without fixing.

Do not ask for permission to fix them.

Do not split work into multiple prompts unless

the user explicitly asks you to.

After completing, provide a full summary of

everything done and verified.
