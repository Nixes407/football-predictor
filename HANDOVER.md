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

- SUPABASE_KEY — service_role key

- PIPELINE_SECRET — pipeline trigger secret

- FRONTEND_URL — https://football-predictor-gamma.vercel.app

- API_FOOTBALL_KEY — api-football.com key

**Vercel (frontend):**

- NEXT_PUBLIC_API_URL — https://football-predictor-kyk5.onrender.com

**GitHub Secrets:**

- Same as Render + RENDER_BACKEND

## Admin Endpoints

- `GET /predict/admin/status` — pipeline + model status

- `GET /predict/admin/fotmob-team-names` — all FotMob team names by league

- `GET /predict/admin/new-tables-check` — DB snapshot

- `GET /predict/admin/diagnostics` — formula engine status

- `DELETE /predict/admin/clear-fixtures-cache?secret=X` — force fixture refresh

- `POST /predict/admin/run-pipeline-fast?secret=X` — retrain models (~30-40 mins)

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
upserts, exit codes). Live FotMob probing is egress-blocked in sandbox
(allowed hosts: github.com only) — real run will occur on GitHub
Actions where FotMob is reachable.

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

### PRIORITY 3 — FIX CORNERS PREDICTION

Currently corners use goals/xG as proxy because

shots/corners columns are null in team_season_stats.

FotMob deep stats has corner data per team.

Map FotMob corner data to correct columns so

corner O/U markets use real data.

### PRIORITY 4 — RETRAIN ML BACKUP MODELS

Current ML models trained on 2021-2024 data.

After a full data collection, run:

`POST /predict/admin/run-pipeline-fast`

This rebuilds features and retrains 5 backup models

with current season data.

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

2. Player stats only 160 rows (old 2023-2024 data)

   Will be fixed by Priority 2

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
