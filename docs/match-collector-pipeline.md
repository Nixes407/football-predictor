# Where data collection and training actually run

Audit date: 2026-10-19 · branch `arena/f593d439-football-predictor` · base `main` @ `12736aa` (PR #78 merged)

## TL;DR

| Stage | Script | Executed by | Trigger |
| --- | --- | --- | --- |
| Match results (11 leagues) | `backend/data/collector.py` | GitHub Actions runner | `Run Match Collector` step — **added in this change** |
| StatsBomb open data | `backend/data/statsbomb_collector.py` | GitHub Actions runner | `Run StatsBomb collector` step |
| FotMob team/player stats | `backend/data/fotmob_collector.py` | GitHub Actions runner | `Run FotMob collector (primary)` step |
| Understat (backup xG) | `backend/data/understat_collector.py` | GitHub Actions runner | only if FotMob fails |
| API-Football | `backend/data/api_football_collector.py` | GitHub Actions runner | `continue-on-error` |
| FBref team stats | `backend/data/fbref_collector.py` | **Render** (subprocess) | `POST /predict/admin/run-pipeline-fast`, `POST /predict/admin/run-pipeline`, app startup (`auto_start_fast_pipeline`), weekly APScheduler job on Monday |
| Feature builder | `backend/features/builder.py` | **Render** (subprocess) | same three entry points |
| Trainer | `backend/models/trainer.py` | **Render** (subprocess) | same three entry points |

`fbref_collector.py`, `features/builder.py` and `models/trainer.py` are **never**
run on the GitHub Actions runner. The only workflow that exists for the daily
pipeline is `.github/workflows/daily_fbref_update.yml`; its `Trigger FBref +
builder + trainer pipeline` step calls the Render backend over HTTP, and the
Render process runs those three scripts as subprocesses.

## 1. Workflows present

```
.github/workflows/daily_fbref_update.yml   # the daily pipeline (cron 03:00 UTC + manual)
.github/workflows/fix_bsa_label.yml        # one-off BSA label repair, workflow_dispatch only
```

Before this change `daily_fbref_update.yml` had exactly these steps:

```
Checkout repository
Set up Python 3.11
Install Python dependencies
Run StatsBomb collector                    <- data/statsbomb_collector.py
Run FotMob collector (primary)             <- data/fotmob_collector.py
Run Understat collector (backup)           <- only when FotMob fails
Run API-Football collector                 <- continue-on-error: true
All collectors succeeded - trigger pipeline  <- POST /predict/admin/run-pipeline-fast
Wait for pipeline completion
Verify models ready
Print final diagnostics
Success summary
```

`backend/data/collector.py` (football-data.co.uk match history) appeared in
**no** workflow and in no other scheduler, so the `matches` table was only ever
populated by hand. That is why the six leagues added in PR #78
(`ENG-Championship`, `POR-Primeira Liga`, `NED-Eredivisie`, `TUR-Süper Lig`,
`NOR-Eliteserien`, `AUT-Bundesliga`) had no rows to train on.

## 2. The new step

`daily_fbref_update.yml`, step 5 of the job (right after dependency install,
before StatsBomb and therefore before anything downstream):

```yaml
      - name: Run Match Collector
        id: match_collector
        continue-on-error: true
        timeout-minutes: 60
        working-directory: backend
        run: |
          echo "==> [1/6] Collecting football-data.co.uk match history (incremental)..."
          python data/collector.py
          echo "==> Match collection complete"
```

* Runs before every other collector and before the builder/trainer trigger.
* `continue-on-error: true`: a source outage cannot block the rest of the run.
* `working-directory: backend` so `database.py` is importable.
* Uses `DATABASE_URL` from the job-level `env` (already configured for the
  other collectors).
* Reads `SEASONS` from `collector.py`, i.e. `2021-2022 … current season`
  (July boundary rule, see `season_range()`), for exactly the 11 leagues in
  `TRAINING_LEAGUES`. Nothing is hard-coded in the YAML, so the season range
  rolls forward automatically.
* A `Cache soccerdata downloads` step (`actions/cache@v4`) caches
  `$SOCCERDATA_DIR` (`${{ runner.temp }}/soccerdata`) so finished-season CSVs
  survive between runs.

## 3. Incremental collection

`collect_match_history()` is now incremental by default:

1. One grouped query reads how many matches the `matches` table holds per
   `(league, season)`.
2. A league-season is skipped when the stored count is at least
   `COMPLETE_SEASON_RATIO` (75 %) of `EXPECTED_MATCHES_PER_SEASON[league]`.
3. The **in-progress** season is never skipped — its football-data.co.uk file
   grows with every matchday, so it is re-fetched and upserted
   (`ON CONFLICT (match_id) DO NOTHING` inserts only what is new).
4. If the database cannot be read, nothing is skipped (fail-open).

A daily run therefore downloads ~11 files (the current season for each league)
instead of 11 leagues × 6 seasons = 66, plus one-off backfills for any
league-season that is missing or only partially stored.

Flags (run from `backend/`):

```bash
python data/collector.py                 # incremental, matches + ClubElo
python data/collector.py --matches-only  # skip ClubElo
python data/collector.py --force         # re-download every league-season
```

The process now exits non-zero when dependencies are missing or when nothing at
all could be collected, so a failed run is visible in the Actions log instead of
silently succeeding.

## 4. fbref / builder / trainer entry points (Render side)

`backend/routers/predict.py`:

| Endpoint / hook | Steps it runs | Called from |
| --- | --- | --- |
| `POST /predict/admin/run-pipeline-fast` (line ~1526) | `fbref_collector`, `builder`, `trainer` | the daily workflow |
| `POST /predict/admin/run-pipeline` (line ~1429) | `collector`, `fbref_collector`, `builder`, `trainer` | manual only |
| `auto_start_fast_pipeline()` (line ~412, from `main.py` lifespan) | `fbref_collector`, `builder`, `trainer` | every Render boot **when no models are found in Supabase Storage** |

`backend/main.py` also starts an APScheduler `BackgroundScheduler`:

* `scheduled_daily_fetch` — 00:00 UTC daily: fetches fixtures and stores
  predictions; on **Mondays** it also calls `scrape_fbref_stats()`.
* `refresh_todays_fixtures` — every 2 hours: refreshes today's fixtures.

Note that `/predict/admin/run-pipeline-fast` deliberately omits `collector.py`
("use this when data is already in the database"). That is now correct for the
daily run because the GitHub Actions workflow collects the matches and the
builder reads them from the same database.

## 5. Bugs found and fixed

1. **`matches` never updated (the reported problem).** No workflow step called
   `data/collector.py`. Fixed by the new `Run Match Collector` step.
2. **Extra-league registration silently ineffective (product bug).**
   `_register_extra_leagues_with_soccerdata()` mutated
   `soccerdata._config.LEAGUE_DICT`, but soccerdata >= 1.7 memoises the derived
   `{canonical: source}` map on the reader class (`_all_leagues_dict`) the first
   time `available_leagues()`/`_all_leagues()` is called. After that, mutation
   had no effect and `MatchHistory(leagues="ENG-Championship", …)` still raised
   `ValueError: Invalid league`; the error was swallowed into the per-season
   "failed" counter, so ELC/PPL/DED/TR1 ingested **zero** rows even with PR #78
   applied. The cached map is now dropped after registration (and covered by
   `SoccerdataCacheInvalidationTests`).
   Reproduced with soccerdata 1.9.1.
3. **Silent success on infrastructure failure.** `run_collection()` returned
   `None` and `__main__` always exited 0 when dependencies were missing or when
   every league-season failed, so CI could not tell a broken run from a good
   one. `main()` now returns an exit code and `sys.exit(main())` propagates it.
4. **Stale/misleading workflow messages.** The API-Football step claimed the
   pipeline "will NOT be triggered" (untrue since it was marked
   `continue-on-error`); step counters said `[1/4] … [4/4]` while six steps
   existed. Both corrected, and the pipeline-trigger step is now named
   `Trigger FBref + builder + trainer pipeline` with a comment explaining that
   those three scripts run on Render, not on the runner.
5. **Test-order hazard.** `SoccerdataRegistrationTests` only passed because it
   ran before anything populated soccerdata's league cache, and it left the
   process-global cache polluted for later tests. Registration now invalidates
   the cache itself, and the new test stores/restores it.

## 6. Verification

```bash
cd backend && python -m unittest discover -s tests   # 61 tests, all pass
```
