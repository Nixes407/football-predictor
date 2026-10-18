# Expanded training leagues — implementation & validation report

Date: 2026-10-18 · Branch: `feature/expand-training-leagues` (opened as PR from
the Arena session branch `arena/46789769-football-predictor`)

## Goal

Train the model set on 11 leagues instead of the Big Five:

`ENG-Premier League, ESP-La Liga, GER-Bundesliga, ITA-Serie A, FRA-Ligue 1,`
plus `ENG-Championship (ELC), POR-Primeira Liga (PPL), NED-Eredivisie (DED),`
`TUR-Süper Lig (TR1), NOR-Eliteserien (NOR), AUT-Bundesliga (AUT)`.

Brazilian Série A (BSA) and the Belgian Jupiler Pro League are deliberately
**not** part of the training set.

## What changed in code

1. **`backend/models/trainer.py`**
   * New `TRAINING_LEAGUES` constant with exactly the 11 leagues above.
   * New `filter_training_leagues()` + `canonical_league()`: league matching is
     case/accent-insensitive and alias-aware (e.g. `TUR-Super Lig`,
     `NOR-OBOS-ligaen`, `POR-Liga Portugal`, `AUT-Admiral Bundesliga`), with
     `AUT-Bundesliga` and `GER-Bundesliga` kept strictly apart.
   * `train_all_models(..., leagues=None)` now filters the features frame to
     `TRAINING_LEAGUES` **before** the chronological split, prints a per-league
     row table, warns about leagues with zero rows, and reports
     `features_rows` in its returned summary (`results["_dataset"]`).
2. **`backend/data/collector.py`** (ingestion — without it the trainer's new
   leagues would never appear in `matches`)
   * `LEAGUES` extended to the same 11 entries.
   * **Fix**: soccerdata's `MatchHistory` only knows the Big Five in its
     `LEAGUE_DICT`; `MatchHistory(leagues="ENG-Championship")` raises
     `ValueError: Invalid league` and every scrape would silently "fail"
     (0 rows). `_register_extra_leagues_with_soccerdata()` now registers
     football-data.co.uk's division codes `E1` (Championship), `P1`
     (Portugal), `N1` (Eredivisie), `T1` (Süper Lig) at import time.
   * **Fix**: football-data.co.uk serves Norway and Austria only as the
     all-time "extra leagues" files `new/NOR.csv` / `new/AUT.csv` (goals +
     odds, **no** shots/corners/cards). Added a dedicated downloader with
     a `Country` guard (unknown codes silently redirect to a different
     country's file), `dd/mm/yyyy` day-first date parsing and
     season-label filtering (calendar-year handling for Eliteserien).
3. **`backend/scripts/analyze_training_leagues.py`** (new ops tool)
   Prints the per-league/per-season completed-match table from `matches` and
   the `fbref_team_stats` coverage for `TRAINING_LEAGUES`; `--json` for
   machine-readable output. Works against any `DATABASE_URL`.
4. **Tests** — `backend/tests/test_trainer_league_filter.py`,
   `backend/tests/test_collector_extra_leagues.py` (35 tests total, all pass;
   the pre-existing FBref cutoff tests remain green).
5. **Frontend: untouched** (as required). `fbref_collector.py` (the writer of
   `fbref_team_stats`) is left at Big Five — see coverage below.

## Data availability (step 1) & FBref coverage (step 2)

The live Supabase database was **not reachable from the dev sandbox** (egress
allowlist), so the audit ran two ways: by reading the ingestion code (which
determines what can ever be in the tables) and by replaying the whole pipeline
against a local database replica built from the real openfootball/football.json
dataset (all 4 target seasons, finished matches).

* `matches` can now hold, per season, e.g.: ELC 552-557 (incl. playoffs),
  POR 306, NED 306, TUR 306, NOR 240 (16 teams, calendar year), AUT 192-195.
  On the **current** production DB these six leagues have **0 rows** — the old
  collector only ingested the Big Five — so a `collector.py` run
  (`python data/collector.py`) is required before retraining; the upserts are
  idempotent (`ON CONFLICT DO NOTHING`).
* `fbref_team_stats` is written by `fbref_collector.py` for **Big Five only**;
  none of the six new leagues have rows. This is expected and acceptable:
  `builder.join_fbref_features()` zero-fills the FBref columns, so the new
  leagues train on ELO + rolling-form/H2H features (their `*_fbref_*` columns
  are 0). Extending `fbref_collector.LEAGUES` (plus soccerdata team-name
  aliases) is a follow-up.

## Training validation (steps 4-5) — local full-pipeline replay

Built once, then trained twice (chronological 65/15/20 split), same
`result_model_xgb_v1` evaluation. `features_rows` counts training rows after
league filtering; the corners models had no source stats in the openfootball
replica and aborted cleanly (they train in production, where football-data
supplies shots/corners).

| run | features_rows | XGB acc (test) | XGB log loss (test) | XGB acc (val) | XGB log loss (val) |
| --- | --- | --- | --- | --- | --- |
| baseline, Big Five only | 7,136 (of 12,929) | 0.4874 | 1.0339 | 0.4794 | 1.0319 |
| expanded, 11-league set | **12,929** (10 leagues with data) | **0.4925** | **1.0224** | **0.4910** | 1.0283 |

Reading: +0.5 pp accuracy and −0.011 log loss on test while doubling the
dataset — the expanded data helps and does not dilute the Big Five signal.
Trade-off to watch in the production retrain: XGB's draw-class recall dropped
(many draws in Championship-style leagues), so macro-F1 favoured logistic
regression on this replica; the trainer keeps logging the logistic-vs-XGB
comparison table on every run.

Exact per-league row counts of the expanded run: Championship 2,228 · EPL
1,520 · La Liga 1,510 · Serie A 1,510 · Ligue 1 1,372 · Portugal 1,224 ·
Bundesliga 1,224 · Eredivisie 1,224 · Austria 779 · Türkiye 338 · Norway 0
(then 0 in this snapshot; after a production collection run: ~960 for
2021-2024 + whatever 2024-25 rounds exist).

## Production retrain runbook

```bash
cd backend
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# 1. backfill matches + elo for the 11 leagues (idempotent upserts):
.venv/bin/python data/collector.py
# 2. verify availability & fbref coverage before/after:
.venv/bin/python scripts/analyze_training_leagues.py --json /tmp/availability.json
# 3. rebuild features and retrain + upload all 9 models to Supabase Storage:
.venv/bin/python -c "from features.builder import build_and_save_features as b; b('data/processed/features.csv')"
.venv/bin/python models/trainer.py
# (or POST /predict/admin/run-pipeline on the deployed backend)
```

## Known limitations / follow-ups

* `NOR-Eliteserien` extra CSV: football-data.co.uk labels its seasons by
  calendar year; rows for e.g. 2021 are stored as season `2021-2022` to align
  with the project's season convention.
* Shots/corners/fouls/cards/odds are NULL for NOR/AUT (source limitation) —
  rolling stat features zero-fill for those leagues and the corners models
  skip their rows automatically (NULL corner target).
* FBref + Understat + FotMob player/xG pipelines were not extended here; only
  the trainer-facing dataset scope changed.
* If the production DB holds league strings written by other collectors, they
  are normalised by `canonical_league()`; anything unknown is dropped from
  training and printed in the run log for a fast fix.
