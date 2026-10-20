# Feature parity audit — training (builder.py) vs inference (predict.py)

Date: 2026-10-20 · Branch: `arena/589b84c6-football-predictor` · Step 8

## Why this audit exists

The models are trained on rows produced by `backend/features/builder.py` and
scored at request time by `backend/routers/predict.py`. If those two paths
compute different values for the same fixture, the model silently receives
inputs it was never trained on. Nothing raises: `/predict/match` still returns
plausible probabilities, just from wrong numbers — the most expensive class of
bug in this project, because it degrades quality without any signal.

## Method

1. Enumerated every column in all four published feature sets
   (`FEATURE_COLS` 108, `BTTS_FEATURE_COLS` 17, `OVER25_FEATURE_COLS` 18,
   `OVER95_CORNERS_FEATURE_COLS` 17 — 124 distinct columns).
2. Read the builder's computation for each, then traced what `predict.py`
   actually put in the model's input row.
3. Compared window sizes, venue splits, date cutoffs, fallback/NaN rules,
   encoder usage and the FBref snapshot rule.
4. Built a synthetic match history and drove both paths over it, which
   confirmed the findings below numerically (`backend/tests/test_feature_parity.py`).

### The headline finding

`predict.py` did not compute a single rolling feature. `_get_team_features()`
copied the values **stored on a past row of `features.csv`**:

```python
team_rows = df[df["home_team"] == team]   # or away_team
latest = team_rows.iloc[-1]               # the team's last home (or away) match
```

That row's columns are the features *as of that older match* — a match that may
have been at a different venue, against different opponents. Every windowed
feature was therefore stale by at least one match, and the H2H columns came
from whatever opponent the team happened to face last.

**Result: 123 of 124 features were wrong. Only `league_encoded` was right.**

## The audit table

*Builder* is the training definition (source of truth). *predict.py (before)*
is what the API fed the model before this change. `MATCH` rows were already
correct.

| Feature | Builder (training) | predict.py (before) | Match? | Fix |
| --- | --- | --- | --- | --- |
| `home_form_wins_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_draws_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_losses_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_points_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_gf_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_ga_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_gd_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_shots_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_shots_ot_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_corners_5` | Rolling 5-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_wins_10` | Rolling 10-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_draws_10` | Rolling 10-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_losses_10` | Rolling 10-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_points_10` | Rolling 10-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_gf_10` | Rolling 10-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_ga_10` | Rolling 10-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_form_gd_10` | Rolling 10-match sum/record for the home team, both venues, strictly before the match date | Copied from the last features.csv row where the team was home — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `home_home_wins_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_home_draws_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_home_losses_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_home_points_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_home_gf_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_home_ga_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_home_gd_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_home_shots_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_home_corners_5` | Rolling 5-match record for the home team in home matches only, before the fixture | Copied from the last row where the team played at home (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_form_wins_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_draws_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_losses_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_points_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_gf_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_ga_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_gd_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_shots_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_shots_ot_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_corners_5` | Rolling 5-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_wins_10` | Rolling 10-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_draws_10` | Rolling 10-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_losses_10` | Rolling 10-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_points_10` | Rolling 10-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_gf_10` | Rolling 10-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_ga_10` | Rolling 10-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_form_gd_10` | Rolling 10-match sum/record for the away team, both venues, strictly before the match date | Copied from the last features.csv row where the team was away — a *different, older* fixture, so the window is one match out of date and drawn from the wrong venue set | **MISMATCH** | Recomputed by `builder.compute_team_rolling_form` at the fixture date |
| `away_away_wins_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_away_draws_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_away_losses_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_away_points_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_away_gf_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_away_ga_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_away_gd_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_away_shots_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `away_away_corners_5` | Rolling 5-match record for the away team in away matches only, before the fixture | Copied from the last row where the team played away (one fixture stale) | **MISMATCH** | Recomputed by `compute_team_rolling_form(venue=...)` at the fixture date |
| `home_elo` | Latest ClubElo rating strictly before the match date, 1500.0 when unknown | Rating stored on the team's last row in that role, and **0.0** (not 1500) when the team had none | **MISMATCH** | Authoritative `elo_ratings` lookup (`builder.get_latest_elo_before`), CSV reconstruction as fallback, 1500 default preserved |
| `away_elo` | Latest ClubElo rating strictly before the match date, 1500.0 when unknown | Rating stored on the team's last row in that role, and **0.0** (not 1500) when the team had none | **MISMATCH** | Authoritative `elo_ratings` lookup (`builder.get_latest_elo_before`), CSV reconstruction as fallback, 1500 default preserved |
| `elo_difference` | home_elo − away_elo from the match date's ratings | home_elo of the last home row − away_elo of the last away row (two different dates); 0.0 whenever either was missing | **MISMATCH** | Derived once, centrally, from the resolved ratings |
| `h2h_home_wins_5` | Last 5 meetings **between these two teams** before the match date | Taken from the home team's last row, i.e. the H2H of a *different opponent* | **WRONG** | Recomputed by `builder.compute_h2h` for the actual pairing |
| `h2h_draws_5` | Last 5 meetings **between these two teams** before the match date | Taken from the home team's last row, i.e. the H2H of a *different opponent* | **WRONG** | Recomputed by `builder.compute_h2h` for the actual pairing |
| `h2h_away_wins_5` | Last 5 meetings **between these two teams** before the match date | Taken from the home team's last row, i.e. the H2H of a *different opponent* | **WRONG** | Recomputed by `builder.compute_h2h` for the actual pairing |
| `h2h_home_gf_5` | Last 5 meetings **between these two teams** before the match date | Taken from the home team's last row, i.e. the H2H of a *different opponent* | **WRONG** | Recomputed by `builder.compute_h2h` for the actual pairing |
| `h2h_away_gf_5` | Last 5 meetings **between these two teams** before the match date | Taken from the home team's last row, i.e. the H2H of a *different opponent* | **WRONG** | Recomputed by `builder.compute_h2h` for the actual pairing |
| `league_encoded` | `league_encoder.joblib` fitted by the builder | Same saved encoder | **MATCH** | Shared `encode_label` helper (case/whitespace tolerant) |
| `season_encoded` | `season_encoder.joblib`, encoded for the match's own season | Same encoder, but the request defaulted to the hardcoded season `2024-2025` | **MISMATCH** | Default is now the current season (`current_season_label`) |
| `home_fbref_games` | FBref team-season stats for the fixture's (team, league, season), only when the snapshot was collected on/before the match date | Copied from the team's last row in that role — not refreshed for the fixture, blind to a snapshot collected after that match | **MISMATCH** | `join_fbref_features` re-applied at the fixture date against the live `fbref_team_stats` snapshot (CSV fallback) |
| `home_fbref_shots` | FBref `shots` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_shots_on_target` | FBref `shots_on_target` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_shot_accuracy` | FBref `shot_accuracy` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_goals_per_shot` | FBref `goals_per_shot` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_goals_per_sot` | FBref `goals_per_sot` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_yellow_cards` | FBref `yellow_cards` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_red_cards` | FBref `red_cards` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_fouls_committed` | FBref `fouls_committed` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_fouls_drawn` | FBref `fouls_drawn` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_offsides` | FBref `offsides` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_crosses` | FBref `crosses` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_interceptions` | FBref `interceptions` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_tackles_won` | FBref `tackles_won` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_pens_won` | FBref `pens_won` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_pens_conceded` | FBref `pens_conceded` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_sota` | FBref `sota` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_goals_against` | FBref `goals_against` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_saves` | FBref `saves` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_save_pct` | FBref `save_pct` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_clean_sheets` | FBref `clean_sheets` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_pk_faced` | FBref `pk_faced` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_fbref_pk_saved` | FBref `pk_saved` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_games` | FBref `games` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_shots` | FBref `shots` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_shots_on_target` | FBref `shots_on_target` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_shot_accuracy` | FBref `shot_accuracy` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_goals_per_shot` | FBref `goals_per_shot` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_goals_per_sot` | FBref `goals_per_sot` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_yellow_cards` | FBref `yellow_cards` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_red_cards` | FBref `red_cards` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_fouls_committed` | FBref `fouls_committed` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_fouls_drawn` | FBref `fouls_drawn` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_offsides` | FBref `offsides` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_crosses` | FBref `crosses` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_interceptions` | FBref `interceptions` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_tackles_won` | FBref `tackles_won` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_pens_won` | FBref `pens_won` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_pens_conceded` | FBref `pens_conceded` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_sota` | FBref `sota` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_goals_against` | FBref `goals_against` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_saves` | FBref `saves` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_save_pct` | FBref `save_pct` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_clean_sheets` | FBref `clean_sheets` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_pk_faced` | FBref `pk_faced` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `away_fbref_pk_saved` | FBref `pk_saved` for the fixture's team-season (date-filtered) | Copied from the team's last row in that role | **MISMATCH** | Same shared join as training |
| `home_btts_rate_5` | Fraction over the team's last 5 matches in its own venue (BTTS / clean sheet / scored) | Copied from the last row in that role — stale, and zero-filled only if the column was absent | **MISMATCH** | Recomputed by `builder.compute_btts_rolling` at the fixture date |
| `away_btts_rate_5` | Fraction over the team's last 5 matches in its own venue (BTTS / clean sheet / scored) | Copied from the last row in that role — stale, and zero-filled only if the column was absent | **MISMATCH** | Recomputed by `builder.compute_btts_rolling` at the fixture date |
| `home_cs_rate_5` | Fraction over the team's last 5 matches in its own venue (BTTS / clean sheet / scored) | Copied from the last row in that role — stale, and zero-filled only if the column was absent | **MISMATCH** | Recomputed by `builder.compute_btts_rolling` at the fixture date |
| `away_cs_rate_5` | Fraction over the team's last 5 matches in its own venue (BTTS / clean sheet / scored) | Copied from the last row in that role — stale, and zero-filled only if the column was absent | **MISMATCH** | Recomputed by `builder.compute_btts_rolling` at the fixture date |
| `home_scored_rate_5` | Fraction over the team's last 5 matches in its own venue (BTTS / clean sheet / scored) | Copied from the last row in that role — stale, and zero-filled only if the column was absent | **MISMATCH** | Recomputed by `builder.compute_btts_rolling` at the fixture date |
| `away_scored_rate_5` | Fraction over the team's last 5 matches in its own venue (BTTS / clean sheet / scored) | Copied from the last row in that role — stale, and zero-filled only if the column was absent | **MISMATCH** | Recomputed by `builder.compute_btts_rolling` at the fixture date |
| `home_over25_rate_5` | Fraction over the team's last 5 matches in its own venue (3+ goals / team scored 2+) | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_over25_rolling` at the fixture date |
| `away_over25_rate_5` | Fraction over the team's last 5 matches in its own venue (3+ goals / team scored 2+) | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_over25_rolling` at the fixture date |
| `home_high_scoring_rate_5` | Fraction over the team's last 5 matches in its own venue (3+ goals / team scored 2+) | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_over25_rolling` at the fixture date |
| `away_high_scoring_rate_5` | Fraction over the team's last 5 matches in its own venue (3+ goals / team scored 2+) | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_over25_rolling` at the fixture date |
| `home_corners_for_pg_5` | Corners for/against per game and 10+ corner share over the team's last 5 matches in its own venue | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_corners_rolling` at the fixture date |
| `away_corners_for_pg_5` | Corners for/against per game and 10+ corner share over the team's last 5 matches in its own venue | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_corners_rolling` at the fixture date |
| `home_corners_against_pg_5` | Corners for/against per game and 10+ corner share over the team's last 5 matches in its own venue | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_corners_rolling` at the fixture date |
| `away_corners_against_pg_5` | Corners for/against per game and 10+ corner share over the team's last 5 matches in its own venue | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_corners_rolling` at the fixture date |
| `home_over95_corners_rate_5` | Corners for/against per game and 10+ corner share over the team's last 5 matches in its own venue | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_corners_rolling` at the fixture date |
| `away_over95_corners_rate_5` | Corners for/against per game and 10+ corner share over the team's last 5 matches in its own venue | Copied from the last row in that role | **MISMATCH** | Recomputed by `builder.compute_corners_rolling` at the fixture date |

## What was fixed

### 1. Inference now runs the builder's own feature functions

`backend/features/inference.py` reconstructs the inputs the builder needs from
the features CSV (match history, Elo history, FBref snapshots) and calls
`builder.compute_match_features()` — the exact function `build_features()` calls
for every training row — using the fixture date as the cutoff. There is no
second implementation of any rolling window in the prediction path.

| Mismatch | Fix |
| --- | --- |
| Rolling windows (`*_5`, `*_10`) were copied from an older row | Recomputed at the fixture date, same `window=` and venue split as training |
| Venue splits (`home_home_*`, `away_away_*`) came from a row chosen by venue, not from the venue-filtered history | Recomputed with `venue="home"` / `venue="away"` |
| H2H came from the last row's opponent | `compute_h2h(home_team, away_team, fixture_date)` |
| Elo was 0.0 when unknown, and taken from two different dates | `builder.get_latest_elo_before()` reads `elo_ratings` (the table the builder reads); CSV reconstruction and the 1500 fallback remain for DB outages |
| `elo_difference` was computed from mismatched ratings (and, in the Over 2.5 builder, from a partially-built dict) | Derived once, centrally, from the resolved ratings |
| FBref stats were the values stored on an old row, never refreshed, no availability check | `join_fbref_features()` re-applied at the fixture date against the live `fbref_team_stats` snapshot |
| Specialist rates (BTTS / Over 2.5 / corners, 16 columns) were copied per-model from different rows | All three classifiers now read the same rebuilt row |
| NaN was passed through (`float(value or 0.0)` keeps `NaN`) | Every model input is finite; anything unavailable is the builder's 0.0 |
| `season_encoded` defaulted to the literal `2024-2025` in `predict.py` and `fixtures.py` | Default is `current_season_label()` (July rule, same as the collectors) |
| Unseen label handling differed between the paths | One `features.inference.encode_label`, case/whitespace tolerant, 0 fallback |

### 2. The features CSV now carries the match history it is replayed from

The builder emitted only `target_*` columns, so a saved CSV could not rebuild
`home_shots` / `away_shots_ot` / … `build_features()` now also writes the nine
`MATCH_HISTORY_COLUMNS` (raw results and stats) as **meta** columns. They are
never model inputs; `models/trainer.py` still selects `FEATURE_COLS` only, so
nothing about training changes.

An older CSV (published before this change) keeps working: goals, result and
corners are mapped back from the `target_*` columns, and the six shot-derived
features fall back to the team's last stored value instead of silently becoming
0. `scripts/check_features.py` warns while a CSV is still in that state, and
`/predict/admin/diagnostics` reports `features_csv_has_match_history: false`.

### 3. One source of truth for the column lists

`backend/features/feature_columns.py` now owns `FEATURE_COLS`,
`BTTS_FEATURE_COLS`, `OVER25_FEATURE_COLS`, `OVER95_CORNERS_FEATURE_COLS`,
`FBREF_STAT_COLUMNS`, `MATCH_HISTORY_COLUMNS`, the required-column lists and an
import-time contract check. `models/trainer.py` and `routers/predict.py` import
the same objects — the copies in both modules are gone.

`validate_feature_column_counts()` runs on import of the trainer and of the API
and raises if a list is ever edited to something other than 108 / 17 / 18 / 17
or contains a duplicate, so a half-finished edit cannot reach production.

Nothing about the *values* of the training features changed: the builder's
formulas, windows, cutoffs and NaN rules are untouched. FEATURE_COLS order is
byte-identical to the previous definition (asserted in the tests), so existing
artifacts stay compatible.

### 4. Other bugs found during the audit

* `_build_over25_feature_vector` derived `elo_difference` from `vector.get(...)`
  while filling the same dict — correct only because `home_elo`/`away_elo`
  happen to precede it in the list. Now centralised.
* `_get_team_features` selected features by `home_*`/`away_*` prefix, so
  `h2h_*` columns were read off the home row only (see above) and any future
  non-prefixed column silently scored 0.
* `_get_btts_probability` / `_get_over25_probability` each duplicated the
  "are the required columns in the CSV?" check with different column lists;
  one shared `_model_feature_vector` now does it.
* `/predict/match`'s `season` field was pinned to `2024-2025`; so was
  `fixtures.DEFAULT_SEASON`. Both now resolve to the current season.
* The builder's per-row code lived inside the `build_features` loop, which is
  why inference could not reuse it. It is now `build_feature_row()` /
  `compute_match_features()`.

## Known limitations (deliberate, documented)

| Limitation | Impact | Mitigation |
| --- | --- | --- |
| Elo reconstruction from the CSV alone uses each team's *pre-match* rating, one match behind the `elo_ratings` table | Only matters when the database is unreachable at inference | The authoritative DB lookup is preferred; the CSV path is the fallback. `/predict/admin/diagnostics` reports which one is active |
| An older features.csv has no raw shot counts | 6 shot-derived features use the team's last stored value for one deployment cycle | Warnings in `check_features.py` and diagnostics; exact again after the next daily builder run |
| A club whose real Elo history predates its first row in the CSV | Not visible to the CSV-only fallback (1500 is used) | Same DB-first rule as above |
| The frontend still sends `season: "2024-2025"` from its own dropdown | `season_encoded` is stale for UI-originated requests until the frontend is updated (out of scope: no frontend changes) | Server-side default is correct; `fixtures.py` (daily pipeline) no longer uses a hardcoded season |

## Regression test

`backend/tests/test_feature_parity.py` (19 tests, no database, no network):

* builds a synthetic 48-match, 2-league history and turns it into a features
  frame with the **builder**;
* cuts that frame at each fixture date and scores it with **predict**;
* asserts every one of the 108 / 17 / 18 / 17 inputs matches within `1e-6`,
  for every synthetic match — not just a cherry-picked one;
* pins the column lists to the shared module and their counts;
* covers the legacy-CSV fallback, the FBref CSV fallback, the 1500 Elo
  fallback, the same-day cutoff rule, and unseen-label encoding.

The daily workflow now runs it right after `check_features.py`, before the
9-model training pass, so drift fails the run instead of shipping.

## Verification after merge

See the end of this document for the exact calls.

```bash
# 1. Local, no services needed
cd backend && python -m unittest tests.test_feature_parity -v

# 2. After the next daily run (or a manual workflow_dispatch)
curl -s "$API/predict/admin/diagnostics" | python -m json.tool
#   "feature_assembly": "builder"                  <- new code path is live
#   "feature_columns_source": "features/feature_columns.py"
#   "features_csv_has_match_history": true         <- next builder run republished the CSV
#   "elo_lookup": "database"                       <- Elo read from elo_ratings
#   "fbref_source": "database"
#   "cols_match": true

# 3. A prediction still answers, and now exposes the exact row it scored
curl -s -X POST "$API/predict/match" -H 'Content-Type: application/json' \
  -d '{"home_team":"Arsenal","away_team":"Chelsea","league":"ENG-Premier League"}'
#   -> goals/corners/1X2 as before, plus "feature_vector_used" whose
#      home_form_* / h2h_* values change from fixture to fixture (they used to
#      be frozen to the team's last stored row).
```
