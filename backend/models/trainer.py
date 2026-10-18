"""
Model training pipeline for the Football Predictor.

Trains all prediction models from the features CSV
produced by builder.py. Uses strict chronological
splitting. Never tunes on the test set.

Run from backend/ directory:
    python models/trainer.py
"""

from __future__ import annotations

import os
import unicodedata
import warnings
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from supabase import create_client
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# --------------------------------------------------------------------------- #
# Feature columns
# --------------------------------------------------------------------------- #

FEATURE_COLS = [
    # Home form 5
    "home_form_wins_5", "home_form_draws_5",
    "home_form_losses_5", "home_form_points_5",
    "home_form_gf_5", "home_form_ga_5",
    "home_form_gd_5", "home_form_shots_5",
    "home_form_shots_ot_5", "home_form_corners_5",
    # Home form 10
    "home_form_wins_10", "home_form_draws_10",
    "home_form_losses_10", "home_form_points_10",
    "home_form_gf_10", "home_form_ga_10",
    "home_form_gd_10",
    # Home at home 5
    "home_home_wins_5", "home_home_draws_5",
    "home_home_losses_5", "home_home_points_5",
    "home_home_gf_5", "home_home_ga_5",
    "home_home_gd_5", "home_home_shots_5",
    "home_home_corners_5",
    # Away form 5
    "away_form_wins_5", "away_form_draws_5",
    "away_form_losses_5", "away_form_points_5",
    "away_form_gf_5", "away_form_ga_5",
    "away_form_gd_5", "away_form_shots_5",
    "away_form_shots_ot_5", "away_form_corners_5",
    # Away form 10
    "away_form_wins_10", "away_form_draws_10",
    "away_form_losses_10", "away_form_points_10",
    "away_form_gf_10", "away_form_ga_10",
    "away_form_gd_10",
    # Away at away 5
    "away_away_wins_5", "away_away_draws_5",
    "away_away_losses_5", "away_away_points_5",
    "away_away_gf_5", "away_away_ga_5",
    "away_away_gd_5", "away_away_shots_5",
    "away_away_corners_5",
    # Elo
    "home_elo", "away_elo", "elo_difference",
    # H2H
    "h2h_home_wins_5", "h2h_draws_5",
    "h2h_away_wins_5", "h2h_home_gf_5",
    "h2h_away_gf_5",
    # League and season
    "league_encoded", "season_encoded",
    # FBref home team stats
    "home_fbref_games",
    "home_fbref_shots", "home_fbref_shots_on_target",
    "home_fbref_shot_accuracy",
    "home_fbref_goals_per_shot", "home_fbref_goals_per_sot",
    "home_fbref_yellow_cards", "home_fbref_red_cards",
    "home_fbref_fouls_committed", "home_fbref_fouls_drawn",
    "home_fbref_offsides", "home_fbref_crosses",
    "home_fbref_interceptions", "home_fbref_tackles_won",
    "home_fbref_pens_won", "home_fbref_pens_conceded",
    "home_fbref_sota", "home_fbref_goals_against",
    "home_fbref_saves", "home_fbref_save_pct",
    "home_fbref_clean_sheets",
    "home_fbref_pk_faced", "home_fbref_pk_saved",
    # FBref away team stats
    "away_fbref_games",
    "away_fbref_shots", "away_fbref_shots_on_target",
    "away_fbref_shot_accuracy",
    "away_fbref_goals_per_shot", "away_fbref_goals_per_sot",
    "away_fbref_yellow_cards", "away_fbref_red_cards",
    "away_fbref_fouls_committed", "away_fbref_fouls_drawn",
    "away_fbref_offsides", "away_fbref_crosses",
    "away_fbref_interceptions", "away_fbref_tackles_won",
    "away_fbref_pens_won", "away_fbref_pens_conceded",
    "away_fbref_sota", "away_fbref_goals_against",
    "away_fbref_saves", "away_fbref_save_pct",
    "away_fbref_clean_sheets",
    "away_fbref_pk_faced", "away_fbref_pk_saved",
]

# The 1X2 target column produced by builder.py.
TARGET_COL = "target_result"


# --------------------------------------------------------------------------- #
# Training leagues
# --------------------------------------------------------------------------- #

#: Leagues whose rows are used to train every model. ``features.csv`` can
#: contain more (or fewer) rows than this list describes: ``train_all_models``
#: filters to exactly these competitions before the chronological split.
#: The Big Five plus the six leagues added when the training dataset was
#: expanded (see backend/data/collector.py, which ingests them).
TRAINING_LEAGUES: list[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "ENG-Championship",
    "POR-Primeira Liga",
    "NED-Eredivisie",
    "TUR-Süper Lig",
    "NOR-Eliteserien",
    "AUT-Bundesliga",
]

#: Canonical spelling per known alias/variant of the league names above.
#: The database stores league strings written by different collectors and
#: sources, so the match against TRAINING_LEAGUES is done after normalising
#: (casefold, whitespace collapsed, diacritics stripped) — which subsumes
#: the case/accent variants — while genuinely different spellings are listed
#: here explicitly.
LEAGUE_ALIASES: dict[str, str] = {
    "tur-super lig":      "TUR-Süper Lig",
    "tur-superlig":       "TUR-Süper Lig",
    "tur-spor toto super lig": "TUR-Süper Lig",
    "turkiye-super lig":  "TUR-Süper Lig",
    "eng-championship":   "ENG-Championship",
    "eng-league championship": "ENG-Championship",
    "eng-d2":             "ENG-Championship",
    "england-championship": "ENG-Championship",
    "por-primeira liga":  "POR-Primeira Liga",
    "por-liga portugal":  "POR-Primeira Liga",
    "por-liga betclic":   "POR-Primeira Liga",
    "ned-eredivisie":     "NED-Eredivisie",
    "ned-holland eredivisie": "NED-Eredivisie",
    "nor-eliteserien":    "NOR-Eliteserien",
    "nor-obos-ligaen":    "NOR-Eliteserien",
    "nor-telenor-ligaen": "NOR-Eliteserien",
    "nor-norway eliteserien": "NOR-Eliteserien",
    "aut-bundesliga":     "AUT-Bundesliga",
    "aut-austrian bundesliga": "AUT-Bundesliga",
    "aut-admiral bundesliga": "AUT-Bundesliga",
    "aut-osterreich bundesliga": "AUT-Bundesliga",
    "ger-bundesliga":     "GER-Bundesliga",
    "ger-1 bundesliga":   "GER-Bundesliga",
    "ger-bundesliga 1":    "GER-Bundesliga",
}


def _normalise_league_name(name: object) -> str:
    """Lowercase, strip diacritics and collapse whitespace for alias lookup."""
    text = str(name).strip()
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    return " ".join(ascii_only.casefold().split())


def canonical_league(name: object) -> str | None:
    """
    Map a stored league string to the canonical TRAINING_LEAGUES spelling.

    Returns ``None`` when the name is not one of the training leagues
    (after normalisation and alias lookup), so callers can decide whether
    an unrecognised league is a mismatch to fix or a league to ignore.
    """
    if name is None:
        return None
    normalised = _normalise_league_name(name)
    if not normalised:
        return None
    aliased = LEAGUE_ALIASES.get(normalised)
    if aliased is not None:
        return aliased
    canonical_normalised = {
        _normalise_league_name(c): c for c in TRAINING_LEAGUES
    }
    return canonical_normalised.get(normalised)


def filter_training_leagues(
    df: pd.DataFrame,
    leagues: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Keep only rows whose ``league`` belongs to the training leagues.

    ``leagues`` defaults to :data:`TRAINING_LEAGUES`. Matching is
    accent/case-insensitive and alias-aware (:func:`canonical_league`), so
    e.g. "TUR-Super Lig" and "NOR-OBOS-ligaen" match their canonical names.
    Prints a per-league row count of the kept data, warns about leagues
    configured but missing from the frame, and about unrecognised leagues
    that are dropped. Rows without a ``league`` column at all are kept
    unchanged (with a warning) so a schema regression cannot silently
    train on nothing.
    """
    wanted = list(TRAINING_LEAGUES if leagues is None else leagues)
    if df is None or df.empty:
        return pd.DataFrame()
    if "league" not in df.columns:
        print("[trainer] WARNING: features frame has no 'league' column — "
              "skipping league filtering")
        return df

    canon_to_wanted = {canonical_league(c): c for c in wanted}
    canon_to_wanted.pop(None, None)

    canonical = df["league"].map(canonical_league)
    keep = canonical.isin(set(canon_to_wanted))

    counts = df.loc[keep, "league"].value_counts()
    renamed = counts.copy()
    renamed.index = [canon_to_wanted.get(c, c) for c in renamed.index]
    print(f"[trainer] League filter: keeping {int(keep.sum())} of {len(df)} "
          f"rows across {len(renamed)} leagues")
    for name, n in renamed.items():
        print(f"[trainer]   {name:<22} {n:5} rows")

    seen_canon = set(canonical.dropna().unique())
    missing = [c for c in wanted if c not in seen_canon]
    if missing:
        print(f"[trainer] WARNING: leagues configured for training but with "
              f"0 rows in features: {missing}")

    unknown = df.loc[~keep & canonical.isna(), "league"].value_counts()
    if len(unknown):
        print(f"[trainer] NOTE: dropped {int((~keep & canonical.isna()).sum())} "
              f"rows from leagues outside the training set:")
        for name, n in unknown.items():
            print(f"[trainer]   {str(name):<28} {n:5} rows")

    return df.loc[keep].reset_index(drop=True)

BTTS_FEATURE_COLS = [
    "home_btts_rate_5", "away_btts_rate_5",
    "home_cs_rate_5", "away_cs_rate_5",
    "home_scored_rate_5", "away_scored_rate_5",
    "home_form_gf_5", "away_form_gf_5",
    "home_form_ga_5", "away_form_ga_5",
    "home_fbref_goals_against", "away_fbref_goals_against",
    "home_fbref_clean_sheets", "away_fbref_clean_sheets",
    "home_elo", "away_elo",
    "league_encoded",
]

TARGET_BTTS_COL = "target_btts"

OVER25_FEATURE_COLS = [
    "home_over25_rate_5", "away_over25_rate_5",
    "home_high_scoring_rate_5", "away_high_scoring_rate_5",
    "home_form_gf_5", "away_form_gf_5",
    "home_form_ga_5", "away_form_ga_5",
    "home_form_gf_10", "away_form_gf_10",
    "home_fbref_goals_against", "away_fbref_goals_against",
    "home_fbref_shots_on_target", "away_fbref_shots_on_target",
    "home_elo", "away_elo",
    "elo_difference",
    "league_encoded",
]

TARGET_OVER25_COL = "target_over25"

# Dedicated feature set for the Over 9.5 corners classifier. Deliberately
# narrow: opponent-adjusted corner rates carry the signal, while the shared
# 108-column vector dilutes it with goal/discipline noise.
OVER95_CORNERS_FEATURE_COLS = [
    # New corner-specific rolling rates (per game / fractions)
    "home_corners_for_pg_5", "away_corners_for_pg_5",
    "home_corners_against_pg_5", "away_corners_against_pg_5",
    "home_over95_corners_rate_5", "away_over95_corners_rate_5",
    # Existing venue-specific corners taken
    "home_home_corners_5", "away_away_corners_5",
    # Existing overall corners form
    "home_form_corners_5", "away_form_corners_5",
    # FBref attacking volume — the closest available corner drivers
    "home_fbref_shots_on_target", "away_fbref_shots_on_target",
    "home_fbref_crosses", "away_fbref_crosses",
    # Team strength and league context
    "home_elo", "away_elo",
    "league_encoded",
]

TARGET_OVER95_CORNERS_COL = "target_over95_corners"


# --------------------------------------------------------------------------- #
# Part A — Feature loading
# --------------------------------------------------------------------------- #


def load_features(
    path: str = "data/processed/features.csv",
) -> pd.DataFrame:
    """
    Load the features CSV produced by builder.py.
    Parses date column, sorts chronologically.
    Returns empty DataFrame on failure.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[trainer] WARNING: features file not found: {path}")
        print("[trainer] Run builder.build_and_save_features() first.")
        return pd.DataFrame()
    except Exception as exc:
        print(f"[trainer] WARNING: could not read {path}: {exc}")
        return pd.DataFrame()

    try:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Chronological order is essential: every split and every
        # downstream evaluation depends on it. Never shuffle.
        if "date" in df.columns:
            df = df.sort_values("date")
        df = df.reset_index(drop=True)
    except Exception as exc:
        print(f"[trainer] WARNING: could not prepare features: {exc}")
        return pd.DataFrame()

    if len(df) == 0:
        print(f"[trainer] WARNING: features file is empty: {path}")
        return df

    first_date = df["date"].min() if "date" in df.columns else None
    last_date = df["date"].max() if "date" in df.columns else None
    print(
        f"[trainer] Loaded {len(df)} rows x {len(df.columns)} columns "
        f"from {path}"
    )
    print(f"[trainer] Date range: {first_date} -> {last_date}")
    return df


# --------------------------------------------------------------------------- #
# Part B — Chronological split
# --------------------------------------------------------------------------- #


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.65,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train, validation, test sets
    by chronological order — never randomly.

    train_frac = 0.65  (first 65% of rows)
    val_frac   = 0.15  (next 15% of rows)
    test       = remaining 20% of rows

    Prints the size and date range of each split.
    Returns (train_df, val_df, test_df).
    """
    if df is None or len(df) == 0:
        print("[trainer] WARNING: chronological_split received an empty DataFrame")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not 0 < train_frac < 1 or not 0 <= val_frac < 1:
        print(
            f"[trainer] WARNING: invalid split fractions "
            f"train={train_frac} val={val_frac}; using defaults"
        )
        train_frac, val_frac = 0.65, 0.15

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    # Slice by position only — no shuffling anywhere in this module.
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    def _describe(name: str, part: pd.DataFrame) -> None:
        if len(part) == 0:
            print(f"[trainer] {name:5} split: 0 rows")
            return
        if "date" in part.columns:
            start = part["date"].min()
            end = part["date"].max()
        else:
            start = end = "n/a"
        print(
            f"[trainer] {name:5} split: {len(part):5} rows "
            f"({len(part) / n:.1%})  {start} -> {end}"
        )

    print(f"[trainer] Chronological split of {n} rows "
          f"(train {train_frac:.0%} / val {val_frac:.0%} / "
          f"test {1 - train_frac - val_frac:.0%})")
    _describe("train", train_df)
    _describe("val", val_df)
    _describe("test", test_df)

    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Part D — 1X2 model training
# --------------------------------------------------------------------------- #


def _build_result_pipeline() -> Pipeline:
    """
    Build the 1X2 classifier pipeline:
    SimpleImputer(median) -> StandardScaler -> LogisticRegression.

    Note on ``multi_class``: scikit-learn deprecated the parameter in 1.5 and
    removed it in 1.7, where the ``lbfgs`` solver is always multinomial. It is
    therefore passed only when the installed version still accepts it, so the
    same hyperparameters work on both old and new scikit-learn instead of
    raising TypeError on import of the estimator.
    """
    params = dict(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    if "multi_class" in LogisticRegression().get_params():
        params["multi_class"] = "multinomial"

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ]
    )


def _extract_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select FEATURE_COLS / TARGET_COL and drop rows with a null target."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=object)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[trainer] WARNING: missing feature columns: {missing}")

    usable_cols = [c for c in FEATURE_COLS if c in df.columns]
    if TARGET_COL not in df.columns:
        print(f"[trainer] WARNING: target column '{TARGET_COL}' not found")
        return df[usable_cols].copy(), pd.Series(dtype=object)

    mask = df[TARGET_COL].notna()
    dropped = int((~mask).sum())
    if dropped:
        print(f"[trainer] Dropped {dropped} rows with a null {TARGET_COL}")

    X = df.loc[mask, usable_cols].copy()
    y = df.loc[mask, TARGET_COL].astype(str).copy()
    return X, y


def _per_class_recall(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes,
) -> dict:
    """
    Per-class recall (H, D, A) computed directly with numpy, so the module
    needs no extra scikit-learn metric imports.
    """
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    recalls = {}
    for label in classes:
        is_class = actual == label
        total = int(is_class.sum())
        hits = int((predicted[is_class] == label).sum()) if total else 0
        recalls[str(label)] = (hits / total) if total else 0.0
    return recalls


def _evaluate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> dict:
    """Compute and print accuracy, balanced accuracy, macro F1, log loss and per-class recall."""
    if len(X) == 0 or len(y) == 0:
        print(f"[trainer] {split_name}: no rows to evaluate")
        return {}

    y_pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)

    accuracy = accuracy_score(y, y_pred)
    balanced = balanced_accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    # labels are passed explicitly so the metric stays correct even if a split
    # happens to contain only one or two of the three outcomes
    loss = log_loss(y, proba, labels=classes)
    recalls = _per_class_recall(y, y_pred, classes)

    print(f"[trainer] --- {split_name} metrics ({len(X)} rows) ---")
    print(f"[trainer]   Accuracy          : {accuracy:.4f}")
    print(f"[trainer]   Balanced accuracy : {balanced:.4f}")
    print(f"[trainer]   Macro F1          : {macro_f1:.4f}")
    print(f"[trainer]   Log loss          : {loss:.4f}")
    for label in sorted(recalls):
        count = int((np.asarray(y) == label).sum())
        print(
            f"[trainer]   Recall {label}          : "
            f"{recalls[label]:.4f}  (n={count})"
        )

    return {
        "rows": int(len(X)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "macro_f1": float(macro_f1),
        "log_loss": float(loss),
        **{f"recall_{label}": value for label, value in recalls.items()},
    }


def train_result_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: str = "models",
) -> dict:
    """
    Train and evaluate the 1X2 match result classifier.

    Steps:
    1. Extract X and y from each split using FEATURE_COLS
       and target column "target_result"
    2. Drop rows where target_result is null
    3. Use a Pipeline:
         SimpleImputer(strategy="median")
         -> StandardScaler()
         -> LogisticRegression(
              max_iter=1000,
              class_weight="balanced",
              random_state=42,
              multi_class="multinomial",
              solver="lbfgs"
            )
    4. Fit pipeline on train only
    5. Evaluate on VALIDATION set:
         - Accuracy
         - Balanced accuracy
         - Macro F1
         - Log loss
         - Per-class recall (H, D, A)
       Print all validation metrics
    6. Refit the SAME pipeline (same hyperparameters)
       on train + validation COMBINED
    7. Evaluate once on TEST set:
         - Same metrics as above
       Print all test metrics
    8. Save the final fitted pipeline to:
         {model_dir}/result_model_v1.joblib
    9. Return a dict with all test metrics and
       the path to the saved model

    Notes:
    - y values are strings: "H", "D", "A"
    - Never tune hyperparameters based on test results
    - Print clearly which step is running
    """
    print("=" * 70)
    print("[trainer] STEP 1/6: extracting features and targets")
    print("=" * 70)
    X_train, y_train = _extract_xy(train_df)
    X_val, y_val = _extract_xy(val_df)
    X_test, y_test = _extract_xy(test_df)
    print(
        f"[trainer] train={len(X_train)}  val={len(X_val)}  test={len(X_test)} "
        f"| {len(FEATURE_COLS)} feature columns"
    )

    if len(X_train) == 0:
        print("[trainer] ERROR: no training rows available, aborting")
        return {}

    print("=" * 70)
    print("[trainer] STEP 2/6: fitting pipeline on TRAIN")
    print("=" * 70)
    pipeline = _build_result_pipeline()
    pipeline.fit(X_train, y_train)
    print(f"[trainer] Fitted on {len(X_train)} rows; "
          f"classes seen: {list(pipeline.classes_)}")

    print("=" * 70)
    print("[trainer] STEP 3/6: validating (model selection happens here)")
    print("=" * 70)
    val_metrics = _evaluate(pipeline, X_val, y_val, "VALIDATION")

    print("=" * 70)
    print("[trainer] STEP 4/6: refitting the SAME pipeline on TRAIN + VALIDATION")
    print("=" * 70)
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    final_pipeline = _build_result_pipeline()
    final_pipeline.fit(X_trainval, y_trainval)
    print(f"[trainer] Refitted on {len(X_trainval)} rows; "
          f"classes seen: {list(final_pipeline.classes_)}")

    print("=" * 70)
    print("[trainer] STEP 5/6: evaluating ONCE on TEST (all choices are locked)")
    print("=" * 70)
    test_metrics = _evaluate(final_pipeline, X_test, y_test, "TEST")

    print("=" * 70)
    print("[trainer] STEP 6/6: saving model artifact")
    print("=" * 70)
    model_path = Path(model_dir) / "result_model_v1.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    print(f"[trainer] Saved pipeline -> {model_path}")

    results = {
        **test_metrics,
        "val_accuracy": val_metrics.get("accuracy"),
        "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
        "val_macro_f1": val_metrics.get("macro_f1"),
        "val_log_loss": val_metrics.get("log_loss"),
        "model_path": str(model_path),
        "n_features": len(FEATURE_COLS),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    return results


def _evaluate_1x2_metrics(
    y_true_str: pd.Series | np.ndarray,
    y_pred_str: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
) -> dict:
    """Compute accuracy, balanced accuracy, macro F1, AUC, log loss, and per-class recall."""
    if len(y_true_str) == 0:
        return {}

    acc = float(accuracy_score(y_true_str, y_pred_str))
    bal_acc = float(balanced_accuracy_score(y_true_str, y_pred_str))
    macro_f1 = float(f1_score(y_true_str, y_pred_str, average="macro", zero_division=0))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_int = np.array([class_to_idx[c] for c in y_true_str])

    try:
        auc = float(
            roc_auc_score(
                y_true_int,
                proba,
                multi_class="ovr",
                average="macro",
                labels=list(range(len(classes))),
            )
        )
    except Exception:
        auc = float("nan")

    try:
        loss = float(log_loss(y_true_int, proba, labels=list(range(len(classes)))))
    except Exception:
        loss = float("nan")

    recalls = _per_class_recall(y_true_str, y_pred_str, ["H", "D", "A"])

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "auc_macro_ovr": auc,
        "log_loss": loss,
        "recall_H": float(recalls.get("H", 0.0)),
        "recall_D": float(recalls.get("D", 0.0)),
        "recall_A": float(recalls.get("A", 0.0)),
    }


def _build_xgb_result_pipeline() -> Pipeline:
    """
    Build the XGBoost 1X2 classifier pipeline:
    SimpleImputer(median) -> XGBClassifier.
    XGBoost handles feature scaling internally; StandardScaler is omitted.
    """
    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**params)),
        ]
    )


def train_xgb_result_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: str = "models",
) -> dict:
    """
    Train and evaluate the XGBoost 1X2 match result classifier, and compare
    side-by-side against the logistic regression model on the identical test split.
    """
    print("=" * 70)
    print("[trainer] XGBOOST 1X2 MODEL — STEP 1/6: extracting features and targets")
    print("=" * 70)
    X_train, y_train = _extract_xy(train_df)
    X_val, y_val = _extract_xy(val_df)
    X_test, y_test = _extract_xy(test_df)
    print(
        f"[trainer] train={len(X_train)}  val={len(X_val)}  test={len(X_test)} "
        f"| {len(FEATURE_COLS)} feature columns"
    )

    if len(X_train) == 0:
        print("[trainer] ERROR: no training rows available for XGBoost, aborting")
        return {}

    # Label encode target_result strings ("H", "D", "A") to integers 0, 1, 2
    le = LabelEncoder()
    unique_targets = sorted(list(set(y_train) | set(y_val) | set(y_test) | {"A", "D", "H"}))
    le.fit(unique_targets)
    classes = list(le.classes_)
    print(f"[trainer] Label encoder classes: {classes}")

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    print("=" * 70)
    print("[trainer] XGBOOST 1X2 MODEL — STEP 2/6: fitting pipeline on TRAIN")
    print("=" * 70)
    pipeline = _build_xgb_result_pipeline()
    pipeline.fit(X_train, y_train_enc)
    print(f"[trainer] Fitted XGBoost on {len(X_train)} rows")

    print("=" * 70)
    print("[trainer] XGBOOST 1X2 MODEL — STEP 3/6: validating")
    print("=" * 70)
    val_pred_enc = pipeline.predict(X_val)
    val_pred_str = le.inverse_transform(val_pred_enc)
    val_proba = pipeline.predict_proba(X_val)
    val_metrics = _evaluate_1x2_metrics(y_val, val_pred_str, val_proba, classes)
    print(f"[trainer] --- VALIDATION metrics ({len(X_val)} rows) ---")
    print(f"[trainer]   Accuracy          : {val_metrics['accuracy']:.4f}")
    print(f"[trainer]   Balanced accuracy : {val_metrics['balanced_accuracy']:.4f}")
    print(f"[trainer]   Macro F1          : {val_metrics['macro_f1']:.4f}")
    print(f"[trainer]   AUC (macro OvR)   : {val_metrics['auc_macro_ovr']:.4f}")
    print(f"[trainer]   Log loss          : {val_metrics['log_loss']:.4f}")

    print("=" * 70)
    print("[trainer] XGBOOST 1X2 MODEL — STEP 4/6: refitting on TRAIN + VALIDATION")
    print("=" * 70)
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval_enc = np.concatenate([y_train_enc, y_val_enc])
    y_trainval = pd.concat([y_train, y_val], axis=0)
    final_xgb_pipeline = _build_xgb_result_pipeline()
    final_xgb_pipeline.fit(X_trainval, y_trainval_enc)
    print(f"[trainer] Refitted XGBoost on {len(X_trainval)} rows")

    print("=" * 70)
    print("[trainer] XGBOOST 1X2 MODEL — STEP 5/6: evaluating ONCE on TEST")
    print("=" * 70)
    xgb_test_pred_enc = final_xgb_pipeline.predict(X_test)
    xgb_test_pred_str = le.inverse_transform(xgb_test_pred_enc)
    xgb_test_proba = final_xgb_pipeline.predict_proba(X_test)
    xgb_metrics = _evaluate_1x2_metrics(y_test, xgb_test_pred_str, xgb_test_proba, classes)

    # Re-evaluate logistic regression model on the identical test split
    logistic_path = Path(model_dir) / "result_model_v1.joblib"
    if logistic_path.exists():
        try:
            logistic_pipeline = joblib.load(logistic_path)
        except Exception as exc:
            print(f"[trainer] WARNING: could not load {logistic_path}: {exc}, refitting logistic")
            logistic_pipeline = _build_result_pipeline()
            logistic_pipeline.fit(X_trainval, y_trainval)
    else:
        logistic_pipeline = _build_result_pipeline()
        logistic_pipeline.fit(X_trainval, y_trainval)

    log_pred_str = logistic_pipeline.predict(X_test)
    raw_log_proba = logistic_pipeline.predict_proba(X_test)
    pipe_classes = list(logistic_pipeline.classes_)
    log_proba = np.zeros((len(X_test), len(classes)), dtype=float)
    for idx, cls_name in enumerate(classes):
        if cls_name in pipe_classes:
            col_idx = pipe_classes.index(cls_name)
            log_proba[:, idx] = raw_log_proba[:, col_idx]

    logistic_metrics = _evaluate_1x2_metrics(y_test, log_pred_str, log_proba, classes)

    # Side-by-side metric comparison logging
    print("\n" + "=" * 70)
    print("1X2 MODEL COMPARISON: LOGISTIC REGRESSION vs XGBOOST (TEST SET)")
    print("=" * 70)
    print(f"| {'Metric':<17} | {'Logistic':<10} | {'XGBoost':<10} |")
    print(f"|{'-' * 19}|{'-' * 12}|{'-' * 12}|")
    table_rows = [
        ("Accuracy", f"{logistic_metrics['accuracy']:.4f}", f"{xgb_metrics['accuracy']:.4f}"),
        ("Balanced Acc", f"{logistic_metrics['balanced_accuracy']:.4f}", f"{xgb_metrics['balanced_accuracy']:.4f}"),
        ("Macro F1", f"{logistic_metrics['macro_f1']:.4f}", f"{xgb_metrics['macro_f1']:.4f}"),
        ("AUC (macro OvR)", f"{logistic_metrics['auc_macro_ovr']:.4f}", f"{xgb_metrics['auc_macro_ovr']:.4f}"),
        ("Log Loss", f"{logistic_metrics['log_loss']:.4f}", f"{xgb_metrics['log_loss']:.4f}"),
        ("Recall H", f"{logistic_metrics['recall_H']:.4f}", f"{xgb_metrics['recall_H']:.4f}"),
        ("Recall D", f"{logistic_metrics['recall_D']:.4f}", f"{xgb_metrics['recall_D']:.4f}"),
        ("Recall A", f"{logistic_metrics['recall_A']:.4f}", f"{xgb_metrics['recall_A']:.4f}"),
    ]
    for m, l, x in table_rows:
        print(f"| {m:<17} | {l:<10} | {x:<10} |")
    print("=" * 70)

    # Compare AUC and F1 to identify the superior model
    xgb_auc = xgb_metrics.get("auc_macro_ovr", 0.0)
    log_auc = logistic_metrics.get("auc_macro_ovr", 0.0)
    xgb_f1 = xgb_metrics.get("macro_f1", 0.0)
    log_f1 = logistic_metrics.get("macro_f1", 0.0)

    if (xgb_auc >= log_auc and xgb_f1 >= log_f1) and (xgb_auc > log_auc or xgb_f1 > log_f1):
        winner = "XGBoost (higher AUC and Macro F1)"
    elif xgb_auc > log_auc and xgb_f1 < log_f1:
        winner = "XGBoost on AUC, Logistic on Macro F1 (XGBoost selected for calibration/ranking)"
    elif xgb_auc < log_auc and xgb_f1 > log_f1:
        winner = "Logistic on AUC, XGBoost on Macro F1"
    elif xgb_auc > log_auc:
        winner = "XGBoost (higher AUC)"
    elif xgb_f1 > log_f1:
        winner = "XGBoost (higher Macro F1)"
    else:
        winner = "Logistic Regression"
    print(f"[trainer] Performance summary: {winner}")

    print("=" * 70)
    print("[trainer] XGBOOST 1X2 MODEL — STEP 6/6: saving artifacts")
    print("=" * 70)
    model_path = Path(model_dir) / "result_model_xgb_v1.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_xgb_pipeline, model_path)
    print(f"[trainer] Saved XGBoost pipeline -> {model_path}")

    encoder_path = Path(model_dir) / "result_model_xgb_v1_label_encoder.joblib"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, encoder_path)
    print(f"[trainer] Saved label encoder -> {encoder_path}")

    return {
        "rows": int(len(X_test)),
        "accuracy": float(xgb_metrics["accuracy"]),
        "balanced_accuracy": float(xgb_metrics["balanced_accuracy"]),
        "macro_f1": float(xgb_metrics["macro_f1"]),
        "auc_macro_ovr": float(xgb_metrics["auc_macro_ovr"]),
        "log_loss": float(xgb_metrics["log_loss"]),
        "recall_H": float(xgb_metrics["recall_H"]),
        "recall_D": float(xgb_metrics["recall_D"]),
        "recall_A": float(xgb_metrics["recall_A"]),
        "val_accuracy": val_metrics.get("accuracy"),
        "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
        "val_macro_f1": val_metrics.get("macro_f1"),
        "val_auc_macro_ovr": val_metrics.get("auc_macro_ovr"),
        "val_log_loss": val_metrics.get("log_loss"),
        "logistic_test_metrics": logistic_metrics,
        "winner": winner,
        "model_path": str(model_path),
        "encoder_path": str(encoder_path),
        "n_features": len(FEATURE_COLS),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }


def train_count_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    model_name: str,
    model_dir: str = "models",
) -> dict:
    """
    Train a Poisson regression model for a single
    count target (goals or corners).

    target_col is one of:
        "target_home_goals"
        "target_away_goals"
        "target_home_corners"
        "target_away_corners"

    model_name is the filename stem e.g. "home_goals_v1"
    Saved to {model_dir}/{model_name}.joblib
    """
    print(f"[trainer] Target: {target_col}")

    def _extract_count_xy(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if df is None or len(df) == 0:
            return (
                pd.DataFrame(columns=FEATURE_COLS),
                pd.Series(dtype=float),
            )
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"[trainer] WARNING: missing feature cols: {missing}")
        usable = [c for c in FEATURE_COLS if c in df.columns]
        if target_col not in df.columns:
            print(f"[trainer] WARNING: target '{target_col}' not found")
            return df[usable].copy(), pd.Series(dtype=float)
        y_raw = pd.to_numeric(df[target_col], errors="coerce")
        mask = y_raw.notna()
        dropped = int((~mask).sum())
        if dropped:
            print(f"[trainer] Dropped {dropped} null rows in {target_col}")
        return df.loc[mask, usable].copy(), y_raw[mask].astype(float)

    def _poisson_deviance(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        y_pred = np.clip(y_pred, 0.01, None)
        y_true = np.asarray(y_true, dtype=float)
        dev = 2.0 * np.mean(
            y_pred - y_true - y_true * np.log(y_pred / (y_true + 1e-8))
        )
        return float(dev)

    def _evaluate_count(
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        split_name: str,
    ) -> dict:
        if len(X) == 0:
            print(f"[trainer] {split_name}: no rows to evaluate")
            return {}
        y_pred = pipeline.predict(X)
        y_arr = np.asarray(y, dtype=float)
        mae = float(mean_absolute_error(y_arr, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_arr, y_pred)))
        dev = _poisson_deviance(y_arr, y_pred)
        print(f"[trainer] --- {split_name} ({len(X)} rows) ---")
        print(f"[trainer]   MAE            : {mae:.4f}")
        print(f"[trainer]   RMSE           : {rmse:.4f}")
        print(f"[trainer]   Poisson dev    : {dev:.4f}")
        return {
            "rows": len(X),
            "mae": mae,
            "rmse": rmse,
            "poisson_deviance": dev,
        }

    # --- Extract splits ---
    X_train, y_train = _extract_count_xy(train_df)
    X_val,   y_val   = _extract_count_xy(val_df)
    X_test,  y_test  = _extract_count_xy(test_df)
    print(
        f"[trainer] Sizes -> train={len(X_train)} "
        f"val={len(X_val)} test={len(X_test)}"
    )

    if len(X_train) == 0:
        print("[trainer] ERROR: no training rows, aborting")
        return {}

    # --- Fit on train ---
    print("[trainer] Fitting on TRAIN...")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   PoissonRegressor(alpha=1.0, max_iter=300)),
    ])
    pipeline.fit(X_train, y_train)

    # --- Validate ---
    print("[trainer] Evaluating on VALIDATION...")
    val_metrics = _evaluate_count(pipeline, X_val, y_val, "VALIDATION")

    # --- Refit on train + val ---
    print("[trainer] Refitting on TRAIN + VALIDATION...")
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = pd.concat([y_train, y_val], axis=0)
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   PoissonRegressor(alpha=1.0, max_iter=300)),
    ])
    final_pipeline.fit(X_tv, y_tv)

    # --- Test (once, locked) ---
    print("[trainer] Evaluating ONCE on TEST...")
    test_metrics = _evaluate_count(
        final_pipeline, X_test, y_test, "TEST")

    # --- Save ---
    model_path = Path(model_dir) / f"{model_name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    print(f"[trainer] Saved -> {model_path}")

    return {
        **test_metrics,
        "val_mae": val_metrics.get("mae"),
        "val_rmse": val_metrics.get("rmse"),
        "val_poisson_deviance": val_metrics.get("poisson_deviance"),
        "model_path": str(model_path),
        "target": target_col,
    }


def train_btts_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: str = "models",
) -> dict:
    """
    Train and evaluate the dedicated BTTS binary classifier.

    Uses BTTS_FEATURE_COLS (6 new rolling rates + selected existing
    features) and target_btts. Same 65/15/20 chronological split.
    Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression.
    Saves to {model_dir}/btts_model_v1.joblib
    """
    print("=" * 70)
    print("[trainer] BTTS MODEL — extracting features and targets")
    print("=" * 70)
    target_col = TARGET_BTTS_COL

    def _extract_btts_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=BTTS_FEATURE_COLS), pd.Series(dtype=float)
        missing = [c for c in BTTS_FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"[trainer] WARNING: missing BTTS feature columns: {missing}")
        usable = [c for c in BTTS_FEATURE_COLS if c in df.columns]
        if target_col not in df.columns:
            print(f"[trainer] WARNING: target \'{target_col}\' not found")
            return df[usable].copy(), pd.Series(dtype=float)
        y_raw = pd.to_numeric(df[target_col], errors="coerce")
        mask = y_raw.notna()
        dropped = int((~mask).sum())
        if dropped:
            print(f"[trainer] Dropped {dropped} rows with a null {target_col}")
        X = df.loc[mask, usable].copy()
        y = y_raw[mask].astype(int)
        return X, y

    def _evaluate_btts(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
        if len(X) == 0 or len(y) == 0:
            print(f"[trainer] {split_name}: no rows to evaluate")
            return {}
        y_pred = pipeline.predict(X)
        try:
            y_proba = pipeline.predict_proba(X)[:, 1]
        except Exception:
            y_proba = y_pred.astype(float)
        acc = float(accuracy_score(y, y_pred))
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        f1 = float(f1_score(y, y_pred, average="binary", zero_division=0))
        try:
            auc = float(roc_auc_score(y, y_proba))
        except Exception:
            auc = float("nan")
        print(f"[trainer] --- {split_name} metrics ({len(X)} rows) ---")
        print(f"[trainer]   Accuracy          : {acc:.4f}")
        print(f"[trainer]   Balanced accuracy : {bal_acc:.4f}")
        print(f"[trainer]   F1 (binary)       : {f1:.4f}")
        print(f"[trainer]   AUC-ROC           : {auc:.4f}")
        return {
            "rows": int(len(X)),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "auc_roc": auc,
        }

    X_train, y_train = _extract_btts_xy(train_df)
    X_val, y_val = _extract_btts_xy(val_df)
    X_test, y_test = _extract_btts_xy(test_df)
    print(f"[trainer] BTTS sizes -> train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    if len(X_train) == 0:
        print("[trainer] ERROR: no BTTS training rows, aborting")
        return {}

    print("[trainer] Fitting BTTS pipeline on TRAIN...")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    print("[trainer] Evaluating BTTS on VALIDATION...")
    val_metrics = _evaluate_btts(pipeline, X_val, y_val, "VALIDATION")

    print("[trainer] Refitting BTTS on TRAIN + VALIDATION...")
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = pd.concat([y_train, y_val], axis=0)
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    final_pipeline.fit(X_tv, y_tv)

    print("[trainer] Evaluating BTTS ONCE on TEST...")
    test_metrics = _evaluate_btts(final_pipeline, X_test, y_test, "TEST")

    model_path = Path(model_dir) / "btts_model_v1.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    print(f"[trainer] Saved BTTS pipeline -> {model_path}")

    results = {
        **test_metrics,
        "val_accuracy": val_metrics.get("accuracy"),
        "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
        "val_f1": val_metrics.get("f1"),
        "val_auc_roc": val_metrics.get("auc_roc"),
        "model_path": str(model_path),
        "n_features": len(BTTS_FEATURE_COLS),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    return results


def train_over25_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: str = "models",
) -> dict:
    """
    Train and evaluate the dedicated Over 2.5 binary classifier.

    Uses OVER25_FEATURE_COLS (4 new rolling rates + selected existing
    features) and target_over25. Same 65/15/20 chronological split.
    Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression.
    Saves to {model_dir}/over25_model_v1.joblib
    """
    print("=" * 70)
    print("[trainer] OVER2.5 MODEL — extracting features and targets")
    print("=" * 70)
    target_col = TARGET_OVER25_COL

    def _extract_over25_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=OVER25_FEATURE_COLS), pd.Series(dtype=float)
        missing = [c for c in OVER25_FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"[trainer] WARNING: missing Over 2.5 feature columns: {missing}")
        usable = [c for c in OVER25_FEATURE_COLS if c in df.columns]
        if target_col not in df.columns:
            print(f"[trainer] WARNING: target '{target_col}' not found")
            return df[usable].copy(), pd.Series(dtype=float)
        y_raw = pd.to_numeric(df[target_col], errors="coerce")
        mask = y_raw.notna()
        dropped = int((~mask).sum())
        if dropped:
            print(f"[trainer] Dropped {dropped} rows with a null {target_col}")
        X = df.loc[mask, usable].copy()
        y = y_raw[mask].astype(int)
        return X, y

    def _evaluate_over25(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
        if len(X) == 0 or len(y) == 0:
            print(f"[trainer] {split_name}: no rows to evaluate")
            return {}
        y_pred = pipeline.predict(X)
        try:
            y_proba = pipeline.predict_proba(X)[:, 1]
        except Exception:
            y_proba = y_pred.astype(float)
        acc = float(accuracy_score(y, y_pred))
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        f1 = float(f1_score(y, y_pred, average="binary", zero_division=0))
        try:
            auc = float(roc_auc_score(y, y_proba))
        except Exception:
            auc = float("nan")
        print(f"[trainer] --- {split_name} metrics ({len(X)} rows) ---")
        print(f"[trainer]   Accuracy          : {acc:.4f}")
        print(f"[trainer]   Balanced accuracy : {bal_acc:.4f}")
        print(f"[trainer]   F1 (binary)       : {f1:.4f}")
        print(f"[trainer]   AUC-ROC           : {auc:.4f}")
        return {
            "rows": int(len(X)),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "auc_roc": auc,
        }

    X_train, y_train = _extract_over25_xy(train_df)
    X_val, y_val = _extract_over25_xy(val_df)
    X_test, y_test = _extract_over25_xy(test_df)
    print(f"[trainer] Over 2.5 sizes -> train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    if len(X_train) == 0:
        print("[trainer] ERROR: no Over 2.5 training rows, aborting")
        return {}

    print("[trainer] Fitting Over 2.5 pipeline on TRAIN...")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    print("[trainer] Evaluating Over 2.5 on VALIDATION...")
    val_metrics = _evaluate_over25(pipeline, X_val, y_val, "VALIDATION")

    print("[trainer] Refitting Over 2.5 on TRAIN + VALIDATION...")
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = pd.concat([y_train, y_val], axis=0)
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    final_pipeline.fit(X_tv, y_tv)

    print("[trainer] Evaluating Over 2.5 ONCE on TEST...")
    test_metrics = _evaluate_over25(final_pipeline, X_test, y_test, "TEST")

    model_path = Path(model_dir) / "over25_model_v1.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    print(f"[trainer] Saved Over 2.5 pipeline -> {model_path}")

    results = {
        **test_metrics,
        "val_accuracy": val_metrics.get("accuracy"),
        "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
        "val_f1": val_metrics.get("f1"),
        "val_auc_roc": val_metrics.get("auc_roc"),
        "model_path": str(model_path),
        "n_features": len(OVER25_FEATURE_COLS),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    return results


def train_over95_corners_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: str = "models",
) -> dict:
    """
    Train and evaluate the dedicated Over 9.5 corners binary classifier.

    Uses OVER95_CORNERS_FEATURE_COLS (6 new corner-specific rolling rates +
    the existing corner form, FBref attacking volume, Elo and league columns)
    and target_over95_corners. Same 65/15/20 chronological split.
    Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression.
    Saves to {model_dir}/over95_corners_model_v1.joblib
    """
    print("=" * 70)
    print("[trainer] OVER 9.5 CORNERS MODEL — extracting features and targets")
    print("=" * 70)
    target_col = TARGET_OVER95_CORNERS_COL

    def _extract_over95_corners_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=OVER95_CORNERS_FEATURE_COLS), pd.Series(dtype=float)
        missing = [c for c in OVER95_CORNERS_FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"[trainer] WARNING: missing Over 9.5 corners feature columns: {missing}")
        usable = [c for c in OVER95_CORNERS_FEATURE_COLS if c in df.columns]
        if target_col not in df.columns:
            print(f"[trainer] WARNING: target '{target_col}' not found")
            return df[usable].copy(), pd.Series(dtype=float)
        y_raw = pd.to_numeric(df[target_col], errors="coerce")
        mask = y_raw.notna()
        dropped = int((~mask).sum())
        if dropped:
            print(f"[trainer] Dropped {dropped} rows with a null {target_col}")
        X = df.loc[mask, usable].copy()
        y = y_raw[mask].astype(int)
        return X, y

    def _evaluate_over95_corners(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
        if len(X) == 0 or len(y) == 0:
            print(f"[trainer] {split_name}: no rows to evaluate")
            return {}
        y_pred = pipeline.predict(X)
        try:
            y_proba = pipeline.predict_proba(X)[:, 1]
        except Exception:
            y_proba = y_pred.astype(float)
        acc = float(accuracy_score(y, y_pred))
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        f1 = float(f1_score(y, y_pred, average="binary", zero_division=0))
        try:
            auc = float(roc_auc_score(y, y_proba))
        except Exception:
            auc = float("nan")
        print(f"[trainer] --- {split_name} metrics ({len(X)} rows) ---")
        print(f"[trainer]   Accuracy          : {acc:.4f}")
        print(f"[trainer]   Balanced accuracy : {bal_acc:.4f}")
        print(f"[trainer]   F1 (binary)       : {f1:.4f}")
        print(f"[trainer]   AUC-ROC           : {auc:.4f}")
        return {
            "rows": int(len(X)),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "auc_roc": auc,
        }

    X_train, y_train = _extract_over95_corners_xy(train_df)
    X_val, y_val = _extract_over95_corners_xy(val_df)
    X_test, y_test = _extract_over95_corners_xy(test_df)
    print(f"[trainer] Over 9.5 corners sizes -> train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    if len(X_train) == 0:
        print("[trainer] ERROR: no Over 9.5 corners training rows, aborting")
        return {}

    print("[trainer] Fitting Over 9.5 corners pipeline on TRAIN...")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    print("[trainer] Evaluating Over 9.5 corners on VALIDATION...")
    val_metrics = _evaluate_over95_corners(pipeline, X_val, y_val, "VALIDATION")

    print("[trainer] Refitting Over 9.5 corners on TRAIN + VALIDATION...")
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = pd.concat([y_train, y_val], axis=0)
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    final_pipeline.fit(X_tv, y_tv)

    print("[trainer] Evaluating Over 9.5 corners ONCE on TEST...")
    test_metrics = _evaluate_over95_corners(final_pipeline, X_test, y_test, "TEST")

    model_path = Path(model_dir) / "over95_corners_model_v1.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    print(f"[trainer] Saved Over 9.5 corners pipeline -> {model_path}")

    results = {
        **test_metrics,
        "val_accuracy": val_metrics.get("accuracy"),
        "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
        "val_f1": val_metrics.get("f1"),
        "val_auc_roc": val_metrics.get("auc_roc"),
        "model_path": str(model_path),
        "n_features": len(OVER95_CORNERS_FEATURE_COLS),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    return results


def upload_models_to_supabase(model_dir: str = "models") -> None:
    """
    Push the saved model artifacts to the "models"
    bucket in Supabase Storage, overwriting existing copies.

    Credentials are read from the SUPABASE_URL and SUPABASE_KEY
    environment variables. This step is best-effort: a missing
    credential or a failed upload prints a message and returns,
    it never interrupts a training run.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        print(
            "[trainer] WARNING: SUPABASE_URL and/or SUPABASE_KEY is not set, "
            "skipping upload to Supabase"
        )
        return

    # backend/ root — one level above this models/ package.
    BACKEND_DIR = Path(__file__).resolve().parent.parent

    # (local path, path inside the "models" bucket)
    model_files = [
        (Path(model_dir) / "result_model_v1.joblib", "result_model_v1.joblib"),
        (
            Path(model_dir) / "result_model_xgb_v1.joblib",
            "result_model_xgb_v1.joblib",
        ),
        (
            Path(model_dir) / "result_model_xgb_v1_label_encoder.joblib",
            "result_model_xgb_v1_label_encoder.joblib",
        ),
        (Path(model_dir) / "home_goals_v1.joblib", "home_goals_v1.joblib"),
        (Path(model_dir) / "away_goals_v1.joblib", "away_goals_v1.joblib"),
        (Path(model_dir) / "home_corners_v1.joblib", "home_corners_v1.joblib"),
        (Path(model_dir) / "away_corners_v1.joblib", "away_corners_v1.joblib"),
        (Path(model_dir) / "btts_model_v1.joblib", "btts_model_v1.joblib"),
        (Path(model_dir) / "over25_model_v1.joblib", "over25_model_v1.joblib"),
        (
            Path(model_dir) / "over95_corners_model_v1.joblib",
            "over95_corners_model_v1.joblib",
        ),
        (
            BACKEND_DIR / "data" / "league_encoder.joblib",
            "league_encoder.joblib",
        ),
        (
            BACKEND_DIR / "data" / "season_encoder.joblib",
            "season_encoder.joblib",
        ),
        (
            BACKEND_DIR / "data" / "processed" / "features.csv",
            "features.csv",
        ),
    ]

    try:
        client = create_client(url, key)
    except Exception as exc:
        print(f"[trainer] WARNING: Supabase client creation failed: {exc}")
        return

    for model_path, file_name in model_files:
        try:
            data = model_path.read_bytes()
            client.storage.from_("models").upload(
                path=file_name,
                file=data,
                file_options={
                    "content-type": "application/octet-stream",
                    "x-upsert": "true",
                },
            )
            print(f"[trainer] Uploaded to Supabase -> models/{file_name}")
        except Exception as exc:
            print(f"[trainer] WARNING: upload failed for {file_name}: {exc}")
            continue


def train_all_models(
    features_path: str = "data/processed/features.csv",
    model_dir: str = "models",
    leagues: Sequence[str] | None = None,
) -> dict:
    """
    Full training pipeline. Trains all models in order:
      1. 1X2 result classifier
      2. Home goals Poisson
      3. Away goals Poisson
      4. Home corners Poisson
      5. Away corners Poisson
      6. BTTS binary classifier
      7. Over 2.5 binary classifier
      8. Over 9.5 corners binary classifier

    Returns a summary dict of all model results.
    Validation controls all choices.
    Test is evaluated exactly once per model.

    ``leagues`` restricts training to the given competitions
    (default: :data:`TRAINING_LEAGUES` — the Big Five plus the six
    expanded-coverage leagues). Filtering happens before the
    chronological split, so every split and metric below describes
    exactly the selected leagues.
    """
    print("=" * 60)
    print("FOOTBALL PREDICTOR — MODEL TRAINING")
    print("=" * 60)

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    df = load_features(features_path)
    if df.empty:
        print("[trainer] ERROR: no features. Run builder.py first.")
        return {}

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[trainer] ERROR: missing feature columns: {missing}")
        return {}

    rows_before_filter = len(df)
    df = filter_training_leagues(df, leagues)
    if df.empty:
        print("[trainer] ERROR: league filtering removed every row — check "
              "that the features CSV uses the expected league names.")
        return {}
    features_rows = len(df)
    print(f"[trainer] features_rows for this training run: {features_rows} "
          f"(of {rows_before_filter} loaded)")

    train_df, val_df, test_df = chronological_split(df)

    results = {
        "_dataset": {
            "features_rows": features_rows,
            "features_rows_all_leagues": rows_before_filter,
            "leagues": list(TRAINING_LEAGUES if leagues is None else leagues),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        },
    }

    print("\n--- 1X2 RESULT MODEL (LOGISTIC) ---")
    results["result"] = train_result_model(
        train_df, val_df, test_df, model_dir)

    print("\n--- 1X2 RESULT MODEL (XGBOOST) ---")
    results["result_xgb"] = train_xgb_result_model(
        train_df, val_df, test_df, model_dir)

    count_targets = [
        ("target_home_goals",   "home_goals_v1"),
        ("target_away_goals",   "away_goals_v1"),
        ("target_home_corners", "home_corners_v1"),
        ("target_away_corners", "away_corners_v1"),
    ]
    for target_col, model_name in count_targets:
        print(f"\n--- {model_name.upper()} ---")
        results[model_name] = train_count_model(
            train_df, val_df, test_df,
            target_col, model_name, model_dir,
        )

    print("\n--- BTTS MODEL ---")
    results["btts"] = train_btts_model(train_df, val_df, test_df, model_dir)

    print("\n--- OVER 2.5 MODEL ---")
    results["over25"] = train_over25_model(train_df, val_df, test_df, model_dir)

    print("\n--- OVER 9.5 CORNERS MODEL ---")
    results["over95_corners"] = train_over95_corners_model(
        train_df, val_df, test_df, model_dir
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        if not metrics:
            continue
        print(f"\n{name}:")
        for k, v in metrics.items():
            if k != "model_path":
                print(f"  {k}: {v}")

    upload_models_to_supabase(model_dir)

    return results


if __name__ == "__main__":
    train_all_models()
