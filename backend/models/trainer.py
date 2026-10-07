"""
Model training pipeline for the Football Predictor.

Trains all prediction models from the features CSV
produced by builder.py. Uses strict chronological
splitting. Never tunes on the test set.

Run from backend/ directory:
    python models/trainer.py
"""

from __future__ import annotations

import warnings
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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
]

# The 1X2 target column produced by builder.py.
TARGET_COL = "target_result"


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


def train_all_models(
    features_path: str = "data/processed/features.csv",
    model_dir: str = "models",
) -> dict:
    """
    Full training pipeline. Trains all models in order:
      1. 1X2 result classifier
      2. Home goals Poisson
      3. Away goals Poisson
      4. Home corners Poisson
      5. Away corners Poisson

    Returns a summary dict of all model results.
    Validation controls all choices.
    Test is evaluated exactly once per model.
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

    train_df, val_df, test_df = chronological_split(df)

    results = {}

    print("\n--- 1X2 RESULT MODEL ---")
    results["result"] = train_result_model(
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

    return results


if __name__ == "__main__":
    train_all_models()
