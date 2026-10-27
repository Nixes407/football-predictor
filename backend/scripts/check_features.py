"""Sanity-check the features CSV before the trainer (and the API) use it.

Run from ``backend/`` after ``features/builder.py``::

    python scripts/check_features.py
    python scripts/check_features.py --min-rows 10000

Exit code 0 means the file exists, parses, has at least ``--min-rows`` rows and
carries the trainer's 62 feature columns. The 46 all-zero team-season snapshot
columns were removed from the ML pipeline, so a legacy CSV that still carries
them is reported rather than silently trained on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# backend/ must be importable when this file is run as ``scripts/check_features.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.feature_columns import MATCH_HISTORY_COLUMNS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="data/processed/features.csv",
        help="features CSV written by features/builder.py",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="fail when fewer rows than this were built (default: 1)",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"[check] ERROR: {path} does not exist — builder.py did not run")
        return 1

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"[check] ERROR: could not read {path}: {exc}")
        return 1

    rows = len(df)
    print(f"[check] file        : {path}")
    print(f"[check] rows        : {rows:,}")
    print(f"[check] columns     : {len(df.columns)}")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print(f"[check] date range  : {dates.min()} -> {dates.max()}")
    if "league" in df.columns:
        print(f"[check] leagues     : {df['league'].nunique()} distinct")

    # The API replays this CSV's rolling windows at inference time; without
    # the raw match columns the shot-derived features cannot be rebuilt and
    # fall back to the team's last stored row (see features/inference.py).
    missing_history = [c for c in MATCH_HISTORY_COLUMNS if c not in df.columns]
    if missing_history:
        print(f"[check] WARNING: features.csv has no raw match history "
              f"({missing_history}); inference will fall back to stored "
              f"shot values until the next builder run")
    else:
        print("[check] match history: all raw columns present")

    # The snapshot columns are no longer model inputs. A CSV that still has
    # them means the builder ran from an old checkout: the trainer ignores them,
    # but the workflow log should say so explicitly.
    legacy_cols = [c for c in df.columns if "fbref" in c]
    if legacy_cols:
        print(f"[check] WARNING: {len(legacy_cols)} legacy team-season snapshot "
              "columns in features.csv — they are ignored by the trainer")
    else:
        print("[check] model inputs: no legacy snapshot columns present")

    if rows < args.min_rows:
        print(f"[check] ERROR: {rows} rows is below --min-rows {args.min_rows}")
        return 1

    print("[check] features.csv OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
