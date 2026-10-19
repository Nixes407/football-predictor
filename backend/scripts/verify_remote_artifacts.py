"""Verify that every artifact reached the Supabase Storage "models" bucket.

Run from ``backend/`` after ``models/trainer.py --strict-upload``::

    python scripts/verify_remote_artifacts.py

The trainer already fails on a failed upload; this checks the bucket *contents*
(object names, sizes, last-modified timestamps) so the workflow can refuse to
tell Render to reload until all 9 models, both label encoders and features.csv
are actually stored.

Exits 0 when every object is present, 1 otherwise (with the missing names).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/verify_remote_artifacts.py` from backend/.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import model_artifacts as artifacts  # noqa: E402


def _list_bucket() -> list[dict]:
    from supabase import create_client

    url, key = artifacts.supabase_credentials()
    if not url or not key:
        raise RuntimeError(artifacts.missing_credentials_message())

    client = create_client(url, key)
    # Supabase Storage returns at most 100 entries per page; the bucket holds
    # 13 objects, so a single call is enough. `options` is optional.
    return client.storage.from_(artifacts.MODEL_BUCKET).list()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-optional", action="store_true",
                        help="treat objects the API downloads best-effort as "
                             "optional (default: require the full set)")
    args = parser.parse_args(argv)

    try:
        entries = _list_bucket()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"[verify] ERROR: could not list bucket "
              f"{artifacts.MODEL_BUCKET!r}: {exc}")
        return 1

    remote = {entry.get("name"): entry for entry in entries if entry.get("name")}

    required = list(artifacts.UPLOAD_OBJECTS)
    if args.allow_missing_optional:
        required = [
            name for name in required
            if name not in artifacts.OPTIONAL_DOWNLOAD_OBJECTS
        ]

    missing = []
    print(f"[verify] bucket: {artifacts.MODEL_BUCKET} "
          f"({len(remote)} object(s) visible)")
    for object_name in required:
        entry = remote.get(object_name)
        if entry is None:
            missing.append(object_name)
            print(f"[verify] MISSING  {object_name}")
            continue
        size = (entry.get("metadata") or {}).get("size")
        size_text = f"{int(size) / 1024:.0f} KiB" if size else "size unknown"
        print(f"[verify] ok       {object_name:<45} {size_text:>12}  "
              f"{entry.get('updated_at', '')}")

    if missing:
        print("[verify] ERROR: missing from Supabase Storage: "
              + ", ".join(missing))
        return 1

    print(f"[verify] all {len(required)} object(s) present — safe to reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
