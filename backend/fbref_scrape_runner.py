"""
Background runner for the StatsBomb -> fbref_team_stats aggregator on Render.

The endpoint and its polling contract are unchanged (``/predict/admin/run-fbref-scrape``
and ``/predict/admin/fbref-scrape-status``), but the work itself is now
``data/statsbomb_aggregator.py``: it aggregates the match-level rows already
collected in ``statsbomb_match_stats`` into ``fbref_team_stats`` — no Chrome,
no CAPTCHA, no scraping.

Why the old runner pointed at ``data/fbref_collector.py`` is history:
- GitHub Actions runners sit on AWS us-east IPs that FBref challenges with an
  unsolvable Cloudflare CAPTCHA.
- Render's free tier cannot install headless Google Chrome, so the direct FBref
  scrape could not run there either.
``data/fbref_collector.py`` now prints a deprecation notice and exits 0.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

BACKEND_DIR = Path(__file__).resolve().parent
COLLECTOR_SCRIPT = BACKEND_DIR / "data" / "statsbomb_aggregator.py"

_lock = threading.Lock()
_state: dict[str, object] = {
    "status": "idle",        # "idle" | "running" | "complete" | "failed"
    "last_run_at": None,     # ISO-8601 UTC timestamp string or None
    "last_result": None,     # "statsbomb_aggregator completed OK" | error string | None
}

RESULT_RUNNING = "statsbomb_aggregator running"
RESULT_COMPLETE = "statsbomb_aggregator completed OK"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_failure_message(lines: list[str], returncode: int) -> str:
    """Pick the most informative error line from captured aggregator output."""
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "ERROR" in stripped or "Exception" in stripped or "Error:" in stripped:
            return stripped[:500]
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return f"statsbomb_aggregator exited with code {returncode}"


def _run_collector_worker(
    on_status_change: Callable[[str, str | None], None] | None,
    on_complete: Callable[[], None] | None,
) -> None:
    """Execute ONLY ``data/statsbomb_aggregator.py`` in a child process and record outcome."""
    tail: deque[str] = deque(maxlen=120)
    cmd = [sys.executable, str(COLLECTOR_SCRIPT)]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BACKEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                tail.append(line)
                print(line, end="", flush=True)
        returncode = proc.wait()
        output_lines = list(tail)

        finished_at = _utc_now_iso()
        if returncode == 0:
            # The aggregator exits 0 on partial data by design: any rows it
            # managed to upsert are usable immediately.
            if on_complete is not None:
                try:
                    on_complete()
                except Exception as cb_exc:  # noqa: BLE001
                    print(f"[fbref-runner] WARNING: post-aggregation refresh hook failed: {cb_exc}")
            with _lock:
                _state["status"] = "complete"
                _state["last_run_at"] = finished_at
                _state["last_result"] = RESULT_COMPLETE
            if on_status_change is not None:
                on_status_change("fbref_scrape_complete", None)
        else:
            err_msg = _extract_failure_message(output_lines, returncode)
            with _lock:
                _state["status"] = "failed"
                _state["last_run_at"] = finished_at
                _state["last_result"] = err_msg
            if on_status_change is not None:
                on_status_change("fbref_scrape_failed", err_msg)
    except Exception as exc:  # noqa: BLE001
        err_msg = f"{type(exc).__name__}: {exc}"[:500]
        print(f"[fbref-runner] ERROR launching statsbomb_aggregator.py: {err_msg}", flush=True)
        with _lock:
            _state["status"] = "failed"
            _state["last_run_at"] = _utc_now_iso()
            _state["last_result"] = err_msg
        if on_status_change is not None:
            on_status_change("fbref_scrape_failed", err_msg)


def start_fbref_scrape(
    on_status_change: Callable[[str, str | None], None] | None = None,
    on_complete: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Start ``data/statsbomb_aggregator.py`` asynchronously if not already running."""
    with _lock:
        if _state["status"] == "running":
            return {"status": "started"}
        _state["status"] = "running"
        _state["last_run_at"] = _utc_now_iso()
        _state["last_result"] = RESULT_RUNNING

    if on_status_change is not None:
        on_status_change("fbref_scrape_running", None)

    worker = threading.Thread(
        target=_run_collector_worker,
        args=(on_status_change, on_complete),
        name="fbref-collector-runner",
        daemon=True,
    )
    worker.start()
    return {"status": "started"}


def get_fbref_scrape_status(
    count_rows_fn: Callable[[], int] | None = None,
) -> dict[str, object]:
    """Return current aggregation status and live ``fbref_team_stats`` DB row count."""
    with _lock:
        status = str(_state["status"])
        last_run_at = _state["last_run_at"]
        last_result = _state["last_result"]

    rows = 0
    if count_rows_fn is not None:
        try:
            rows = int(count_rows_fn())
        except Exception as exc:  # noqa: BLE001
            print(f"[fbref-runner] WARNING: could not count fbref_team_stats rows: {exc}")
            rows = 0

    return {
        "status": status,
        "last_run_at": last_run_at,
        "last_result": last_result,
        "fbref_team_stats_rows": rows,
    }


def reset_runner_state() -> None:
    """Reset module state (used by unit tests)."""
    with _lock:
        _state["status"] = "idle"
        _state["last_run_at"] = None
        _state["last_result"] = None
