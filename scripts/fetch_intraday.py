"""Fetch VIX 1-min and SPY 1-min bars from Polygon — incremental.

Incremental mode: if CSVs already exist (restored from cache), only
data from `(last_date - 3 days)` onward is fetched and appended.
Full mode: fetches the full date range from config.toml (cold start).

Run: uv run python scripts/fetch_intraday.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.fetchers.intraday import get_1min_bars
from app.io import append_csv, last_date_in_csv


def _incremental_start(csv_path: Path, full_start: str, overlap_days: int = 3) -> str:
    last = last_date_in_csv(csv_path)
    if last:
        start = (date.fromisoformat(last) - timedelta(days=overlap_days)).isoformat()
        print(f"  Cache hit: last date={last}, fetching from {start}")
        return start
    print(f"  Cache miss: full fetch from {full_start}")
    return full_start


def main():
    data_dir = Path("data")

    # ── VIX 1-min ─────────────────────────────────────────────────────────────
    vix_path = data_dir / settings.vix_1min_csv
    vix_start = _incremental_start(vix_path, settings.vix_1min_start_date)
    print(f"Fetching VIX 1-min ({vix_start} → {settings.vix_1min_end_date})…")
    try:
        vix_1min = get_1min_bars(
            settings.vix_1min_ticker,
            vix_start,
            settings.vix_1min_end_date,
            adjusted=False,
        )
        if not vix_1min.empty:
            append_csv(vix_1min, vix_path)
            print()
        else:
            print("  No VIX 1-min data returned (plan may not include indices at 1-min)\n")
    except Exception as e:
        print(f"  VIX 1-min fetch failed: {e}\n")

    # ── SPY 1-min ─────────────────────────────────────────────────────────────
    spy_path = data_dir / settings.spy_1min_csv
    spy_start = _incremental_start(spy_path, settings.spy_start_date)
    print(f"Fetching SPY 1-min ({spy_start} → {settings.spy_end_date})…")
    try:
        spy_1min = get_1min_bars(
            settings.spy_ticker,
            spy_start,
            settings.spy_end_date,
            adjusted=True,
        )
        if not spy_1min.empty:
            append_csv(spy_1min, spy_path)
            print()
        else:
            print("  No SPY 1-min data returned\n")
    except Exception as e:
        print(f"  SPY 1-min fetch failed: {e}\n")


if __name__ == "__main__":
    main()
