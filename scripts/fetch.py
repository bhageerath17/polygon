"""Entry point: fetch SPX 1-min price, options snapshot, and VIX daily.

Incremental mode: if CSVs already exist (restored from cache), only the
data from `(last_date - 3 days)` onward is fetched and appended.
Full mode: fetches the full date range from config.toml (cold start).
"""
from __future__ import annotations

import json
import requests
from datetime import date, timedelta
from pathlib import Path

from app.config import settings
from app.fetchers.price import get_spx_1min
from app.fetchers.options import get_spx_options_snapshot, parse_options_snapshot
from app.fetchers.vix import get_vix_daily
from app.io import save_csv, append_csv, last_date_in_csv


def _incremental_start(csv_path: Path, full_start: str, overlap_days: int = 3) -> str:
    """Return the start date to fetch: last_date-overlap if CSV exists, else full_start."""
    last = last_date_in_csv(csv_path)
    if last:
        start = (date.fromisoformat(last) - timedelta(days=overlap_days)).isoformat()
        print(f"  Cache hit: last date={last}, fetching from {start}")
        return start
    print(f"  Cache miss: full fetch from {full_start}")
    return full_start


def main() -> None:
    data_dir = Path("data")

    # ── 1. SPX 1-min ──────────────────────────────────────────────────────────
    spx_path = data_dir / settings.price_csv
    start = _incremental_start(spx_path, settings.start_date)
    print(f"Fetching SPX 1-min ({start} → {settings.end_date})...")
    spx = get_spx_1min(start_date=start, end_date=settings.end_date)
    if not spx.empty:
        append_csv(spx, spx_path)
        print()

    # ── 2. VIX daily ──────────────────────────────────────────────────────────
    vix_path = data_dir / settings.vix_csv
    vix_start = _incremental_start(vix_path, settings.vix_start_date)
    print(f"Fetching VIX daily ({vix_start} → {settings.vix_end_date})...")
    vix = get_vix_daily(start_date=vix_start, end_date=settings.vix_end_date)
    if not vix.empty:
        append_csv(vix, vix_path)
        print()

    # ── 3. Options snapshot (always fresh — point-in-time) ────────────────────
    print("Fetching SPX options snapshot...")
    try:
        results, raw = get_spx_options_snapshot()
        opts = parse_options_snapshot(results)
        if not opts.empty:
            save_csv(opts, data_dir / settings.options_csv, index=False)
            print(f"Shape: {opts.shape}")
        else:
            print("No options data:", json.dumps(raw, indent=2)[:500])
    except requests.HTTPError as e:
        print(f"HTTP error: {e}\n{e.response.text[:500]}")


if __name__ == "__main__":
    main()
