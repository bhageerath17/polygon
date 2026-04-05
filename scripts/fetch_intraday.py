"""
Fetch VIX 1-min and SPY 1-min bars from Polygon.
SPY is used as a volume proxy for SPX (index has no traded volume).
VIX 1-min gives us intraday volatility regime data.

Run: uv run python scripts/fetch_intraday.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.fetchers.intraday import get_1min_bars
from app.io import save_csv


def main():
    # ── VIX 1-min ─────────────────────────────────────────────────────────────
    print(f"Fetching VIX 1-min ({settings.vix_1min_start_date} → {settings.vix_1min_end_date})…")
    try:
        vix_1min = get_1min_bars(
            settings.vix_1min_ticker,
            settings.vix_1min_start_date,
            settings.vix_1min_end_date,
            adjusted=False,
        )
        if not vix_1min.empty:
            save_csv(vix_1min, settings.vix_1min_csv)
            print(f"  VIX 1-min: {vix_1min.shape[0]:,} bars\n")
        else:
            print("  No VIX 1-min data returned (plan may not include indices at 1-min)\n")
    except Exception as e:
        print(f"  VIX 1-min fetch failed: {e}\n")

    # ── SPY 1-min ─────────────────────────────────────────────────────────────
    print(f"Fetching SPY 1-min ({settings.spy_start_date} → {settings.spy_end_date})…")
    try:
        spy_1min = get_1min_bars(
            settings.spy_ticker,
            settings.spy_start_date,
            settings.spy_end_date,
            adjusted=True,
        )
        if not spy_1min.empty:
            save_csv(spy_1min, settings.spy_1min_csv)
            print(f"  SPY 1-min: {spy_1min.shape[0]:,} bars\n")
        else:
            print("  No SPY 1-min data returned\n")
    except Exception as e:
        print(f"  SPY 1-min fetch failed: {e}\n")


if __name__ == "__main__":
    main()
