"""Entry point: fetch SPX 1-min price, options snapshot, and VIX daily."""
import json
import requests
from app.config import settings
from app.fetchers.price import get_spx_1min
from app.fetchers.options import get_spx_options_snapshot, parse_options_snapshot
from app.fetchers.vix import get_vix_daily
from app.io import save_csv


def main() -> None:
    # 1. SPX 1-minute price bars
    print(f"Fetching SPX 1-min data ({settings.start_date} → {settings.end_date})...")
    spx = get_spx_1min()
    if not spx.empty:
        save_csv(spx, settings.price_csv)
        print(f"Shape: {spx.shape}\n")

    # 2. VIX daily
    print(f"Fetching VIX daily ({settings.vix_start_date} → {settings.vix_end_date})...")
    vix = get_vix_daily()
    if not vix.empty:
        save_csv(vix, settings.vix_csv)
        print(f"Shape: {vix.shape}\n")

    # 3. Options snapshot
    print("Fetching SPX options snapshot...")
    try:
        results, raw = get_spx_options_snapshot()
        opts = parse_options_snapshot(results)
        if not opts.empty:
            save_csv(opts, settings.options_csv, index=False)
            print(f"Shape: {opts.shape}")
        else:
            print("No options data:", json.dumps(raw, indent=2)[:500])
    except requests.HTTPError as e:
        print(f"HTTP error: {e}\n{e.response.text[:500]}")


if __name__ == "__main__":
    main()
