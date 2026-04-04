"""Entry point: fetch SPX price data and options snapshot, write to data/."""
import json
import requests
from app.config import settings
from app.fetchers.price import get_spx_1min
from app.fetchers.options import get_spx_options_snapshot, parse_options_snapshot
from app.io import save_csv


def main() -> None:
    # 1. SPX 1-minute price bars
    print(f"Fetching SPX 1-min data ({settings.start_date} -> {settings.end_date})...")
    spx_1min = get_spx_1min()
    if not spx_1min.empty:
        save_csv(spx_1min, settings.price_csv)
        print(f"Shape: {spx_1min.shape}")
        print(spx_1min.head(5).to_string())
    else:
        print("No SPX 1-min data fetched.")

    print("\n" + "=" * 60 + "\n")

    # 2. Options snapshot
    print("Fetching SPX options snapshot...")
    try:
        results, raw = get_spx_options_snapshot()
        opts_df = parse_options_snapshot(results)
        if not opts_df.empty:
            save_csv(opts_df, settings.options_csv, index=False)
            print(f"Shape: {opts_df.shape}")
            print(opts_df.head(5).to_string(index=False))
            print("\nExpiration dates:", opts_df["expiration_date"].dt.date.unique())
        else:
            print("No options data. Raw response:")
            print(json.dumps(raw, indent=2)[:2000])
    except requests.HTTPError as e:
        print(f"HTTP error: {e}\n{e.response.text[:500]}")


if __name__ == "__main__":
    main()
