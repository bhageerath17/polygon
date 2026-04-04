import os
import requests
import pandas as pd
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["POLYGON_API_KEY"]
BASE_URL = "https://api.polygon.io"

def get_spx_1min(start_date="2026-01-02", end_date="2026-01-31"):
    """Fetch SPX 1-minute OHLCV bars for a date range."""
    url = f"{BASE_URL}/v2/aggs/ticker/I:SPX/range/1/minute/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }
    all_results = []
    while url:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        all_results.extend(results)
        print(f"  Fetched {len(results)} bars (total so far: {len(all_results)})")
        # Handle pagination
        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": API_KEY}  # next_url already has other params
        else:
            break
        time.sleep(0.2)  # be polite to the API

    if not all_results:
        print("No SPX 1-min data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "num_trades"})
    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume", "vwap", "num_trades"] if c in df.columns]
    df = df[keep].set_index("datetime")
    return df


def get_spx_options_snapshot(limit=250):
    """Fetch the latest options snapshot for SPX (most recent available data)."""
    url = f"{BASE_URL}/v3/snapshot/options/I:SPX"
    params = {
        "limit": limit,
        "apiKey": API_KEY,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    print(f"  Fetched {len(results)} options contracts in snapshot")
    return results, data


def parse_options_snapshot(results):
    """Flatten options snapshot results into a DataFrame."""
    rows = []
    for item in results:
        details = item.get("details", {})
        day = item.get("day", {})
        greeks = item.get("greeks", {})
        quote = item.get("last_quote", {})
        trade = item.get("last_trade", {})
        rows.append({
            "ticker": item.get("ticker"),
            "contract_type": details.get("contract_type"),
            "expiration_date": details.get("expiration_date"),
            "strike_price": details.get("strike_price"),
            "shares_per_contract": details.get("shares_per_contract"),
            "open_interest": item.get("open_interest"),
            "implied_volatility": item.get("implied_volatility"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
            "day_open": day.get("open"),
            "day_high": day.get("high"),
            "day_low": day.get("low"),
            "day_close": day.get("close"),
            "day_volume": day.get("volume"),
            "day_vwap": day.get("vwap"),
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "last_trade_price": trade.get("price"),
            "last_trade_size": trade.get("size"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["expiration_date"] = pd.to_datetime(df["expiration_date"])
        df = df.sort_values(["expiration_date", "contract_type", "strike_price"])
    return df


if __name__ == "__main__":
    # --- 1. SPX 1-minute price data for January 2026 ---
    print("Fetching SPX 1-minute data for January 2026...")
    spx_1min = get_spx_1min("2026-01-02", "2026-01-31")
    if not spx_1min.empty:
        out_path = "spx_1min_jan2026.csv"
        spx_1min.to_csv(out_path)
        print(f"\nSPX 1-min data saved to {out_path}")
        print(f"Shape: {spx_1min.shape}")
        print(spx_1min.head(10).to_string())
    else:
        print("No SPX 1-min data fetched.")

    print("\n" + "="*60 + "\n")

    # --- 2. SPX Options snapshot (latest available) ---
    print("Fetching SPX options snapshot...")
    try:
        results, raw = get_spx_options_snapshot(limit=250)
        opts_df = parse_options_snapshot(results)
        if not opts_df.empty:
            out_path = "spx_options_snapshot.csv"
            opts_df.to_csv(out_path, index=False)
            print(f"\nSPX options snapshot saved to {out_path}")
            print(f"Shape: {opts_df.shape}")
            print("\nSample (first 10 rows):")
            print(opts_df.head(10).to_string(index=False))
            print("\nExpiration dates available:")
            print(opts_df["expiration_date"].dt.date.unique())
        else:
            print("No options snapshot data. Raw response:")
            print(json.dumps(raw, indent=2)[:2000])
    except requests.HTTPError as e:
        print(f"HTTP error fetching options: {e}")
        print(e.response.text[:1000])
