from __future__ import annotations
import pandas as pd
from app.client import client
from app.config import settings


def get_option_1min(option_ticker: str, date: str) -> pd.DataFrame:
    """Fetch 1-minute OHLCV bars for a single options contract on a given date.

    Args:
        option_ticker: e.g. "O:SPX260102C06875000"
        date: "YYYY-MM-DD"
    """
    url = (
        f"{settings.base_url}/v2/aggs/ticker/{option_ticker}"
        f"/range/1/minute/{date}/{date}"
    )
    results = client.get_paginated(url, {"adjusted": "true", "sort": "asc", "limit": 50000})
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].set_index("datetime")


def get_spx_options_snapshot(limit: int | None = None) -> tuple[list[dict], dict]:
    """Fetch the latest options snapshot for SPX."""
    url = f"{settings.base_url}/v3/snapshot/options/{settings.ticker}"
    data = client.get(url, {"limit": limit or settings.options_limit})
    results = data.get("results", [])
    print(f"  Fetched {len(results)} options contracts in snapshot")
    return results, data


def parse_options_snapshot(results: list[dict]) -> pd.DataFrame:
    """Flatten options snapshot results into a DataFrame."""
    rows = []
    for item in results:
        details = item.get("details", {})
        day = item.get("day", {})
        greeks = item.get("greeks", {})
        quote = item.get("last_quote", {})
        trade = item.get("last_trade", {})
        rows.append({
            "ticker":               item.get("ticker"),
            "contract_type":        details.get("contract_type"),
            "expiration_date":      details.get("expiration_date"),
            "strike_price":         details.get("strike_price"),
            "shares_per_contract":  details.get("shares_per_contract"),
            "open_interest":        item.get("open_interest"),
            "implied_volatility":   item.get("implied_volatility"),
            "delta":  greeks.get("delta"),
            "gamma":  greeks.get("gamma"),
            "theta":  greeks.get("theta"),
            "vega":   greeks.get("vega"),
            "day_open":   day.get("open"),
            "day_high":   day.get("high"),
            "day_low":    day.get("low"),
            "day_close":  day.get("close"),
            "day_volume": day.get("volume"),
            "day_vwap":   day.get("vwap"),
            "bid":              quote.get("bid"),
            "ask":              quote.get("ask"),
            "last_trade_price": trade.get("price"),
            "last_trade_size":  trade.get("size"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["expiration_date"] = pd.to_datetime(df["expiration_date"])
        df = df.sort_values(["expiration_date", "contract_type", "strike_price"])
    return df
