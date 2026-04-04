from __future__ import annotations
import pandas as pd
from app.client import client
from app.config import settings


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
