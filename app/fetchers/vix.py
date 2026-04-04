from __future__ import annotations
import pandas as pd
from app.client import client
from app.config import settings


def get_vix_daily(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch VIX daily OHLC bars."""
    start = start_date or settings.vix_start_date
    end = end_date or settings.vix_end_date
    url = f"{settings.base_url}/v2/aggs/ticker/{settings.vix_ticker}/range/1/day/{start}/{end}"
    results = client.get_paginated(url, {"adjusted": "false", "sort": "asc", "limit": 5000})

    if not results:
        print("No VIX data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    keep = [c for c in ["date", "open", "high", "low", "close"] if c in df.columns]
    return df[keep].set_index("date")
