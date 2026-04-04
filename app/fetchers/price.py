from __future__ import annotations
import pandas as pd
from app.client import client
from app.config import settings


def get_spx_1min(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch SPX 1-minute OHLC bars for a date range."""
    start = start_date or settings.start_date
    end = end_date or settings.end_date
    url = (
        f"{settings.base_url}/v2/aggs/ticker/{settings.ticker}"
        f"/range/1/minute/{start}/{end}"
    )
    results = client.get_paginated(url, {"adjusted": "true", "sort": "asc", "limit": 50000})

    if not results:
        print("No SPX 1-min data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["datetime"] = (
        pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    )
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap", "n": "num_trades",
    })
    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume", "vwap", "num_trades"] if c in df.columns]
    return df[keep].set_index("datetime")
