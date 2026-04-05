"""Generic 1-minute bar fetcher for any Polygon ticker (stocks, indices)."""
from __future__ import annotations
import pandas as pd
from app.client import client
from app.config import settings


def get_1min_bars(
    ticker: str,
    start_date: str,
    end_date: str,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Fetch 1-minute OHLCV bars for any Polygon ticker over a date range."""
    url = (
        f"{settings.base_url}/v2/aggs/ticker/{ticker}"
        f"/range/1/minute/{start_date}/{end_date}"
    )
    results = client.get_paginated(
        url,
        {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000},
    )
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["datetime"] = (
        pd.to_datetime(df["t"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
    )
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap",
    })
    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume", "vwap"] if c in df.columns]
    return df[keep].set_index("datetime")
