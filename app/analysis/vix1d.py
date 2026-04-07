"""
VIX1D — Compute 1-day implied volatility from ATM 0DTE straddle prices.

For each trading day:
  1. Determine the ATM strike from SPX opening price.
  2. Fetch 1-min bars for the ATM 0DTE call and put (with local cache).
  3. Compute the straddle mid-price at the open (average of first few bars).
  4. Derive VIX1D:  vix1d = (straddle / SPX) × √252 × 100

This replaces the VIX / √252 approximation with an actual market-implied
1-day volatility, which captures term-structure effects, event premia,
and intraday supply/demand that the 30-day VIX misses.
"""
from __future__ import annotations

from datetime import date
from math import sqrt
from pathlib import Path

import pandas as pd

from app.analysis.reversals import build_0dte_ticker
from app.fetchers.options import get_option_1min


SQRT252 = sqrt(252)
CACHE_DIR = Path("data/options_cache")


def _fetch_with_cache(ticker: str, trade_date: date, cache_dir: Path) -> pd.DataFrame:
    """Fetch 1-min option bars, caching to disk."""
    safe_name  = ticker.replace(":", "_")
    cache_path = cache_dir / f"{safe_name}_{trade_date}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col="datetime", parse_dates=True)
        if not df.empty:
            df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        return df

    try:
        df = get_option_1min(ticker, str(trade_date))
        if not df.empty:
            df.to_csv(cache_path)
        return df
    except Exception as e:
        print(f"  Straddle fetch skipped for {ticker} ({type(e).__name__}: {e})")
        return pd.DataFrame()


def _straddle_open_price(
    call_bars: pd.DataFrame,
    put_bars: pd.DataFrame,
    n_bars: int = 5,
) -> float | None:
    """Compute the straddle mid-price from the first n RTH bars.

    Uses the average of the first n 1-min close prices for each leg
    to smooth out the opening tick noise.
    """
    if call_bars.empty or put_bars.empty:
        return None

    call_rth = call_bars.between_time("09:31", "09:40")
    put_rth  = put_bars.between_time("09:31", "09:40")

    call_slice = call_rth.iloc[:n_bars]
    put_slice  = put_rth.iloc[:n_bars]

    if len(call_slice) == 0 or len(put_slice) == 0:
        return None

    call_mid = call_slice["close"].mean()
    put_mid  = put_slice["close"].mean()
    return float(call_mid + put_mid)


def compute_vix1d_series(
    spx_1min: pd.DataFrame,
    strike_rounding: int = 25,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute daily VIX1D from ATM 0DTE straddle prices.

    Args:
        spx_1min: SPX 1-min OHLC with DatetimeIndex (ET-aware or UTC).
        strike_rounding: strike rounding for ATM (default 25).
        cache_dir: cache directory for option bar CSVs.

    Returns:
        DataFrame indexed by date with columns:
            straddle_price  — ATM straddle mid at open
            vix1d           — annualised 1-day IV (VIX-scale, e.g. 18.5)
            em_1d           — 1-day expected move in points (1σ)
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    spx = spx_1min.copy()
    spx.index = pd.to_datetime(spx.index, utc=True).tz_convert("America/New_York")
    trading_days = sorted(set(spx.index.date))

    rows: list[dict] = []
    for day in trading_days:
        day_bars = spx[spx.index.date == day]
        if len(day_bars) < 5:
            continue

        spx_open = float(day_bars.iloc[0]["open"])

        call_ticker = build_0dte_ticker(spx_open, day, "C", strike_rounding)
        put_ticker  = build_0dte_ticker(spx_open, day, "P", strike_rounding)

        call_bars = _fetch_with_cache(call_ticker, day, cache_dir)
        put_bars  = _fetch_with_cache(put_ticker, day, cache_dir)

        straddle = _straddle_open_price(call_bars, put_bars)
        if straddle is None or straddle <= 0:
            rows.append({"date": str(day), "straddle_price": None, "vix1d": None, "em_1d": None})
            continue

        # VIX1D = (straddle / SPX) × √252 × 100
        vix1d = (straddle / spx_open) * SQRT252 * 100
        # 1σ expected move in SPX points
        em_1d = straddle  # straddle ≈ 0.798 × σ × SPX, but raw straddle is a better EM

        rows.append({
            "date":            str(day),
            "straddle_price":  round(straddle, 2),
            "vix1d":           round(vix1d, 2),
            "em_1d":           round(em_1d, 2),
        })

        if rows and len(rows) % 50 == 0:
            print(f"  VIX1D computed for {len(rows)}/{len(trading_days)} days")

    print(f"  VIX1D computed for {len(rows)}/{len(trading_days)} days (done)")
    df = pd.DataFrame(rows).set_index("date")
    return df
