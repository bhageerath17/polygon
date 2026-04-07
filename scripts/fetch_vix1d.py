"""Fetch ATM 0DTE straddle prices and compute VIX1D for all trading days.

Reads SPX 1-min data, fetches call+put bars per day (cached), and outputs
data/vix1d.csv with columns: straddle_price, vix1d, em_1d.
"""
from __future__ import annotations

import pandas as pd

from app.config import settings
from app.analysis.vix1d import compute_vix1d_series
from app.io import save_csv


def main() -> None:
    print("Loading SPX 1-min data...")
    spx = pd.read_csv(settings.price_csv, index_col="datetime")
    spx.index = pd.to_datetime(spx.index)
    print(f"  {len(spx):,} bars, {len(set(spx.index.date)):,} trading days\n")

    print("Computing VIX1D from ATM 0DTE straddles...")
    vix1d_df = compute_vix1d_series(
        spx,
        strike_rounding=settings.strike_rounding,
    )

    valid = vix1d_df["vix1d"].notna().sum()
    total = len(vix1d_df)
    print(f"\n  VIX1D coverage: {valid}/{total} days ({valid/total*100:.0f}%)")

    if valid:
        save_csv(vix1d_df, settings.vix1d_csv)
        mean_vix1d = vix1d_df["vix1d"].dropna().mean()
        mean_em    = vix1d_df["em_1d"].dropna().mean()
        print(f"  Avg VIX1D: {mean_vix1d:.1f}  |  Avg EM (1σ): {mean_em:.1f} pts")
    else:
        print("  No straddle data available — VIX1D CSV not written")


if __name__ == "__main__":
    main()
