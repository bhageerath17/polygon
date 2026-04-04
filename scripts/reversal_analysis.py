"""Entry point: detect SPX reversals, fetch ATM call data, run squeeze analysis."""
import json
import pandas as pd
from app.config import settings
from app.analysis.reversals import run_reversal_analysis
from app.io import save_csv, clean_for_json


def main() -> None:
    print("Loading SPX 1-min data...")
    spx = pd.read_csv(settings.price_csv, index_col="datetime", parse_dates=True)

    print("Loading VIX daily data...")
    vix = pd.read_csv(settings.vix_csv, index_col="date", parse_dates=True)
    vix.index = vix.index.date

    print(f"Running reversal analysis (window={settings.reversal_window} mins)...")
    events_df, summary = run_reversal_analysis(
        spx, vix,
        window_mins=settings.reversal_window,
        squeeze_lookback=settings.squeeze_lookback,
        strike_rounding=settings.strike_rounding,
    )

    if not events_df.empty:
        save_csv(events_df, settings.reversal_csv)

    settings.reversal_json.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.reversal_json, "w") as f:
        json.dump(clean_for_json(summary), f, indent=2, default=str)
    print(f"Saved reversal JSON → {settings.reversal_json}")

    print(f"\n{'='*55}")
    print(f"  Total reversal events : {summary['total_events']}")
    if summary['total_events']:
        print(f"  Date range            : {summary['date_range']}")
        print(f"  Avg fall / reversal   : {summary['avg_fall_pts']} / {summary['avg_reversal_pts']} pts")
        print(f"  Fall categories       : {summary['fall_category_counts']}")
        print(f"  Reversal categories   : {summary['reversal_category_counts']}")
        print(f"  Squeeze counts        : {summary['squeeze_strength_counts']}")
        print(f"  % with squeeze        : {summary['pct_preceded_by_squeeze']}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
