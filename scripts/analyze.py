"""Entry point: run patches analysis and write CSV + JSON to data/."""
import json
import pandas as pd
from app.config import settings
from app.analysis.patches import run_patches_analysis
from app.io import save_csv


def main() -> None:
    print("Loading SPX 1-min data...")
    spx = pd.read_csv(settings.price_csv, index_col="datetime", parse_dates=True)

    print("Loading VIX daily data...")
    vix = pd.read_csv(settings.vix_csv, index_col="date", parse_dates=True)
    vix.index = vix.index.date  # ensure date objects for lookup

    print(f"Running patches analysis (lookback={settings.lookback_mins} mins)...")
    daily_df, summary = run_patches_analysis(spx, vix, settings.lookback_mins)

    # Save CSV
    save_csv(daily_df, settings.analysis_csv)

    # Save JSON — sanitise NaN/Inf to null so output is valid JSON
    def _clean(obj):
        if isinstance(obj, float) and (obj != obj or obj in (float('inf'), float('-inf'))):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    settings.analysis_json.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.analysis_json, "w") as f:
        json.dump(_clean(summary), f, indent=2, default=str)
    print(f"Saved analysis JSON → {settings.analysis_json}")

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Trading days analysed : {summary['trading_days']}")
    print(f"  Date range            : {summary['date_range']}")
    print(f"  Breach above 1σ       : {summary['breach_above_1sig_%']}%")
    print(f"  Breach below 1σ       : {summary['breach_below_1sig_%']}%")
    print(f"  Any 1σ breach         : {summary['any_breach_1sig_%']}%")
    print(f"  Close inside bands    : {summary['close_zone_dist'].get('inside', 0)} days")
    print(f"  VIX avg / median      : {summary['vix_stats']['mean']} / {summary['vix_stats']['median']}")
    print(f"  Gap ups / Gap downs   : {summary['gap_stats']['gap_up_count']} / {summary['gap_stats']['gap_dn_count']}")
    print(f"{'='*55}\n")
    print("VIX Quartile Breakdown:")
    for q in summary["vix_quartile_breakdown"]:
        print(f"  {q['quartile']:12s}  VIX {q['vix_range']:12s}  "
              f"breach={q['any_breach_%']:4.1f}%  "
              f"inside={q['close_inside_%']:4.1f}%  "
              f"gap↑={q['gap_up_%']:4.1f}%  gap↓={q['gap_dn_%']:4.1f}%")


if __name__ == "__main__":
    main()
