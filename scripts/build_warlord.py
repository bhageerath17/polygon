"""
Build Warlord v2 pin-prediction analysis.
LOCAL ONLY — this file and the dashboard are gitignored.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.warlord import run_warlord_analysis
from app.config import settings

WARLORD_JSON = Path("data/warlord_results.json")


def _load_csv(path: Path, index_col: str, parse_dates: bool = True) -> pd.DataFrame:
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    print(f"  Loaded {path.name}: {len(df):,} rows")
    return df


def _clean(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    return obj


def main():
    sep = "=" * 68
    print(f"\n{sep}")
    print("  WARLORD v2 — EOD Pin Prediction (Stacking + Heuristics)")
    print(f"{sep}\n")

    print("Loading data...")
    spx_1min = _load_csv(settings.price_csv, index_col="datetime")
    vix_daily = _load_csv(settings.vix_csv, index_col="date")
    spy_1min = _load_csv(settings.spy_1min_csv, index_col="datetime")
    vix_1min = _load_csv(settings.vix_1min_csv, index_col="datetime")

    vix1d_df = None
    if settings.vix1d_csv.exists():
        vix1d_df = _load_csv(settings.vix1d_csv, index_col="date", parse_dates=False)
        if vix1d_df.empty:
            vix1d_df = None
    print()

    results = run_warlord_analysis(
        spx_1min=spx_1min if not spx_1min.empty else None,
        vix_daily=vix_daily if not vix_daily.empty else None,
        spy_1min=spy_1min if not spy_1min.empty else None,
        vix_1min=vix_1min if not vix_1min.empty else None,
        vix1d_df=vix1d_df,
    )

    if "error" in results:
        print(f"ERROR: {results['error']}")
        sys.exit(1)

    results = _clean(results)
    WARLORD_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(WARLORD_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{sep}")
    print(f"  WARLORD v2 RESULTS")
    print(f"{sep}")
    print(f"  Time       : {results['prediction_time']}")
    print(f"  Zones      : {results['zone_width']}pt bins, {results['n_candidate_zones']} candidates")
    print(f"  Models     : {', '.join(results['models_used'])}")
    print(f"  Features   : {results['n_features_original']} -> {results['n_features_selected']} selected")
    print(f"  Days       : {results['n_total_days']} (Train:{results['n_train']} Holdout:{results['n_holdout']})")
    print()
    print(f"  HOLDOUT METRICS")
    print(f"  ────────────────────────────────────────")
    print(f"  Top-1 Accuracy : {results['holdout_top1_accuracy']*100:5.1f}%")
    print(f"  Top-3 Hit Rate : {results['holdout_top3_accuracy']*100:5.1f}%")
    print(f"  Avg Distance   : {results['holdout_avg_distance']:5.1f} pts")
    print(f"  Median Distance: {results['holdout_median_distance']:5.1f} pts")
    print(f"  Train Top-1    : {results['train_top1_accuracy']*100:.1f}% (overfit gap: {(results['train_top1_accuracy']-results['holdout_top1_accuracy'])*100:.0f}pp)")
    print()
    print(f"  WARLORD P&L")
    print(f"  ────────────────────────────────────────")
    print(f"  Total   : ${results['warlord_total_pnl']:,.0f}")
    print(f"  Avg/Day : ${results['warlord_avg_pnl']:,.1f}")
    print(f"  Win Rate: {results['warlord_win_rate']*100:.1f}%")

    # Multi-timepoint
    tp = results.get("timepoint_comparison", {})
    if tp:
        print(f"\n  MULTI-TIMEPOINT (closer to close = better)")
        print(f"  ────────────────────────────────────────")
        for label in sorted(tp.keys()):
            t = tp[label]
            print(f"    {label:12s}  Top-1={t['top1_accuracy']*100:5.1f}%  Top-3={t['top3_accuracy']*100:5.1f}%  "
                  f"Avg={t['avg_distance']:5.1f}pts  P&L=${t['avg_pnl']:+.1f}/day")

    # VIX regime
    vrp = results.get("vix_regime_results", {})
    if vrp:
        print(f"\n  VIX REGIME PERFORMANCE")
        print(f"  ────────────────────────────────────────")
        for regime, perf in vrp.items():
            print(f"    {regime:18s}  n={perf['n']:3d}  Top-1={perf['top1_accuracy']*100:5.1f}%  "
                  f"Top-3={perf['top3_accuracy']*100:5.1f}%  P&L=${perf['avg_pnl']:+.1f}/day")

    # Heuristic buckets
    hb = results.get("heuristic_buckets", {})
    if hb:
        print(f"\n  HEURISTIC SCORE PERFORMANCE")
        print(f"  ────────────────────────────────────────")
        for bucket, perf in hb.items():
            print(f"    {bucket:18s}  n={perf['n']:3d}  Top-1={perf['top1_accuracy']*100:5.1f}%  "
                  f"Top-3={perf['top3_accuracy']*100:5.1f}%  P&L=${perf['avg_pnl']:+.1f}/day")

    # Top features
    print(f"\n  TOP 10 FEATURES")
    print(f"  ────────────────────────────────────────")
    for feat in results["feature_importance"][:10]:
        bar = "#" * max(1, int(feat["importance"] / max(results["feature_importance"][0]["importance"], 1e-6) * 30))
        print(f"    {feat['feature']:28s} {bar} {feat['importance']:.4f}")

    # Execution rules
    print(f"\n  EXECUTION HEURISTICS")
    print(f"  ────────────────────────────────────────")
    for h in results.get("heuristics", []):
        print(f"    {h['rule']:35s} -> {h['action']}")

    print(f"\n{sep}")
    print(f"  Saved -> {WARLORD_JSON}")
    print(f"  Dashboard -> open warlord_dashboard.html")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
