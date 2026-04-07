"""
Build reversal prediction model v2 (triple-barrier + calibration).
Loads SPX 1-min, SPY 1-min, VIX 1-min, VIX daily + reversal events.
Outputs model2_results.json to data/.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.model2 import run_model2
from app.config import settings

REVERSAL_JSON = Path("data/reversal_analysis.json")
MODEL2_JSON   = Path("data/model2_results.json")


def _load_csv(path: Path, index_col: str, parse_dates: bool = True) -> pd.DataFrame:
    if not path.exists():
        print(f"  WARNING: {path} not found — feature group disabled")
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
    if not REVERSAL_JSON.exists():
        print(f"ERROR: {REVERSAL_JSON} not found. Run 'make reversal' first.")
        sys.exit(1)

    with open(REVERSAL_JSON) as f:
        reversal_data = json.load(f)
    events = reversal_data.get("events", [])
    print(f"Loaded {len(events)} reversal events\n")

    print("Loading price/vol data...")
    spx_1min  = _load_csv(settings.price_csv,    index_col="datetime")
    vix_daily = _load_csv(settings.vix_csv,      index_col="date")
    spy_1min  = _load_csv(settings.spy_1min_csv, index_col="datetime")
    vix_1min  = _load_csv(settings.vix_1min_csv, index_col="datetime")

    vix1d_df = None
    if settings.vix1d_csv.exists():
        vix1d_df = _load_csv(settings.vix1d_csv, index_col="date", parse_dates=False)
        if vix1d_df.empty:
            vix1d_df = None
    print()

    results = run_model2(
        events,
        vix_daily=vix_daily  if not vix_daily.empty else None,
        spx_1min=spx_1min    if not spx_1min.empty  else None,
        spy_1min=spy_1min    if not spy_1min.empty   else None,
        vix_1min=vix_1min    if not vix_1min.empty   else None,
        vix1d_df=vix1d_df,
    )

    if "error" in results:
        print(f"Model error: {results['error']}")
        sys.exit(1)

    results = _clean(results)
    MODEL2_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL2_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  MODEL 2 — Triple-Barrier + Calibration")
    print(f"{sep}")
    print(f"  Label Method  : {results['label_method']}  (k={results['k_barrier']})")
    print(f"  Models        : {', '.join(results['models_used'])}")
    print(f"  Features      : {results['n_features']}")
    print(f"  Events        : {results['n_events']}  |  Positive: {results['n_significant_reversals']} ({results['base_rate']*100:.1f}%)")
    print(f"  Threshold     : {results['optimal_threshold']}")
    print(f"  OOF Accuracy  : {results['loocv_accuracy']*100:.1f}%")
    print(f"  Precision     : {results['precision']*100:.1f}%")
    print(f"  Recall        : {results['recall']*100:.1f}%")
    print(f"  Confusion     : {results['confusion_matrix']}")
    print(f"  Calibration   : {'Yes' if results['calibration_applied'] else 'No'}")
    print(f"  ToD Normalize : {'Yes' if results['tod_normalization_applied'] else 'No'}")
    print(f"  Rank Scaling  : {'Yes' if results['robust_scaling_applied'] else 'No'}")

    fold_dist = results.get("fold_precision_distribution", [])
    if fold_dist:
        print(f"\n  Fold Precision Distribution:")
        for i, p in enumerate(fold_dist):
            bar = "#" * max(1, int(p * 30))
            print(f"    Fold {i+1}: {p*100:5.1f}%  {bar}")
        print(f"    Mean:   {sum(fold_dist)/len(fold_dist)*100:.1f}%")

    print(f"\n  Feature Stability Score: {results['feature_stability_score']*100:.1f}%")

    print(f"\n  VIX Regime Performance:")
    for regime, perf in results.get("vix_regime_performance", {}).items():
        print(f"    {regime:6s}: prec={perf['precision']*100:5.1f}%  rec={perf['recall']*100:5.1f}%  n={perf['n']}")

    print(f"\n  Spread Regime Performance:")
    for regime, perf in results.get("spread_regime_performance", {}).items():
        print(f"    {regime:12s}: prec={perf['precision']*100:5.1f}%  rec={perf['recall']*100:5.1f}%  n={perf['n']}")

    print(f"\n  Top 12 features:")
    for feat in results["feature_importance"][:12]:
        bar = "#" * max(1, int(feat["importance"] / max(results["feature_importance"][0]["importance"], 1e-6) * 30))
        print(f"    {feat['feature']:28s} {bar} {feat['importance']:.4f}")

    print(f"\n  Squeeze -> avg prob reversal:")
    for k, v in results.get("squeeze_impact", {}).items():
        print(f"    {k:12s} -> {v*100:.1f}%")
    print(f"{sep}")
    print(f"\nSaved -> {MODEL2_JSON}")


if __name__ == "__main__":
    main()
