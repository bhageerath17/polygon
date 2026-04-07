"""
Build reversal prediction model v3 (holdout validation + feature selection).
Loads SPX 1-min, SPY 1-min, VIX 1-min, VIX daily + reversal events.
Outputs model3_results.json to data/.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.model3 import run_model3
from app.config import settings

REVERSAL_JSON = Path("data/reversal_analysis.json")
MODEL3_JSON   = Path("data/model3_results.json")


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

    results = run_model3(
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
    MODEL3_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL3_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  MODEL 3 — Holdout Validation + Feature Selection")
    print(f"{sep}")
    print(f"  Label Method  : {results['label_method']}  (k={results['k_barrier']})")
    print(f"  Models        : {', '.join(results['models_used'])}")
    print(f"  Features      : {results['n_features_original']} → {results['n_features_selected']} (corr>{results['corr_threshold']} dropped)")
    print(f"  Events        : {results['n_events']}  |  Train: {results['n_train']}  |  Holdout: {results['n_holdout']}")
    print(f"  Threshold     : {results['optimal_threshold']} (tuned on train set)")
    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  HOLDOUT METRICS (model never saw this data) │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Precision : {results['holdout_precision']*100:5.1f}%                          │")
    print(f"  │  Recall    : {results['holdout_recall']*100:5.1f}%                          │")
    print(f"  │  Accuracy  : {results['holdout_accuracy']*100:5.1f}%                          │")
    print(f"  │  Confusion : {results['holdout_confusion_matrix']}  │")
    print(f"  └─────────────────────────────────────────────┘")
    print()
    print(f"  Train metrics (for overfitting check):")
    print(f"    Precision : {results['train_precision']*100:.1f}%")
    print(f"    Recall    : {results['train_recall']*100:.1f}%")
    print(f"    Confusion : {results['train_confusion_matrix']}")
    overfit_gap = results['train_precision'] - results['holdout_precision']
    if overfit_gap > 0.15:
        print(f"    ⚠ Overfitting detected: train prec {overfit_gap*100:.0f}pp higher than holdout")
    else:
        print(f"    ✓ Overfit gap: {overfit_gap*100:+.1f}pp (acceptable)")

    print(f"\n  Holdout dates : {results['holdout_dates']}")
    print(f"  Train dates   : {results['train_dates']}")

    print(f"\n  VIX Regime Performance (holdout):")
    for regime, perf in results.get("vix_regime_performance", {}).items():
        print(f"    {regime:6s}: prec={perf['precision']*100:5.1f}%  rec={perf['recall']*100:5.1f}%  n={perf['n']}")

    print(f"\n  Top 15 features:")
    for feat in results["feature_importance"][:15]:
        bar = "#" * max(1, int(feat["importance"] / max(results["feature_importance"][0]["importance"], 1e-6) * 30))
        print(f"    {feat['feature']:28s} {bar} {feat['importance']:.4f}")

    print(f"{sep}")
    print(f"\nSaved -> {MODEL3_JSON}")


if __name__ == "__main__":
    main()
