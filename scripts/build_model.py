"""
Build reversal prediction model from cached reversal_analysis.json.
Outputs model_results.json to data/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.model import run_model

REVERSAL_JSON = Path("data/reversal_analysis.json")
MODEL_JSON    = Path("data/model_results.json")


def main():
    if not REVERSAL_JSON.exists():
        print(f"ERROR: {REVERSAL_JSON} not found. Run 'make reversal' first.")
        sys.exit(1)

    with open(REVERSAL_JSON) as f:
        reversal_data = json.load(f)

    events = reversal_data.get("events", [])
    print(f"Loaded {len(events)} reversal events")

    results = run_model(events)

    if "error" in results:
        print(f"Model error: {results['error']}")
        sys.exit(1)

    # Clean NaN/inf for JSON
    def _clean(obj):
        if isinstance(obj, float):
            import math
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        return obj

    results = _clean(results)

    MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Model results saved → {MODEL_JSON}")
    print(f"\n{'='*55}")
    print(f"  Events          : {results['n_events']}")
    print(f"  Significant rev : {results['n_significant_reversals']} ({results['base_rate']*100:.1f}%)")
    print(f"  LOOCV Accuracy  : {results['loocv_accuracy']*100:.1f}%")
    print(f"  Precision       : {results['precision']*100:.1f}%")
    print(f"  Recall          : {results['recall']*100:.1f}%")
    print(f"  Confusion matrix: {results['confusion_matrix']}")
    print(f"\n  Top features by |coefficient|:")
    for f in results["feature_importance"][:5]:
        print(f"    {f['feature']:25s} {f['coefficient']:+.4f}")
    print(f"\n  Squeeze → avg prob(strong reversal):")
    for k, v in results["squeeze_impact"].items():
        print(f"    {k:10s} → {v*100:.1f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
