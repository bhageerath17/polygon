"""
Reversal Prediction Model v3 — Holdout Validation + Feature Selection.

Key differences from Models 1 & 2:
  1. True temporal holdout: first 80% train, last 20% test (never seen)
  2. Feature selection: drop correlated (|r|>0.85), keep top 30 by importance
  3. Faster: only RF + LightGBM, fewer trees
  4. All eval metrics measured on holdout only — no data leakage
  5. Triple-barrier labels (same as Model 2)

Architecture:
  Base layer  : RandomForest + LightGBM
  Meta layer  : LogisticRegression on base OOF probabilities (train set only)
  Threshold   : tuned on train set, applied to holdout
"""
from __future__ import annotations

import copy
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _prep_1min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df = df.copy()
    df.index = idx
    return df.between_time("09:30", "15:59")


# ── Triple-barrier labels (reused from model2) ──────────────────────────────

def _compute_triple_barrier_labels(
    events: list[dict],
    spx_1min: pd.DataFrame,
    k: float = 1.5,
    vb_bars: int = 60,
    min_fall_pts: float = 30.0,
) -> np.ndarray:
    spx = _prep_1min(spx_1min)
    spx_daily = {str(d): g for d, g in spx.groupby(spx.index.date)}
    labels = np.zeros(len(events), dtype=int)

    for i, ev in enumerate(events):
        fall_pts = float(ev.get("fall_pts", 0))
        if fall_pts < min_fall_pts:
            continue

        date_str = str(ev.get("date", ""))
        time_str = str(ev.get("time_of_low", "12:00:00"))[:5]
        swing_low = float(ev.get("swing_low", ev.get("spx_at_low", 5000)))

        day_bars = spx_daily.get(date_str, pd.DataFrame())
        if day_bars.empty:
            continue

        bars_at_T = day_bars[day_bars.index.strftime("%H:%M") <= time_str]
        if len(bars_at_T) < 5:
            continue

        trail_closes = bars_at_T["close"].values[-30:]
        if len(trail_closes) < 3:
            continue
        log_rets = np.diff(np.log(trail_closes))
        sigma_t = max(float(np.std(log_rets)), 1e-4)

        pt_level = swing_low * np.exp(+k * sigma_t)
        sl_level = swing_low * np.exp(-k * sigma_t)

        bars_after = day_bars[day_bars.index.strftime("%H:%M") > time_str]
        if len(bars_after) == 0:
            continue

        for _, bar in bars_after.iloc[:vb_bars].iterrows():
            if float(bar["low"]) <= sl_level:
                break
            if float(bar["high"]) >= pt_level:
                labels[i] = 1
                break

    return labels


# ── Feature selection ────────────────────────────────────────────────────────

def _select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_features: int = 30,
    corr_threshold: float = 0.85,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    1. Remove features correlated > corr_threshold (keep the one with higher
       univariate importance).
    2. Keep top max_features by RF importance.
    Returns (X_selected, selected_names, dropped_names).
    """
    n_features = X.shape[1]
    if n_features <= max_features:
        return X, list(feature_names), []

    # Quick RF for importance ranking
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    importances = rf.feature_importances_

    # Step 1: correlation-based elimination
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr)
    dropped = set()
    for i in range(n_features):
        if i in dropped:
            continue
        for j in range(i + 1, n_features):
            if j in dropped:
                continue
            if abs(corr[i, j]) > corr_threshold:
                # Drop the less important one
                if importances[i] >= importances[j]:
                    dropped.add(j)
                else:
                    dropped.add(i)
                    break

    surviving = [i for i in range(n_features) if i not in dropped]

    # Step 2: keep top max_features by importance
    surviving_with_imp = sorted(surviving, key=lambda i: importances[i], reverse=True)
    selected = surviving_with_imp[:max_features]
    selected.sort()  # preserve original order

    selected_names = [feature_names[i] for i in selected]
    dropped_names = [feature_names[i] for i in range(n_features) if i not in selected]

    return X[:, selected], selected_names, dropped_names


# ── Base models ──────────────────────────────────────────────────────────────

def _base_models() -> dict:
    m: dict = {}
    m["rf"] = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    if _HAS_LGBM:
        m["lgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.7, class_weight="balanced",
            num_leaves=15, min_child_samples=3,
            random_state=42, verbosity=-1,
        )
    return m


def _clone_with_imbalance(name: str, template, y_tr: np.ndarray):
    return copy.deepcopy(template)


# ── Purged CV on training set only ───────────────────────────────────────────

def _purged_cv_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 5,
    embargo: int = 3,
) -> np.ndarray:
    """
    Expanding-window purged CV on training set only.
    Returns OOF base probabilities for the meta-learner.
    """
    templates = _base_models()
    names = list(templates.keys())
    n = len(y_train)
    step = n // (n_folds + 1)

    oof_base = np.full((n, len(names)), float(y_train.mean()))

    for fold in range(1, n_folds + 1):
        test_start = fold * step
        test_end = min(test_start + step, n)
        train_end = max(test_start - embargo, 0)
        if train_end < 10 or test_end <= test_start:
            continue
        tr = np.arange(0, train_end)
        te = np.arange(test_start, test_end)
        y_tr = y_train[tr]
        if y_tr.sum() == 0 or (len(y_tr) - y_tr.sum()) == 0:
            continue

        for j, (nm, tmpl) in enumerate(templates.items()):
            m = _clone_with_imbalance(nm, tmpl, y_tr)
            m.fit(X_train[tr], y_tr)
            oof_base[te, j] = m.predict_proba(X_train[te])[:, 1]

    return oof_base


# ── Precision-first threshold ────────────────────────────────────────────────

def _tune_threshold(y: np.ndarray, probs: np.ndarray, min_recall: float = 0.20) -> float:
    best_t, best_prec = 0.5, 0.0
    for t in np.arange(0.99, 0.29, -0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_t = round(float(t), 2)
    return best_t


# ── Feature importance ───────────────────────────────────────────────────────

def _feat_importance(models: dict, feature_names: list[str]) -> list[dict]:
    imps = [m.feature_importances_ for m in models.values() if hasattr(m, "feature_importances_")]
    if not imps:
        return []
    avg = np.mean(imps, axis=0)
    pairs = sorted(zip(feature_names, avg.tolist()), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]


# ── Main entry ───────────────────────────────────────────────────────────────

def run_model3(
    events: list[dict],
    vix_daily: pd.DataFrame | None = None,
    spx_1min: pd.DataFrame | None = None,
    spy_1min: pd.DataFrame | None = None,
    vix_1min: pd.DataFrame | None = None,
    vix1d_df: pd.DataFrame | None = None,
    holdout_frac: float = 0.20,
    max_features: int = 30,
    corr_threshold: float = 0.85,
) -> dict:
    from app.analysis.features import build_feature_matrix

    if len(events) < 20:
        return {"error": "Not enough events", "n_events": len(events)}

    spx_df = spx_1min if spx_1min is not None else pd.DataFrame()

    # ── 1. Build full feature matrix ────────────────────────────────────────
    print("  Building feature matrix...")
    X_full, y_fixed, feature_names = build_feature_matrix(
        events, spx_df,
        vix_daily=vix_daily, spy_1min=spy_1min,
        vix_1min=vix_1min, vix1d_df=vix1d_df,
    )
    feature_names = list(feature_names)

    # ── 2. Triple-barrier labels ────────────────────────────────────────────
    print("  Computing triple-barrier labels (k=1.5)...")
    y_tb = _compute_triple_barrier_labels(events, spx_df, k=1.5, vb_bars=60)

    if y_tb.sum() < 10:
        print(f"  WARNING: Triple-barrier gave {y_tb.sum()} positives, falling back to fixed")
        y_full = y_fixed
        label_method = "fixed_threshold_fallback"
    else:
        y_full = y_tb
        label_method = "triple_barrier"

    # ── 3. Temporal train/holdout split ─────────────────────────────────────
    n = len(events)
    split_idx = int(n * (1 - holdout_frac))
    # Ensure holdout has at least 10 events
    split_idx = min(split_idx, n - 10)
    split_idx = max(split_idx, 20)

    X_train_full, X_hold_full = X_full[:split_idx], X_full[split_idx:]
    y_train, y_hold = y_full[:split_idx], y_full[split_idx:]
    events_train, events_hold = events[:split_idx], events[split_idx:]

    holdout_dates = f"{events_hold[0]['date']} to {events_hold[-1]['date']}"
    train_dates = f"{events_train[0]['date']} to {events_train[-1]['date']}"

    print(f"  Train: {len(y_train)} events ({train_dates})")
    print(f"  Holdout: {len(y_hold)} events ({holdout_dates})")
    print(f"  Train pos: {y_train.sum()}/{len(y_train)}  |  Holdout pos: {y_hold.sum()}/{len(y_hold)}")

    # ── 4. Feature selection (on train set only!) ───────────────────────────
    print(f"  Feature selection: {len(feature_names)} features → ", end="")
    X_train, sel_names, dropped = _select_features(
        X_train_full, y_train, feature_names,
        max_features=max_features, corr_threshold=corr_threshold,
    )
    # Apply same selection to holdout
    sel_idx = [feature_names.index(n) for n in sel_names]
    X_hold = X_hold_full[:, sel_idx]
    print(f"{len(sel_names)} selected, {len(dropped)} dropped")

    # Replace NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_hold = np.nan_to_num(X_hold, nan=0.0, posinf=1e6, neginf=-1e6)

    # ── 5. Train stacking ensemble on train set ─────────────────────────────
    print(f"  Training stacking ensemble (purged 5-fold CV on train set)...")
    oof_base_train = _purged_cv_train(X_train, y_train, n_folds=5, embargo=3)

    # Meta-learner on train OOF
    meta = LogisticRegression(C=0.5, max_iter=2000, random_state=42, solver="lbfgs")
    meta.fit(oof_base_train, y_train)
    oof_meta_train = meta.predict_proba(oof_base_train)[:, 1]

    # Tune threshold on train set
    threshold = _tune_threshold(y_train, oof_meta_train, min_recall=0.20)

    # ── 6. Retrain base models on full train set ────────────────────────────
    templates = _base_models()
    full_base: dict = {}
    for nm, tmpl in templates.items():
        m = _clone_with_imbalance(nm, tmpl, y_train)
        m.fit(X_train, y_train)
        full_base[nm] = m

    # ── 7. Predict on holdout ───────────────────────────────────────────────
    print("  Predicting on holdout set...")
    hold_base = np.column_stack([
        m.predict_proba(X_hold)[:, 1] for m in full_base.values()
    ])
    hold_probs = meta.predict_proba(hold_base)[:, 1]
    hold_preds = (hold_probs >= threshold).astype(int)

    # Also get train set predictions for reference
    train_base = np.column_stack([
        m.predict_proba(X_train)[:, 1] for m in full_base.values()
    ])
    train_probs = meta.predict_proba(train_base)[:, 1]
    train_preds = (train_probs >= threshold).astype(int)

    # ── 8. Compute holdout metrics ──────────────────────────────────────────
    tp = int(((hold_preds == 1) & (y_hold == 1)).sum())
    fp = int(((hold_preds == 1) & (y_hold == 0)).sum())
    tn = int(((hold_preds == 0) & (y_hold == 0)).sum())
    fn = int(((hold_preds == 0) & (y_hold == 1)).sum())
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    accuracy = round((tp + tn) / len(y_hold), 3)

    # Train metrics for comparison (detect overfitting)
    tr_tp = int(((train_preds == 1) & (y_train == 1)).sum())
    tr_fp = int(((train_preds == 1) & (y_train == 0)).sum())
    tr_fn = int(((train_preds == 0) & (y_train == 1)).sum())
    tr_tn = int(((train_preds == 0) & (y_train == 0)).sum())
    train_precision = round(tr_tp / (tr_tp + tr_fp), 3) if (tr_tp + tr_fp) > 0 else 0.0
    train_recall = round(tr_tp / (tr_tp + tr_fn), 3) if (tr_tp + tr_fn) > 0 else 0.0

    # ── 9. VIX regime metrics on holdout ────────────────────────────────────
    vix_vals_hold = np.array([float(ev.get("vix_prev") or 20.0) for ev in events_hold])
    # Use train-set percentiles for regime boundaries (no holdout leakage)
    vix_vals_train = np.array([float(ev.get("vix_prev") or 20.0) for ev in events_train])
    p33 = float(np.percentile(vix_vals_train, 33.33))
    p66 = float(np.percentile(vix_vals_train, 66.67))

    vix_regime_perf = {}
    for regime_name, lo, hi in [("low", -999, p33), ("mid", p33, p66), ("high", p66, 9999)]:
        mask = (vix_vals_hold >= lo) & (vix_vals_hold < hi) if regime_name != "high" \
            else (vix_vals_hold >= lo)
        n_r = int(mask.sum())
        if n_r == 0:
            vix_regime_perf[regime_name] = {"precision": 0.0, "recall": 0.0, "n": 0}
            continue
        y_r = y_hold[mask]
        preds_r = hold_preds[mask]
        r_tp = int(((preds_r == 1) & (y_r == 1)).sum())
        r_fp = int(((preds_r == 1) & (y_r == 0)).sum())
        r_fn = int(((preds_r == 0) & (y_r == 1)).sum())
        prec = round(r_tp / (r_tp + r_fp), 3) if (r_tp + r_fp) > 0 else 0.0
        rec = round(r_tp / (r_tp + r_fn), 3) if (r_tp + r_fn) > 0 else 0.0
        vix_regime_perf[regime_name] = {"precision": prec, "recall": rec, "n": n_r}

    # ── 10. Per-event predictions (holdout only) ────────────────────────────
    event_preds = []
    for i, ev in enumerate(events_hold):
        event_preds.append({
            "date": ev["date"],
            "time_of_low": ev.get("time_of_low", ""),
            "fall_pts": ev.get("fall_pts", 0),
            "reversal_pts": ev.get("reversal_pts", 0),
            "reversal_category": ev.get("reversal_category", "weak"),
            "squeeze_strength": ev.get("squeeze_strength", "none"),
            "vix_prev": ev.get("vix_prev"),
            "pred_prob_significant": round(float(hold_probs[i]), 3),
            "pred_significant": int(hold_preds[i]),
            "actual_significant": int(y_hold[i]),
            "hit": int(hold_preds[i] == y_hold[i]),
            "gap_pts": float(ev.get("gap_pts", 0)),
            "gap_dir": int(ev.get("gap_dir", 0)),
            "split": "holdout",
        })

    # Also include train events with their predictions (marked as train)
    for i, ev in enumerate(events_train):
        event_preds.append({
            "date": ev["date"],
            "time_of_low": ev.get("time_of_low", ""),
            "fall_pts": ev.get("fall_pts", 0),
            "reversal_pts": ev.get("reversal_pts", 0),
            "reversal_category": ev.get("reversal_category", "weak"),
            "squeeze_strength": ev.get("squeeze_strength", "none"),
            "vix_prev": ev.get("vix_prev"),
            "pred_prob_significant": round(float(train_probs[i]), 3),
            "pred_significant": int(train_preds[i]),
            "actual_significant": int(y_train[i]),
            "hit": int(train_preds[i] == y_train[i]),
            "gap_pts": float(ev.get("gap_pts", 0)),
            "gap_dir": int(ev.get("gap_dir", 0)),
            "split": "train",
        })

    # Sort all event predictions by date
    event_preds.sort(key=lambda e: e["date"])

    # ── 11. Squeeze impact on holdout ───────────────────────────────────────
    squeeze_map = {"none": 0, "no_data": 0, "mild": 1, "moderate": 2, "strong": 3}
    squeeze_impact: dict = {}
    for label, code in squeeze_map.items():
        bucket = [hold_probs[i] for i, ev in enumerate(events_hold)
                  if squeeze_map.get(str(ev.get("squeeze_strength", "none")), 0) == code]
        if bucket:
            squeeze_impact[label] = round(float(np.mean(bucket)), 3)

    return {
        "n_events": len(events),
        "n_train": len(events_train),
        "n_holdout": len(events_hold),
        "n_significant_reversals": int(y_full.sum()),
        "base_rate": round(float(y_full.mean()), 3),
        "train_dates": train_dates,
        "holdout_dates": holdout_dates,
        "target": "triple_barrier_PT_hit" if label_method == "triple_barrier"
                  else "medium_or_strong (reversal_pts >= 20)",
        "label_method": label_method,
        "k_barrier": 1.5,
        "models_used": list(full_base.keys()) + ["meta:logistic"],
        "n_features_original": len(feature_names),
        "n_features_selected": len(sel_names),
        "selected_features": sel_names,
        "dropped_features": dropped,
        "corr_threshold": corr_threshold,
        "optimal_threshold": threshold,
        # Holdout metrics (the ones that matter)
        "holdout_precision": precision,
        "holdout_recall": recall,
        "holdout_accuracy": accuracy,
        "holdout_confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        # Train metrics (for overfitting comparison)
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_confusion_matrix": {"tp": tr_tp, "fp": tr_fp, "tn": tr_tn, "fn": tr_fn},
        "feature_importance": _feat_importance(full_base, sel_names),
        "squeeze_impact": squeeze_impact,
        "event_predictions": event_preds,
        "vix_regime_performance": vix_regime_perf,
        "holdout_frac": holdout_frac,
    }
