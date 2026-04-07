"""
Reversal Prediction Ensemble — Stacking with Purged Time-Series CV.

Architecture:
  Base layer   : RandomForest + GradientBoosting + XGBoost + LightGBM
  Meta layer   : LogisticRegression on OOF proba (stacking)
  Threshold    : tuned to maximise precision @ recall >= 0.25

Evaluation:
  Purged walk-forward CV (no overlapping label windows, 5-fold rolling).
  Primary metric: precision on positive class ("if we say reversal → it reverses").

Feature engineering is delegated to app.analysis.features.build_feature_matrix.
"""
from __future__ import annotations

import copy
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# ── base models ────────────────────────────────────────────────────────────────

def _base_models() -> dict:
    m: dict = {}
    m["rf"] = RandomForestClassifier(
        n_estimators=400, max_depth=5, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    m["gb"] = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, min_samples_leaf=2, random_state=42,
    )
    if _HAS_XGB:
        m["xgb"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=2,
            eval_metric="logloss", random_state=42,
            verbosity=0, use_label_encoder=False,
        )
    if _HAS_LGBM:
        m["lgbm"] = LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, class_weight="balanced",
            num_leaves=15, min_child_samples=3,
            random_state=42, verbosity=-1,
        )
    return m


def _clone_with_imbalance(name: str, template, y_tr: np.ndarray):
    n_neg  = int((y_tr == 0).sum())
    n_pos  = int((y_tr == 1).sum())
    ratio  = max(n_neg / max(n_pos, 1), 1.0)
    cloned = copy.deepcopy(template)
    if name == "xgb" and _HAS_XGB:
        cloned.set_params(scale_pos_weight=ratio)
    return cloned


# ── purged walk-forward CV ─────────────────────────────────────────────────────

def _purged_cv_splits(
    n: int,
    n_folds: int = 5,
    embargo: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window purged CV:
    - Events are ordered chronologically (index = time order).
    - Each fold trains on the first k * step events,
      tests on the next block, with `embargo` events removed
      from the end of training to avoid label bleed.
    """
    step = n // (n_folds + 1)
    splits = []
    for fold in range(1, n_folds + 1):
        test_start = fold * step
        test_end   = min(test_start + step, n)
        if test_end <= test_start:
            continue
        train_end  = max(test_start - embargo, 0)
        if train_end < 5:
            continue
        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits


# ── stacking with purged CV ────────────────────────────────────────────────────

def _stacking_purged(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    embargo: int = 3,
) -> tuple[np.ndarray, dict, LogisticRegression, StandardScaler]:
    """
    Expanding-window purged CV for n >= 100.
    Train on [0..fold_end-embargo], test on [fold_end..fold_end+step].
    """
    scaler    = StandardScaler()
    X_sc      = scaler.fit_transform(X)
    templates = _base_models()
    names     = list(templates.keys())
    n         = len(y)
    step      = n // (n_folds + 1)

    oof_base = np.full((n, len(names)), float(y.mean()))

    for fold in range(1, n_folds + 1):
        test_start = fold * step
        test_end   = min(test_start + step, n)
        train_end  = max(test_start - embargo, 0)
        if train_end < 10 or test_end <= test_start:
            continue
        tr = np.arange(0, train_end)
        te = np.arange(test_start, test_end)
        y_tr = y[tr]
        if y_tr.sum() == 0 or (len(y_tr) - y_tr.sum()) == 0:
            continue
        for j, (nm, tmpl) in enumerate(templates.items()):
            m = _clone_with_imbalance(nm, tmpl, y_tr)
            m.fit(X_sc[tr], y_tr)
            oof_base[te, j] = m.predict_proba(X_sc[te])[:, 1]

    meta = LogisticRegression(C=0.5, max_iter=2000, random_state=42, solver="lbfgs")
    meta.fit(oof_base, y)
    oof_meta = meta.predict_proba(oof_base)[:, 1]

    full_base: dict = {}
    for nm, tmpl in templates.items():
        m = _clone_with_imbalance(nm, tmpl, y)
        m.fit(X_sc, y)
        full_base[nm] = m

    return oof_meta, full_base, meta, scaler


def _stacking_loocv(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, dict, LogisticRegression, StandardScaler]:
    """
    Leave-One-Out stacking for small datasets (< 100 events).
    Each event is left out in turn; base models trained on N-1 events
    produce OOF probabilities. Meta learner trained on those OOF probs.
    Returns (oof_meta_probs, full_base_models, meta_model, scaler).

    Note: LOOCV is the gold standard for n < 60; purged expanding-window
    CV starves early folds when n is this small.
    """
    scaler    = StandardScaler()
    X_sc      = scaler.fit_transform(X)
    templates = _base_models()
    names     = list(templates.keys())
    n         = len(y)

    oof_base = np.zeros((n, len(names)))

    for i in range(n):
        tr_idx = np.array([j for j in range(n) if j != i])
        te_idx = np.array([i])
        X_tr, X_te = X_sc[tr_idx], X_sc[te_idx]
        y_tr = y[tr_idx]
        if y_tr.sum() == 0 or (len(y_tr) - y_tr.sum()) == 0:
            oof_base[i] = 0.5
            continue
        for j, (nm, tmpl) in enumerate(templates.items()):
            m = _clone_with_imbalance(nm, tmpl, y_tr)
            m.fit(X_tr, y_tr)
            oof_base[i, j] = float(m.predict_proba(X_te)[0, 1])

    # Meta learner on OOF base preds
    meta = LogisticRegression(C=0.5, max_iter=2000, random_state=42, solver="lbfgs")
    meta.fit(oof_base, y)
    oof_meta = meta.predict_proba(oof_base)[:, 1]

    # Full models on all data
    full_base: dict = {}
    for nm, tmpl in templates.items():
        m = _clone_with_imbalance(nm, tmpl, y)
        m.fit(X_sc, y)
        full_base[nm] = m

    return oof_meta, full_base, meta, scaler


def _ensemble_proba(X_sc, full_base, meta):
    base = np.column_stack([m.predict_proba(X_sc)[:, 1] for m in full_base.values()])
    return meta.predict_proba(base)[:, 1]


# ── precision-first threshold ─────────────────────────────────────────────────

def _tune_threshold(y: np.ndarray, probs: np.ndarray, min_recall: float = 0.20) -> float:
    best_t, best_prec = 0.5, 0.0
    for t in np.arange(0.99, 0.29, -0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_t    = round(float(t), 2)
    return best_t


# ── feature importance (mean across tree models) ──────────────────────────────

def _feat_importance(full_base: dict, feature_names: list[str]) -> list[dict]:
    imps = [m.feature_importances_ for m in full_base.values() if hasattr(m, "feature_importances_")]
    if not imps:
        return []
    avg = np.mean(imps, axis=0)
    pairs = sorted(zip(feature_names, avg.tolist()), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]


# ── main entry ─────────────────────────────────────────────────────────────────

def run_model(
    events: list[dict],
    vix_daily: pd.DataFrame | None = None,
    spx_1min:  pd.DataFrame | None = None,
    spy_1min:  pd.DataFrame | None = None,
    vix_1min:  pd.DataFrame | None = None,
    vix1d_df:  pd.DataFrame | None = None,
) -> dict:
    from app.analysis.features import build_feature_matrix

    if len(events) < 10:
        return {"error": "Not enough events", "n_events": len(events)}

    print("  Building feature matrix…")
    X, y, feature_names = build_feature_matrix(
        events,
        spx_1min  if spx_1min  is not None else pd.DataFrame(),
        vix_daily = vix_daily,
        spy_1min  = spy_1min,
        vix_1min  = vix_1min,
        vix1d_df  = vix1d_df,
    )
    print(f"  Features: {len(feature_names)}  |  Pos: {y.sum()}/{len(y)}")

    if len(y) >= 100:
        # Enough data: use purged expanding-window 5-fold CV
        print(f"  Training stacking ensemble with purged 5-fold CV (n={len(y)})…")
        oof_probs, full_base, meta, scaler = _stacking_purged(X, y, n_folds=5, embargo=3)
    else:
        print(f"  Training stacking ensemble with LOOCV (n={len(y)}, too small for k-fold)…")
        oof_probs, full_base, meta, scaler = _stacking_loocv(X, y)

    threshold = _tune_threshold(y, oof_probs, min_recall=0.20)
    oof_preds = (oof_probs >= threshold).astype(int)

    tp = int(((oof_preds == 1) & (y == 1)).sum())
    fp = int(((oof_preds == 1) & (y == 0)).sum())
    tn = int(((oof_preds == 0) & (y == 0)).sum())
    fn = int(((oof_preds == 0) & (y == 1)).sum())
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    accuracy  = round((tp + tn) / len(y), 3)

    # Full-model predictions (for display)
    X_sc      = scaler.transform(X)
    full_probs = _ensemble_proba(X_sc, full_base, meta)

    event_preds = []
    for i, ev in enumerate(events):
        event_preds.append({
            "date":                  ev["date"],
            "time_of_low":           ev.get("time_of_low", ""),
            "fall_pts":              ev.get("fall_pts", 0),
            "reversal_pts":          ev.get("reversal_pts", 0),
            "reversal_category":     ev.get("reversal_category", "weak"),
            "squeeze_strength":      ev.get("squeeze_strength", "none"),
            "vix_prev":              ev.get("vix_prev"),
            "pred_prob_significant": round(float(oof_probs[i]), 3),
            "pred_significant":      int(oof_probs[i] >= threshold),
            "actual_significant":    int(y[i]),
            "hit":                   int(oof_preds[i] == y[i]),
            "gap_pts":               float(ev.get("gap_pts", 0)),
            "gap_dir":               int(ev.get("gap_dir", 0)),
        })

    # Squeeze impact summary
    squeeze_map = {"none": 0, "no_data": 0, "mild": 1, "moderate": 2, "strong": 3}
    squeeze_impact: dict = {}
    for label, code in squeeze_map.items():
        bucket = [oof_probs[i] for i, ev in enumerate(events)
                  if squeeze_map.get(str(ev.get("squeeze_strength", "none")), 0) == code]
        if bucket:
            squeeze_impact[label] = round(float(np.mean(bucket)), 3)

    return {
        "n_events":                len(events),
        "n_significant_reversals": int(y.sum()),
        "base_rate":               round(float(y.mean()), 3),
        "target":                  "medium_or_strong (reversal_pts >= 20)",
        "models_used":             list(full_base.keys()) + ["meta:logistic"],
        "n_features":              len(feature_names),
        "optimal_threshold":       threshold,
        "loocv_accuracy":          accuracy,
        "precision":               precision,
        "recall":                  recall,
        "confusion_matrix":        {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "feature_importance":      _feat_importance(full_base, feature_names),
        "squeeze_impact":          squeeze_impact,
        "event_predictions":       event_preds,
    }
