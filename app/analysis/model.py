"""
Reversal Prediction Model — LogisticRegression + Leave-One-Out CV.

Features:
  - vix_prev          : prior day VIX (higher = more volatility, higher reversal prob)
  - fall_pts          : magnitude of the intraday fall
  - fall_normalized   : fall_pts / vix_prev (normalised by vol regime)
  - hour_of_low       : hour the low occurred (morning vs afternoon dynamics)
  - squeeze_encoded   : 0=none, 1=mild, 2=moderate, 3=strong
  - is_monday         : Monday tends to be trend-continuation
  - is_friday         : Friday tends to see end-of-week reversions

Target:
  reversal_significant = 1 if reversal_category in ["medium", "strong"]  (reversal_pts >= 20)
  (only 3 "strong" events out of 42 → too sparse; medium+strong gives a workable 32/42 split)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


SQUEEZE_MAP = {"none": 0, "no_data": 0, "mild": 1, "moderate": 2, "strong": 3}


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) from the events DataFrame."""
    df = df.copy()

    # Fill missing VIX with median
    vix_median = df["vix_prev"].median()
    df["vix_prev"] = df["vix_prev"].fillna(vix_median)

    df["squeeze_encoded"] = df["squeeze_strength"].map(SQUEEZE_MAP).fillna(0)
    df["fall_normalized"] = df["fall_pts"] / df["vix_prev"].clip(lower=1)
    df["is_monday"] = (df["day_of_week"] == "Monday").astype(int)
    df["is_friday"] = (df["day_of_week"] == "Friday").astype(int)

    feature_names = [
        "vix_prev",
        "fall_pts",
        "fall_normalized",
        "hour_of_low",
        "squeeze_encoded",
        "is_monday",
        "is_friday",
    ]

    X = df[feature_names].values.astype(float)
    y = df["reversal_category"].isin(["medium", "strong"]).astype(int).values
    return X, y, feature_names


def run_model(events: list[dict]) -> dict:
    """Train logistic regression with LOOCV; return model results + per-event predictions."""
    df = pd.DataFrame(events)

    if len(df) < 10:
        return {"error": "Not enough events for modelling", "n_events": len(df)}

    X, y, feature_names = _build_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full-dataset model for feature importances + final predictions
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_scaled, y)

    # LOOCV for unbiased accuracy
    loo = LeaveOneOut()
    loo_probs: list[float] = []
    loo_preds: list[int]   = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr = y[train_idx]
        m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        m.fit(X_tr, y_tr)
        prob = float(m.predict_proba(X_te)[0, 1])
        pred = int(prob >= 0.5)
        loo_probs.append(prob)
        loo_preds.append(pred)

    loo_preds_arr = np.array(loo_preds)
    loo_acc = float((loo_preds_arr == y).mean())
    n_pos = int(y.sum())

    # Feature coefficients (from full model)
    coef_pairs = sorted(
        zip(feature_names, clf.coef_[0].tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    feature_importance = [{"feature": n, "coefficient": round(c, 4)} for n, c in coef_pairs]

    # Full-model predictions for all events (used on dashboard with actual probability)
    full_probs = clf.predict_proba(X_scaled)[:, 1].tolist()

    # Per-event output
    event_predictions = []
    for i, ev in enumerate(events):
        event_predictions.append({
            "date":               ev["date"],
            "time_of_low":        ev["time_of_low"],
            "fall_pts":           ev["fall_pts"],
            "reversal_pts":       ev["reversal_pts"],
            "reversal_category":  ev["reversal_category"],
            "squeeze_strength":   ev["squeeze_strength"],
            "vix_prev":           ev["vix_prev"],
            "pred_prob_significant": round(loo_probs[i], 3),   # LOOCV probability (medium or strong)
            "pred_significant":      loo_preds[i],              # LOOCV prediction
            "actual_significant":    int(y[i]),
            "hit":                   int(loo_preds[i] == y[i]),
        })

    # Confusion matrix values
    tp = int(((loo_preds_arr == 1) & (y == 1)).sum())
    fp = int(((loo_preds_arr == 1) & (y == 0)).sum())
    tn = int(((loo_preds_arr == 0) & (y == 0)).sum())
    fn = int(((loo_preds_arr == 0) & (y == 1)).sum())
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0

    # Squeeze impact: avg loo_prob by squeeze bucket
    squeeze_impact = {}
    for label, code in [("none", 0), ("mild", 1), ("moderate", 2), ("strong", 3)]:
        mask = df["squeeze_strength"].map(SQUEEZE_MAP).fillna(0) == code
        probs_bucket = [p for p, m in zip(loo_probs, mask) if m]
        if probs_bucket:
            squeeze_impact[label] = round(float(np.mean(probs_bucket)), 3)

    return {
        "n_events":                len(df),
        "n_significant_reversals": n_pos,
        "base_rate":               round(n_pos / len(df), 3),
        "target":                  "medium_or_strong (reversal_pts >= 20)",
        "loocv_accuracy":     round(loo_acc, 3),
        "precision":          precision,
        "recall":             recall,
        "confusion_matrix":   {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "feature_importance": feature_importance,
        "squeeze_impact":     squeeze_impact,
        "event_predictions":  event_predictions,
    }
