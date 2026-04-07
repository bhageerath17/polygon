"""
Reversal Prediction Ensemble v2 — Triple-Barrier Labels + Calibration.

Improvements over Model 1:
  A  Triple-barrier labelling (vol-scaled PT/SL + vertical barrier)
  B  Time-of-day feature normalization (rolling 60-day window)
  C  Robust rank scaling (replaces StandardScaler)
  D  Purged expanding-window CV with N=6 folds, embargo=5
  E  Platt calibration (LogisticRegression on OOF meta probs)
  F  VIX regime conditioning (3 regimes, per-regime metrics)
  G  Roll spread proxy feature
  H  Feature importance stability tracking

Architecture:
  Base layer    : RandomForest + GradientBoosting + XGBoost + LightGBM
  Meta layer    : LogisticRegression on OOF base proba (stacking)
  Calibrator    : LogisticRegression on OOF meta proba (Platt scaling)
  Threshold     : tuned to maximise precision @ recall >= 0.25
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _prep_1min(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeTZ-aware ET index, RTH only."""
    if df.empty:
        return df
    idx = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df = df.copy()
    df.index = idx
    return df.between_time("09:30", "15:59")


# ── A. Triple-barrier labels ─────────────────────────────────────────────────

def _compute_triple_barrier_labels(
    events: list[dict],
    spx_1min: pd.DataFrame,
    k: float = 1.5,
    vb_bars: int = 60,
    min_fall_pts: float = 30.0,
) -> np.ndarray:
    """
    For each event, compute triple-barrier label.
    PT = swing_low * exp(+k * sigma_t)
    SL = swing_low * exp(-k * sigma_t)
    VB = vb_bars bars after swing low
    Label = 1 if PT hit first, 0 otherwise.
    """
    spx = _prep_1min(spx_1min)
    spx_daily = {str(d): g for d, g in spx.groupby(spx.index.date)}

    labels = np.zeros(len(events), dtype=int)

    for i, ev in enumerate(events):
        fall_pts = float(ev.get("fall_pts", 0))
        if fall_pts < min_fall_pts:
            labels[i] = 0
            continue

        date_str = str(ev.get("date", ""))
        time_str = str(ev.get("time_of_low", "12:00:00"))[:5]
        swing_low = float(ev.get("swing_low", ev.get("spx_at_low", 5000)))

        day_bars = spx_daily.get(date_str, pd.DataFrame())
        if day_bars.empty:
            labels[i] = 0
            continue

        # Bars up to T for trailing vol computation
        bars_at_T = day_bars[day_bars.index.strftime("%H:%M") <= time_str]
        if len(bars_at_T) < 5:
            labels[i] = 0
            continue

        # Realized vol from trailing 30 bars (log returns)
        trail_closes = bars_at_T["close"].values[-30:]
        if len(trail_closes) < 3:
            labels[i] = 0
            continue
        log_rets = np.diff(np.log(trail_closes))
        sigma_t = float(np.std(log_rets))  # per-bar vol
        if sigma_t < 1e-8:
            sigma_t = 1e-4  # floor

        pt_level = swing_low * np.exp(+k * sigma_t)
        sl_level = swing_low * np.exp(-k * sigma_t)

        # Forward bars after the swing low
        bars_after = day_bars[day_bars.index.strftime("%H:%M") > time_str]
        if len(bars_after) == 0:
            labels[i] = 0
            continue

        look_ahead = bars_after.iloc[:vb_bars]

        # Check which barrier is hit first
        hit_label = 0  # default: VB or SL
        for _, bar in look_ahead.iterrows():
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            # Check SL first (conservative)
            if bar_low <= sl_level:
                hit_label = 0
                break
            if bar_high >= pt_level:
                hit_label = 1
                break
        # If neither hit within vb_bars, label = 0 (VB)

        labels[i] = hit_label

    return labels


# ── B. Time-of-day normalization ──────────────────────────────────────────────

def _tod_normalize(
    X: np.ndarray,
    events: list[dict],
    feature_names: list[str],
    spx_1min: pd.DataFrame,
    lookback_days: int = 60,
) -> np.ndarray:
    """
    For return/vol/rsi features, normalize by time-of-day statistics
    estimated from trailing lookback_days trading days.
    """
    # Identify feature columns to normalize (stretch, rv, rsi, ret_z, autocorr)
    tod_features = set()
    for idx, name in enumerate(feature_names):
        for prefix in ("stretch_", "ret_z_", "rv_", "rsi_", "autocorr_",
                        "jump_share_", "down_sv_", "range_vol_", "rv_spike"):
            if name.startswith(prefix) or name == prefix:
                tod_features.add(idx)
                break

    if not tod_features:
        return X

    # Get minute-of-day for each event
    minutes = []
    dates = []
    for ev in events:
        time_str = str(ev.get("time_of_low", "12:00:00"))[:5]
        try:
            hh, mm = int(time_str[:2]), int(time_str[3:5])
        except ValueError:
            hh, mm = 12, 0
        minutes.append(hh * 60 + mm)
        dates.append(str(ev.get("date", "")))

    X_norm = X.copy().astype(float)
    eps = 1e-6

    # For each event, normalize selected features using stats from
    # trailing events at same minute-of-day (within a band of +/- 15 min)
    for i in range(len(events)):
        # Find prior events within lookback window and similar time-of-day
        m_i = minutes[i]
        d_i = dates[i]

        prior_mask = []
        for j in range(i):
            if dates[j] < d_i and abs(minutes[j] - m_i) <= 15:
                prior_mask.append(j)

        # Need at least 5 prior observations
        if len(prior_mask) < 5:
            continue

        prior_idx = np.array(prior_mask[-lookback_days:])  # most recent

        for col in tod_features:
            if col >= X.shape[1]:
                continue
            prior_vals = X[prior_idx, col]
            mu = float(np.mean(prior_vals))
            sigma = float(np.std(prior_vals))
            X_norm[i, col] = (X[i, col] - mu) / (sigma + eps)

    return X_norm


# ── C. Robust rank scaling ───────────────────────────────────────────────────

def _rank_scale(X: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Replace each value with its rank among the last W events / W.
    Expanding window for early events.
    """
    n, p = X.shape
    X_ranked = np.zeros_like(X, dtype=float)

    for i in range(n):
        start = max(0, i - window + 1)
        block = X[start:i + 1]  # includes current
        w = len(block)
        for j in range(p):
            col = block[:, j]
            # Rank of the current value among the block
            rank = float(np.sum(col <= col[-1]))
            X_ranked[i, j] = rank / w

    return X_ranked


# ── G. Roll spread estimator ─────────────────────────────────────────────────

def _compute_roll_spread(
    events: list[dict],
    spx_1min: pd.DataFrame,
    n_bars: int = 30,
) -> np.ndarray:
    """
    Roll spread = 2 * sqrt(max(-Cov(dP_t, dP_{t-1}), 0))
    using SPX 1-min returns in the 30 bars before each event.
    """
    spx = _prep_1min(spx_1min)
    spx_daily = {str(d): g for d, g in spx.groupby(spx.index.date)}
    spreads = np.zeros(len(events))

    for i, ev in enumerate(events):
        date_str = str(ev.get("date", ""))
        time_str = str(ev.get("time_of_low", "12:00:00"))[:5]
        day_bars = spx_daily.get(date_str, pd.DataFrame())
        if day_bars.empty:
            continue

        bars_at_T = day_bars[day_bars.index.strftime("%H:%M") <= time_str]
        closes = bars_at_T["close"].values[-n_bars:]
        if len(closes) < 5:
            continue

        dp = np.diff(closes)
        if len(dp) < 3:
            continue

        # Autocovariance at lag 1
        cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
        spreads[i] = 2.0 * sqrt(max(-cov, 0.0))

    return spreads


# ── base models ───────────────────────────────────────────────────────────────

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
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    ratio = max(n_neg / max(n_pos, 1), 1.0)
    cloned = copy.deepcopy(template)
    if name == "xgb" and _HAS_XGB:
        cloned.set_params(scale_pos_weight=ratio)
    return cloned


# ── D. Purged expanding-window CV (N=6, embargo=5) ────────────────────────────

def _purged_cv_splits(
    n: int,
    n_folds: int = 6,
    embargo: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    step = n // (n_folds + 1)
    splits = []
    for fold in range(1, n_folds + 1):
        test_start = fold * step
        test_end = min(test_start + step, n)
        if test_end <= test_start:
            continue
        train_end = max(test_start - embargo, 0)
        if train_end < 5:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits


# ── H. Feature importance stability ──────────────────────────────────────────

def _track_feature_stability(
    fold_top10s: list[list[str]],
) -> float:
    """Mean overlap of top-10 features between consecutive folds."""
    if len(fold_top10s) < 2:
        return 1.0
    overlaps = []
    for i in range(1, len(fold_top10s)):
        s1 = set(fold_top10s[i - 1])
        s2 = set(fold_top10s[i])
        overlaps.append(len(s1 & s2) / 10.0)
    return float(np.mean(overlaps))


# ── stacking with purged CV + calibration ─────────────────────────────────────

def _stacking_purged_v2(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_folds: int = 6,
    embargo: int = 5,
) -> tuple[np.ndarray, dict, LogisticRegression, LogisticRegression, list[float], list[list[str]]]:
    """
    Expanding-window purged CV with calibration.
    Returns (calibrated_oof_probs, full_base_models, meta_model, calibrator,
             per_fold_precisions, fold_top10s).
    """
    templates = _base_models()
    names = list(templates.keys())
    n = len(y)
    step = n // (n_folds + 1)

    oof_base = np.full((n, len(names)), float(y.mean()))
    fold_precisions: list[float] = []
    fold_top10s: list[list[str]] = []

    for fold in range(1, n_folds + 1):
        test_start = fold * step
        test_end = min(test_start + step, n)
        train_end = max(test_start - embargo, 0)
        if train_end < 10 or test_end <= test_start:
            continue
        tr = np.arange(0, train_end)
        te = np.arange(test_start, test_end)
        y_tr = y[tr]
        if y_tr.sum() == 0 or (len(y_tr) - y_tr.sum()) == 0:
            continue

        fold_imps = []
        for j, (nm, tmpl) in enumerate(templates.items()):
            m = _clone_with_imbalance(nm, tmpl, y_tr)
            m.fit(X[tr], y_tr)
            oof_base[te, j] = m.predict_proba(X[te])[:, 1]
            if hasattr(m, "feature_importances_"):
                fold_imps.append(m.feature_importances_)

        # Per-fold top-10 features
        if fold_imps and len(feature_names) > 0:
            avg_imp = np.mean(fold_imps, axis=0)
            top10_idx = np.argsort(avg_imp)[-10:][::-1]
            fold_top10s.append([feature_names[idx] for idx in top10_idx if idx < len(feature_names)])

    # Meta learner
    meta = LogisticRegression(C=0.5, max_iter=2000, random_state=42, solver="lbfgs")
    meta.fit(oof_base, y)
    oof_meta = meta.predict_proba(oof_base)[:, 1]

    # E. Platt calibration
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="lbfgs")
    calibrator.fit(oof_meta.reshape(-1, 1), y)
    oof_calibrated = calibrator.predict_proba(oof_meta.reshape(-1, 1))[:, 1]

    # Per-fold precision using calibrated probs (use threshold=0.5 for fold metrics)
    for fold in range(1, n_folds + 1):
        test_start = fold * step
        test_end = min(test_start + step, n)
        if test_end <= test_start:
            continue
        te = np.arange(test_start, test_end)
        y_te = y[te]
        preds_te = (oof_calibrated[te] >= 0.5).astype(int)
        if preds_te.sum() > 0 and y_te.sum() > 0:
            fold_precisions.append(float(precision_score(y_te, preds_te, zero_division=0)))

    # Full models on all data
    full_base: dict = {}
    for nm, tmpl in templates.items():
        m = _clone_with_imbalance(nm, tmpl, y)
        m.fit(X, y)
        full_base[nm] = m

    return oof_calibrated, full_base, meta, calibrator, fold_precisions, fold_top10s


def _stacking_loocv_v2(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict, LogisticRegression, LogisticRegression, list[float], list[list[str]]]:
    """LOOCV stacking for small datasets (< 100 events)."""
    templates = _base_models()
    names = list(templates.keys())
    n = len(y)

    oof_base = np.zeros((n, len(names)))

    for i in range(n):
        tr_idx = np.array([j for j in range(n) if j != i])
        te_idx = np.array([i])
        y_tr = y[tr_idx]
        if y_tr.sum() == 0 or (len(y_tr) - y_tr.sum()) == 0:
            oof_base[i] = 0.5
            continue
        for j, (nm, tmpl) in enumerate(templates.items()):
            m = _clone_with_imbalance(nm, tmpl, y_tr)
            m.fit(X[tr_idx], y_tr)
            oof_base[i, j] = float(m.predict_proba(X[te_idx])[0, 1])

    meta = LogisticRegression(C=0.5, max_iter=2000, random_state=42, solver="lbfgs")
    meta.fit(oof_base, y)
    oof_meta = meta.predict_proba(oof_base)[:, 1]

    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="lbfgs")
    calibrator.fit(oof_meta.reshape(-1, 1), y)
    oof_calibrated = calibrator.predict_proba(oof_meta.reshape(-1, 1))[:, 1]

    full_base: dict = {}
    for nm, tmpl in templates.items():
        m = _clone_with_imbalance(nm, tmpl, y)
        m.fit(X, y)
        full_base[nm] = m

    # Single fold metrics (not meaningful for LOOCV, report overall)
    fold_precisions = []
    fold_top10s = []

    # Compute one global top-10 for stability
    imps = [m.feature_importances_ for m in full_base.values() if hasattr(m, "feature_importances_")]
    if imps:
        avg = np.mean(imps, axis=0)
        top10_idx = np.argsort(avg)[-10:][::-1]
        fold_top10s.append([feature_names[idx] for idx in top10_idx if idx < len(feature_names)])

    return oof_calibrated, full_base, meta, calibrator, fold_precisions, fold_top10s


# ── precision-first threshold ─────────────────────────────────────────────────

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


# ── feature importance ────────────────────────────────────────────────────────

def _feat_importance(full_base: dict, feature_names: list[str]) -> list[dict]:
    imps = [m.feature_importances_ for m in full_base.values() if hasattr(m, "feature_importances_")]
    if not imps:
        return []
    avg = np.mean(imps, axis=0)
    pairs = sorted(zip(feature_names, avg.tolist()), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]


# ── F. VIX regime performance ─────────────────────────────────────────────────

def _vix_regime_metrics(
    events: list[dict],
    y: np.ndarray,
    oof_preds: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    """
    Split events into 3 VIX regimes, compute per-regime precision/recall.
    Returns (regime_performance_dict, regime_array).
    """
    vix_vals = np.array([float(ev.get("vix_prev") or 20.0) for ev in events])
    p33 = np.percentile(vix_vals, 33.33)
    p66 = np.percentile(vix_vals, 66.67)

    regimes = np.zeros(len(events), dtype=int)
    regimes[vix_vals >= p66] = 2
    regimes[(vix_vals >= p33) & (vix_vals < p66)] = 1

    result = {}
    for regime_idx, regime_name in enumerate(["low", "mid", "high"]):
        mask = regimes == regime_idx
        n_r = int(mask.sum())
        if n_r == 0:
            result[regime_name] = {"precision": 0.0, "recall": 0.0, "n": 0}
            continue
        y_r = y[mask]
        preds_r = (oof_preds[mask] >= threshold).astype(int)
        tp = int(((preds_r == 1) & (y_r == 1)).sum())
        fp = int(((preds_r == 1) & (y_r == 0)).sum())
        fn = int(((preds_r == 0) & (y_r == 1)).sum())
        prec = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        result[regime_name] = {"precision": prec, "recall": rec, "n": n_r}

    return result, regimes


# ── spread regime performance ─────────────────────────────────────────────────

def _spread_regime_metrics(
    roll_spreads: np.ndarray,
    y: np.ndarray,
    oof_preds: np.ndarray,
    threshold: float,
) -> dict:
    median_spread = float(np.median(roll_spreads[roll_spreads > 0])) if (roll_spreads > 0).any() else 0.0
    result = {}
    for regime_name, mask in [
        ("low_spread", roll_spreads <= median_spread),
        ("high_spread", roll_spreads > median_spread),
    ]:
        n_r = int(mask.sum())
        if n_r == 0:
            result[regime_name] = {"precision": 0.0, "recall": 0.0, "n": 0}
            continue
        y_r = y[mask]
        preds_r = (oof_preds[mask] >= threshold).astype(int)
        tp = int(((preds_r == 1) & (y_r == 1)).sum())
        fp = int(((preds_r == 1) & (y_r == 0)).sum())
        fn = int(((preds_r == 0) & (y_r == 1)).sum())
        prec = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        result[regime_name] = {"precision": prec, "recall": rec, "n": n_r}

    return result


# ── main entry ────────────────────────────────────────────────────────────────

def run_model2(
    events: list[dict],
    vix_daily: pd.DataFrame | None = None,
    spx_1min: pd.DataFrame | None = None,
    spy_1min: pd.DataFrame | None = None,
    vix_1min: pd.DataFrame | None = None,
    vix1d_df: pd.DataFrame | None = None,
) -> dict:
    from app.analysis.features import build_feature_matrix

    if len(events) < 10:
        return {"error": "Not enough events", "n_events": len(events)}

    spx_df = spx_1min if spx_1min is not None else pd.DataFrame()

    # ── 1. Build feature matrix (same as Model 1) ────────────────────────────
    print("  Building feature matrix...")
    X, y_fixed, feature_names = build_feature_matrix(
        events,
        spx_df,
        vix_daily=vix_daily,
        spy_1min=spy_1min,
        vix_1min=vix_1min,
        vix1d_df=vix1d_df,
    )
    feature_names = list(feature_names)

    # ── A. Triple-barrier labels ─────────────────────────────────────────────
    print("  Computing triple-barrier labels (k=1.5)...")
    y_tb = _compute_triple_barrier_labels(events, spx_df, k=1.5, vb_bars=60)

    # Fallback: if too few positives with triple-barrier, use fixed threshold
    if y_tb.sum() < 10:
        print(f"  WARNING: Triple-barrier produced only {y_tb.sum()} positives, falling back to fixed labels")
        y = y_fixed
        label_method = "fixed_threshold_fallback"
    else:
        y = y_tb
        label_method = "triple_barrier"

    print(f"  Features: {len(feature_names)}  |  Pos: {y.sum()}/{len(y)}  |  Labels: {label_method}")

    # ── G. Roll spread feature ───────────────────────────────────────────────
    print("  Computing Roll spread proxy...")
    roll_spreads = _compute_roll_spread(events, spx_df)
    X = np.column_stack([X, roll_spreads])
    feature_names.append("roll_spread")

    # ── F. VIX regime feature ────────────────────────────────────────────────
    vix_vals = np.array([float(ev.get("vix_prev") or 20.0) for ev in events])
    p33 = np.percentile(vix_vals, 33.33)
    p66 = np.percentile(vix_vals, 66.67)
    vix_regime = np.zeros(len(events), dtype=float)
    vix_regime[vix_vals >= p66] = 2.0
    vix_regime[(vix_vals >= p33) & (vix_vals < p66)] = 1.0
    X = np.column_stack([X, vix_regime])
    feature_names.append("vix_regime")

    # ── B. Time-of-day normalization ─────────────────────────────────────────
    print("  Applying time-of-day normalization...")
    X = _tod_normalize(X, events, feature_names, spx_df)

    # ── C. Robust rank scaling ───────────────────────────────────────────────
    print("  Applying robust rank scaling...")
    X = _rank_scale(X, window=100)

    # Replace any NaN/inf
    X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)

    # ── D+E. Stacking with purged CV + Platt calibration ─────────────────────
    if len(y) >= 100:
        print(f"  Training stacking ensemble with purged 6-fold CV (n={len(y)})...")
        oof_probs, full_base, meta, calibrator, fold_precisions, fold_top10s = \
            _stacking_purged_v2(X, y, feature_names, n_folds=6, embargo=5)
    else:
        print(f"  Training stacking ensemble with LOOCV (n={len(y)}, too small for k-fold)...")
        oof_probs, full_base, meta, calibrator, fold_precisions, fold_top10s = \
            _stacking_loocv_v2(X, y, feature_names)

    # ── Threshold tuning ─────────────────────────────────────────────────────
    threshold = _tune_threshold(y, oof_probs, min_recall=0.20)
    oof_preds = (oof_probs >= threshold).astype(int)

    tp = int(((oof_preds == 1) & (y == 1)).sum())
    fp = int(((oof_preds == 1) & (y == 0)).sum())
    tn = int(((oof_preds == 0) & (y == 0)).sum())
    fn = int(((oof_preds == 0) & (y == 1)).sum())
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    accuracy = round((tp + tn) / len(y), 3)

    # ── H. Feature stability ────────────────────────────────────────────────
    feature_stability = _track_feature_stability(fold_top10s)

    # ── F. VIX regime performance ────────────────────────────────────────────
    vix_regime_perf, _ = _vix_regime_metrics(events, y, oof_probs, threshold)

    # ── Spread regime performance ────────────────────────────────────────────
    spread_regime_perf = _spread_regime_metrics(roll_spreads, y, oof_probs, threshold)

    # ── Per-event predictions ────────────────────────────────────────────────
    event_preds = []
    for i, ev in enumerate(events):
        event_preds.append({
            "date": ev["date"],
            "time_of_low": ev.get("time_of_low", ""),
            "fall_pts": ev.get("fall_pts", 0),
            "reversal_pts": ev.get("reversal_pts", 0),
            "reversal_category": ev.get("reversal_category", "weak"),
            "squeeze_strength": ev.get("squeeze_strength", "none"),
            "vix_prev": ev.get("vix_prev"),
            "pred_prob_significant": round(float(oof_probs[i]), 3),
            "pred_significant": int(oof_probs[i] >= threshold),
            "actual_significant": int(y[i]),
            "hit": int(oof_preds[i] == y[i]),
            "gap_pts": float(ev.get("gap_pts", 0)),
            "gap_dir": int(ev.get("gap_dir", 0)),
        })

    # ── Squeeze impact ───────────────────────────────────────────────────────
    squeeze_map = {"none": 0, "no_data": 0, "mild": 1, "moderate": 2, "strong": 3}
    squeeze_impact: dict = {}
    for label, code in squeeze_map.items():
        bucket = [oof_probs[i] for i, ev in enumerate(events)
                  if squeeze_map.get(str(ev.get("squeeze_strength", "none")), 0) == code]
        if bucket:
            squeeze_impact[label] = round(float(np.mean(bucket)), 3)

    return {
        # Same keys as Model 1
        "n_events": len(events),
        "n_significant_reversals": int(y.sum()),
        "base_rate": round(float(y.mean()), 3),
        "target": "triple_barrier_PT_hit" if label_method == "triple_barrier" else "medium_or_strong (reversal_pts >= 20)",
        "models_used": list(full_base.keys()) + ["meta:logistic", "calibrator:platt"],
        "n_features": len(feature_names),
        "optimal_threshold": threshold,
        "loocv_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "feature_importance": _feat_importance(full_base, feature_names),
        "squeeze_impact": squeeze_impact,
        "event_predictions": event_preds,
        # Model 2 specific keys
        "label_method": label_method,
        "k_barrier": 1.5,
        "fold_precision_distribution": fold_precisions,
        "vix_regime_performance": vix_regime_perf,
        "spread_regime_performance": spread_regime_perf,
        "feature_stability_score": round(feature_stability, 3),
        "calibration_applied": True,
        "tod_normalization_applied": True,
        "robust_scaling_applied": True,
    }
