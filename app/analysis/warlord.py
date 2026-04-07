"""
Warlord Analysis v2 — EOD SPX Pin Prediction.

Strategy: Sell 1 ATM call + 1 ATM put, buy 1 call +wing, 1 put -wing.
Wing width configurable (20 or 25 pts). Max profit at ATM pin.

Improvements over v1:
  1. Multi-timepoint evaluation (12:00, 12:30, 1:00, 1:30 PM CST)
  2. Volume-profile POC (point-of-control) as pin magnet
  3. VWAP reversion features with VIX regime conditioning
  4. Stacking ensemble: RF + GB + LGBM → LogisticRegression meta-learner
  5. Heuristic execution scoring (VIX regime, range compression, straddle ratio)
  6. Feature selection (drop correlated >0.90)
  7. Bollinger position, momentum layers, improved microstructure

Research basis:
  - Dealer gamma hedging creates pin magnets at high-OI strikes (SSRN 519044)
  - VWAP reversion strengthens after 2pm on low-vol days
  - 0DTE straddle decay rate signals hedging pressure
  - Volume POC acts as secondary attractor when near gamma strike
"""
from __future__ import annotations

import copy
import warnings
from math import pi, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# ── constants ─────────────────────────────────────────────────────────────────

# Multi-timepoint prediction times (ET)
TIMEPOINTS_ET = {
    "12:00 CST": "13:00",
    "12:30 CST": "13:30",
    "1:00 CST": "14:00",
    "1:30 CST": "14:30",
}
PRIMARY_TIME_ET = "14:30"  # 2:30 PM ET = 1:30 PM CST
ZONE_WIDTH = 5
N_CANDIDATE_ZONES = 9
HOLDOUT_FRAC = 0.20
WING_WIDTH = 20


# ── helpers ───────────────────────────────────────────────────────────────────

def _prep_1min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df = df.copy()
    df.index = idx
    return df.between_time("09:30", "15:59")


def _realized_var(rets: np.ndarray) -> float:
    return float(np.sum(rets**2)) if len(rets) > 0 else 0.0


def _bipower_var(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 0.0
    mu1 = sqrt(2 / pi)
    return float(mu1**-2 * np.sum(np.abs(rets[1:]) * np.abs(rets[:-1])))


def _autocorr_lag1(rets: np.ndarray) -> float:
    if len(rets) < 4:
        return 0.0
    try:
        c = float(pd.Series(rets).autocorr(lag=1))
        return c if not np.isnan(c) else 0.0
    except Exception:
        return 0.0


def _round_to_zone(price: float, width: int = ZONE_WIDTH) -> float:
    return round(price / width) * width


def _rsi(closes: np.ndarray, n: int = 14) -> float:
    if len(closes) < n + 1:
        return 50.0
    diffs = np.diff(closes[-(n + 1):])
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


# ── volume profile ────────────────────────────────────────────────────────────

def _volume_poc(spy_to_T: pd.DataFrame) -> float | None:
    """Point of control: price level with highest volume (1-pt buckets)."""
    if spy_to_T.empty or "volume" not in spy_to_T.columns:
        return None
    c = spy_to_T["close"].values.astype(float)
    v = spy_to_T["volume"].values.astype(float)
    if len(c) < 10:
        return None
    # Bucket by rounded price
    buckets: dict[float, float] = {}
    for price, vol in zip(c, v):
        key = round(price, 0)
        buckets[key] = buckets.get(key, 0) + vol
    if not buckets:
        return None
    return max(buckets, key=buckets.get)


# ── per-day feature builder ───────────────────────────────────────────────────

def _build_day_features(
    date_str: str,
    spx_day: pd.DataFrame,
    spy_day: pd.DataFrame,
    vix_day: pd.DataFrame,
    vix_daily: pd.DataFrame | None,
    vix1d_df: pd.DataFrame | None,
    prev_close: float | None,
    cutoff_time: str = PRIMARY_TIME_ET,
) -> dict | None:
    """
    Build features from data up to cutoff_time on a single day.
    Returns None if insufficient data.
    """
    if spx_day.empty:
        return None

    spx_to_T = spx_day[spx_day.index.strftime("%H:%M") <= cutoff_time]
    if len(spx_to_T) < 30:
        return None

    # EOD close (target)
    eod_bar = spx_day[spx_day.index.strftime("%H:%M") >= "15:55"]
    if eod_bar.empty:
        return None
    eod_close = float(eod_bar["close"].iloc[-1])

    spot = float(spx_to_T["close"].iloc[-1])
    day_open = float(spx_day["open"].iloc[0])
    day_high = float(spx_to_T["high"].max())
    day_low = float(spx_to_T["low"].min())
    session_range = day_high - day_low
    closes = spx_to_T["close"].values.astype(float)
    highs = spx_to_T["high"].values.astype(float)
    lows = spx_to_T["low"].values.astype(float)
    opens = spx_to_T["open"].values.astype(float)
    rets = np.diff(np.log(closes))

    f: dict[str, float] = {}

    # ═══ A. STRIKE PINNING / DEALER POSITIONING ══════════════════════════════
    nearest_strike = _round_to_zone(spot, 5)
    f["nearest_strike_dist"] = spot - nearest_strike
    f["abs_strike_dist"] = abs(spot - nearest_strike)

    nearest_10 = _round_to_zone(spot, 10)
    f["dist_to_round_10"] = spot - nearest_10
    f["abs_dist_round_10"] = abs(spot - nearest_10)

    nearest_25 = _round_to_zone(spot, 25)
    f["dist_to_round_25"] = spot - nearest_25

    nearest_50 = _round_to_zone(spot, 50)
    f["dist_to_round_50"] = spot - nearest_50

    # Fractional zone position (0 = at strike, ±0.5 = edge)
    f["zone_position"] = (spot - nearest_strike) / max(ZONE_WIDTH, 1)

    # Approximate VWAP (equal-weight close mean as proxy)
    vwap = float(spx_to_T["close"].mean())
    f["vwap_to_nearest_strike"] = vwap - nearest_strike

    # Gamma-weighted pin score: inverse-distance weighting to nearby 5-pt strikes
    nearby_strikes = [nearest_strike + i * 5 for i in range(-4, 5)]
    inv_dists = [1.0 / max(abs(spot - s), 0.5) for s in nearby_strikes]
    total_inv = sum(inv_dists)
    f["gamma_pin_score"] = sum(s * w for s, w in zip(nearby_strikes, inv_dists)) / total_inv - spot

    # ═══ B. VWAP REVERSION & PRICE LOCATION ══════════════════════════════════
    f["spot_vs_vwap"] = spot - vwap
    f["spot_vs_vwap_pct"] = (spot - vwap) / max(abs(vwap), 1)
    f["spot_vs_midpoint"] = spot - (day_high + day_low) / 2
    f["spot_vs_open"] = spot - day_open

    # VWAP reversion signal: when close to VWAP and low vol, strong pin
    f["vwap_reversion_signal"] = 1.0 / (1.0 + abs(spot - vwap))

    # Time spent near dominant strike
    near_5 = np.abs(closes - nearest_strike) <= 5.0
    f["time_near_strike_5"] = float(near_5.sum()) / len(closes)
    near_10 = np.abs(closes - nearest_strike) <= 10.0
    f["time_near_strike_10"] = float(near_10.sum()) / len(closes)

    # Strike crossings
    above = closes > nearest_strike
    f["strike_crossings"] = float(np.sum(np.diff(above.astype(int)) != 0))

    # Noon-to-current drift
    noon_bars = spx_to_T[spx_to_T.index.strftime("%H:%M") <= "12:00"]
    f["noon_to_current"] = (spot - float(noon_bars["close"].iloc[-1])) if len(noon_bars) > 0 else 0.0

    # Trend slope
    x_reg = np.arange(len(closes))
    f["morning_trend_slope"] = float(np.polyfit(x_reg, closes, 1)[0]) if len(closes) > 5 else 0.0

    # Range compression: recent vs morning
    recent = spx_to_T[spx_to_T.index.strftime("%H:%M") > "13:30"]
    morning = spx_to_T[spx_to_T.index.strftime("%H:%M") <= "12:00"]
    if len(recent) > 5 and len(morning) > 5:
        f["range_compression"] = float(recent["high"].max() - recent["low"].min()) / max(
            float(morning["high"].max() - morning["low"].min()), 0.1
        )
    else:
        f["range_compression"] = 1.0

    # ═══ C. REALIZED VOLATILITY & MICROSTRUCTURE ═════════════════════════════
    f["rv_morning"] = float(np.std(rets)) if len(rets) > 1 else 0.0

    last_30 = closes[-30:]
    last_30_rets = np.diff(np.log(last_30)) if len(last_30) > 1 else np.array([])
    f["rv_last_30m"] = float(np.std(last_30_rets)) if len(last_30_rets) > 1 else 0.0

    last_60 = closes[-60:]
    last_60_rets = np.diff(np.log(last_60)) if len(last_60) > 1 else np.array([])
    f["rv_last_60m"] = float(np.std(last_60_rets)) if len(last_60_rets) > 1 else 0.0

    f["vol_accel"] = f["rv_last_30m"] / max(f["rv_morning"], 1e-6)
    f["vol_accel_60"] = f["rv_last_60m"] / max(f["rv_morning"], 1e-6)

    # Jump proxy
    f["jump_proxy"] = max(_realized_var(rets) - _bipower_var(rets), 0) / max(
        _realized_var(rets), 1e-10
    )

    # Return autocorrelation (negative = mean-reverting = pin friendly)
    f["ret_autocorr"] = _autocorr_lag1(rets)
    f["ret_autocorr_30"] = _autocorr_lag1(last_30_rets)

    # ═══ D. BOLLINGER / RSI / TECHNICAL ══════════════════════════════════════
    if len(closes) >= 20:
        bb_mean = float(np.mean(closes[-20:]))
        bb_std = float(np.std(closes[-20:]))
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std
        f["bb_pct_b"] = (spot - bb_lower) / max(bb_upper - bb_lower, 0.01)
        f["bb_bandwidth"] = (bb_upper - bb_lower) / max(bb_mean, 1.0)
    else:
        f["bb_pct_b"] = 0.5
        f["bb_bandwidth"] = 0.0

    f["rsi_14"] = _rsi(closes)

    # Price momentum layers
    for lookback in (15, 30, 60, 120):
        if len(closes) > lookback:
            f[f"momentum_{lookback}m"] = spot - float(closes[-lookback])
        else:
            f[f"momentum_{lookback}m"] = 0.0

    # ═══ E. VIX / IMPLIED VOLATILITY ═════════════════════════════════════════
    if not vix_day.empty:
        vix_to_T = vix_day[vix_day.index.strftime("%H:%M") <= cutoff_time]
        if len(vix_to_T) > 0:
            f["vix_level"] = float(vix_to_T["close"].iloc[-1])
            f["vix_change_from_open"] = float(vix_to_T["close"].iloc[-1] - vix_to_T["close"].iloc[0])
            # VIX regime flags
            f["vix_low"] = float(f["vix_level"] < 16)
            f["vix_mid"] = float(16 <= f["vix_level"] < 24)
            f["vix_high"] = float(f["vix_level"] >= 24)
            # VIX acceleration (last 30m)
            vix_closes = vix_to_T["close"].values.astype(float)
            if len(vix_closes) >= 30:
                f["vix_chg_30m"] = float(vix_closes[-1] - vix_closes[-30])
            else:
                f["vix_chg_30m"] = 0.0
        else:
            f["vix_level"] = 20.0
            f["vix_change_from_open"] = 0.0
            f["vix_low"] = 0.0
            f["vix_mid"] = 1.0
            f["vix_high"] = 0.0
            f["vix_chg_30m"] = 0.0
    else:
        f["vix_level"] = 20.0
        f["vix_change_from_open"] = 0.0
        f["vix_low"] = 0.0
        f["vix_mid"] = 1.0
        f["vix_high"] = 0.0
        f["vix_chg_30m"] = 0.0

    # Straddle implied move / VIX1D
    if vix1d_df is not None and date_str in vix1d_df.index:
        row_v = vix1d_df.loc[date_str]
        straddle = float(row_v.get("straddle_price", 0)) if pd.notna(row_v.get("straddle_price")) else 0.0
        f["straddle_implied_move"] = straddle
        f["straddle_vs_range"] = straddle / max(session_range, 0.1) if straddle > 0 else 1.0
        # Straddle decay: compare implied move to remaining range potential
        f["straddle_decay_ratio"] = session_range / max(straddle, 0.1)
    else:
        daily_sigma = spot * f["vix_level"] / (100 * sqrt(252))
        f["straddle_implied_move"] = daily_sigma
        f["straddle_vs_range"] = daily_sigma / max(session_range, 0.1)
        f["straddle_decay_ratio"] = session_range / max(daily_sigma, 0.1)

    # ═══ F. GAP / STRUCTURE ══════════════════════════════════════════════════
    if prev_close:
        f["gap_pts"] = day_open - prev_close
        f["dist_from_prev_close"] = spot - prev_close
        f["abs_gap"] = abs(day_open - prev_close)
    else:
        f["gap_pts"] = 0.0
        f["dist_from_prev_close"] = 0.0
        f["abs_gap"] = 0.0

    # ═══ G. VOLUME PROFILE (SPY) ═════════════════════════════════════════════
    spy_to_T = spy_day[spy_day.index.strftime("%H:%M") <= cutoff_time] if not spy_day.empty else pd.DataFrame()

    if not spy_to_T.empty and "volume" in spy_to_T.columns and len(spy_to_T) > 30:
        spy_v = spy_to_T["volume"].values.astype(float)
        spy_c = spy_to_T["close"].values.astype(float)

        # Volume POC (SPY price level with highest volume)
        poc = _volume_poc(spy_to_T)
        if poc is not None:
            # Scale POC to SPX: SPY POC * (SPX/SPY ratio)
            spy_spot = float(spy_c[-1])
            spx_poc_estimate = poc * (spot / max(spy_spot, 1))
            f["poc_dist"] = spot - spx_poc_estimate
            f["abs_poc_dist"] = abs(spot - spx_poc_estimate)
        else:
            f["poc_dist"] = 0.0
            f["abs_poc_dist"] = 0.0

        # Signed order-flow imbalance
        dc = np.diff(spy_c[-30:])
        signs = np.sign(dc)
        v_chg = spy_v[-len(dc):][:len(dc)]
        total_v = float(v_chg.sum())
        f["signed_flow_imb"] = float(np.dot(signs, v_chg)) / max(total_v, 1.0)

        # Volume concentration ratio (are we at a high-volume level?)
        # How much of today's volume traded near current price
        near_vol_mask = np.abs(spy_c - spy_spot) <= 0.5  # within $0.50 of SPY spot
        f["vol_concentration_at_spot"] = float(spy_v[near_vol_mask].sum()) / max(total_v, 1.0)
    else:
        f["poc_dist"] = 0.0
        f["abs_poc_dist"] = 0.0
        f["signed_flow_imb"] = 0.0
        f["vol_concentration_at_spot"] = 0.0

    # ═══ H. TIME / CALENDAR ══════════════════════════════════════════════════
    try:
        dt = pd.Timestamp(date_str)
        f["dow"] = float(dt.dayofweek)
        f["is_friday"] = float(dt.dayofweek == 4)
        f["is_monday"] = float(dt.dayofweek == 0)
    except Exception:
        f["dow"] = 2.0
        f["is_friday"] = 0.0
        f["is_monday"] = 0.0

    # ═══ I. INTERACTION TERMS ════════════════════════════════════════════════
    f["vwap_x_low_vol"] = f["vwap_reversion_signal"] * f["vix_low"]
    f["strike_time_x_compression"] = f["time_near_strike_5"] * (1 - f["range_compression"])
    f["autocorr_x_strike_dist"] = f["ret_autocorr"] * f["abs_strike_dist"]

    # ═══ HEURISTIC EXECUTION SCORE ═══════════════════════════════════════════
    # Score 0-100: higher = better conditions for warlord
    score = 50.0
    # Low VIX → pin-friendly (+15)
    if f["vix_level"] < 16:
        score += 15
    elif f["vix_level"] < 20:
        score += 8
    elif f["vix_level"] > 28:
        score -= 15
    # Range compression < 0.5 → consolidating (+12)
    if f["range_compression"] < 0.5:
        score += 12
    elif f["range_compression"] < 0.8:
        score += 5
    elif f["range_compression"] > 1.5:
        score -= 10
    # Close to VWAP → pin attractor (+10)
    if abs(f["spot_vs_vwap"]) < 3:
        score += 10
    elif abs(f["spot_vs_vwap"]) < 7:
        score += 5
    # Close to strike → already pinning (+10)
    if f["abs_strike_dist"] < 1.5:
        score += 10
    elif f["abs_strike_dist"] < 3:
        score += 5
    # Straddle mostly consumed → limited remaining move (+8)
    if f["straddle_decay_ratio"] > 1.2:
        score += 8
    elif f["straddle_decay_ratio"] > 0.8:
        score += 3
    # Negative autocorrelation → mean-reverting (+5)
    if f["ret_autocorr"] < -0.1:
        score += 5
    # High volume at spot → price anchored (+5)
    if f["vol_concentration_at_spot"] > 0.15:
        score += 5
    score = max(0, min(100, score))

    return {
        "features": f,
        "spot_at_T": spot,
        "eod_close": eod_close,
        "date": date_str,
        "nearest_strike": nearest_strike,
        "vwap": vwap,
        "session_range": session_range,
        "heuristic_score": round(score, 1),
    }


# ── zone construction ─────────────────────────────────────────────────────────

def _build_candidate_zones(spot: float) -> list[float]:
    center = _round_to_zone(spot, ZONE_WIDTH)
    half = N_CANDIDATE_ZONES // 2
    return [center + i * ZONE_WIDTH for i in range(-half, half + 1)]


def _zone_label(eod_close: float, zones: list[float]) -> int:
    dists = [abs(eod_close - z) for z in zones]
    return int(np.argmin(dists))


# ── warlord payoff ────────────────────────────────────────────────────────────

def _warlord_payoff(atm: float, close: float, credit: float, wing: float = WING_WIDTH) -> float:
    """Short iron butterfly: sell 1 ATM straddle, buy 1 strangle at ±wing."""
    intrinsic = -max(close - atm, 0) - max(atm - close, 0)
    protection = max(close - atm - wing, 0) + max(atm - wing - close, 0)
    return credit + intrinsic + protection


# ── feature selection ─────────────────────────────────────────────────────────

def _select_features(
    X: np.ndarray, y: np.ndarray, names: list[str],
    max_features: int = 35, corr_threshold: float = 0.90,
) -> tuple[np.ndarray, list[str]]:
    """Drop correlated features, keep top by RF importance."""
    if X.shape[1] <= max_features:
        return X, names

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    importances = rf.feature_importances_

    # Drop correlated
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr)
    dropped = set()
    for i in range(len(names)):
        if i in dropped:
            continue
        for j in range(i + 1, len(names)):
            if j in dropped:
                continue
            if abs(corr[i, j]) > corr_threshold:
                if importances[i] >= importances[j]:
                    dropped.add(j)
                else:
                    dropped.add(i)
                    break

    surviving = [i for i in range(len(names)) if i not in dropped]
    surviving.sort(key=lambda i: importances[i], reverse=True)
    selected = sorted(surviving[:max_features])

    return X[:, selected], [names[i] for i in selected]


# ── stacking ensemble ─────────────────────────────────────────────────────────

def _build_base_models() -> dict:
    m = {}
    m["rf"] = RandomForestClassifier(
        n_estimators=250, max_depth=5, min_samples_leaf=6,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    m["gb"] = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.04,
        min_samples_leaf=6, subsample=0.75, random_state=42,
    )
    if _HAS_LGBM:
        m["lgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.04,
            num_leaves=12, min_child_samples=8, subsample=0.75,
            class_weight="balanced", random_state=42, verbosity=-1,
        )
    return m


def _train_stacking(X_train, y_train, n_folds=5, embargo=3):
    """
    Purged expanding-window CV on train set → OOF base proba → meta-learner.
    Returns (base_models_dict, meta_model).
    """
    templates = _build_base_models()
    names = list(templates.keys())
    n = len(y_train)
    step = n // (n_folds + 1)

    oof_base = np.full((n, len(names) * N_CANDIDATE_ZONES), 1.0 / N_CANDIDATE_ZONES)

    for fold in range(1, n_folds + 1):
        test_start = fold * step
        test_end = min(test_start + step, n)
        train_end = max(test_start - embargo, 0)
        if train_end < 10 or test_end <= test_start:
            continue
        tr = np.arange(0, train_end)
        te = np.arange(test_start, test_end)
        y_tr = y_train[tr]
        if len(np.unique(y_tr)) < 2:
            continue

        for j, (nm, tmpl) in enumerate(templates.items()):
            m = copy.deepcopy(tmpl)
            m.fit(X_train[tr], y_tr)
            proba = m.predict_proba(X_train[te])
            full_proba = np.zeros((len(te), N_CANDIDATE_ZONES))
            for ci, cls in enumerate(m.classes_):
                if 0 <= cls < N_CANDIDATE_ZONES:
                    full_proba[:, cls] = proba[:, ci]
            oof_base[te, j * N_CANDIDATE_ZONES:(j + 1) * N_CANDIDATE_ZONES] = full_proba

    # Meta-learner on OOF
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="lbfgs")
    meta.fit(oof_base, y_train)

    # Retrain base models on full train set
    final_base = {}
    for nm, tmpl in templates.items():
        m = copy.deepcopy(tmpl)
        m.fit(X_train, y_train)
        final_base[nm] = m

    return final_base, meta


def _predict_stacking(base_models, meta, X):
    """Get zone probabilities from stacking ensemble."""
    names = list(base_models.keys())
    base_proba = np.zeros((len(X), len(names) * N_CANDIDATE_ZONES))
    for j, (nm, m) in enumerate(base_models.items()):
        proba = m.predict_proba(X)
        full_proba = np.zeros((len(X), N_CANDIDATE_ZONES))
        for ci, cls in enumerate(m.classes_):
            if 0 <= cls < N_CANDIDATE_ZONES:
                full_proba[:, cls] = proba[:, ci]
        base_proba[:, j * N_CANDIDATE_ZONES:(j + 1) * N_CANDIDATE_ZONES] = full_proba

    meta_proba = meta.predict_proba(base_proba)
    # Align to N_CANDIDATE_ZONES
    full = np.zeros((len(X), N_CANDIDATE_ZONES))
    for ci, cls in enumerate(meta.classes_):
        if 0 <= cls < N_CANDIDATE_ZONES:
            full[:, cls] = meta_proba[:, ci]
    # Normalize
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return full / row_sums


# ── evaluate predictions ──────────────────────────────────────────────────────

def _evaluate(samples, zones_list, proba, y):
    """Compute per-day predictions and aggregate metrics."""
    predictions = []
    top1_hits = top3_hits = 0
    distances = []
    pnls = []

    for i, s in enumerate(samples):
        zones = zones_list[i]
        probs = proba[i]
        actual_close = s["eod_close"]
        actual_zone_idx = y[i]

        top_indices = np.argsort(probs)[::-1][:3]
        top_zones = [
            {"strike": zones[idx], "probability": round(float(probs[idx]), 3), "zone_idx": int(idx)}
            for idx in top_indices
        ]

        predicted_zone_idx = top_indices[0]
        top1_hit = int(predicted_zone_idx == actual_zone_idx)
        top3_hit = int(actual_zone_idx in top_indices[:3])
        top1_hits += top1_hit
        top3_hits += top3_hit

        predicted_strike = zones[predicted_zone_idx]
        dist = abs(actual_close - predicted_strike)
        distances.append(dist)

        straddle_credit = s["features"].get("straddle_implied_move", 10.0)
        credit = straddle_credit * 0.8
        pnl = _warlord_payoff(predicted_strike, actual_close, credit)
        pnls.append(pnl)

        feat = s["features"]
        predictions.append({
            "date": s["date"],
            "spot_at_T": round(s["spot_at_T"], 2),
            "eod_close": round(actual_close, 2),
            "nearest_strike_at_T": s["nearest_strike"],
            "predicted_zones": top_zones,
            "actual_zone_strike": zones[actual_zone_idx],
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "distance_to_actual": round(dist, 2),
            "warlord_pnl": round(pnl, 2),
            "credit": round(credit, 2),
            "heuristic_score": s.get("heuristic_score", 50),
            # 6 segmentation dimensions
            "vix_level": round(feat.get("vix_level", 20), 1),
            "range_compression": round(feat.get("range_compression", 1), 2),
            "straddle_vs_range": round(feat.get("straddle_vs_range", 1), 2),
            "ret_autocorr": round(feat.get("ret_autocorr", 0), 3),
            "abs_strike_dist": round(feat.get("abs_strike_dist", 2.5), 2),
            "dow": int(feat.get("dow", 2)),
        })

    n = len(y)
    return {
        "top1_accuracy": round(top1_hits / n, 3),
        "top3_accuracy": round(top3_hits / n, 3),
        "avg_distance": round(float(np.mean(distances)), 2),
        "median_distance": round(float(np.median(distances)), 2),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(float(np.mean(pnls)), 2),
        "win_rate": round(sum(1 for p in pnls if p > 0) / n, 3),
        "cumulative_pnl": [round(sum(pnls[:i + 1]), 2) for i in range(n)],
        "predictions": predictions,
    }


# ── main entry ────────────────────────────────────────────────────────────────

def run_warlord_analysis(
    spx_1min: pd.DataFrame,
    vix_daily: pd.DataFrame | None = None,
    spy_1min: pd.DataFrame | None = None,
    vix_1min: pd.DataFrame | None = None,
    vix1d_df: pd.DataFrame | None = None,
    holdout_frac: float = HOLDOUT_FRAC,
) -> dict:
    spx = _prep_1min(spx_1min)
    spy = _prep_1min(spy_1min) if spy_1min is not None and not spy_1min.empty else pd.DataFrame()
    vix = _prep_1min(vix_1min) if vix_1min is not None and not vix_1min.empty else pd.DataFrame()

    spx_daily = {str(d): g for d, g in spx.groupby(spx.index.date)}
    spy_daily = {str(d): g for d, g in spy.groupby(spy.index.date)} if not spy.empty else {}
    vix_daily_1min = {str(d): g for d, g in vix.groupby(vix.index.date)} if not vix.empty else {}
    sorted_dates = sorted(spx_daily.keys())

    # ── 1. Build samples at primary time ──────────────────────────────────────
    print("  Building per-day features (primary: 1:30 PM CST)...")
    samples = []
    for i, date_str in enumerate(sorted_dates):
        prev_close = None
        if i > 0:
            prev_day = spx_daily[sorted_dates[i - 1]]
            if not prev_day.empty:
                prev_close = float(prev_day["close"].iloc[-1])

        result = _build_day_features(
            date_str, spx_daily[date_str],
            spy_daily.get(date_str, pd.DataFrame()),
            vix_daily_1min.get(date_str, pd.DataFrame()),
            vix_daily, vix1d_df, prev_close, PRIMARY_TIME_ET,
        )
        if result is not None:
            samples.append(result)

    n_samples = len(samples)
    print(f"  Built {n_samples} day samples")
    if n_samples < 30:
        return {"error": f"Not enough samples ({n_samples}), need >=30"}

    # ── 2. Feature matrix + zone labels ───────────────────────────────────────
    feature_names = list(samples[0]["features"].keys())
    X = np.array([[s["features"][f] for f in feature_names] for s in samples])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    zone_labels = []
    all_zones = []
    for s in samples:
        zones = _build_candidate_zones(s["spot_at_T"])
        zone_labels.append(_zone_label(s["eod_close"], zones))
        all_zones.append(zones)
    y = np.array(zone_labels)

    # ── 3. Temporal split ─────────────────────────────────────────────────────
    split_idx = int(n_samples * (1 - holdout_frac))
    split_idx = min(split_idx, n_samples - 10)
    split_idx = max(split_idx, 20)

    X_train_full, X_hold_full = X[:split_idx], X[split_idx:]
    y_train, y_hold = y[:split_idx], y[split_idx:]
    samples_train = samples[:split_idx]
    samples_hold = samples[split_idx:]
    zones_train = all_zones[:split_idx]
    zones_hold = all_zones[split_idx:]

    print(f"  Train: {len(y_train)} days ({samples_train[0]['date']} to {samples_train[-1]['date']})")
    print(f"  Holdout: {len(y_hold)} days ({samples_hold[0]['date']} to {samples_hold[-1]['date']})")

    # ── 4. Feature selection (train only) ─────────────────────────────────────
    print(f"  Feature selection: {len(feature_names)} features → ", end="")
    X_train, sel_names = _select_features(X_train_full, y_train, feature_names)
    sel_idx = [feature_names.index(n) for n in sel_names]
    X_hold = X_hold_full[:, sel_idx]
    print(f"{len(sel_names)} selected")

    # ── 5. Train stacking ensemble ────────────────────────────────────────────
    print("  Training stacking ensemble (purged 5-fold CV)...")
    base_models, meta = _train_stacking(X_train, y_train)

    # ── 6. Predict holdout ────────────────────────────────────────────────────
    print("  Predicting on holdout...")
    hold_proba = _predict_stacking(base_models, meta, X_hold)
    hold_results = _evaluate(samples_hold, zones_hold, hold_proba, y_hold)

    # Train metrics for overfit check
    train_proba = _predict_stacking(base_models, meta, X_train)
    train_preds = np.argmax(train_proba, axis=1)
    train_top1 = round(float(accuracy_score(y_train, train_preds)), 3)

    # ── 7. Feature importance ─────────────────────────────────────────────────
    imps = [m.feature_importances_ for m in base_models.values() if hasattr(m, "feature_importances_")]
    avg_imp = np.mean(imps, axis=0) if imps else np.zeros(len(sel_names))
    feat_importance = sorted(
        [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(sel_names, avg_imp)],
        key=lambda x: x["importance"], reverse=True,
    )

    # ── 8. Multi-timepoint evaluation ─────────────────────────────────────────
    print("  Multi-timepoint evaluation...")
    timepoint_results = {}
    for label, cutoff in TIMEPOINTS_ET.items():
        tp_samples = []
        for i, date_str in enumerate(sorted_dates):
            prev_close = None
            if i > 0:
                prev_day = spx_daily[sorted_dates[i - 1]]
                if not prev_day.empty:
                    prev_close = float(prev_day["close"].iloc[-1])
            result = _build_day_features(
                date_str, spx_daily[date_str],
                spy_daily.get(date_str, pd.DataFrame()),
                vix_daily_1min.get(date_str, pd.DataFrame()),
                vix_daily, vix1d_df, prev_close, cutoff,
            )
            if result is not None:
                tp_samples.append(result)

        if len(tp_samples) < n_samples * 0.9:
            continue

        # Align to same dates as primary
        primary_dates = {s["date"] for s in samples}
        tp_samples = [s for s in tp_samples if s["date"] in primary_dates]

        if len(tp_samples) != n_samples:
            continue

        tp_X = np.array([[s["features"][f] for f in feature_names] for s in tp_samples])
        tp_X = np.nan_to_num(tp_X, nan=0.0, posinf=1e6, neginf=-1e6)
        tp_X_hold = tp_X[split_idx:][:, sel_idx]

        tp_zones = []
        tp_y = []
        for s in tp_samples[split_idx:]:
            z = _build_candidate_zones(s["spot_at_T"])
            tp_zones.append(z)
            tp_y.append(_zone_label(s["eod_close"], z))
        tp_y = np.array(tp_y)

        tp_proba = _predict_stacking(base_models, meta, tp_X_hold)
        tp_eval = _evaluate(tp_samples[split_idx:], tp_zones, tp_proba, tp_y)

        timepoint_results[label] = {
            "top1_accuracy": tp_eval["top1_accuracy"],
            "top3_accuracy": tp_eval["top3_accuracy"],
            "avg_distance": tp_eval["avg_distance"],
            "median_distance": tp_eval["median_distance"],
            "avg_pnl": tp_eval["avg_pnl"],
        }

    # ── 9. 6-dimension heuristic segmentation ───────────────────────────────
    print("  6-dimension heuristic segmentation...")
    preds = hold_results["predictions"]

    def _segment_stats(subset):
        n = len(subset)
        if n < 2:
            return None
        return {
            "n": n,
            "top1_accuracy": round(sum(p["top1_hit"] for p in subset) / n, 3),
            "top3_accuracy": round(sum(p["top3_hit"] for p in subset) / n, 3),
            "avg_distance": round(sum(p["distance_to_actual"] for p in subset) / n, 2),
            "avg_pnl": round(sum(p["warlord_pnl"] for p in subset) / n, 2),
            "win_rate": round(sum(1 for p in subset if p["warlord_pnl"] > 0) / n, 3),
        }

    # DIM 1: VIX Regime
    vix_regime_results = {}
    for label, lo, hi in [("Low (<16)", 0, 16), ("Mid (16-24)", 16, 24), ("High (>24)", 24, 999)]:
        s = _segment_stats([p for p in preds if lo <= p["vix_level"] < hi])
        if s:
            vix_regime_results[label] = s

    # DIM 2: Range Compression (afternoon vs morning range)
    range_comp_results = {}
    for label, lo, hi in [("Compressed (<0.5)", 0, 0.5), ("Normal (0.5-1.0)", 0.5, 1.0), ("Expanding (>1.0)", 1.0, 999)]:
        s = _segment_stats([p for p in preds if lo <= p["range_compression"] < hi])
        if s:
            range_comp_results[label] = s

    # DIM 3: Straddle Decay (how much of implied move is consumed)
    straddle_results = {}
    for label, lo, hi in [("Mostly Left (sv<0.6)", 0, 0.6), ("Partial (0.6-1.0)", 0.6, 1.0), ("Consumed (>1.0)", 1.0, 999)]:
        s = _segment_stats([p for p in preds if lo <= p["straddle_vs_range"] < hi])
        if s:
            straddle_results[label] = s

    # DIM 4: Trend State (return autocorrelation)
    trend_results = {}
    for label, lo, hi in [("Mean-Revert (<-0.05)", -999, -0.05), ("Neutral", -0.05, 0.05), ("Trending (>0.05)", 0.05, 999)]:
        s = _segment_stats([p for p in preds if lo <= p["ret_autocorr"] < hi])
        if s:
            trend_results[label] = s

    # DIM 5: Strike Proximity (how close spot is to nearest 5-pt strike)
    strike_prox_results = {}
    for label, lo, hi in [("At Strike (<1pt)", 0, 1), ("Near (1-2.5pt)", 1, 2.5), ("Far (>2.5pt)", 2.5, 999)]:
        s = _segment_stats([p for p in preds if lo <= p["abs_strike_dist"] < hi])
        if s:
            strike_prox_results[label] = s

    # DIM 6: Day of Week
    dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    dow_results = {}
    for dow_val, dow_label in dow_names.items():
        s = _segment_stats([p for p in preds if p["dow"] == dow_val])
        if s:
            dow_results[dow_label] = s

    # Composite heuristic score buckets
    heuristic_buckets = {}
    for label, lo, hi in [("Excellent (>75)", 75, 101), ("Good (60-75)", 60, 75),
                          ("Fair (45-60)", 45, 60), ("Poor (<45)", 0, 45)]:
        s = _segment_stats([p for p in preds if lo <= p["heuristic_score"] < hi])
        if s:
            heuristic_buckets[label] = s

    # ── 10. Execution heuristics summary ──────────────────────────────────────
    heuristics = [
        {"rule": "VIX < 16", "mechanism": "Low vol = dealer gamma dominant, strong pin tendency",
         "action": "SELL warlord aggressively at top-1 predicted strike"},
        {"rule": "VIX 16-20 + range compression < 0.5", "mechanism": "Moderate vol but consolidating into close",
         "action": "SELL warlord at top-1, reduce size 25%"},
        {"rule": "VIX > 24", "mechanism": "High vol = breakout risk, pin weak",
         "action": "SKIP or use wider wings (25pts)"},
        {"rule": "Straddle decay ratio > 1.2", "mechanism": "Session range already exceeded implied move, limited upside move left",
         "action": "SELL warlord — move is mostly done"},
        {"rule": "Spot within 2pts of VWAP", "mechanism": "VWAP reversion + dealer hedging = double magnet",
         "action": "SELL warlord at nearest strike to VWAP"},
        {"rule": "Strike crossings > 5", "mechanism": "Price oscillating around a level = pinning in progress",
         "action": "SELL warlord at the crossed strike"},
        {"rule": "Heuristic score > 75", "mechanism": "Multiple pin-friendly conditions aligned",
         "action": "SELL max size — best conditions"},
        {"rule": "Heuristic score < 45", "mechanism": "Trending / high vol / away from strikes",
         "action": "SKIP — wait for better day"},
    ]

    return {
        "prediction_time": "1:30 PM CST / 2:30 PM ET",
        "zone_width": ZONE_WIDTH,
        "n_candidate_zones": N_CANDIDATE_ZONES,
        "wing_width": WING_WIDTH,
        "n_total_days": n_samples,
        "n_train": len(y_train),
        "n_holdout": len(y_hold),
        "train_dates": f"{samples_train[0]['date']} to {samples_train[-1]['date']}",
        "holdout_dates": f"{samples_hold[0]['date']} to {samples_hold[-1]['date']}",
        "models_used": list(base_models.keys()) + ["meta:logistic"],
        "n_features_original": len(feature_names),
        "n_features_selected": len(sel_names),
        "selected_features": sel_names,
        "feature_names": feature_names,
        # Primary holdout metrics
        "holdout_top1_accuracy": hold_results["top1_accuracy"],
        "holdout_top3_accuracy": hold_results["top3_accuracy"],
        "holdout_avg_distance": hold_results["avg_distance"],
        "holdout_median_distance": hold_results["median_distance"],
        "train_top1_accuracy": train_top1,
        # P&L
        "warlord_total_pnl": hold_results["total_pnl"],
        "warlord_avg_pnl": hold_results["avg_pnl"],
        "warlord_win_rate": hold_results["win_rate"],
        "warlord_cumulative_pnl": hold_results["cumulative_pnl"],
        # Multi-timepoint
        "timepoint_comparison": timepoint_results,
        # 6-dimension segmentation
        "vix_regime_results": vix_regime_results,
        "range_comp_results": range_comp_results,
        "straddle_results": straddle_results,
        "trend_results": trend_results,
        "strike_prox_results": strike_prox_results,
        "dow_results": dow_results,
        "heuristic_buckets": heuristic_buckets,
        # Execution rules
        "heuristics": heuristics,
        # Features
        "feature_importance": feat_importance,
        "predictions": hold_results["predictions"],
    }
