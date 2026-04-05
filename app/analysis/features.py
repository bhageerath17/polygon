"""
Comprehensive feature engineering for intraday SPX reversal prediction.

For each reversal event at (date, time_of_low = T), features are computed
from data in the lookback window [T - 60 min, T] — no lookahead.

Feature groups (matches research taxonomy):
  A  Price momentum / overextension vs moving average
  B  Realized variance, jump proxy, semivariance
  C  Volume / order-flow proxies (SPY as volume source)
  D  Candle patterns + technical oscillators
  E  VIX intraday + cross-signal SPX-VIX
  F  Market structure anchors (VWAP, gap, key levels)
  G  Opening range + gap features
  H  Options-derived: straddle proxy + implied-realized spread
  I  Existing squeeze / options features from reversal analysis
  J  Time-of-day encoding
  K  Interaction terms
"""
from __future__ import annotations

import warnings
from math import pi, sqrt

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── helpers ────────────────────────────────────────────────────────────────────

_SQRT_2_PI = sqrt(2 / pi)   # bipower variation constant


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to RTH 09:30–16:00 ET."""
    return df.between_time("09:30", "15:59")


def _prep_1min(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeTZ-aware ET index, RTH only."""
    if df.empty:
        return df
    # parse with utc=True to handle mixed-tz strings (e.g. "-05:00" / "-04:00")
    idx = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df = df.copy()
    df.index = idx
    return _rth(df)


def _realized_var(rets: np.ndarray) -> float:
    return float(np.sum(rets ** 2)) if len(rets) > 0 else 0.0


def _bipower_var(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 0.0
    return float(_SQRT_2_PI ** -2 * np.sum(np.abs(rets[1:]) * np.abs(rets[:-1])))


def _ema(prices: np.ndarray, span: int) -> float:
    if len(prices) == 0:
        return float("nan")
    alpha = 2 / (span + 1)
    val = prices[0]
    for p in prices[1:]:
        val = alpha * p + (1 - alpha) * val
    return float(val)


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
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def _stoch_k(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, n: int = 14) -> float:
    if len(closes) < n:
        return 50.0
    c, h, l = closes[-1], highs[-n:].max(), lows[-n:].min()
    if h == l:
        return 50.0
    return float((c - l) / (h - l) * 100)


def _macd_hist(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    if len(closes) < slow + signal:
        return 0.0
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    macd = ema_fast - ema_slow
    sig  = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
    return float(macd[-1] - sig[-1])


def _slope_and_curvature(prices: np.ndarray, n: int) -> tuple[float, float]:
    """OLS slope and curvature (2nd diff of slope) of log prices over n bars."""
    if len(prices) < n:
        n = len(prices)
    if n < 3:
        return 0.0, 0.0
    lp = np.log(prices[-n:])
    x  = np.arange(n)
    # Slope
    slope = float(np.polyfit(x, lp, 1)[0])
    # Curvature: fit quadratic, take 2nd coeff
    curvature = float(np.polyfit(x, lp, 2)[0])
    return slope, curvature


def _autocorr_lag1(rets: np.ndarray) -> float:
    if len(rets) < 4:
        return 0.0
    try:
        c = float(pd.Series(rets).autocorr(lag=1))
        return c if not np.isnan(c) else 0.0
    except Exception:
        return 0.0


# ── time-of-day volume baseline ────────────────────────────────────────────────

def _build_tod_vol_baseline(spy: pd.DataFrame) -> dict[int, float]:
    """Compute expected volume by minute-of-day from all historical SPY bars."""
    if spy.empty or "volume" not in spy.columns:
        return {}
    s = spy.copy()
    s["minute_of_day"] = s.index.hour * 60 + s.index.minute
    baseline = s.groupby("minute_of_day")["volume"].median().to_dict()
    return baseline


# ── main feature builder ───────────────────────────────────────────────────────

SQUEEZE_MAP = {"none": 0, "no_data": 0, "mild": 1, "moderate": 2, "strong": 3}
DOW_MAP     = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}


def build_feature_matrix(
    events: list[dict],
    spx_1min: pd.DataFrame,
    vix_daily: pd.DataFrame | None = None,
    spy_1min:  pd.DataFrame | None = None,
    vix_1min:  pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    For each event build a feature vector at T = time_of_low.
    Returns (X, y, feature_names).
    """
    spx = _prep_1min(spx_1min)
    spy = _prep_1min(spy_1min) if (spy_1min is not None and not spy_1min.empty) else None
    vix = _prep_1min(vix_1min) if (vix_1min is not None and not vix_1min.empty) else None

    # Pre-compute time-of-day volume baseline for SPY
    tod_baseline = _build_tod_vol_baseline(spy) if spy is not None else {}

    # Pre-build daily group dicts for O(1) per-event lookup
    spx_daily = {str(d): g for d, g in spx.groupby(spx.index.date)}
    spy_daily = ({str(d): g for d, g in spy.groupby(spy.index.date)} if spy is not None else {})
    vix_daily_1min = ({str(d): g for d, g in vix.groupby(vix.index.date)} if vix is not None else {})

    rows: list[dict] = []

    for ev in events:
        row = _compute_event_features(
            ev, spx_daily, spy_daily, vix_daily_1min,
            vix_daily, tod_baseline,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Target: 1 if medium or strong reversal (>= 20 pts)
    y = np.array([
        int(ev["reversal_category"] in ("medium", "strong"))
        for ev in events
    ])

    feature_names = [c for c in df.columns]
    X = df.fillna(0.0).values.astype(float)

    return X, y, feature_names


def _compute_event_features(
    ev: dict,
    spx_daily: dict,
    spy_daily:  dict,
    vix_daily_1min: dict,
    vix_daily:  pd.DataFrame | None,
    tod_baseline: dict,
) -> dict:
    date_str = str(ev.get("date", ""))
    time_str = str(ev.get("time_of_low", "12:00:00"))[:5]   # "HH:MM"

    spx_day = spx_daily.get(date_str, pd.DataFrame())
    spy_day = spy_daily.get(date_str, pd.DataFrame())
    vix_day = vix_daily_1min.get(date_str, pd.DataFrame())

    # Bars up to and including T
    spx_at_T  = spx_day[spx_day.index.strftime("%H:%M") <= time_str] if not spx_day.empty else pd.DataFrame()
    spy_at_T  = spy_day[spy_day.index.strftime("%H:%M") <= time_str] if not spy_day.empty else pd.DataFrame()
    vix_at_T  = vix_day[vix_day.index.strftime("%H:%M") <= time_str] if not vix_day.empty else pd.DataFrame()

    spx_c = spx_at_T["close"].values if not spx_at_T.empty else np.array([])
    spx_h = spx_at_T["high"].values  if not spx_at_T.empty else np.array([])
    spx_l = spx_at_T["low"].values   if not spx_at_T.empty else np.array([])
    spx_o = spx_at_T["open"].values  if not spx_at_T.empty else np.array([])
    spx_rng = (spx_h - spx_l) if len(spx_h) else np.array([])

    spy_c = spy_at_T["close"].values   if not spy_at_T.empty else np.array([])
    spy_v = spy_at_T["volume"].values  if (not spy_at_T.empty and "volume" in spy_at_T.columns) else np.array([])

    vix_c = vix_at_T["close"].values if not vix_at_T.empty else np.array([])

    f: dict[str, float] = {}

    # ── A. Price stretch vs EMA / return z-scores ─────────────────────────────
    for w in (5, 10, 20, 60):
        prices = spx_c[-w:] if len(spx_c) >= 2 else np.array([])
        if len(prices) < 2:
            f[f"stretch_{w}"] = 0.0
            f[f"ret_z_{w}"]   = 0.0
            continue
        rets   = np.diff(np.log(prices))
        rv     = _realized_var(rets)
        rv_std = sqrt(max(rv, 1e-10))
        ema_v  = _ema(prices, span=w)
        last   = float(prices[-1])
        f[f"stretch_{w}"] = (last - ema_v) / (rv_std * last) if last > 0 else 0.0
        ret_L  = float(np.log(last / prices[0]))
        # Rolling z-score: use same window as baseline
        ret_std = float(np.std(rets)) if len(rets) > 1 else 1e-6
        ret_mu  = float(np.mean(rets)) * len(rets)
        f[f"ret_z_{w}"] = (ret_L - ret_mu) / max(ret_std * sqrt(w), 1e-10)

    # ── B. Realized variance, jump proxy, semivariance ───────────────────────
    for w in (5, 20, 60):
        prices = spx_c[-w:] if len(spx_c) >= 2 else np.array([])
        rets   = np.diff(np.log(prices)) if len(prices) > 1 else np.array([])
        rv     = _realized_var(rets)
        bv     = _bipower_var(rets)
        f[f"rv_{w}"]          = rv
        f[f"jump_share_{w}"]  = max(rv - bv, 0) / max(rv, 1e-10) if rv > 0 else 0.0
        f[f"down_sv_{w}"]     = float(np.sum(rets[rets < 0] ** 2)) / max(rv, 1e-10) if rv > 0 else 0.5
        f[f"range_vol_{w}"]   = float(np.mean(np.log(spx_h[-w:] / np.maximum(spx_l[-w:], 1e-1)))) \
                                 if (len(spx_h) >= w and len(spx_l) >= w) else 0.0

    # RV spike: ratio of recent (5-bar) RV vs longer (60-bar) RV
    f["rv_spike"] = f["rv_5"] / max(f["rv_60"], 1e-10)

    # Autocorrelation of returns (reversal indicator when negative)
    for w in (10, 20, 30):
        rets = np.diff(np.log(spx_c[-w:])) if len(spx_c) > w else np.array([])
        f[f"autocorr_{w}"] = _autocorr_lag1(rets)

    # ── C. Volume / order-flow (SPY) ─────────────────────────────────────────
    for w in (5, 10, 20, 30):
        if len(spy_v) < 2 or len(spy_c) < 2:
            f[f"rel_vol_{w}"]          = 1.0
            f[f"signed_vol_imb_{w}"]   = 0.0
            continue
        v_w = spy_v[-w:]
        c_w = spy_c[-w:]
        # Relative volume vs TOD baseline
        if tod_baseline:
            spy_times = spy_at_T.index[-w:] if not spy_at_T.empty else []
            if len(spy_times) > 0:
                last_min = spy_times[-1].hour * 60 + spy_times[-1].minute
                expected = tod_baseline.get(last_min, float(np.median(list(tod_baseline.values()))))
                f[f"rel_vol_{w}"] = float(v_w.mean()) / max(expected, 1.0)
            else:
                f[f"rel_vol_{w}"] = 1.0
        else:
            f[f"rel_vol_{w}"] = 1.0
        # Signed volume imbalance = Σ sign(ΔP) * V / Σ V
        dc = np.diff(c_w)
        signs = np.sign(dc)
        vol_of_changes = v_w[1:] if len(v_w) > len(dc) else v_w[:len(dc)]
        total_v = vol_of_changes.sum()
        f[f"signed_vol_imb_{w}"] = float(np.dot(signs, vol_of_changes)) / max(total_v, 1.0)

    # CVD slope: linear regression slope of cumulative signed volume delta
    if len(spy_v) >= 10 and len(spy_c) >= 10:
        dc = np.diff(spy_c[-30:])
        signs = np.sign(dc)
        v_chg = spy_v[-len(dc):][:len(dc)]
        cvd   = np.cumsum(signs * v_chg)
        x_reg = np.arange(len(cvd))
        f["cvd_slope"] = float(np.polyfit(x_reg, cvd, 1)[0]) / max(np.abs(cvd).mean(), 1.0) \
                         if len(cvd) > 2 else 0.0
    else:
        f["cvd_slope"] = 0.0

    # ── D. Candle patterns + technical oscillators ────────────────────────────
    if len(spx_c) >= 1 and len(spx_h) >= 1:
        last_body   = abs(float(spx_c[-1]) - float(spx_o[-1])) if len(spx_o) >= 1 else 0.0
        last_range  = max(float(spx_h[-1]) - float(spx_l[-1]), 1e-3) if len(spx_l) >= 1 else 1.0
        last_lo     = float(spx_l[-1]) if len(spx_l) >= 1 else 0.0
        last_hi     = float(spx_h[-1]) if len(spx_h) >= 1 else 0.0
        last_o      = float(spx_o[-1]) if len(spx_o) >= 1 else float(spx_c[-1])
        last_c      = float(spx_c[-1])

        # Lower wick (bullish rejection)
        lower_wick  = min(last_o, last_c) - last_lo
        upper_wick  = last_hi - max(last_o, last_c)
        f["lower_wick_ratio"] = lower_wick / last_range
        f["upper_wick_ratio"] = upper_wick / last_range
        f["body_to_range"]    = last_body  / last_range   # low = doji

        # 5-bar average wick ratios
        if len(spx_h) >= 5:
            bodies5  = np.abs(spx_c[-5:] - spx_o[-5:]) if len(spx_o) >= 5 else np.zeros(5)
            ranges5  = np.maximum(spx_h[-5:] - spx_l[-5:], 1e-3)
            f["avg_body_range_5"]   = float(np.mean(bodies5 / ranges5))
            lower5 = np.minimum(spx_c[-5:], spx_o[-5:]) - spx_l[-5:] if len(spx_o) >= 5 else np.zeros(5)
            f["avg_lower_wick_5"]   = float(np.mean(lower5 / ranges5))
        else:
            f["avg_body_range_5"]  = f["body_to_range"]
            f["avg_lower_wick_5"]  = f["lower_wick_ratio"]
    else:
        f["lower_wick_ratio"] = 0.0
        f["upper_wick_ratio"] = 0.0
        f["body_to_range"]    = 0.5
        f["avg_body_range_5"] = 0.5
        f["avg_lower_wick_5"] = 0.0

    # RSI (14)
    f["rsi_14"] = _rsi(spx_c, n=14)

    # Stochastic %K (14)
    f["stoch_k_14"] = _stoch_k(spx_c, spx_h, spx_l, n=14)

    # Bollinger %b and bandwidth (20 bars)
    if len(spx_c) >= 20:
        b20_mean  = float(np.mean(spx_c[-20:]))
        b20_std   = float(np.std(spx_c[-20:]))
        b20_upper = b20_mean + 2 * b20_std
        b20_lower = b20_mean - 2 * b20_std
        band_range = b20_upper - b20_lower
        f["bb_pct_b"]    = (float(spx_c[-1]) - b20_lower) / max(band_range, 1e-3)
        f["bb_bandwidth"]= band_range / max(b20_mean, 1.0)
        # ATR for Keltner Channel (BB-KC squeeze)
        if len(spx_h) >= 20:
            tr20 = np.maximum(spx_h[-20:] - spx_l[-20:],
                   np.maximum(np.abs(spx_h[-20:] - np.roll(spx_c[-20:], 1)),
                              np.abs(spx_l[-20:] - np.roll(spx_c[-20:], 1))))
            atr20 = float(np.mean(tr20[1:]))
            kc_upper = b20_mean + 1.5 * atr20
            kc_lower = b20_mean - 1.5 * atr20
            f["bb_kc_squeeze"] = float(b20_upper < kc_upper and b20_lower > kc_lower)
        else:
            f["bb_kc_squeeze"] = 0.0
    else:
        f["bb_pct_b"]    = 0.5
        f["bb_bandwidth"]= 0.0
        f["bb_kc_squeeze"] = 0.0

    # MACD histogram
    f["macd_hist"] = _macd_hist(spx_c)

    # Trend slope and curvature (20 and 60 bars)
    for w in (20, 60):
        sl, cv = _slope_and_curvature(spx_c, w)
        f[f"trend_slope_{w}"]    = sl
        f[f"trend_curvature_{w}"]= cv

    # ── E. VIX intraday features ───────────────────────────────────────────────
    if len(vix_c) >= 2:
        vix_now = float(vix_c[-1])
        f["vix_intraday"] = vix_now

        # VIX z-score vs daily baseline
        vix_d = vix_daily
        if vix_d is not None and not vix_d.empty:
            vd = vix_d.copy()
            vd.index = pd.to_datetime(vd.index)
            prior = vd[vd.index < pd.Timestamp(date_str)]["close"].tail(60)
            if len(prior) >= 5:
                f["vix_z_60d"] = (vix_now - float(prior.mean())) / max(float(prior.std()), 0.1)
            else:
                f["vix_z_60d"] = 0.0
        else:
            f["vix_z_60d"] = 0.0

        # VIX change over 5 / 30 mins
        f["vix_chg_5m"]  = float(vix_c[-1] - vix_c[max(-6, -len(vix_c))])
        f["vix_chg_30m"] = float(vix_c[-1] - vix_c[max(-31, -len(vix_c))])
        f["vix_accel"]   = float((vix_c[-1] - vix_c[-2]) - (vix_c[-2] - vix_c[-3])) \
                            if len(vix_c) >= 3 else 0.0

        # SPX-VIX reaction ratio (dVIX per dSPX) — leverage effect proxy
        if len(spx_c) >= 6 and len(vix_c) >= 6:
            d_spx = float(spx_c[-1] - spx_c[-6])
            d_vix = float(vix_c[-1] - vix_c[-6])
            f["spx_vix_beta"] = d_vix / abs(d_spx) if abs(d_spx) > 0.5 else 0.0
        else:
            f["spx_vix_beta"] = 0.0
    else:
        # Fall back to prev-day VIX
        vix_now = float(ev.get("vix_prev") or 20.0)
        f["vix_intraday"] = vix_now
        f["vix_z_60d"]    = 0.0
        f["vix_chg_5m"]   = 0.0
        f["vix_chg_30m"]  = 0.0
        f["vix_accel"]    = 0.0
        f["spx_vix_beta"] = 0.0

    # ── F. Market structure: VWAP, prev close, day open ──────────────────────
    last_price = float(spx_c[-1]) if len(spx_c) > 0 else float(ev.get("swing_low", 5000))

    # Session VWAP from open: use SPY vwap if available, else equal-weight
    if not spy_day.empty and "vwap" in spy_day.columns:
        spx_vwap_bars = spy_day[spy_day.index.strftime("%H:%M") <= time_str]
        session_vwap = float(spx_vwap_bars["vwap"].iloc[-1]) * (last_price / max(float(spy_day["close"].iloc[0]), 1e-3))
    else:
        if not spx_day.empty:
            session_vwap = float(spx_at_T["close"].mean())
        else:
            session_vwap = last_price
    f["vwap_dev"] = (last_price - session_vwap) / max(session_vwap, 1.0)

    # Previous day close
    prev_close = _get_prev_close(date_str, spx_daily)
    if prev_close:
        f["dist_to_prev_close"] = (last_price - prev_close) / prev_close
    else:
        f["dist_to_prev_close"] = 0.0

    # Day open and day high
    if not spx_day.empty:
        day_open = float(spx_day["open"].iloc[0])
        day_high = float(spx_day["high"].max())
        f["dist_to_open"]     = (last_price - day_open) / max(day_open, 1.0)
        f["dist_to_day_high"] = (day_high - last_price) / max(last_price, 1.0)
    else:
        f["dist_to_open"]     = 0.0
        f["dist_to_day_high"] = 0.0

    # ── G. Opening range + gap ────────────────────────────────────────────────
    if not spx_day.empty and prev_close:
        # Gap
        day_open_px = float(spx_day["open"].iloc[0])
        gap_pts     = day_open_px - prev_close
        gap_pct     = gap_pts / prev_close * 100
        gap_dir     = 1 if gap_pts > 0.5 else (-1 if gap_pts < -0.5 else 0)
        # Opening 5-min range
        first5 = spx_day.iloc[:5]
        open5_range = float(first5["high"].max() - first5["low"].min())
        open5_dir   = int(float(first5["close"].iloc[-1]) > float(first5["open"].iloc[0]))
    else:
        gap_pts     = float(ev.get("gap_pts", 0))
        gap_pct     = 0.0
        gap_dir     = int(ev.get("gap_dir", 0))
        open5_range = 0.0
        open5_dir   = 0
    f["gap_pts"]     = gap_pts
    f["gap_pct"]     = gap_pct
    f["gap_dir"]     = float(gap_dir)
    f["open5_range"] = open5_range
    f["open5_dir"]   = float(open5_dir)

    # Fall speed: fall_pts / bars_elapsed
    bars_elapsed = max(len(spx_at_T) - 5, 1)  # subtract first 5 opening bars
    f["fall_speed"]      = float(ev.get("fall_pts", 0)) / bars_elapsed
    f["pct_day_at_low"]  = len(spx_at_T) / (6.5 * 60)

    # ── H. Straddle proxy + implied-realized spread ───────────────────────────
    spx_price   = last_price
    vix_level   = f["vix_intraday"]
    daily_sigma = spx_price * vix_level / (100 * sqrt(252))   # 1σ daily move
    f["straddle_proxy"]      = daily_sigma
    f["fall_vs_straddle"]    = float(ev.get("fall_pts", 0)) / max(daily_sigma, 0.1)
    # Intraday VRP: annualised IV^2 - annualised RV_{20}
    iv2 = (vix_level / 100) ** 2
    rv20_ann = f["rv_20"] * 252 * 390   # scale 1-min RV to annual
    f["intraday_vrp"] = iv2 - rv20_ann

    # ── I. Squeeze / existing features ───────────────────────────────────────
    f["squeeze_encoded"] = float(SQUEEZE_MAP.get(str(ev.get("squeeze_strength", "none")), 0))
    f["squeeze_duration"]= float(ev.get("squeeze_duration", 0))
    f["expansion_body"]  = float(ev.get("expansion_body") or 0)
    f["squeeze_min_body"]= float(ev.get("squeeze_min_body") or 0.5)
    f["expansion_ratio"] = f["expansion_body"] / max(f["squeeze_min_body"], 0.1)

    # Fall characteristics
    f["fall_pts"]        = float(ev.get("fall_pts", 0))
    f["vix_prev"]        = float(ev.get("vix_prev") or vix_level)
    f["fall_normalized"] = f["fall_pts"] / max(f["vix_prev"], 1.0)

    # ── J. VIX daily momentum ─────────────────────────────────────────────────
    if vix_daily is not None and not vix_daily.empty:
        vd2 = vix_daily.copy()
        vd2.index = pd.to_datetime(vd2.index)
        prior = vd2[vd2.index < pd.Timestamp(date_str)]["close"].tail(25)
        if len(prior) >= 2:
            f["vix_1d_chg"]  = float(prior.iloc[-1] - prior.iloc[-2])
            f["vix_5d_chg"]  = float(prior.iloc[-1] - prior.iloc[max(0, len(prior)-5)])
            f["vix_vs_20d"]  = float(prior.iloc[-1] - prior.mean())
            f["vix_trend_up"]= float(f["vix_5d_chg"] > 0)
        else:
            f["vix_1d_chg"]  = 0.0
            f["vix_5d_chg"]  = 0.0
            f["vix_vs_20d"]  = 0.0
            f["vix_trend_up"]= 0.0
    else:
        f["vix_1d_chg"]  = 0.0
        f["vix_5d_chg"]  = 0.0
        f["vix_vs_20d"]  = 0.0
        f["vix_trend_up"]= 0.0

    # ── K. Time of day ────────────────────────────────────────────────────────
    try:
        hh, mm = int(time_str[:2]), int(time_str[3:5])
    except ValueError:
        hh, mm = 12, 0
    minute_of_day = hh * 60 + mm
    f["minute_of_low"]  = float(minute_of_day)
    f["time_sin"]       = float(np.sin(2 * pi * (minute_of_day - 570) / 390))
    f["time_cos"]       = float(np.cos(2 * pi * (minute_of_day - 570) / 390))
    f["is_morning"]     = float(hh < 11)
    f["is_afternoon"]   = float(hh >= 13)
    f["is_open30"]      = float(hh == 9 and mm <= 59 or hh == 10 and mm == 0)
    f["is_close30"]     = float(hh == 15 and mm >= 30)

    # Day of week
    dow_str = str(ev.get("day_of_week", "Wednesday"))
    f["dow_encoded"] = float(DOW_MAP.get(dow_str, 2))
    f["is_monday"]   = float(dow_str == "Monday")
    f["is_friday"]   = float(dow_str == "Friday")
    f["is_tuesday"]  = float(dow_str == "Tuesday")
    f["hour_of_low"] = float(hh)

    # ── L. Interaction terms ───────────────────────────────────────────────────
    f["stretch_x_squeeze"]      = f["stretch_20"] * f["squeeze_encoded"]
    f["rv_spike_x_vol_imb"]     = f["rv_spike"]   * f["signed_vol_imb_10"]
    f["vix_chg_x_fall_norm"]    = f["vix_chg_30m"] * f["fall_normalized"]
    f["gap_dir_x_fall"]         = f["gap_dir"]     * f["fall_normalized"]
    f["autocorr_x_stretch"]     = f["autocorr_20"] * f["stretch_20"]
    f["lower_wick_x_rv_spike"]  = f["lower_wick_ratio"] * f["rv_spike"]

    return f


def _get_prev_close(date_str: str, spx_daily: dict) -> float | None:
    """Get the previous trading day's close."""
    all_dates = sorted(spx_daily.keys())
    try:
        idx = all_dates.index(date_str)
    except ValueError:
        return None
    if idx == 0:
        return None
    prev = spx_daily[all_dates[idx - 1]]
    return float(prev["close"].iloc[-1]) if not prev.empty else None
