"""
Reversal Analysis — SPX intraday reversals + ATM 0DTE call squeeze detection.

Logic:
  1. Detect intraday drawdown-then-bounce events in SPX 1-min data.
     Categorise fall: small (30-50), medium (50-80), large (80+)
     Categorise reversal: weak (0-20), medium (20-50), strong (50+)

  2. For each reversal event, identify the ATM 0DTE call option ticker
     (SPX price at swing low, rounded to nearest 25 strike).

  3. Fetch 1-min OHLCV bars for that option contract (with local cache).

  4. Detect "squeeze" in ATM call candle bodies in the 20 bars before the low:
     - Squeeze = consecutive decrease in rolling-max of body sizes
     - Expansion = first post-low bar with body > 2× min-body during squeeze
     - Classify: none / mild / moderate / strong
"""
from __future__ import annotations

import json
from datetime import date, time
from math import sqrt
from pathlib import Path
from typing import NamedTuple

import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
FALL_BINS = [
    ("small",  30,  50),
    ("medium", 50,  80),
    ("large",  80,  9999),
]
REVERSAL_BINS = [
    ("weak",   0,  20),
    ("medium", 20, 50),
    ("strong", 50, 9999),
]


# ── Ticker builder ────────────────────────────────────────────────────────────

def build_0dte_call_ticker(spx_price: float, trade_date: date, strike_rounding: int = 25) -> str:
    """Build a Polygon 0DTE ATM call ticker for SPX.

    Format: O:SPX{YYMMDD}C{strike*1000:08d}
    e.g. SPX at 5875 on 2026-01-02 → O:SPX260102C05875000
    """
    atm_strike = round(spx_price / strike_rounding) * strike_rounding
    date_str   = trade_date.strftime("%y%m%d")
    strike_int = int(atm_strike * 1000)
    # Polygon uses SPXW prefix for 0DTE weeklies (Mon/Wed/Fri/Tue/Thu)
    return f"O:SPXW{date_str}C{strike_int:08d}"


# ── Reversal event ─────────────────────────────────────────────────────────────

class ReversalEvent(NamedTuple):
    date:               date
    time_of_high:       time
    time_of_low:        time
    swing_high:         float
    swing_low:          float
    fall_pts:           float
    reversal_pts:       float
    fall_category:      str
    reversal_category:  str
    vix_prev:           float | None
    spx_at_low:         float


def _categorize(val: float, bins: list) -> str | None:
    for name, lo, hi in bins:
        if lo <= val < hi:
            return name
    return None


def detect_reversals(
    spx_1min: pd.DataFrame,
    vix_daily: pd.DataFrame,
    window_mins: int = 30,
) -> list[ReversalEvent]:
    """Scan SPX 1-min RTH data for intraday fall-then-bounce events."""
    spx = spx_1min.copy()
    spx.index = pd.to_datetime(spx.index, utc=True).tz_convert("America/New_York")
    spx = spx.between_time("09:30", "16:00")

    trading_days = sorted(set(spx.index.date))
    events: list[ReversalEvent] = []

    for day in trading_days:
        day_bars = spx[spx.index.date == day]
        if len(day_bars) < window_mins + 5:
            continue

        prev_vix_rows = vix_daily[vix_daily.index < day]
        vix_prev = float(prev_vix_rows.iloc[-1]["close"]) if len(prev_vix_rows) else None

        # Slide window across the day; record every qualifying drawdown
        day_events = []
        for end_i in range(window_mins, len(day_bars)):
            window = day_bars.iloc[end_i - window_mins: end_i + 1]

            high_i = window["high"].argmax()
            # Only look for the low AFTER the high (fall-then-bounce)
            post_high = window.iloc[high_i:]
            if len(post_high) < 3:
                continue

            low_i_rel = post_high["low"].argmin()
            swing_high = float(window["high"].iloc[high_i])
            swing_low  = float(post_high["low"].iloc[low_i_rel])
            fall_pts   = swing_high - swing_low

            fall_cat = _categorize(fall_pts, FALL_BINS)
            if fall_cat is None:
                continue

            # Reversal = max high AFTER the swing low (within the remaining day bars)
            low_abs_idx  = post_high.index[low_i_rel]
            bars_after_low = day_bars[day_bars.index > low_abs_idx]
            if len(bars_after_low) == 0:
                reversal_pts = 0.0
            else:
                # Look at up to 60 bars after the low for max recovery
                recovery_window = bars_after_low.iloc[:60]
                max_high_after  = float(recovery_window["high"].max())
                reversal_pts    = max(max_high_after - swing_low, 0.0)
            reversal_cat = _categorize(reversal_pts, REVERSAL_BINS) or "weak"

            time_of_high = window.index[high_i].time()
            time_of_low  = post_high.index[low_i_rel].time()

            day_events.append(ReversalEvent(
                date=day,
                time_of_high=time_of_high,
                time_of_low=time_of_low,
                swing_high=swing_high,
                swing_low=swing_low,
                fall_pts=round(fall_pts, 2),
                reversal_pts=round(reversal_pts, 2),
                fall_category=fall_cat,
                reversal_category=reversal_cat,
                vix_prev=vix_prev,
                spx_at_low=swing_low,
            ))

        # Keep only the largest fall event per day to avoid double-counting
        if day_events:
            best = max(day_events, key=lambda e: e.fall_pts)
            events.append(best)

    return events


# ── ATM options fetch with cache ───────────────────────────────────────────────

def fetch_atm_call_data(
    events: list[ReversalEvent],
    strike_rounding: int = 25,
    cache_dir: Path | None = None,
) -> dict[date, pd.DataFrame]:
    """Fetch 1-min bars for the ATM 0DTE call at each reversal event.
    Results are cached to data/options_cache/ to avoid repeat API calls.
    """
    from app.fetchers.options import get_option_1min

    cache_dir = cache_dir or Path("data/options_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[date, pd.DataFrame] = {}

    for event in events:
        ticker     = build_0dte_call_ticker(event.spx_at_low, event.date, strike_rounding)
        safe_name  = ticker.replace(":", "_")
        cache_path = cache_dir / f"{safe_name}_{event.date}.csv"

        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col="datetime", parse_dates=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        else:
            print(f"  Fetching {ticker} for {event.date}…")
            df = get_option_1min(ticker, str(event.date))
            if not df.empty:
                df.to_csv(cache_path)

        results[event.date] = df if not df.empty else pd.DataFrame()

    return results


# ── Squeeze detection ──────────────────────────────────────────────────────────

class SqueezeResult(NamedTuple):
    strength:        str
    duration_mins:   int
    body_sequence:   list
    min_body:        float
    expansion_body:  float | None


def detect_squeeze(
    option_bars: pd.DataFrame,
    event_time: time,
    lookback: int = 20,
) -> SqueezeResult:
    """Detect volatility compression in ATM call bodies before a reversal."""
    if option_bars.empty:
        return SqueezeResult("no_data", 0, [], 0.0, None)

    event_str = event_time.strftime("%H:%M")

    try:
        bars_before = option_bars.between_time("09:30", event_str)
    except Exception:
        return SqueezeResult("none", 0, [], 0.0, None)

    bars_before = bars_before.iloc[-lookback:]
    if len(bars_before) < 5:
        return SqueezeResult("none", 0, [], 0.0, None)

    bodies = (bars_before["close"] - bars_before["open"]).abs()
    rolling_max = bodies.rolling(5, min_periods=1).max()

    # Count consecutive bars (from end) where rolling_max is non-increasing
    squeeze_len = 0
    for i in range(len(rolling_max) - 1, 0, -1):
        if rolling_max.iloc[i] <= rolling_max.iloc[i - 1] + 0.05:  # small tolerance
            squeeze_len += 1
        else:
            break

    # Expansion candle: first bar after the low
    expansion_body = None
    try:
        post_low = option_bars.between_time(event_str, "16:00")
        if len(post_low) >= 2:
            first_post = post_low.iloc[1]
            expansion_body = abs(float(first_post["close"]) - float(first_post["open"]))
    except Exception:
        pass

    min_body = float(bodies.iloc[-squeeze_len:].min()) if squeeze_len > 0 else 0.0
    floor    = max(min_body, 0.5)

    if squeeze_len >= 8 and expansion_body and expansion_body > 3 * floor:
        strength = "strong"
    elif squeeze_len >= 5 and expansion_body and expansion_body > 2 * floor:
        strength = "moderate"
    elif squeeze_len >= 3:
        strength = "mild"
    else:
        strength = "none"

    return SqueezeResult(
        strength=strength,
        duration_mins=squeeze_len,
        body_sequence=[round(b, 2) for b in bodies.tolist()],
        min_body=round(min_body, 2),
        expansion_body=round(expansion_body, 2) if expansion_body is not None else None,
    )


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_reversal_analysis(
    spx_1min: pd.DataFrame,
    vix_daily: pd.DataFrame,
    window_mins: int = 30,
    squeeze_lookback: int = 20,
    strike_rounding: int = 25,
) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: detect reversals → fetch ATM calls → squeeze detection.

    Returns (events_df, summary_dict).
    """
    events = detect_reversals(spx_1min, vix_daily, window_mins)
    print(f"Found {len(events)} reversal events")

    atm_data = fetch_atm_call_data(events, strike_rounding)
    hits = sum(1 for v in atm_data.values() if not v.empty)
    print(f"ATM call data available for {hits}/{len(events)} events")

    rows = []
    for event in events:
        opt_df = atm_data.get(event.date, pd.DataFrame())
        squeeze = detect_squeeze(opt_df, event.time_of_low, squeeze_lookback)
        ticker  = build_0dte_call_ticker(event.spx_at_low, event.date, strike_rounding)

        rows.append({
            "date":               str(event.date),
            "time_of_high":       str(event.time_of_high),
            "time_of_low":        str(event.time_of_low),
            "swing_high":         event.swing_high,
            "swing_low":          event.swing_low,
            "fall_pts":           event.fall_pts,
            "reversal_pts":       event.reversal_pts,
            "fall_category":      event.fall_category,
            "reversal_category":  event.reversal_category,
            "vix_prev":           event.vix_prev,
            "atm_ticker":         ticker,
            "squeeze_strength":   squeeze.strength,
            "squeeze_duration":   squeeze.duration_mins,
            "squeeze_min_body":   squeeze.min_body,
            "expansion_body":     squeeze.expansion_body,
            "day_of_week":        event.date.strftime("%A"),
            "hour_of_low":        event.time_of_low.hour,
        })

    events_df = pd.DataFrame(rows)
    if not events_df.empty:
        events_df = events_df.set_index("date")

    summary = _build_reversal_summary(events_df)
    return events_df, summary


# ── Summary builder ────────────────────────────────────────────────────────────

def _build_reversal_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "No reversal events found", "total_events": 0}

    n = len(df)
    has_squeeze = df["squeeze_strength"].isin(["mild", "moderate", "strong"])
    squeeze_df  = df[has_squeeze]

    # Squeeze by strength breakdown
    squeeze_breakdown = []
    for strength in ["strong", "moderate", "mild", "none", "no_data"]:
        grp = df[df["squeeze_strength"] == strength]
        if not len(grp):
            continue
        squeeze_breakdown.append({
            "squeeze_strength":    strength,
            "count":               len(grp),
            "avg_fall_pts":        round(grp["fall_pts"].mean(), 1),
            "avg_reversal_pts":    round(grp["reversal_pts"].mean(), 1),
            "pct_strong_reversal": round((grp["reversal_category"] == "strong").mean() * 100, 1),
            "avg_expansion_body":  round(grp["expansion_body"].dropna().mean(), 2)
                                   if grp["expansion_body"].dropna().any() else None,
        })

    # VIX regime breakdown
    vix_valid = df[df["vix_prev"].notna()].copy()
    vix_breakdown = []
    if len(vix_valid) >= 3:
        try:
            vix_valid["vix_regime"] = pd.qcut(
                vix_valid["vix_prev"], q=3,
                labels=["low", "medium", "high"], duplicates="drop"
            )
            for regime, grp in vix_valid.groupby("vix_regime", observed=True):
                vix_breakdown.append({
                    "regime":              str(regime),
                    "vix_range":           f"{grp['vix_prev'].min():.1f}–{grp['vix_prev'].max():.1f}",
                    "count":               len(grp),
                    "avg_fall_pts":        round(grp["fall_pts"].mean(), 1),
                    "avg_reversal_pts":    round(grp["reversal_pts"].mean(), 1),
                    "pct_with_squeeze":    round(grp["squeeze_strength"].isin(["mild","moderate","strong"]).mean() * 100, 1),
                    "pct_strong_reversal": round((grp["reversal_category"] == "strong").mean() * 100, 1),
                })
        except Exception:
            pass

    return {
        "total_events":              n,
        "date_range":                f"{df.index[0]} to {df.index[-1]}",
        "avg_fall_pts":              round(df["fall_pts"].mean(), 1),
        "avg_reversal_pts":          round(df["reversal_pts"].mean(), 1),
        "median_fall_pts":           round(df["fall_pts"].median(), 1),
        "median_reversal_pts":       round(df["reversal_pts"].median(), 1),
        "fall_category_counts":      df["fall_category"].value_counts().to_dict(),
        "reversal_category_counts":  df["reversal_category"].value_counts().to_dict(),
        "squeeze_strength_counts":   df["squeeze_strength"].value_counts().to_dict(),
        "pct_preceded_by_squeeze":   round(has_squeeze.mean() * 100, 1),
        "avg_squeeze_duration_mins": round(squeeze_df["squeeze_duration"].mean(), 1) if len(squeeze_df) else 0,
        "squeeze_breakdown":         squeeze_breakdown,
        "vix_regime_breakdown":      vix_breakdown,
        "fall_reversal_crosstab":    pd.crosstab(df["fall_category"], df["reversal_category"]).to_dict(),
        "day_of_week_distribution":  df["day_of_week"].value_counts().to_dict(),
        "time_of_day_distribution":  df["hour_of_low"].value_counts().sort_index().to_dict(),
        "events":                    df.reset_index().to_dict(orient="records"),
    }
