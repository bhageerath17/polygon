"""
Patches Analysis — Python translation of the Pine Script:
  "SPX 0DTE EM (Skewed VWAP + Midpoint)"

Logic:
  - For each trading day, use the first `lookback_mins` bars (9:31–9:35) to build
    4 patch boxes using: sig = bar.open × (prev_day_VIX / 100) / √252
      • Box 1.0σ Up  : [min(open+sig),  max(open+sig)]
      • Box 0.8σ Up  : [min(open+0.8sig), max(open+0.8sig)]
      • Box 0.8σ Dn  : [min(open-0.8sig), max(open-0.8sig)]
      • Box 1.0σ Dn  : [min(open-sig),  max(open-sig)]
  - PDC EM: prev_close ± (prev_close × prev_VIX/100 / √252)
  - Skewed VWAP: volume-weighted EM biased by buy/sell volume during lookback
  - Then analyse rest of day: breaches, close zone, gap
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


SQRT252 = sqrt(252)


# ── helpers ──────────────────────────────────────────────────────────────────

def _sig(open_price: float, vix: float, em_1d: float | None = None) -> float:
    """1σ daily expected move in SPX points.

    If em_1d (straddle-derived 1-day EM) is available, use it directly.
    Otherwise fall back to VIX / √252 approximation.
    """
    if em_1d is not None and em_1d > 0:
        return em_1d
    return open_price * (vix / 100.0) / SQRT252


def _classify_close_zone(
    close: float,
    b10u_max: float, b10u_min: float,
    b08u_max: float, b08u_min: float,
    b08d_max: float, b08d_min: float,
    b10d_max: float, b10d_min: float,
) -> str:
    if close >= b10u_max:                          return "above_1sig"
    if close >= b10u_min:                          return "in_1sig_up_box"
    if close >= b08u_max:                          return "gap_upper"
    if close >= b08u_min:                          return "in_08sig_up_box"
    if close >= b08d_max:                          return "inside"
    if close >= b08d_min:                          return "in_08sig_dn_box"
    if close >= b10d_max:                          return "gap_lower"
    if close >= b10d_min:                          return "in_1sig_dn_box"
    return "below_1sig"


def _classify_gap_zone(
    day_open: float,
    prev_close: float,
    pdc_em: float,
) -> str:
    gap = day_open - prev_close
    if   gap >  pdc_em: return "gap_up_beyond_em"
    elif gap >  0:      return "gap_up_within_em"
    elif gap == 0:      return "flat"
    elif gap > -pdc_em: return "gap_dn_within_em"
    else:               return "gap_dn_beyond_em"


# ── core per-day computation ──────────────────────────────────────────────────

def _analyse_day(
    day_bars: pd.DataFrame,
    prev_close: float | None,
    prev_vix: float | None,
    lookback_mins: int,
    em_1d: float | None = None,
) -> dict | None:
    if prev_vix is None or len(day_bars) < lookback_mins + 1:
        return None

    # ── Lookback window: bars 1..lookback_mins (skip bar 0 = 9:30 open bar)
    # Pine Script: m1_time > start_ts (9:30:00) and <= end_ts (9:35:00)
    # In our 1-min data bar labelled 09:31 is the bar AFTER 09:30
    lookback = day_bars.iloc[1: lookback_mins + 1]
    rest     = day_bars.iloc[lookback_mins + 1:]

    # ── Patch boxes
    sigs      = lookback["open"].apply(lambda o: _sig(o, prev_vix, em_1d))
    up10      = lookback["open"] + sigs
    up08      = lookback["open"] + sigs * 0.8
    dn08      = lookback["open"] - sigs * 0.8
    dn10      = lookback["open"] - sigs

    b10u_max, b10u_min = float(up10.max()), float(up10.min())
    b08u_max, b08u_min = float(up08.max()), float(up08.min())
    b08d_max, b08d_min = float(dn08.max()), float(dn08.min())
    b10d_max, b10d_min = float(dn10.max()), float(dn10.min())

    # ── Skewed VWAP (buy-weighted upper / sell-weighted lower)
    buy_vol = sell_vol = 0.0
    sum_vw_buy_up = sum_vw_sell_dn = 0.0
    for _, row in lookback.iterrows():
        s   = _sig(row["open"], prev_vix, em_1d)
        bv  = row["close"] > row["open"] and float(row.get("volume", 0) or 0)
        sv  = row["close"] < row["open"] and float(row.get("volume", 0) or 0)
        ev  = float(row.get("volume", 0) or 0) * 0.5 if row["close"] == row["open"] else 0
        bv  = float(row.get("volume", 0) or 0) if row["close"] > row["open"] else ev
        sv  = float(row.get("volume", 0) or 0) if row["close"] < row["open"] else ev
        if bv:
            buy_vol       += bv
            sum_vw_buy_up += (row["open"] + s) * bv
        if sv:
            sell_vol       += sv
            sum_vw_sell_dn += (row["open"] - s) * sv

    vw_buy_em  = sum_vw_buy_up  / buy_vol  if buy_vol  else None
    vw_sell_em = sum_vw_sell_dn / sell_vol if sell_vol else None
    skew_mid   = (vw_buy_em + vw_sell_em) / 2 if vw_buy_em and vw_sell_em else None

    # ── PDC EM
    pdc_em        = _sig(prev_close, prev_vix, em_1d) if prev_close else None
    pdc_em_upper  = prev_close + pdc_em        if pdc_em    else None
    pdc_em_lower  = prev_close - pdc_em        if pdc_em    else None

    # ── Day OHLC
    day_open  = float(day_bars.iloc[0]["open"])
    day_close = float(day_bars.iloc[-1]["close"])
    day_high  = float(day_bars["high"].max())
    day_low   = float(day_bars["low"].min())

    # ── Breach detection (on bars AFTER the lookback window)
    if len(rest):
        breach_above_1sig  = bool((rest["high"]  > b10u_max).any())
        breach_below_1sig  = bool((rest["low"]   < b10d_min).any())
        breach_above_08sig = bool((rest["high"]  > b08u_max).any())
        breach_below_08sig = bool((rest["low"]   < b08d_min).any())
    else:
        breach_above_1sig = breach_below_1sig = False
        breach_above_08sig = breach_below_08sig = False

    # ── Close zone
    close_zone = _classify_close_zone(
        day_close,
        b10u_max, b10u_min,
        b08u_max, b08u_min,
        b08d_max, b08d_min,
        b10d_max, b10d_min,
    )

    # ── Gap zone
    gap          = day_open - prev_close       if prev_close else None
    gap_pct      = gap / prev_close * 100      if prev_close else None
    gap_zone     = _classify_gap_zone(day_open, prev_close, pdc_em) if pdc_em else None

    return {
        "vix_prev":          round(prev_vix, 2),
        "vix1d":             round(em_1d / float(day_bars.iloc[0]["open"]) * SQRT252 * 100, 2) if em_1d else None,
        "em_source":         "straddle" if em_1d else "vix",
        "sig":               round(float(sigs.mean()), 2),
        "open":              round(day_open, 2),
        "high":              round(day_high, 2),
        "low":               round(day_low, 2),
        "close":             round(day_close, 2),
        "prev_close":        round(prev_close, 2)   if prev_close else None,
        "gap":               round(gap, 2)           if gap is not None else None,
        "gap_pct":           round(gap_pct, 3)       if gap_pct is not None else None,
        "gap_zone":          gap_zone,
        "pdc_em":            round(pdc_em, 2)        if pdc_em else None,
        "pdc_em_upper":      round(pdc_em_upper, 2)  if pdc_em_upper else None,
        "pdc_em_lower":      round(pdc_em_lower, 2)  if pdc_em_lower else None,
        "b10u_max":          round(b10u_max, 2),
        "b10u_min":          round(b10u_min, 2),
        "b08u_max":          round(b08u_max, 2),
        "b08u_min":          round(b08u_min, 2),
        "b08d_max":          round(b08d_max, 2),
        "b08d_min":          round(b08d_min, 2),
        "b10d_max":          round(b10d_max, 2),
        "b10d_min":          round(b10d_min, 2),
        "vw_buy_em":         round(vw_buy_em,  2)   if vw_buy_em  else None,
        "vw_sell_em":        round(vw_sell_em, 2)   if vw_sell_em else None,
        "skew_mid":          round(skew_mid,   2)   if skew_mid   else None,
        "breach_above_1sig":  breach_above_1sig,
        "breach_below_1sig":  breach_below_1sig,
        "breach_above_08sig": breach_above_08sig,
        "breach_below_08sig": breach_below_08sig,
        "close_zone":         close_zone,
    }


# ── main entry ────────────────────────────────────────────────────────────────

def run_patches_analysis(
    spx_1min: pd.DataFrame,
    vix_daily: pd.DataFrame,
    lookback_mins: int = 5,
    vix1d_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return (daily_df, summary_dict).

    Args:
        vix1d_df: Optional DataFrame indexed by date str with 'em_1d' column
                  (straddle-derived 1-day expected move). When provided, uses
                  actual market-implied EM instead of VIX / √252.
    """
    spx_1min.index = pd.to_datetime(spx_1min.index, utc=True).tz_convert("America/New_York")
    trading_days = sorted(set(spx_1min.index.date))
    rows = []

    for i, day in enumerate(trading_days):
        day_bars = spx_1min[spx_1min.index.date == day]

        # prev VIX close
        prev_vix_candidates = vix_daily[vix_daily.index < day]
        prev_vix   = float(prev_vix_candidates.iloc[-1]["close"]) if len(prev_vix_candidates) else None

        # prev SPX close
        prev_close = None
        if i > 0:
            prev_bars  = spx_1min[spx_1min.index.date == trading_days[i - 1]]
            prev_close = float(prev_bars.iloc[-1]["close"]) if len(prev_bars) else None

        # Straddle-derived 1-day EM (if available)
        em_1d = None
        if vix1d_df is not None and str(day) in vix1d_df.index:
            val = vix1d_df.loc[str(day), "em_1d"]
            if pd.notna(val):
                em_1d = float(val)

        result = _analyse_day(day_bars, prev_close, prev_vix, lookback_mins, em_1d)
        if result:
            rows.append({"date": str(day), **result})

    daily_df = pd.DataFrame(rows).set_index("date")

    summary = _build_summary(daily_df)
    return daily_df, summary


def _build_summary(df: pd.DataFrame) -> dict:
    n = len(df)

    # VIX quartile breakdown
    df["vix_quartile"] = pd.qcut(df["vix_prev"], q=4, labels=["Q1 Low", "Q2", "Q3", "Q4 High"])
    q_stats = []
    for q, grp in df.groupby("vix_quartile", observed=True):
        q_stats.append({
            "quartile":            str(q),
            "days":                len(grp),
            "vix_range":           f"{grp['vix_prev'].min():.1f}–{grp['vix_prev'].max():.1f}",
            "breach_above_1sig_%": round(grp["breach_above_1sig"].mean() * 100, 1),
            "breach_below_1sig_%": round(grp["breach_below_1sig"].mean() * 100, 1),
            "any_breach_%":        round(((grp["breach_above_1sig"] | grp["breach_below_1sig"]).mean()) * 100, 1),
            "close_inside_%":      round((grp["close_zone"] == "inside").mean() * 100, 1),
            "avg_gap_pct":         round(grp["gap_pct"].abs().mean(), 3),
            "gap_up_%":            round((grp["gap_pct"] > 0).mean() * 100, 1),
            "gap_dn_%":            round((grp["gap_pct"] < 0).mean() * 100, 1),
        })

    return {
        "trading_days":            n,
        "date_range":              f"{df.index[0]} to {df.index[-1]}",
        "breach_above_1sig_%":     round(df["breach_above_1sig"].mean() * 100, 1),
        "breach_below_1sig_%":     round(df["breach_below_1sig"].mean() * 100, 1),
        "any_breach_1sig_%":       round(((df["breach_above_1sig"] | df["breach_below_1sig"]).mean()) * 100, 1),
        "breach_above_08sig_%":    round(df["breach_above_08sig"].mean() * 100, 1),
        "breach_below_08sig_%":    round(df["breach_below_08sig"].mean() * 100, 1),
        "close_zone_dist":         df["close_zone"].value_counts().to_dict(),
        "gap_zone_dist":           df["gap_zone"].dropna().value_counts().to_dict(),
        "vix_stats": {
            "mean":   round(df["vix_prev"].mean(), 2),
            "median": round(df["vix_prev"].median(), 2),
            "min":    round(df["vix_prev"].min(), 2),
            "max":    round(df["vix_prev"].max(), 2),
            "q25":    round(df["vix_prev"].quantile(0.25), 2),
            "q75":    round(df["vix_prev"].quantile(0.75), 2),
        },
        "gap_stats": {
            "mean_abs_gap_pct": round(df["gap_pct"].abs().mean(), 3),
            "gap_up_count":     int((df["gap_pct"] > 0).sum()),
            "gap_dn_count":     int((df["gap_pct"] < 0).sum()),
            "flat_count":       int((df["gap_pct"] == 0).sum()),
        },
        "vix_quartile_breakdown":  q_stats,
        "regime_bins":             _build_regime_bins(df),
        "daily":                   df.reset_index().to_dict(orient="records"),
    }


def _build_regime_bins(df: pd.DataFrame) -> list[dict]:
    """
    6 bins: gap_dir (up / dn) × vix_regime (low / med / high)
    VIX terciles: low = bottom 33%, med = middle 33%, high = top 33%

    For each bin, compute:
    - sample size & % of all days
    - breach / close-zone probabilities
    - suggested strategy + rationale
    - simple backtest win rate (strategy-specific)
    """
    vix_terciles = df["vix_prev"].quantile([1/3, 2/3]).values

    def vix_regime(v):
        if v <= vix_terciles[0]: return "low"
        if v <= vix_terciles[1]: return "med"
        return "high"

    def gap_dir(g):
        if g is None or (isinstance(g, float) and g != g): return None
        return "up" if g > 0 else "dn"

    df = df.copy()
    df["vix_regime"] = df["vix_prev"].apply(vix_regime)
    df["gap_dir"]    = df["gap_pct"].apply(gap_dir)
    df = df[df["gap_dir"].notna()]

    STRATEGIES = {
        ("up", "low"): {
            "name":      "Sell Upper 0.8σ Call Spread",
            "rationale": "Low VIX gap-ups rarely breach 1σ upper band (93% inside rate). "
                         "Sell credit spread at 0.8σ–1.0σ, collect premium, let time decay work.",
            "win_condition": lambda grp: (~grp["breach_above_1sig"]),
        },
        ("up", "med"): {
            "name":      "Iron Condor (bias upper)",
            "rationale": "Med VIX gap-up days frequently stay inside bands. "
                         "Sell condor — tighter on call side, wider on put side.",
            "win_condition": lambda grp: (grp["close_zone"] == "inside") | (grp["close_zone"].str.contains("up_box", na=False)),
        },
        ("up", "high"): {
            "name":      "Sell Put / Buy Call Debit Spread",
            "rationale": "High VIX gap-up with elevated IV — sell downside put for credit "
                         "funded by long call debit spread. Benefit from vol crush + upside.",
            "win_condition": lambda grp: (~grp["breach_below_1sig"]),
        },
        ("dn", "low"): {
            "name":      "Sell Lower 0.8σ Put Spread",
            "rationale": "Low VIX gap-downs with tight bands — sell put credit spread at 0.8σ–1.0σ lower. "
                         "Low vol means cheap hedging, high probability of staying inside.",
            "win_condition": lambda grp: (~grp["breach_below_1sig"]),
        },
        ("dn", "med"): {
            "name":      "Iron Condor (bias lower)",
            "rationale": "Med VIX gap-down days tend to consolidate inside bands. "
                         "Sell condor — tighter on put side, wider on call side.",
            "win_condition": lambda grp: (grp["close_zone"] == "inside") | (grp["close_zone"].str.contains("dn_box", na=False)),
        },
        ("dn", "high"): {
            "name":      "Buy Put Debit Spread / Sell Call",
            "rationale": "High VIX gap-down signals elevated fear — buy directional put spread "
                         "toward 1σ lower target. Sell call at 0.8σ upper to fund premium.",
            "win_condition": lambda grp: (~grp["breach_above_1sig"]),
        },
    }

    VIX_LABELS = {"low": f"Low (≤{vix_terciles[0]:.1f})", "med": f"Med ({vix_terciles[0]:.1f}–{vix_terciles[1]:.1f})", "high": f"High (>{vix_terciles[1]:.1f})"}
    bins = []

    for gap in ("up", "dn"):
        for vix in ("low", "med", "high"):
            grp = df[(df["gap_dir"] == gap) & (df["vix_regime"] == vix)]
            n = len(grp)
            if n == 0:
                continue
            key = (gap, vix)
            strat = STRATEGIES[key]
            win_mask = strat["win_condition"](grp)
            win_rate = round(win_mask.mean() * 100, 1)

            bins.append({
                "bin":              f"Gap {'↑' if gap=='up' else '↓'} + VIX {vix.capitalize()}",
                "gap_dir":          gap,
                "vix_regime":       vix,
                "vix_label":        VIX_LABELS[vix],
                "days":             n,
                "pct_of_all":       round(n / len(df) * 100, 1),
                "breach_above_%":   round(grp["breach_above_1sig"].mean() * 100, 1),
                "breach_below_%":   round(grp["breach_below_1sig"].mean() * 100, 1),
                "any_breach_%":     round(((grp["breach_above_1sig"] | grp["breach_below_1sig"]).mean()) * 100, 1),
                "close_inside_%":   round((grp["close_zone"] == "inside").mean() * 100, 1),
                "close_zone_dist":  grp["close_zone"].value_counts().to_dict(),
                "avg_gap_pct":      round(grp["gap_pct"].mean(), 3),
                "strategy":         strat["name"],
                "rationale":        strat["rationale"],
                "backtest_win_%":   win_rate,
            })

    return bins
