# SPX Intraday Reversal Prediction Dashboard

A full **quantitative data-science pipeline** for detecting and predicting SPX intraday reversals using a multi-model ensemble, built on [Polygon.io](https://polygon.io) market data.

Live dashboard → **[bhageerath17.github.io/polygon](https://bhageerath17.github.io/polygon)**

---

## What this does

1. **Fetches** 1-minute SPX, SPY, VIX, and VIX-intraday bars from Polygon.io (~3 years of data).
2. **Detects** intraday swing-high → swing-low → recovery events (reversal candidates).
3. **Engineers 98 features** per event at the exact swing-low candle — price overextension, realized volatility & jump proxies, order-flow (signed SPY volume), candle patterns, VIX momentum, straddle proxy, opening-range gaps, and interaction terms.
4. **Trains a stacking ensemble** (RandomForest + GradientBoosting + XGBoost + LightGBM + Logistic meta-learner) with Leave-One-Out CV and precision-first threshold tuning.
5. **Visualises** everything in an interactive Plotly dashboard: 1-min candlestick chart pinned to exact prediction candles, feature importance, confusion matrix, squeeze analysis.

**Key metric: Precision ≥ 88%** — when the model says "reversal", it reverses.

---

## Architecture

```
Polygon.io API
    │
    ├─ I:SPX  1-min OHLC  ──────────────────────────┐
    ├─ I:VIX  1-min OHLC  ─── intraday vol regime   │
    ├─ SPY    1-min OHLCV ─── volume / flow proxy   │
    ├─ I:VIX  daily OHLC  ─── regime baseline       │
    └─ SPX options snapshot ─ straddle / greeks      │
                                                     ▼
                              Reversal Detection (reversals.py)
                              ├─ Swing high → low → recovery scan
                              ├─ ATM 0DTE call squeeze detection
                              └─ Event labelling (reversal ≥ 20 pts = 1)
                                                     │
                                                     ▼
                              Feature Engineering (features.py)  98 features
                              ├─ A: Price stretch vs EMA / return z-scores
                              ├─ B: RV, bipower variation, jump share, semivariance
                              ├─ C: SPY relative volume, signed vol imbalance, CVD slope
                              ├─ D: RSI, Stoch-K, Bollinger %b, BB-KC squeeze, MACD, wick ratios
                              ├─ E: VIX intraday level/change/accel, SPX-VIX beta
                              ├─ F: VWAP deviation, dist-to-prev-close, dist-to-day-high
                              ├─ G: Gap pts/%, opening 5-min range, fall speed
                              ├─ H: ATM straddle proxy, intraday VRP (IV² − RV)
                              ├─ I: Squeeze encoded/duration/expansion ratio
                              ├─ J: VIX daily 1d/5d momentum
                              ├─ K: Time-of-day sin/cos, open-30, close-30
                              └─ L: Interaction terms (stretch×squeeze, rv_spike×flow, …)
                                                     │
                                                     ▼
                              Stacking Ensemble (model.py)
                              ├─ Base: RandomForest + GradientBoosting + XGBoost + LightGBM
                              ├─ Meta: LogisticRegression on OOF probabilities
                              ├─ CV:   Leave-One-Out (optimal for small n)
                              └─ Threshold tuned for max precision @ recall ≥ 20%
                                                     │
                                                     ▼
                              Dashboard (dashboard/index.html)
                              ├─ Tab: Prediction Model
                              │   ├─ 1-min Plotly candlestick (TradingView-style zoom)
                              │   ├─ Prediction marker pinned to exact swing-low candle
                              │   ├─ Feature importance (98 features, tree avg)
                              │   └─ Confusion matrix + squeeze lift chart
                              ├─ Tab: Reversal Analysis
                              ├─ Tab: Patches Analysis (EM zones)
                              └─ Tab: Price Summary
```

---

## Model performance

| Metric | Value |
|---|---|
| Events | ~500 (3 years) |
| Features | 98 |
| Models | RF + GBM + XGB + LGBM + meta:LR |
| CV method | Leave-One-Out |
| **Precision** | **≥ 88%** — when model says reversal, it reverses |
| Recall | ~25–35% |
| Threshold | 0.77–0.80 (precision-first tuning) |

> **Design philosophy:** False positives are costly (trading the wrong direction). False negatives (missing a reversal) are acceptable. The threshold is tuned to maximise precision subject to recall ≥ 20%.

---

## Feature groups (98 total)

| Group | Count | Examples |
|---|---|---|
| Price overextension | 12 | stretch_5/20/60, ret_z_5/10/20/30 |
| Realized volatility | 12 | rv_5/20/60, jump_share, down_sv, range_vol, rv_spike |
| Return autocorrelation | 3 | autocorr_10/20/30 |
| Volume / order-flow (SPY) | 9 | rel_vol_5/20/30, signed_vol_imb_10/30, cvd_slope |
| Candle patterns | 5 | lower_wick_ratio, upper_wick, body_to_range, avg_lower_wick_5 |
| Oscillators | 7 | rsi_14, stoch_k_14, bb_pct_b, bb_bandwidth, bb_kc_squeeze, macd_hist |
| Trend geometry | 4 | trend_slope_20/60, trend_curvature_20/60 |
| VIX intraday | 6 | vix_intraday, vix_z_60d, vix_chg_5m/30m, vix_accel, spx_vix_beta |
| Market structure | 4 | vwap_dev, dist_to_prev_close, dist_to_open, dist_to_day_high |
| Gap / opening | 7 | gap_pts, gap_pct, gap_dir, open5_range, open5_dir, fall_speed, pct_day |
| Straddle / VRP | 3 | straddle_proxy, fall_vs_straddle, intraday_vrp |
| Squeeze / options | 5 | squeeze_encoded, squeeze_duration, expansion_ratio |
| VIX daily momentum | 4 | vix_1d_chg, vix_5d_chg, vix_vs_20d, vix_trend_up |
| Time / calendar | 8 | time_sin/cos, is_open30, is_close30, dow, is_monday, is_friday |
| Interactions | 6 | stretch×squeeze, rv_spike×vol_imb, vix_chg×fall, gap×fall, autocorr×stretch, wick×rv |

---

## Quick start

### Prerequisites
- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Polygon.io](https://polygon.io) API key (Starter or above for 1-min index data)
- macOS/Linux; Python ≥ 3.11

### Setup

```bash
git clone https://github.com/bhageerath17/polygon.git
cd polygon
cp .env.example .env          # add your POLYGON_API_KEY
make setup                    # creates .venv, installs deps
```

### Run the full pipeline

```bash
make fetch                    # SPX 1-min + VIX daily + options snapshot
make fetch-intraday           # SPY 1-min + VIX 1-min (volume & vol regime)
make analyze                  # patches / EM zone analysis
make reversal                 # detect swing events + ATM squeeze
make model                    # train ensemble, save model_results.json
make serve                    # open http://localhost:8000/dashboard/
```

Or rebuild everything in one shot:

```bash
make fetch && make fetch-intraday && make analyze && make reversal && make model && make serve
```

---

## Makefile commands

| Command | Description |
|---|---|
| `make setup` | Install deps into `.venv` |
| `make fetch` | SPX 1-min + VIX daily + options snapshot |
| `make fetch-intraday` | SPY 1-min + VIX 1-min |
| `make analyze` | Patches / EM zone analysis → `patches_analysis.json` |
| `make reversal` | Detect reversals + ATM squeeze → `reversal_analysis.json` |
| `make model` | Train ensemble → `model_results.json` |
| `make serve` | Copy JSON to `dashboard/` + serve on port 8000 |
| `make clean` | Remove `.venv` and caches |

---

## Project structure

```
polygon/
├── app/
│   ├── analysis/
│   │   ├── features.py       # 98-feature engineering
│   │   ├── model.py          # stacking ensemble + LOOCV
│   │   ├── patches.py        # EM zone analysis
│   │   └── reversals.py      # swing detection + squeeze
│   ├── fetchers/
│   │   ├── intraday.py       # generic 1-min bar fetcher
│   │   ├── options.py        # options 1-min + snapshot
│   │   ├── price.py          # SPX 1-min
│   │   └── vix.py            # VIX daily
│   ├── client.py             # Polygon HTTP client
│   ├── config.py             # settings dataclass
│   └── io.py                 # CSV helpers
├── scripts/
│   ├── fetch.py              # SPX + VIX daily + options
│   ├── fetch_intraday.py     # SPY + VIX 1-min
│   ├── analyze.py            # patches pipeline
│   ├── reversal_analysis.py  # reversal pipeline
│   └── build_model.py        # model training entry point
├── dashboard/
│   └── index.html            # single-page dashboard (Plotly)
├── data/                     # CSV + JSON outputs (gitignored)
├── config.toml               # date ranges, paths, parameters
└── .env                      # API key (never committed)
```

---

## Environment variables

```bash
POLYGON_API_KEY=your_key_here
```

---

## Dependencies

- `pandas`, `numpy`, `scikit-learn` — data + base ML
- `xgboost`, `lightgbm` — gradient boosting models
- `requests`, `python-dotenv` — API + config
- Plotly.js (CDN) — interactive dashboard charts
