from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent

load_dotenv(_ROOT / ".env")

with open(_ROOT / "config.toml", "rb") as _f:
    _cfg = tomllib.load(_f)


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    # SPX
    ticker: str
    start_date: str
    end_date: str
    options_limit: int
    # VIX daily
    vix_ticker: str
    vix_start_date: str
    vix_end_date: str
    # SPY 1-min
    spy_ticker: str
    spy_start_date: str
    spy_end_date: str
    # VIX 1-min
    vix_1min_ticker: str
    vix_1min_start_date: str
    vix_1min_end_date: str
    # Analysis
    lookback_mins: int
    sig_levels: list
    # Reversals
    reversal_window: int
    squeeze_lookback: int
    strike_rounding: int
    # Paths
    data_dir: Path
    price_csv: Path
    options_csv: Path
    vix_csv: Path
    spy_1min_csv: Path
    vix_1min_csv: Path
    vix1d_csv: Path
    analysis_csv: Path
    analysis_json: Path
    reversal_csv: Path
    reversal_json: Path


settings = Settings(
    api_key=os.environ["POLYGON_API_KEY"],
    base_url=_cfg["api"]["base_url"],
    ticker=_cfg["data"]["ticker"],
    start_date=_cfg["data"]["start_date"],
    end_date=_cfg["data"]["end_date"],
    options_limit=_cfg["data"]["options_limit"],
    vix_ticker=_cfg["vix"]["ticker"],
    vix_start_date=_cfg["vix"]["start_date"],
    vix_end_date=_cfg["vix"]["end_date"],
    spy_ticker=_cfg["spy"]["ticker"],
    spy_start_date=_cfg["spy"]["start_date"],
    spy_end_date=_cfg["spy"]["end_date"],
    vix_1min_ticker=_cfg["vix_1min"]["ticker"],
    vix_1min_start_date=_cfg["vix_1min"]["start_date"],
    vix_1min_end_date=_cfg["vix_1min"]["end_date"],
    lookback_mins=_cfg["analysis"]["lookback_mins"],
    sig_levels=_cfg["analysis"]["sig_levels"],
    reversal_window=_cfg["reversals"]["rolling_window_mins"],
    squeeze_lookback=_cfg["reversals"]["squeeze_lookback"],
    strike_rounding=_cfg["reversals"]["strike_rounding"],
    data_dir=_ROOT / _cfg["paths"]["data_dir"],
    price_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["price_csv"],
    options_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["options_csv"],
    vix_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["vix_csv"],
    spy_1min_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["spy_1min_csv"],
    vix_1min_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["vix_1min_csv"],
    vix1d_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["vix1d_csv"],
    analysis_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["analysis_csv"],
    analysis_json=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["analysis_json"],
    reversal_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["reversal_csv"],
    reversal_json=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["reversal_json"],
)
