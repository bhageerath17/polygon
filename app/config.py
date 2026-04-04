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
    ticker: str
    start_date: str
    end_date: str
    options_limit: int
    data_dir: Path
    price_csv: Path
    options_csv: Path


settings = Settings(
    api_key=os.environ["POLYGON_API_KEY"],
    base_url=_cfg["api"]["base_url"],
    ticker=_cfg["data"]["ticker"],
    start_date=_cfg["data"]["start_date"],
    end_date=_cfg["data"]["end_date"],
    options_limit=_cfg["data"]["options_limit"],
    data_dir=_ROOT / _cfg["paths"]["data_dir"],
    price_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["price_csv"],
    options_csv=_ROOT / _cfg["paths"]["data_dir"] / _cfg["paths"]["options_csv"],
)
