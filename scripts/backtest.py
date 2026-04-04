"""Entry point: load fetched data and run backtest strategy."""
import pandas as pd
from app.config import settings


def main() -> None:
    print(f"Loading price data from {settings.price_csv}...")
    spx = pd.read_csv(settings.price_csv, index_col="datetime", parse_dates=True)

    print(f"Loading options data from {settings.options_csv}...")
    opts = pd.read_csv(settings.options_csv)

    print(f"Price bars: {spx.shape}  |  Options rows: {opts.shape}")
    # TODO: implement backtest strategy here


if __name__ == "__main__":
    main()
