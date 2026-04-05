from pathlib import Path
import pandas as pd


def save_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Saved {len(df):,} rows -> {path}")


def append_csv(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append new_df to existing CSV, dedup by index, return merged DataFrame.

    If the file doesn't exist yet, behaves like save_csv.
    Index is expected to be a datetime string (the CSV index column).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, index_col=0)
        merged = pd.concat([existing, new_df])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        merged.to_csv(path)
        print(f"Appended {len(new_df):,} rows → {len(merged):,} total -> {path}")
        return merged
    else:
        save_csv(new_df, path)
        return new_df


def last_date_in_csv(path: Path) -> str | None:
    """Return the latest date (YYYY-MM-DD) found in the CSV index, or None.

    Works for both datetime-with-timezone indexes (1-min bars) and plain
    date indexes (VIX daily).
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        idx = pd.read_csv(path, usecols=[0], header=0).iloc[:, 0]
        # Try UTC-aware first (1-min bars have mixed DST offsets),
        # fall back to naive for plain-date indexes (VIX daily).
        try:
            last = pd.to_datetime(idx, utc=True, errors="coerce").dropna().max()
        except Exception:
            last = pd.to_datetime(idx, utc=False, errors="coerce").dropna().max()
        if pd.isna(last):
            return None
        return last.strftime("%Y-%m-%d")
    except Exception:
        return None


def clean_for_json(obj):
    """Recursively replace NaN/Inf with None so output is valid JSON."""
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return obj
