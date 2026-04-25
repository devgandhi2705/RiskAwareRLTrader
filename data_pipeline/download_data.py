"""
download_data.py
----------------
Downloads historical OHLCV data for a defined asset universe from Yahoo Finance.

Part of: Safe RL for Risk-Constrained Portfolio Management
Stage:   Data Pipeline — Step 1: Raw Data Download
"""

import os
import time
import logging
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Asset Universe ────────────────────────────────────────────────────────────
# Maps Yahoo Finance ticker → short friendly name used as column prefix
ASSET_UNIVERSE: dict[str, str] = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SPY":     "SPY",
    "GLD":     "GLD",
    "SI=F":    "Silver",
    "^NSEI":   "Nifty50",
    "^BSESN":  "Sensex",
}

START_DATE      = "2016-01-01"
END_DATE        = "2025-12-31"
INTERVAL        = "1d"
RAW_OUTPUT_PATH = os.path.join("data", "raw_market_data.csv")

# ── Retry configuration ───────────────────────────────────────────────────────
MAX_RETRIES   = 5   # maximum attempts per ticker
RETRY_BACKOFF = 5   # initial wait in seconds (doubles each retry)


# ── Functions ─────────────────────────────────────────────────────────────────

def download_asset_data(
    ticker: str,
    friendly_name: str,
    start: str = START_DATE,
    end: str = END_DATE,
    interval: str = INTERVAL,
) -> pd.DataFrame | None:
    """
    Download OHLCV data for a single ticker from Yahoo Finance.

    Retries up to MAX_RETRIES times with exponential backoff when a
    YFRateLimitError is raised.

    Parameters
    ----------
    ticker        : Yahoo Finance ticker symbol (e.g. 'BTC-USD').
    friendly_name : Short name used as a column prefix (e.g. 'BTC').
    start         : Inclusive start date string 'YYYY-MM-DD'.
    end           : Exclusive end date string 'YYYY-MM-DD'.
    interval      : Candle interval; '1d' for daily.

    Returns
    -------
    pd.DataFrame indexed by Date with columns
        {name}_Open, {name}_High, {name}_Low, {name}_Close, {name}_Volume.
    Returns None on download failure or empty response.
    """
    logger.info("  Fetching %-10s  [%s]", friendly_name, ticker)

    raw = None
    wait = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            break  # success — exit retry loop

        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = (
                "YFRateLimitError" in type(exc).__name__
                or "Too Many Requests" in exc_str
                or "Rate limited" in exc_str
            )
            if is_rate_limit and attempt < MAX_RETRIES:
                logger.warning(
                    "  ⚠ Rate-limited on %s (attempt %d/%d). "
                    "Waiting %ds before retry…",
                    ticker, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                wait *= 2  # exponential backoff
            else:
                logger.error("  ✗ Download failed for %s: %s", ticker, exc)
                return None

    if raw is None or raw.empty:
        logger.warning("  ✗ Empty response for %s — skipping.", friendly_name)
        return None

    # yfinance may return MultiIndex columns when downloading a single ticker
    # in certain versions; flatten to single level.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only standard OHLCV columns
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in ohlcv if c in raw.columns]
    raw = raw[available].copy()

    # Namespace columns: Open → BTC_Open, etc.
    raw.columns = [f"{friendly_name}_{col}" for col in raw.columns]

    # Standardise index
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "Date"

    logger.info(
        "  ✓ %-10s  rows=%d  range=[%s → %s]",
        friendly_name,
        len(raw),
        raw.index.min().date(),
        raw.index.max().date(),
    )
    return raw


def download_all_assets(
    asset_universe: dict[str, str] = ASSET_UNIVERSE,
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Iterate over the full asset universe and download OHLCV data for each.

    A 1-second courtesy delay is inserted between tickers to reduce the
    likelihood of hitting Yahoo Finance rate limits.

    Parameters
    ----------
    asset_universe : Mapping of {yahoo_ticker: friendly_name}.
    start          : Data start date.
    end            : Data end date.

    Returns
    -------
    dict of {friendly_name: DataFrame}, one entry per successfully downloaded asset.
    """
    logger.info("=" * 60)
    logger.info("RAW DATA DOWNLOAD  (%s → %s)", start, end)
    logger.info("Asset universe: %s", list(asset_universe.values()))
    logger.info("=" * 60)

    results: dict[str, pd.DataFrame] = {}

    for ticker, name in tqdm(asset_universe.items(), desc="Downloading", unit="asset"):
        df = download_asset_data(ticker, name, start, end)
        if df is not None:
            results[name] = df
        # Small courtesy delay between tickers to reduce rate-limit risk
        time.sleep(1)

    n_ok = len(results)
    n_total = len(asset_universe)
    logger.info("Download complete: %d/%d assets succeeded.", n_ok, n_total)

    if n_ok < n_total:
        missing = set(asset_universe.values()) - set(results.keys())
        logger.warning("Failed assets: %s", missing)

    return results


def merge_raw_data(asset_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge per-asset DataFrames into a single wide DataFrame aligned by Date.

    Uses an outer join so that no dates are silently dropped when assets have
    different trading calendars (e.g. crypto trades 7 days a week; equities
    do not). Missing cells introduced by the outer join are left as NaN for
    the feature engineering step to handle.

    Parameters
    ----------
    asset_data : dict returned by download_all_assets().

    Returns
    -------
    Wide pd.DataFrame indexed by Date with all asset columns side by side.
    """
    if not asset_data:
        raise ValueError("asset_data is empty — nothing to merge.")

    frames = list(asset_data.values())
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how="outer")

    merged.sort_index(inplace=True)
    merged.index = pd.to_datetime(merged.index)

    logger.info(
        "Merged raw dataset: %d rows × %d cols  (%s → %s)",
        len(merged),
        len(merged.columns),
        merged.index.min().date(),
        merged.index.max().date(),
    )
    return merged


def save_raw_data(df: pd.DataFrame, path: str = RAW_OUTPUT_PATH) -> None:
    """
    Save the raw merged DataFrame to CSV.

    Parameters
    ----------
    df   : DataFrame to persist.
    path : Destination file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path)
    logger.info("Saved raw data → %s  (%d rows × %d cols)", path, *df.shape)


def run_download_pipeline() -> pd.DataFrame:
    """
    End-to-end orchestration for the raw-data download step.

    Steps
    -----
    1. Download OHLCV data for every asset (with retry/backoff on rate limits).
    2. Merge into a single wide DataFrame.
    3. Save to disk.

    Returns
    -------
    Raw merged DataFrame.
    """
    asset_data = download_all_assets()
    raw_df     = merge_raw_data(asset_data)
    save_raw_data(raw_df)
    return raw_df


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_download_pipeline()
