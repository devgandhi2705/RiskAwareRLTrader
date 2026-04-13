"""
feature_engineering.py
-----------------------
Computes per-asset technical indicators and return features from raw OHLCV data.

Features generated per asset
-----------------------------
  - Daily return      : percentage change of Close
  - RSI (14)          : Relative Strength Index
  - MA20              : 20-day simple moving average of Close
  - Volatility (20)   : 20-day rolling std of daily returns (annualised)

Part of: Safe RL for Risk-Constrained Portfolio Management
Stage:   Data Pipeline — Step 2: Feature Engineering
"""

import logging
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
RSI_PERIOD       = 14
MA_PERIOD        = 20
VOLATILITY_WINDOW = 20

# Approximate annualisation factor for daily volatility
# (crypto trades ~365 days; equities ~252 — using 252 as conservative default)
ANNUALISE_FACTOR = np.sqrt(252)


# ── Functions ─────────────────────────────────────────────────────────────────

def _get_asset_names(df: pd.DataFrame) -> list[str]:
    """
    Infer unique asset prefix names from column headers.

    Column naming convention:  {AssetName}_{Field}
    Example: 'BTC_Open' → prefix = 'BTC'

    Parameters
    ----------
    df : Wide merged DataFrame.

    Returns
    -------
    Sorted list of unique asset prefix strings.
    """
    prefixes: set[str] = set()
    for col in df.columns:
        parts = col.split("_", maxsplit=1)
        if len(parts) == 2:
            prefixes.add(parts[0])
    return sorted(prefixes)


def compute_features(
    df: pd.DataFrame,
    asset_name: str,
    rsi_period: int = RSI_PERIOD,
    ma_period: int = MA_PERIOD,
    vol_window: int = VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """
    Add technical-indicator columns for a single asset to the DataFrame in-place.

    Columns added
    -------------
    {name}_Return     : Daily percentage return of Close price.
    {name}_RSI        : RSI(rsi_period) on Close prices.
    {name}_MA{n}      : Simple moving average over ma_period days of Close.
    {name}_Volatility : Rolling std of daily returns (vol_window days),
                        annualised by sqrt(252).

    Parameters
    ----------
    df         : Wide DataFrame containing at minimum {asset_name}_Close.
    asset_name : Asset prefix (e.g. 'BTC').
    rsi_period : Look-back for RSI computation.
    ma_period  : Look-back for the moving average.
    vol_window : Rolling window for volatility.

    Returns
    -------
    df with new indicator columns appended (modified in-place, also returned).

    Raises
    ------
    KeyError if the expected Close column is not found.
    """
    close_col = f"{asset_name}_Close"
    if close_col not in df.columns:
        raise KeyError(
            f"Expected column '{close_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    close = df[close_col]

    # ── Daily returns ──────────────────────────────────────────────────────────
    df[f"{asset_name}_Return"] = close.pct_change()

    # ── RSI ────────────────────────────────────────────────────────────────────
    rsi_indicator = RSIIndicator(close=close, window=rsi_period, fillna=False)
    df[f"{asset_name}_RSI"] = rsi_indicator.rsi()

    # ── Moving Average ─────────────────────────────────────────────────────────
    ma_indicator = SMAIndicator(close=close, window=ma_period, fillna=False)
    df[f"{asset_name}_MA{ma_period}"] = ma_indicator.sma_indicator()

    # ── Rolling Volatility ─────────────────────────────────────────────────────
    daily_returns = df[f"{asset_name}_Return"]
    df[f"{asset_name}_Volatility"] = (
        daily_returns.rolling(window=vol_window).std() * ANNUALISE_FACTOR
    )

    return df


def engineer_all_features(
    raw_df: pd.DataFrame,
    rsi_period: int = RSI_PERIOD,
    ma_period: int = MA_PERIOD,
    vol_window: int = VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """
    Apply feature engineering to every asset found in the wide DataFrame.

    Iterates over detected asset prefixes and calls compute_features() for each,
    accumulating indicator columns alongside the raw OHLCV data.

    Parameters
    ----------
    raw_df     : Output of merge_raw_data() — wide OHLCV DataFrame.
    rsi_period : RSI window length.
    ma_period  : SMA window length.
    vol_window : Rolling volatility window.

    Returns
    -------
    Enriched DataFrame with indicator columns added for every asset.
    """
    df = raw_df.copy()
    asset_names = _get_asset_names(df)

    if not asset_names:
        raise ValueError(
            "No asset prefixes detected in DataFrame columns. "
            "Expected format: '{AssetName}_{Field}' (e.g. 'BTC_Close')."
        )

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("Assets detected: %s", asset_names)
    logger.info(
        "Indicators: RSI(%d)  MA(%d)  Volatility(%d)",
        rsi_period, ma_period, vol_window,
    )
    logger.info("=" * 60)

    for name in tqdm(asset_names, desc="Engineering features", unit="asset"):
        close_col = f"{name}_Close"
        if close_col not in df.columns:
            logger.warning(
                "Skipping '%s': column '%s' not found.", name, close_col
            )
            continue

        try:
            df = compute_features(df, name, rsi_period, ma_period, vol_window)
            logger.info("  ✓ %-10s  indicators computed.", name)
        except Exception as exc:
            logger.error("  ✗ Failed to compute features for %s: %s", name, exc)

    n_indicator_cols = len(df.columns) - len(raw_df.columns)
    logger.info(
        "Feature engineering complete.  Added %d indicator columns.",
        n_indicator_cols,
    )
    return df


# ── Column ordering helper ────────────────────────────────────────────────────

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns so all fields for a given asset are grouped together.

    Output order per asset:
        Open, High, Low, Close, Volume, Return, RSI, MA20, Volatility

    Parameters
    ----------
    df : Enriched wide DataFrame.

    Returns
    -------
    DataFrame with columns reordered by asset group.
    """
    field_priority = {
        "Open":       0,
        "High":       1,
        "Low":        2,
        "Close":      3,
        "Volume":     4,
        "Return":     5,
        "RSI":        6,
        "MA20":       7,
        "Volatility": 8,
    }

    def sort_key(col: str) -> tuple[str, int]:
        parts = col.split("_", maxsplit=1)
        if len(parts) == 2:
            asset, field = parts
            # partial match: MA20, MA50, etc.
            priority = next(
                (v for k, v in field_priority.items() if field.startswith(k)),
                99,
            )
            return (asset, priority)
        return (col, 99)

    sorted_cols = sorted(df.columns, key=sort_key)
    return df[sorted_cols]
