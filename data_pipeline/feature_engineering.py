"""
feature_engineering.py  (v2 — Regime-Pipeline Ready)
------------------------------------------------------
Computes per-asset technical indicators and return features from raw OHLCV data.

New in v2
---------
  F1 : MA50 and MA200 columns added per asset — required by regime_detection v2
       for per-asset trend_strength = (MA50 − MA200) / MA200.
  F2 : WARMUP_ROWS exported (= 200, MA_SLOW) so build_dataset.py can use
       MA200 as the burn-in window.
  F3 : reorder_columns updated to include MA50, MA200 in column sort priority.

Features generated per asset
-----------------------------
  {name}_Return       : Daily pct_change of Close
  {name}_RSI          : RSI(14) on Close
  {name}_MA20         : 20-day SMA of Close  (existing)
  {name}_MA50         : 50-day SMA of Close  (new — for regime trend_strength)
  {name}_MA200        : 200-day SMA of Close (new — for regime trend_strength)
  {name}_Volatility   : Rolling 20-day std of daily returns (annualised)

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
RSI_PERIOD        = 14
MA_PERIOD         = 20    # existing MA20
MA_FAST           = 50    # F1: MA50 for regime detection
MA_SLOW           = 200   # F1: MA200 for regime detection
VOLATILITY_WINDOW = 20

# Warmup rows needed before indicators are valid.
# MA200 is the longest window — use it as the burn-in period.
# Exported so build_dataset.py can drop exactly this many rows.
WARMUP_ROWS = MA_SLOW   # 200

ANNUALISE_FACTOR = np.sqrt(252)


# ── Internal helper ────────────────────────────────────────────────────────────

def _get_asset_names(df: pd.DataFrame) -> list:
    """
    Infer unique asset prefix names from column headers.
    Convention: '{AssetName}_{Field}'  →  prefix = 'AssetName'.
    """
    prefixes: set = set()
    for col in df.columns:
        parts = col.split("_", maxsplit=1)
        if len(parts) == 2:
            prefixes.add(parts[0])
    return sorted(prefixes)


# ── Per-asset feature computation ─────────────────────────────────────────────

def compute_features(
    df: pd.DataFrame,
    asset_name: str,
    rsi_period: int = RSI_PERIOD,
    ma_period: int  = MA_PERIOD,
    vol_window: int = VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """
    Add technical-indicator columns for a single asset to the DataFrame.

    Columns added
    -------------
    {name}_Return     : Daily percentage return of Close price.
    {name}_RSI        : RSI(rsi_period) on Close prices.
    {name}_MA20       : 20-day SMA of Close.
    {name}_MA50       : 50-day SMA of Close  (F1).
    {name}_MA200      : 200-day SMA of Close (F1).
    {name}_Volatility : Rolling std of daily returns (vol_window days, annualised).

    Parameters
    ----------
    df         : Wide DataFrame with at minimum {asset_name}_Close.
    asset_name : Asset prefix (e.g. 'BTC', 'SPY').
    rsi_period : Look-back for RSI computation.
    ma_period  : Look-back for the short moving average (MA20).
    vol_window : Rolling window for volatility (annualised).

    Returns
    -------
    df with new indicator columns appended (modified in-place, also returned).
    """
    close_col = f"{asset_name}_Close"
    if close_col not in df.columns:
        raise KeyError(
            f"Expected column '{close_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    close = df[close_col]

    # ── Daily returns ──────────────────────────────────────────────────────────
    df[f"{asset_name}_Return"] = close.pct_change()

    # ── RSI ────────────────────────────────────────────────────────────────────
    rsi_indicator = RSIIndicator(close=close, window=rsi_period, fillna=False)
    df[f"{asset_name}_RSI"] = rsi_indicator.rsi()

    # ── MA20 (existing short-term moving average) ──────────────────────────────
    ma20 = SMAIndicator(close=close, window=ma_period, fillna=False)
    df[f"{asset_name}_MA{ma_period}"] = ma20.sma_indicator()

    # ── MA50 (F1 — regime trend calculation) ──────────────────────────────────
    ma50 = SMAIndicator(close=close, window=MA_FAST, fillna=False)
    df[f"{asset_name}_MA50"] = ma50.sma_indicator()

    # ── MA200 (F1 — regime trend calculation) ─────────────────────────────────
    ma200 = SMAIndicator(close=close, window=MA_SLOW, fillna=False)
    df[f"{asset_name}_MA200"] = ma200.sma_indicator()

    # ── Rolling Volatility (annualised) ───────────────────────────────────────
    daily_returns = df[f"{asset_name}_Return"]
    df[f"{asset_name}_Volatility"] = (
        daily_returns.rolling(window=vol_window).std() * ANNUALISE_FACTOR
    )

    return df


def engineer_all_features(
    raw_df: pd.DataFrame,
    rsi_period: int = RSI_PERIOD,
    ma_period: int  = MA_PERIOD,
    vol_window: int = VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """
    Apply feature engineering to every asset found in the wide DataFrame.

    Parameters
    ----------
    raw_df     : Output of merge_raw_data() — wide OHLCV DataFrame.
    rsi_period : RSI window length.
    ma_period  : SMA window length (MA20).
    vol_window : Rolling volatility window.

    Returns
    -------
    Enriched DataFrame with indicator columns added for every detected asset.
    """
    df = raw_df.copy()
    asset_names = _get_asset_names(df)

    if not asset_names:
        raise ValueError(
            "No asset prefixes detected in DataFrame columns. "
            "Expected format: '{AssetName}_{Field}' (e.g. 'BTC_Close')."
        )

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING  (v2)")
    logger.info("Assets detected: %s", asset_names)
    logger.info(
        "Indicators: RSI(%d)  MA%d  MA50  MA200  Volatility(%d)",
        rsi_period, ma_period, vol_window,
    )
    logger.info("=" * 60)

    for name in tqdm(asset_names, desc="Engineering features", unit="asset"):
        close_col = f"{name}_Close"
        if close_col not in df.columns:
            logger.warning(
                "Skipping '%s': column '%s' not found in DataFrame.", name, close_col
            )
            continue

        try:
            df = compute_features(df, name, rsi_period, ma_period, vol_window)
            logger.info(
                "  ✓ %-10s  indicators computed (Return/RSI/MA20/MA50/MA200/Vol).",
                name,
            )
        except Exception as exc:
            logger.error("  ✗ Failed to compute features for %s: %s", name, exc)

    n_new = len(df.columns) - len(raw_df.columns)
    logger.info(
        "Feature engineering complete.  Added %d indicator columns.", n_new
    )
    return df


# ── Column ordering helper ─────────────────────────────────────────────────────

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns so all fields for a given asset are grouped together.

    Output order per asset:
        Open, High, Low, Close, Volume, Return, RSI,
        MA20, MA50, MA200, Volatility,
        trend_strength, volatility (regime), regime,
        bull_prob, neutral_prob, bear_prob
    """
    field_priority = {
        "Open":           0,
        "High":           1,
        "Low":            2,
        "Close":          3,
        "Volume":         4,
        "Return":         5,
        "RSI":            6,
        "MA20":           7,
        "MA50":           8,
        "MA200":          9,
        "Volatility":     10,   # feature engineering vol (annualised)
        "trend_strength": 11,   # regime detection
        "volatility":     12,   # regime detection vol (30-day, lowercase)
        "regime":         13,
        "bull_prob":      14,
        "neutral_prob":   15,
        "bear_prob":      16,
    }

    def sort_key(col: str):
        parts = col.split("_", maxsplit=1)
        if len(parts) == 2:
            asset, field = parts
            priority = next(
                (v for k, v in field_priority.items() if field.startswith(k)),
                99,
            )
            return (asset, priority, field)
        return (col, 99, col)

    sorted_cols = sorted(df.columns, key=sort_key)
    return df[sorted_cols]