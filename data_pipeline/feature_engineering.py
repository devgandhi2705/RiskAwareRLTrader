"""
feature_engineering.py  (v3 — Simplified, No Regime Features)
--------------------------------------------------------------
Computes exactly 5 per-asset technical indicators from raw OHLCV data.

Features per asset
------------------
  {A}_return     : daily pct-change of Close
  {A}_MA20       : 20-day SMA
  {A}_MA50       : 50-day SMA
  {A}_RSI        : RSI(14)
  {A}_volatility : 20-day rolling std of returns (annualised)

Total: 7 assets × 5 = 35 feature columns.

Regime features (trend_strength, *_regime, *_bull_prob, *_neutral_prob,
*_bear_prob, market_volatility) are NOT computed or added.

WARMUP_ROWS = 50  (MA50 warm-up window).

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
RSI_PERIOD        = 14
MA_FAST           = 20
MA_SLOW           = 50
VOLATILITY_WINDOW = 20
ANNUALISE_FACTOR  = np.sqrt(252)
WARMUP_ROWS       = MA_SLOW   # 50 rows before indicators are reliable

ASSETS = ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty50", "Sensex"]
FEATURES_PER_ASSET = ["return", "MA20", "MA50", "RSI", "volatility"]
N_ASSET_FEATURES   = len(ASSETS) * len(FEATURES_PER_ASSET)   # 35


# ── RSI helper (no external dependency) ───────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


# ── Per-asset feature computation ─────────────────────────────────────────────

def compute_features(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Add 5 indicator columns for one asset to the DataFrame.

    Requires column: {asset}_Close
    Adds columns:
        {asset}_return
        {asset}_MA20
        {asset}_MA50
        {asset}_RSI
        {asset}_volatility
    """
    close_col = f"{asset}_Close"
    if close_col not in df.columns:
        logger.warning("Skipping %s: column %s not found.", asset, close_col)
        return df

    close = df[close_col]

    df[f"{asset}_return"]     = close.pct_change()
    df[f"{asset}_MA20"]       = close.rolling(MA_FAST,  min_periods=1).mean()
    df[f"{asset}_MA50"]       = close.rolling(MA_SLOW,  min_periods=1).mean()
    df[f"{asset}_RSI"]        = _compute_rsi(close, RSI_PERIOD)
    df[f"{asset}_volatility"] = (
        df[f"{asset}_return"]
        .rolling(VOLATILITY_WINDOW, min_periods=5).std()
        * ANNUALISE_FACTOR
    )
    return df


def engineer_all_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to every asset in ASSETS.

    Parameters
    ----------
    raw_df : Wide OHLCV DataFrame with columns like {asset}_Close.

    Returns
    -------
    DataFrame enriched with 35 indicator columns.
    """
    df = raw_df.copy()

    logger.info("=" * 55)
    logger.info("FEATURE ENGINEERING  (v3 — 5 indicators per asset)")
    logger.info("Assets: %s", ASSETS)
    logger.info("=" * 55)

    for asset in tqdm(ASSETS, desc="Engineering features", unit="asset"):
        df = compute_features(df, asset)
        logger.info("  ✓ %-10s  return / MA20 / MA50 / RSI / volatility", asset)

    new_cols = [f"{a}_{f}" for a in ASSETS for f in FEATURES_PER_ASSET]
    added = sum(1 for c in new_cols if c in df.columns)
    logger.info("Feature engineering complete.  Added %d indicator columns.", added)
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group columns by asset.

    Order per asset: return, MA20, MA50, RSI, volatility
    All other columns (OHLCV) follow at the end.
    """
    priority = {f: i for i, f in enumerate(FEATURES_PER_ASSET)}

    indicator_cols = []
    for asset in ASSETS:
        for feat in FEATURES_PER_ASSET:
            col = f"{asset}_{feat}"
            if col in df.columns:
                indicator_cols.append(col)

    other_cols = [c for c in df.columns if c not in indicator_cols]
    return df[indicator_cols + other_cols]