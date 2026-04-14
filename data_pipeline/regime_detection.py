"""
regime_detection.py  (v2 — Multi-Asset Regime Probabilities)
-------------------------------------------------------------
Replaces single-BTC regime detection with per-asset regime signals,
probability features, and a backward-compatible global regime column.

New in v2
---------
  P1 : Per-asset trend_strength   = (MA50 − MA200) / MA200
  P2 : Per-asset volatility       = rolling 30-day std of returns
  P3 : Per-asset discrete regime  = {-1, 0, +1} from trend thresholds
  P4 : Per-asset regime probabilities  via logistic transform of trend_strength
         {asset}_bull_prob, {asset}_neutral_prob, {asset}_bear_prob
  P5 : Global "regime" column retained for backward-compat (driven by BTC)
  P6 : Global trend_strength / market_volatility retained for backward-compat

Columns added per asset (7 assets × 6 cols = 42 new cols)
----------------------------------------------------------
  {A}_trend_strength   float   (MA50 − MA200) / MA200
  {A}_volatility       float   30-day rolling std of {A}_Return
  {A}_regime           int     {-1, 0, 1}
  {A}_bull_prob        float   [0, 1]
  {A}_neutral_prob     float   [0, 1]
  {A}_bear_prob        float   [0, 1]

Backward-compatible global columns (unchanged names)
----------------------------------------------------
  regime              int     {-1, 0, 1}   — driven by BTC as before
  trend_strength      float   — BTC trend_strength
  market_volatility   float   — BTC 30-day vol

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
ASSETS          = ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty50", "Sensex"]
MA_FAST         = 50          # fast MA window
MA_SLOW         = 200         # slow MA window
VOL_WINDOW      = 30          # rolling volatility window (days)
BULL_THRESHOLD  =  0.02       # trend_strength > +2 % → Bull
BEAR_THRESHOLD  = -0.02       # trend_strength < -2 % → Bear

# Logistic steepness for probability transformation (higher = sharper boundary)
LOGISTIC_K      = 20.0

# Fallback proxy for global regime when BTC is absent
REGIME_PROXY_COL = "BTC_Close"


# ── Private helpers ─────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))),
    )


def _regime_probs(
    trend: pd.Series,
) -> tuple:
    """
    Convert a trend_strength series into (bull_prob, neutral_prob, bear_prob).

    bull_prob    = sigmoid( k * trend_strength)
    bear_prob    = sigmoid(-k * trend_strength)
    neutral_prob = 1 − bull_prob − bear_prob   (clipped to [0, 1])

    The three values sum to 1 by construction after re-normalisation.
    """
    t     = trend.fillna(0.0).values.astype(float)
    bull  = pd.Series(_sigmoid( LOGISTIC_K * t), index=trend.index)
    bear  = pd.Series(_sigmoid(-LOGISTIC_K * t), index=trend.index)
    neut  = (1.0 - bull - bear).clip(lower=0.0)

    # Re-normalise to guarantee exact sum = 1 despite floating-point rounding
    total = bull + neut + bear
    bull  = bull / total
    neut  = neut / total
    bear  = bear / total

    return bull, neut, bear


def _asset_regime_features(
    df: pd.DataFrame,
    asset: str,
) -> pd.DataFrame:
    """
    Compute and attach all regime-related columns for one asset.

    Requires column {asset}_Close (or falls back to {asset}_Return for
    synthetic / test DataFrames).

    Columns written
    ---------------
      {asset}_trend_strength
      {asset}_volatility
      {asset}_regime
      {asset}_bull_prob
      {asset}_neutral_prob
      {asset}_bear_prob

    Returns
    -------
    df with new columns attached in-place (also returned for chaining).
    """
    close_col  = f"{asset}_Close"
    return_col = f"{asset}_Return"

    # ── Price series ──────────────────────────────────────────────────────────
    if close_col in df.columns:
        price = df[close_col].ffill().bfill()
    elif return_col in df.columns:
        # Reconstruct pseudo-price from returns for regime calculation
        r     = df[return_col].fillna(0.0)
        price = (1.0 + r).cumprod() * 100.0
        logger.debug(
            "Asset '%s': Close column missing, using reconstructed price from returns.",
            asset,
        )
    else:
        # Asset not present — fill all columns with zeros / neutral
        logger.warning(
            "Asset '%s': neither '%s' nor '%s' found — filling regime cols with zeros.",
            asset, close_col, return_col,
        )
        for col in [
            f"{asset}_trend_strength",
            f"{asset}_volatility",
            f"{asset}_bull_prob",
            f"{asset}_neutral_prob",
            f"{asset}_bear_prob",
        ]:
            df[col] = 0.0
        df[f"{asset}_regime"] = 0
        return df

    # ── Moving averages (MA50 and MA200) ──────────────────────────────────────
    # Use min_periods = half the window so early rows still get a value
    ma_fast = price.rolling(window=MA_FAST,  min_periods=MA_FAST  // 2).mean()
    ma_slow = price.rolling(window=MA_SLOW,  min_periods=MA_SLOW  // 2).mean()

    # ── Trend strength: (MA50 − MA200) / MA200 ────────────────────────────────
    # Replace zero MA200 values with NaN to avoid division-by-zero, then fill
    trend_strength = (
        (ma_fast - ma_slow) / ma_slow.replace(0.0, np.nan)
    ).fillna(0.0)

    # ── Volatility: 30-day rolling std of returns ─────────────────────────────
    if return_col in df.columns:
        ret_series = df[return_col].fillna(0.0)
    else:
        ret_series = price.pct_change().fillna(0.0)

    asset_vol = (
        ret_series.rolling(window=VOL_WINDOW, min_periods=5).std().fillna(0.0)
    )

    # ── Discrete regime ───────────────────────────────────────────────────────
    # Bull: price above slow MA AND trend_strength > BULL_THRESHOLD
    # Bear: price below slow MA AND trend_strength < BEAR_THRESHOLD
    # Neutral: everything else
    regime = pd.Series(0, index=df.index, dtype=int)
    bull_mask = (price > ma_slow) & (trend_strength > BULL_THRESHOLD)
    bear_mask = (price < ma_slow) & (trend_strength < BEAR_THRESHOLD)
    regime[bull_mask] =  1
    regime[bear_mask] = -1

    # ── Regime probabilities ──────────────────────────────────────────────────
    bull_prob, neut_prob, bear_prob = _regime_probs(trend_strength)

    # ── Write columns to DataFrame ────────────────────────────────────────────
    df[f"{asset}_trend_strength"] = trend_strength.values
    df[f"{asset}_volatility"]     = asset_vol.values
    df[f"{asset}_regime"]         = regime.values
    df[f"{asset}_bull_prob"]      = bull_prob.values
    df[f"{asset}_neutral_prob"]   = neut_prob.values
    df[f"{asset}_bear_prob"]      = bear_prob.values

    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach multi-asset regime features to a wide market DataFrame.

    For every asset in ASSETS the following columns are added:
        {asset}_trend_strength, {asset}_volatility, {asset}_regime,
        {asset}_bull_prob, {asset}_neutral_prob, {asset}_bear_prob

    For backward-compatibility with train_ppo.py / train_dqn.py /
    evaluate_agent.py the three global columns are also written using
    BTC as the market proxy:
        regime              — integer {-1, 0, 1}
        trend_strength      — float
        market_volatility   — float

    Parameters
    ----------
    df : Wide DataFrame with date index.  Must contain at least one
         {asset}_Close or {asset}_Return column per asset.

    Returns
    -------
    Copy of df with all new regime columns appended.
    """
    df = df.copy()

    logger.info("Running multi-asset regime detection for %d assets.", len(ASSETS))

    # ── Per-asset regime features ─────────────────────────────────────────────
    for asset in ASSETS:
        df = _asset_regime_features(df, asset)
        logger.debug("Regime features attached for asset: %s", asset)

    # ── Global regime columns (BTC proxy, backward-compat) ────────────────────
    # Re-use columns already computed for BTC to avoid redundant computation.
    if "BTC_trend_strength" in df.columns:
        df["trend_strength"]    = df["BTC_trend_strength"].values
        df["market_volatility"] = df["BTC_volatility"].values
        df["regime"]            = df["BTC_regime"].values
        logger.info(
            "Global regime column set from BTC.  "
            "Bull=%d  Neutral=%d  Bear=%d",
            int((df["regime"] ==  1).sum()),
            int((df["regime"] ==  0).sum()),
            int((df["regime"] == -1).sum()),
        )
    else:
        # Hard fallback: neutral everywhere
        df["trend_strength"]    = 0.0
        df["market_volatility"] = 0.0
        df["regime"]            = 0
        logger.warning(
            "BTC_trend_strength not found — global regime set to Neutral (0) everywhere."
        )

    return df


def get_regime_stats(df: pd.DataFrame) -> dict:
    """
    Count global regime occurrences from a DataFrame that has a 'regime' column.

    Returns
    -------
    dict with keys:
        bull, bear, neutral, total,
        bull_pct, bear_pct, neutral_pct
    """
    if "regime" not in df.columns:
        return {}

    counts = df["regime"].value_counts()
    total  = max(len(df), 1)
    bull   = int(counts.get( 1, 0))
    bear   = int(counts.get(-1, 0))
    neut   = int(counts.get( 0, 0))

    return {
        "bull":        bull,
        "bear":        bear,
        "neutral":     neut,
        "total":       total,
        "bull_pct":    bull / total * 100,
        "bear_pct":    bear / total * 100,
        "neutral_pct": neut / total * 100,
    }


def get_per_asset_regime_stats(df: pd.DataFrame) -> dict:
    """
    Return regime distribution statistics for every asset individually.

    Returns
    -------
    dict of {asset_name: {bull, bear, neutral, bull_pct, bear_pct, neutral_pct}}
    """
    stats = {}
    for asset in ASSETS:
        col = f"{asset}_regime"
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        total  = max(len(df), 1)
        stats[asset] = {
            "bull":        int(counts.get( 1, 0)),
            "neutral":     int(counts.get( 0, 0)),
            "bear":        int(counts.get(-1, 0)),
            "bull_pct":    counts.get( 1, 0) / total * 100,
            "neutral_pct": counts.get( 0, 0) / total * 100,
            "bear_pct":    counts.get(-1, 0) / total * 100,
        }
    return stats