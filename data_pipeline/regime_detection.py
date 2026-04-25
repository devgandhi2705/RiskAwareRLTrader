"""
regime_detection.py
--------------------
Market regime detection module.

Computes three regime signals from a price series and attaches them
as new columns to any DataFrame that contains BTC_Close data.

Regime labels
-------------
  -1  Bear  (price < MA200, strong downtrend)
   0  Neutral / Sideways
   1  Bull   (price > MA200, strong uptrend)

Columns added to DataFrame
--------------------------
  regime           : int  {-1, 0, 1}
  trend_strength   : float  (MA50 - MA200) / MA200
  market_volatility: float  rolling 30-day std of BTC returns

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────
REGIME_PROXY_COL  = "BTC_Close"     # column used for regime detection
MA_FAST           = 50              # fast moving average window
MA_SLOW           = 200             # slow moving average window
VOL_WINDOW        = 30              # rolling volatility window (days)
BULL_THRESHOLD    = 0.02            # trend_strength > +2% → bull
BEAR_THRESHOLD    = -0.02           # trend_strength < -2% → bear


def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime features to a wide market DataFrame.

    Regime is determined from BTC_Close (crypto acts as risk-on/off proxy).
    If BTC_Close is absent, falls back to the first available Close column.

    Parameters
    ----------
    df : Wide DataFrame with date index and asset columns.

    Returns
    -------
    df with three new columns added in-place:
        regime            {-1, 0, 1}
        trend_strength    float
        market_volatility float

    Notes
    -----
    All rolling windows are strictly backward-looking (.shift(1) is NOT applied
    because the raw rolling().mean() already uses only data up to each row
    in a non-centered window — no look-ahead bias introduced).
    """
    df = df.copy()

    # ── Select price series ────────────────────────────────────────────────────
    if REGIME_PROXY_COL in df.columns:
        price = df[REGIME_PROXY_COL].copy()
    else:
        # Fallback: first *_Close column found
        close_cols = [c for c in df.columns if c.endswith("_Close")]
        if not close_cols:
            # No price data at all — fill with neutral
            df["regime"]            = 0
            df["trend_strength"]    = 0.0
            df["market_volatility"] = 0.0
            return df
        price = df[close_cols[0]].copy()

    # Forward-fill to handle weekend / holiday gaps (crypto vs equities)
    price = price.ffill().bfill()

    # ── Moving averages ────────────────────────────────────────────────────────
    ma_fast = price.rolling(window=MA_FAST,  min_periods=MA_FAST // 2).mean()
    ma_slow = price.rolling(window=MA_SLOW, min_periods=MA_SLOW // 2).mean()

    # ── Trend strength: (MA50 - MA200) / MA200 ────────────────────────────────
    trend_strength = ((ma_fast - ma_slow) / ma_slow.replace(0, np.nan)).fillna(0.0)

    # ── Rolling volatility of BTC returns ─────────────────────────────────────
    btc_returns = price.pct_change().fillna(0.0)
    market_vol  = btc_returns.rolling(window=VOL_WINDOW, min_periods=5).std().fillna(0.0)

    # ── Regime classification ──────────────────────────────────────────────────
    regime = pd.Series(0, index=df.index, dtype=int)  # default: neutral

    bull_mask = (price > ma_slow) & (trend_strength > BULL_THRESHOLD)
    bear_mask = (price < ma_slow) & (trend_strength < BEAR_THRESHOLD)

    regime[bull_mask] =  1
    regime[bear_mask] = -1

    # ── Write to DataFrame ─────────────────────────────────────────────────────
    df["regime"]            = regime.values
    df["trend_strength"]    = trend_strength.values
    df["market_volatility"] = market_vol.values

    return df


def get_regime_stats(df: pd.DataFrame) -> dict:
    """
    Count regime occurrences in a DataFrame that already has regime column.

    Returns
    -------
    dict with keys: bull, bear, neutral, total, bull_pct, bear_pct, neutral_pct
    """
    if "regime" not in df.columns:
        return {}
    counts = df["regime"].value_counts()
    total  = len(df)
    bull   = int(counts.get(1,  0))
    bear   = int(counts.get(-1, 0))
    neut   = int(counts.get(0,  0))
    return {
        "bull": bull, "bear": bear, "neutral": neut, "total": total,
        "bull_pct":    bull / total * 100,
        "bear_pct":    bear / total * 100,
        "neutral_pct": neut / total * 100,
    }
