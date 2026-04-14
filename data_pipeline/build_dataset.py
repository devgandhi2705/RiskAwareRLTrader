"""
build_dataset.py  (v2 — Multi-Asset Regime Pipeline)
------------------------------------------------------
Master pipeline script for the Safe RL Portfolio Management project.

Orchestrates the full data preparation workflow:
  1. Download raw OHLCV data           (via download_data.py)
  2. Compute technical indicators      (via feature_engineering.py v2)
     — now includes MA50 and MA200 per asset
  3. Detect market regimes             (via regime_detection.py v2)
     — now per-asset regime, volatility, and regime probabilities
  4. Merge all assets into one wide DataFrame
  5. Clean and align the dataset
  6. Split into train / test sets
  7. Save all outputs to disk

Changes from v1
---------------
  B1 : WARMUP_ROWS now uses feature_engineering.WARMUP_ROWS (= 200, up from 20)
       to account for the MA200 warm-up window.
  B2 : clean_dataset preserves all per-asset regime integer columns
       ({asset}_regime, regime) during numeric coercion.
  B3 : print_dataset_summary prints per-asset regime distribution.
  B4 : Step 3 log now reports per-asset regime stats.

Run
---
    python data_pipeline/build_dataset.py
    python data_pipeline/build_dataset.py --use-cached-raw

Expected outputs (in data/)
----------------------------
  raw_market_data.csv    – merged raw OHLCV (from download step)
  train_dataset.csv      – 2016-01-01 to 2022-12-31
  test_dataset.csv       – 2023-01-01 to 2024-12-31

Part of: Safe RL for Risk-Constrained Portfolio Management
Stage:   Data Pipeline — Step 3: Dataset Construction
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.download_data import (
    run_download_pipeline,
    ASSET_UNIVERSE,
    RAW_OUTPUT_PATH,
)
from data_pipeline.feature_engineering import (
    engineer_all_features,
    reorder_columns,
    MA_PERIOD,
    WARMUP_ROWS,   # B1: 200 (MA200 warm-up)
)
from data_pipeline.regime_detection import (
    detect_market_regime,
    get_regime_stats,
    get_per_asset_regime_stats,   # B3
    ASSETS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_START = "2016-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2024-12-31"

DATA_DIR     = "data"
TRAIN_OUTPUT = os.path.join(DATA_DIR, "train_dataset.csv")
TEST_OUTPUT  = os.path.join(DATA_DIR,  "test_dataset.csv")

# ── Regime integer columns ─────────────────────────────────────────────────────
# These must survive blanket pd.to_numeric(errors='coerce') without change,
# but we still pop-and-restore them to be explicit and avoid casting issues.
_REGIME_INT_COLS = ["regime"] + [f"{a}_regime" for a in ASSETS]


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def merge_assets(asset_data: dict) -> pd.DataFrame:
    """
    Merge per-asset DataFrames into a single wide DataFrame aligned by Date.

    Parameters
    ----------
    asset_data : dict mapping asset_name → DataFrame.

    Returns
    -------
    Wide DataFrame with a DatetimeIndex, all assets outer-joined by date.
    """
    if not asset_data:
        raise ValueError("asset_data is empty — nothing to merge.")

    frames = list(asset_data.values())
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how="outer")

    merged.sort_index(inplace=True)
    merged.index = pd.to_datetime(merged.index)
    return merged


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data-quality rules to the enriched wide DataFrame.

    Steps (in order)
    ----------------
    1. Convert index to DatetimeIndex.
    2. Sort ascending.
    3. Remove duplicate timestamps (keep last).
    4. Forward-fill missing values (limit=5).
    5. Drop all-NaN rows.
    6. Drop warm-up rows (first WARMUP_ROWS = 200) — MA200 burn-in.
    7. Coerce all non-regime columns to numeric; regime int columns preserved (B2).
    8. Drop any columns that became entirely NaN after coercion, excluding MA columns.
    9. Sanity check for required MA columns.
    """
    logger.info("=" * 60)
    logger.info("DATA CLEANING  (WARMUP_ROWS=%d)", WARMUP_ROWS)
    logger.info("=" * 60)

    original_rows = len(df)
    logger.info("Input shape: %d rows × %d cols", *df.shape)

    # Step 1-2: index and sort
    df.index      = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.sort_index(inplace=True)

    # Step 3: deduplicate
    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        df = df[~df.index.duplicated(keep="last")]
        logger.info("  ✓ Removed %d duplicate timestamps.", n_dupes)

    # Step 4: forward-fill
    df = df.ffill(limit=5)
    logger.info("  ✓ Forward-filled missing values (limit=5).")

    # Step 5: drop all-NaN rows
    all_nan_mask = df.isna().all(axis=1)
    n_all_nan    = int(all_nan_mask.sum())
    if n_all_nan:
        df = df[~all_nan_mask]
        logger.info("  ✓ Dropped %d all-NaN rows.", n_all_nan)

    # Step 6: drop MA200 warm-up rows (B1)
    df = df.iloc[WARMUP_ROWS:]
    logger.info("  ✓ Dropped first %d warm-up rows (MA200 window).", WARMUP_ROWS)

    # Step 7: coerce to numeric — preserve regime int columns (B2)
    saved_regime_cols = {}
    for col in _REGIME_INT_COLS:
        if col in df.columns:
            saved_regime_cols[col] = df.pop(col)

    # Selective numeric conversion
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)

    for col, series in saved_regime_cols.items():
        df[col] = series.values

    # Step 8: drop fully-NaN columns post-coercion, excluding MA columns
    drop_cols = [
        c for c in df.columns
        if df[c].isna().all() and "_MA" not in c
    ]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.warning(
            "  ⚠ Dropped fully-NaN columns after coercion (excluding MA): %s", drop_cols
        )

    logger.info("  ✓ Numeric coercion complete; regime integer columns preserved.")

    # Step 9: sanity check for required MA columns
    required_ma_cols = [
        f"{asset}_MA50"
        for asset in ["BTC","ETH","SPY","GLD","Silver","Nifty50","Sensex"]
    ]
    missing = [c for c in required_ma_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing MA columns after cleaning: {missing}")

    dropped = original_rows - len(df)
    logger.info(
        "Cleaning complete.  Final shape: %d rows × %d cols  (%d rows removed).",
        *df.shape, dropped,
    )
    return df


def split_dataset(
    df: pd.DataFrame,
    train_start: str = TRAIN_START,
    train_end: str   = TRAIN_END,
    test_start: str  = TEST_START,
    test_end: str    = TEST_END,
) -> tuple:
    """
    Chronologically split the dataset into training and test sets.

    Returns
    -------
    (train_df, test_df) — both are copies.
    """
    logger.info("=" * 60)
    logger.info("TRAIN / TEST SPLIT")
    logger.info("=" * 60)

    train = df.loc[train_start:train_end].copy()
    test  = df.loc[test_start:test_end].copy()

    logger.info(
        "Training set : %d rows  [%s → %s]",
        len(train),
        train.index.min().date() if not train.empty else "N/A",
        train.index.max().date() if not train.empty else "N/A",
    )
    logger.info(
        "Test set     : %d rows  [%s → %s]",
        len(test),
        test.index.min().date() if not test.empty else "N/A",
        test.index.max().date() if not test.empty else "N/A",
    )

    if train.empty:
        logger.warning("Training set is empty — check date filters.")
    if test.empty:
        logger.warning("Test set is empty — check date filters.")

    return train, test


def save_dataset(df: pd.DataFrame, path: str, label: str = "Dataset") -> None:
    """Persist a DataFrame to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path)
    logger.info(
        "Saved %-16s → %s  (%d rows × %d cols)", label, path, *df.shape
    )


def print_dataset_summary(
    full: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print a concise human-readable summary including per-asset regime stats (B3)."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DATASET SUMMARY")
    print(sep)
    print(f"  Full dataset   : {len(full):>5} rows × {len(full.columns):>3} cols")
    print(f"  Training set   : {len(train):>5} rows  [{TRAIN_START} → {TRAIN_END}]")
    print(f"  Test set       : {len(test):>5} rows  [{TEST_START} → {TEST_END}]")

    all_cols = list(full.columns)
    preview  = (all_cols[:6] + ["..."] + all_cols[-3:]) if len(all_cols) > 9 else all_cols
    print(f"\n  Column preview : {preview}")

    nan_pct = full.isna().mean().mean() * 100
    print(f"\n  NaN rate (full dataset): {nan_pct:.2f}%")

    # ── Global regime distribution ─────────────────────────────────────────────
    if "regime" in full.columns:
        print(f"\n  Global Market Regime Distribution (full dataset):")
        stats = get_regime_stats(full)
        print(f"    Bull    (+1) : {stats['bull']:>4} days  ({stats['bull_pct']:5.1f}%)")
        print(f"    Neutral ( 0) : {stats['neutral']:>4} days  ({stats['neutral_pct']:5.1f}%)")
        print(f"    Bear    (-1) : {stats['bear']:>4} days  ({stats['bear_pct']:5.1f}%)")

        train_stats = get_regime_stats(train)
        test_stats  = get_regime_stats(test)
        print(f"\n  Global regime — Train:")
        print(
            f"    Bull {train_stats['bull_pct']:5.1f}%  |  "
            f"Neutral {train_stats['neutral_pct']:5.1f}%  |  "
            f"Bear {train_stats['bear_pct']:5.1f}%"
        )
        print(f"  Global regime — Test:")
        print(
            f"    Bull {test_stats['bull_pct']:5.1f}%  |  "
            f"Neutral {test_stats['neutral_pct']:5.1f}%  |  "
            f"Bear {test_stats['bear_pct']:5.1f}%"
        )

    # ── Per-asset regime distribution (B3) ────────────────────────────────────
    per_asset = get_per_asset_regime_stats(full)
    if per_asset:
        print(f"\n  Per-Asset Regime Distribution (full dataset):")
        print(f"  {'Asset':<12}  {'Bull%':>7}  {'Neutral%':>9}  {'Bear%':>7}")
        print(f"  {'-'*42}")
        for asset, s in per_asset.items():
            print(
                f"  {asset:<12}  {s['bull_pct']:>7.1f}  "
                f"{s['neutral_pct']:>9.1f}  {s['bear_pct']:>7.1f}"
            )

    print(sep)
    print(f"\n  Output files:")
    print(f"    {RAW_OUTPUT_PATH}")
    print(f"    {TRAIN_OUTPUT}")
    print(f"    {TEST_OUTPUT}")
    print(sep + "\n")


# ── Master Orchestrator ────────────────────────────────────────────────────────

def build_full_pipeline(use_cached_raw: bool = False) -> tuple:
    """
    Run the complete data preparation pipeline end-to-end.

    Steps
    -----
    1. Download raw OHLCV data (or load cached CSV).
    2. Compute technical indicators — now includes MA50, MA200 per asset (F1).
    3. Detect multi-asset regimes — per-asset probabilities + global compat cols (P1-P6).
    4. Clean and align the dataset — warm-up 200 rows (B1).
    5. Reorder columns by asset group.
    6. Chronological train / test split.
    7. Save all three CSVs to disk.

    Returns
    -------
    (full_df, train_df, test_df)
    """
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   SAFE RL PORTFOLIO MANAGEMENT — DATA PIPELINE  v2       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Step 1: Raw data
    if use_cached_raw and os.path.exists(RAW_OUTPUT_PATH):
        logger.info("Loading cached raw data from %s …", RAW_OUTPUT_PATH)
        raw_df = pd.read_csv(RAW_OUTPUT_PATH, index_col="Date", parse_dates=True)
        logger.info("Loaded cached data: %d rows × %d cols", *raw_df.shape)
    else:
        raw_df = run_download_pipeline()

    # Step 2: Feature engineering — adds Return, RSI, MA20, MA50, MA200, Volatility
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING  (v2 — MA50 + MA200 added)")
    logger.info("=" * 60)
    enriched_df = engineer_all_features(raw_df)

    # Step 3: Multi-asset regime detection
    logger.info("=" * 60)
    logger.info("MULTI-ASSET REGIME DETECTION  (v2 — per-asset probabilities)")
    logger.info("=" * 60)
    enriched_df = detect_market_regime(enriched_df)

    # B4: Log per-asset regime stats
    per_asset_stats = get_per_asset_regime_stats(enriched_df)
    for asset, s in per_asset_stats.items():
        logger.info(
            "  %-12s  Bull=%.0f%%  Neutral=%.0f%%  Bear=%.0f%%",
            asset, s["bull_pct"], s["neutral_pct"], s["bear_pct"],
        )
    global_counts = enriched_df["regime"].value_counts().to_dict()
    logger.info(
        "Global regime  Bull=%d  Neutral=%d  Bear=%d",
        global_counts.get( 1, 0),
        global_counts.get( 0, 0),
        global_counts.get(-1, 0),
    )

    # Step 4: Clean — drop duplicates, NaN rows, warm-up rows
    clean_df = clean_dataset(enriched_df)

    # Step 5: Reorder columns by asset group
    clean_df = reorder_columns(clean_df)
    logger.info("Columns reordered by asset group.")

    # Step 6: Chronological split
    train_df, test_df = split_dataset(clean_df)

    # Step 7: Save
    save_dataset(train_df, TRAIN_OUTPUT, label="Training set")
    save_dataset(test_df,  TEST_OUTPUT,  label="Test set")

    # Print human-readable summary
    print_dataset_summary(clean_df, train_df, test_df)

    return clean_df, train_df, test_df


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Safe RL portfolio management dataset (v2 — multi-asset regime)."
    )
    parser.add_argument(
        "--use-cached-raw",
        action="store_true",
        default=False,
        help="Skip re-download and reuse existing raw_market_data.csv.",
    )
    args = parser.parse_args()

    build_full_pipeline(use_cached_raw=args.use_cached_raw)