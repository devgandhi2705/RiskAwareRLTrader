"""
build_dataset.py
----------------
Master pipeline script for the Safe RL Portfolio Management project.

Orchestrates the full data preparation workflow:
  1. Download raw OHLCV data  (via download_data.py)
  2. Compute technical indicators  (via feature_engineering.py)
  3. Detect market regimes  (via regime_detection.py)
  4. Merge all assets into one wide DataFrame
  5. Clean and align the dataset
  6. Split into train / test sets
  7. Save all outputs to disk

Run
---
    python data_pipeline/build_dataset.py

Expected outputs (in data/)
----------------------------
  raw_market_data.csv    – merged raw OHLCV (from download step)
  train_dataset.csv      – 2016-01-01 to 2022-12-31
  test_dataset.csv       – 2023-01-01 to 2024-12-31

Part of: Safe RL for Risk-Constrained Portfolio Management
Stage:   Data Pipeline — Step 3: Dataset Construction
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running from the project root or from inside data_pipeline/
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
)
from data_pipeline.regime_detection import (
    detect_market_regime,
    get_regime_stats,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_START = "2016-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2024-12-31"

DATA_DIR         = "data"
TRAIN_OUTPUT     = os.path.join(DATA_DIR, "train_dataset.csv")
TEST_OUTPUT      = os.path.join(DATA_DIR, "test_dataset.csv")

# Number of leading rows to drop due to indicator warm-up (MA20 needs 20 rows,
# RSI needs 14 — use the largest window as the burn-in period).
WARMUP_ROWS = MA_PERIOD  # 20


# ── Pipeline Functions ────────────────────────────────────────────────────────

def merge_assets(asset_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge per-asset DataFrames into a single wide DataFrame aligned by Date.

    Thin wrapper kept here so build_dataset.py is self-contained should the
    caller want to supply pre-computed asset DataFrames directly.

    Parameters
    ----------
    asset_data : dict of {friendly_name: DataFrame with namespaced columns}.

    Returns
    -------
    Outer-joined wide DataFrame sorted by ascending Date.
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

    Steps applied (in order)
    ------------------------
    1. Convert index to pandas DatetimeIndex.
    2. Sort by date ascending.
    3. Remove duplicated timestamps (keep last).
    4. Forward-fill missing values (up to 5 consecutive NaNs).
    5. Drop rows where *all* columns are NaN.
    6. Drop the warm-up rows at the head (insufficient indicator history).
    7. Ensure every column is numeric (coerce non-numeric to NaN, then drop
       any column that becomes fully NaN after coercion).

    Parameters
    ----------
    df : Enriched wide DataFrame.

    Returns
    -------
    Cleaned DataFrame.
    """
    logger.info("=" * 60)
    logger.info("DATA CLEANING")
    logger.info("=" * 60)

    original_rows = len(df)
    logger.info("Input shape: %d rows × %d cols", *df.shape)

    # 1. DateTime index
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    logger.info("  ✓ DatetimeIndex set.")

    # 2. Sort ascending
    df.sort_index(inplace=True)
    logger.info("  ✓ Sorted by date.")

    # 3. Drop duplicated timestamps
    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        df = df[~df.index.duplicated(keep="last")]
        logger.info("  ✓ Removed %d duplicate timestamps.", n_dupes)
    else:
        logger.info("  ✓ No duplicate timestamps found.")

    # 4. Forward-fill (limit=5 to avoid propagating stale data too far)
    df = df.ffill(limit=5)
    logger.info("  ✓ Forward-filled missing values (limit=5).")

    # 5. Drop rows where ALL columns are NaN
    all_nan_mask = df.isna().all(axis=1)
    n_all_nan = all_nan_mask.sum()
    if n_all_nan:
        df = df[~all_nan_mask]
        logger.info("  ✓ Dropped %d all-NaN rows.", n_all_nan)
    else:
        logger.info("  ✓ No all-NaN rows found.")

    # 6. Drop warm-up rows (first WARMUP_ROWS rows may have NaN in indicators)
    df = df.iloc[WARMUP_ROWS:]
    logger.info("  ✓ Dropped first %d warm-up rows.", WARMUP_ROWS)

    # 7. Coerce to numeric — exclude regime column (int, not float) before
    #    blanket coercion so it is not accidentally overwritten.
    regime_col = df.pop("regime") if "regime" in df.columns else None
    df = df.apply(pd.to_numeric, errors="coerce")
    if regime_col is not None:
        df["regime"] = regime_col.values

    fully_nan_cols = df.columns[df.isna().all()].tolist()
    if fully_nan_cols:
        df.drop(columns=fully_nan_cols, inplace=True)
        logger.warning("  ⚠ Dropped fully-NaN columns after coercion: %s", fully_nan_cols)
    logger.info("  ✓ All columns confirmed numeric.")

    dropped = original_rows - len(df)
    logger.info(
        "Cleaning complete.  Final shape: %d rows × %d cols  (%d rows removed).",
        *df.shape,
        dropped,
    )
    return df


def split_dataset(
    df: pd.DataFrame,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split the dataset into training and test sets.

    Parameters
    ----------
    df          : Cleaned full dataset.
    train_start : Inclusive start date for training set.
    train_end   : Inclusive end date for training set.
    test_start  : Inclusive start date for test set.
    test_end    : Inclusive end date for test set.

    Returns
    -------
    (train_df, test_df) — both indexed by Date.
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


def save_dataset(
    df: pd.DataFrame,
    path: str,
    label: str = "Dataset",
) -> None:
    """
    Persist a DataFrame to a CSV file.

    Parameters
    ----------
    df    : DataFrame to save.
    path  : Destination file path.
    label : Human-readable label used in log messages.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path)
    logger.info(
        "Saved %-16s → %s  (%d rows × %d cols)",
        label,
        path,
        *df.shape,
    )


def print_dataset_summary(
    full: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Print a concise summary of the final datasets to stdout.

    Includes regime distribution statistics if regime columns are present.

    Parameters
    ----------
    full  : Full cleaned dataset.
    train : Training split.
    test  : Test split.
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print("  DATASET SUMMARY")
    print(separator)
    print(f"  Full dataset   : {len(full):>5} rows × {len(full.columns):>3} cols")
    print(f"  Training set   : {len(train):>5} rows  [{TRAIN_START} → {TRAIN_END}]")
    print(f"  Test set       : {len(test):>5} rows  [{TEST_START} → {TEST_END}]")
    print(f"\n  Column groups  :")
    all_cols = list(full.columns)
    preview  = all_cols[:6] + ["..."] + all_cols[-3:] if len(all_cols) > 9 else all_cols
    print(f"    {preview}")
    print(f"\n  NaN summary (full dataset):")
    nan_pct = full.isna().mean().mean() * 100
    print(f"    Overall NaN rate : {nan_pct:.2f}%")

    # ── Regime statistics ──────────────────────────────────────────────────────
    if "regime" in full.columns:
        print(f"\n  Market Regime Distribution (full dataset):")
        stats = get_regime_stats(full)
        print(f"    Bull    (1) : {stats['bull']:>4} days  ({stats['bull_pct']:5.1f}%)")
        print(f"    Neutral (0) : {stats['neutral']:>4} days  ({stats['neutral_pct']:5.1f}%)")
        print(f"    Bear   (-1) : {stats['bear']:>4} days  ({stats['bear_pct']:5.1f}%)")

        train_stats = get_regime_stats(train)
        test_stats  = get_regime_stats(test)
        print(f"\n  Regime split — Train:")
        print(f"    Bull {train_stats['bull_pct']:5.1f}%  |  "
              f"Neutral {train_stats['neutral_pct']:5.1f}%  |  "
              f"Bear {train_stats['bear_pct']:5.1f}%")
        print(f"  Regime split — Test:")
        print(f"    Bull {test_stats['bull_pct']:5.1f}%  |  "
              f"Neutral {test_stats['neutral_pct']:5.1f}%  |  "
              f"Bear {test_stats['bear_pct']:5.1f}%")

    print(separator)
    print(f"\n  Output files:")
    print(f"    {RAW_OUTPUT_PATH}")
    print(f"    {TRAIN_OUTPUT}")
    print(f"    {TEST_OUTPUT}")
    print(separator + "\n")


# ── Master Orchestrator ───────────────────────────────────────────────────────

def build_full_pipeline(
    use_cached_raw: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete data preparation pipeline end-to-end.

    Steps
    -----
    1. Download raw OHLCV data (or load from cache if use_cached_raw=True).
    2. Compute technical indicators for every asset.
    3. Detect market regimes (adds regime, trend_strength, market_volatility).
    4. Clean and align the dataset.
    5. Reorder columns for readability.
    6. Split into train / test.
    7. Save all three CSVs to disk.

    Parameters
    ----------
    use_cached_raw : If True and raw_market_data.csv already exists on disk,
                     skip the download step and read from the cached file.
                     Useful for iterating on feature engineering without
                     re-downloading every time.

    Returns
    -------
    (full_df, train_df, test_df) as pandas DataFrames.
    """
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   SAFE RL PORTFOLIO MANAGEMENT — DATA PIPELINE           ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── Step 1: Raw data ──────────────────────────────────────────────────────
    if use_cached_raw and os.path.exists(RAW_OUTPUT_PATH):
        logger.info("Loading cached raw data from %s …", RAW_OUTPUT_PATH)
        raw_df = pd.read_csv(RAW_OUTPUT_PATH, index_col="Date", parse_dates=True)
        logger.info(
            "Loaded cached data: %d rows × %d cols", *raw_df.shape
        )
    else:
        raw_df = run_download_pipeline()

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    enriched_df = engineer_all_features(raw_df)

    # ── Step 3: Regime detection ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("REGIME DETECTION")
    logger.info("=" * 60)
    enriched_df = detect_market_regime(enriched_df)
    regime_counts = enriched_df["regime"].value_counts().to_dict()
    logger.info(
        "Regime labels assigned.  Bull=%d  Neutral=%d  Bear=%d",
        regime_counts.get(1,  0),
        regime_counts.get(0,  0),
        regime_counts.get(-1, 0),
    )

    # ── Step 4: Clean ─────────────────────────────────────────────────────────
    clean_df = clean_dataset(enriched_df)

    # ── Step 5: Reorder columns ───────────────────────────────────────────────
    clean_df = reorder_columns(clean_df)
    logger.info("Columns reordered by asset group.")

    # ── Step 6: Split ─────────────────────────────────────────────────────────
    train_df, test_df = split_dataset(clean_df)

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    save_dataset(train_df, TRAIN_OUTPUT, label="Training set")
    save_dataset(test_df,  TEST_OUTPUT,  label="Test set")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_dataset_summary(clean_df, train_df, test_df)

    return clean_df, train_df, test_df


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the Safe RL portfolio management dataset."
    )
    parser.add_argument(
        "--use-cached-raw",
        action="store_true",
        default=False,
        help=(
            "Skip the download step and reuse an existing raw_market_data.csv "
            "if one already exists in the data/ directory."
        ),
    )
    args = parser.parse_args()

    build_full_pipeline(use_cached_raw=args.use_cached_raw)
