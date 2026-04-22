"""
build_dataset.py  (v3 — Simplified, No Regime Features)
--------------------------------------------------------
Master data preparation pipeline.

Steps
-----
1. Download / load raw OHLCV data.
2. Compute 5 technical indicators per asset  (feature_engineering v3).
3. Clean, align, and trim the dataset.
4. Split into train (2016–2022) and test (2023–2024) sets.
5. Save all outputs to data/.

Dataset shape
-------------
  35 indicator features  (7 assets × 5 indicators)
  No regime columns.

Run
---
    python data_pipeline/build_dataset.py
    python data_pipeline/build_dataset.py --use-cached-raw

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.download_data import (
    run_download_pipeline,
    RAW_OUTPUT_PATH,
)
from data_pipeline.feature_engineering import (
    engineer_all_features,
    reorder_columns,
    WARMUP_ROWS,
    ASSETS,
    FEATURES_PER_ASSET,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs",
    "dataset_build_log.txt",
)

def log(msg: str) -> None:
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START = "2016-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2025-12-31"

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(_THIS_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data")
TRAIN_OUT  = os.path.join(DATA_DIR, "train_dataset.csv")
TEST_OUT   = os.path.join(DATA_DIR, "test_dataset.csv")


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the enriched DataFrame.

    1. Ensure DatetimeIndex.
    2. Sort ascending.
    3. Remove duplicate timestamps.
    4. Forward-fill missing values (limit 5).
    5. Drop all-NaN rows.
    6. Drop first WARMUP_ROWS rows.
    7. Coerce all columns to numeric.
    8. Drop any fully-NaN columns after coercion.
    """
    log(f"  Cleaning dataset — input shape: {df.shape}")

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.sort_index(inplace=True)

    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        df = df[~df.index.duplicated(keep="last")]
        log(f"  Removed {n_dupes} duplicate timestamps.")

    df = df.ffill(limit=5)

    all_nan = df.isna().all(axis=1).sum()
    if all_nan:
        df = df[~df.isna().all(axis=1)]
        log(f"  Dropped {all_nan} all-NaN rows.")

    df = df.iloc[WARMUP_ROWS:]
    log(f"  Dropped first {WARMUP_ROWS} warm-up rows (MA50 window).")

    df = df.apply(pd.to_numeric, errors="coerce")

    dead_cols = df.columns[df.isna().all()].tolist()
    if dead_cols:
        df.drop(columns=dead_cols, inplace=True)
        log(f"  Dropped fully-NaN columns after coercion: {dead_cols}")

    log(f"  Clean complete — final shape: {df.shape}")
    return df


def split_dataset(df: pd.DataFrame):
    train = df.loc[TRAIN_START:TRAIN_END].copy()
    test  = df.loc[TEST_START:TEST_END].copy()
    log(f"  Train: {len(train)} rows  [{TRAIN_START} → {TRAIN_END}]")
    log(f"  Test : {len(test)}  rows  [{TEST_START} → {TEST_END}]")
    if train.empty:
        log("  WARNING: training set is empty — check date filters.")
    if test.empty:
        log("  WARNING: test set is empty — check date filters.")
    return train, test


def print_summary(full: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame):
    sep = "=" * 55
    log(f"\n{sep}")
    log("  DATASET SUMMARY")
    log(sep)
    log(f"  Full  : {len(full):>5} rows × {len(full.columns):>3} cols")
    log(f"  Train : {len(train):>5} rows")
    log(f"  Test  : {len(test):>5} rows")

    # Expected indicator columns
    expected = [f"{a}_{f}" for a in ASSETS for f in FEATURES_PER_ASSET]
    present  = [c for c in expected if c in full.columns]
    log(f"  Indicator columns : {len(present)} / {len(expected)} expected")

    nan_pct = full[present].isna().mean().mean() * 100
    log(f"  NaN rate (indicators) : {nan_pct:.2f}%")

    log(f"  Output files:")
    log(f"    {TRAIN_OUT}")
    log(f"    {TEST_OUT}")
    log(sep + "\n")


# ── Master pipeline ────────────────────────────────────────────────────────────

def build_full_pipeline(use_cached_raw: bool = False):
    """
    Run the complete data preparation pipeline.

    Returns (full_df, train_df, test_df).
    """
    # Clear log for this run
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    log("╔══════════════════════════════════════════════════════╗")
    log("║  SAFE RL PORTFOLIO — DATA PIPELINE v3               ║")
    log("╚══════════════════════════════════════════════════════╝\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Raw data
    if use_cached_raw and os.path.exists(RAW_OUTPUT_PATH):
        log(f"Loading cached raw data from {RAW_OUTPUT_PATH} …")
        raw_df = pd.read_csv(RAW_OUTPUT_PATH, index_col="Date", parse_dates=True)
        log(f"Loaded: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols")
    else:
        log("Step 1: Downloading raw OHLCV data …")
        raw_df = run_download_pipeline()

    # Step 2: Feature engineering (no regime features)
    log("\nStep 2: Computing technical indicators …")
    enriched_df = engineer_all_features(raw_df)

    # Step 3: Clean
    log("\nStep 3: Cleaning dataset …")
    clean_df = clean_dataset(enriched_df)

    # Step 4: Keep only indicator columns + preserve OHLCV for baselines
    log("\nStep 4: Reordering columns …")
    clean_df = reorder_columns(clean_df)

    # Step 5: Split
    log("\nStep 5: Splitting train / test …")
    train_df, test_df = split_dataset(clean_df)

    # Step 6: Save
    log("\nStep 6: Saving datasets …")
    train_df.to_csv(TRAIN_OUT)
    test_df.to_csv(TEST_OUT)
    log(f"  Saved train → {TRAIN_OUT}  ({len(train_df)} rows × {len(train_df.columns)} cols)")
    log(f"  Saved test  → {TEST_OUT}   ({len(test_df)}  rows × {len(test_df.columns)} cols)")

    print_summary(clean_df, train_df, test_df)
    log(f"Log saved → {LOG_FILE}")

    return clean_df, train_df, test_df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Safe RL portfolio dataset (v3, no regime features)."
    )
    parser.add_argument(
        "--use-cached-raw",
        action="store_true",
        default=False,
        help="Skip download and reuse existing raw_market_data.csv.",
    )
    args = parser.parse_args()
    build_full_pipeline(use_cached_raw=args.use_cached_raw)