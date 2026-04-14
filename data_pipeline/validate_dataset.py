"""
validate_datasets.py
--------------------
Validates the train and test datasets produced by the Safe RL
Portfolio Management data pipeline.

Checks performed
----------------
  1.  File existence
  2.  File is non-empty and readable
  3.  DatetimeIndex integrity (parseable, named 'Date', sorted, no duplicates)
  4.  Date-range coverage (train 2016–2022, test 2023–2024, strict matching)
  5.  No unexpected overlap between train and test
  6.  Expected column schema (all OHLCV + indicator columns present)
  7.  Numeric dtype enforcement
  8.  NaN audit (per-column and per-row summary, thresholds: PASS<10%, WARN 10-20%, FAIL>20%)
  9.  OHLCV sanity checks (High ≥ Low, Close within [Low, High], Volume ≥ 0)
 10.  Price sanity (Close > 0)
 11.  Return sanity (no extreme outliers suggesting calculation errors)
 12.  RSI bounds (0 ≤ RSI ≤ 100)
 13.  MA / Volatility presence and sign checks (MA20, MA50, MA200 > 0, Volatility ≥ 0)
 14.  Regime feature validation (trend_strength finite, volatility ≥ 0, regime in {-1,0,1}, probs in [0,1] and sum ≈ 1)
 15.  Feature variance checks (Return, trend_strength, volatility columns have std > 1e-8)
 16.  Cross-split consistency (same columns, compatible value ranges)
 17.  Row-count plausibility

Usage
-----
    python validate_datasets.py
    python validate_datasets.py --train path/to/train.csv --test path/to/test.csv
"""

import os
import sys
import argparse
import logging
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_TRAIN_PATH = os.path.join("data", "train_dataset.csv")
DEFAULT_TEST_PATH  = os.path.join("data", "test_dataset.csv")

TRAIN_EXPECTED_START = "2016-01-01"
TRAIN_EXPECTED_END   = "2022-12-31"
TEST_EXPECTED_START  = "2023-01-01"
TEST_EXPECTED_END    = "2024-12-31"

# All assets and their expected fields
ASSET_UNIVERSE = ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty50", "Sensex"]

# Fields every asset must have
REQUIRED_FIELDS_ALL = ["Open", "High", "Low", "Close", "Volume", "Return", "RSI"]

# Fields that should exist for most assets (warn if missing, don't hard-fail)
OPTIONAL_FIELDS     = ["MA20", "MA50", "MA200", "Volatility"]

# Plausible row counts (trading days in range ± buffer for weekends/holidays)
TRAIN_MIN_ROWS = 1000
TRAIN_MAX_ROWS = 1900
TEST_MIN_ROWS  = 400
TEST_MAX_ROWS  = 750

# Return outlier threshold: flags daily returns beyond ±150% as suspicious
MAX_DAILY_RETURN = 1.50

# ── Result tracking ───────────────────────────────────────────────────────────

class CheckResult(NamedTuple):
    name:    str
    passed:  bool
    message: str


results: list[CheckResult] = []

def record(name: str, passed: bool, message: str) -> None:
    results.append(CheckResult(name, passed, message))
    symbol = "✓" if passed else "✗"
    level  = logging.INFO if passed else logging.WARNING
    logger.log(level, "  [%s] %-45s %s", symbol, name, message)


# ── Individual checks ─────────────────────────────────────────────────────────

def check_file_exists(path: str, label: str) -> bool:
    ok = os.path.isfile(path)
    record(f"{label} file exists", ok,
           f"{path}" if ok else f"NOT FOUND: {path}")
    return ok


def load_csv(path: str, label: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        record(f"{label} readable / parseable", True,
               f"{len(df)} rows × {len(df.columns)} cols")
        return df
    except Exception as exc:
        record(f"{label} readable / parseable", False, str(exc))
        return None


def check_index(df: pd.DataFrame, label: str) -> None:
    # DatetimeIndex type
    is_dt = isinstance(df.index, pd.DatetimeIndex)
    record(f"{label} DatetimeIndex type", is_dt,
           "Index is DatetimeIndex" if is_dt else f"Got {type(df.index).__name__}")

    # No NaT in index
    nat_count = df.index.isna().sum()
    record(f"{label} index has no NaT", nat_count == 0,
           "No NaT values" if nat_count == 0 else f"{nat_count} NaT timestamps found")

    # Sorted ascending
    sorted_ok = df.index.is_monotonic_increasing
    record(f"{label} index sorted ascending", sorted_ok,
           "Sorted" if sorted_ok else "NOT sorted — check date order")

    # No duplicates
    dupes = df.index.duplicated().sum()
    record(f"{label} no duplicate timestamps", dupes == 0,
           "No duplicates" if dupes == 0 else f"{dupes} duplicate dates found")


def check_date_range(
    df: pd.DataFrame,
    label: str,
    expected_start: str,
    expected_end: str,
) -> None:
    if df.empty:
        record(f"{label} date range", False, "DataFrame is empty")
        return

    actual_start = df.index.min().date()
    actual_end   = df.index.max().date()
    exp_s        = pd.Timestamp(expected_start).date()
    exp_e        = pd.Timestamp(expected_end).date()

    # Strict matching: actual must equal expected
    start_ok = actual_start == exp_s
    record(
        f"{label} start date == {expected_start}",
        start_ok,
        f"Actual start: {actual_start}  (expected: {exp_s})",
    )

    end_ok = actual_end == exp_e
    record(
        f"{label} end date == {expected_end}",
        end_ok,
        f"Actual end: {actual_end}  (expected: {exp_e})",
    )


def check_row_count(df: pd.DataFrame, label: str, min_rows: int, max_rows: int) -> None:
    n = len(df)
    ok = min_rows <= n <= max_rows
    record(
        f"{label} row count plausible",
        ok,
        f"{n} rows  (expected {min_rows}–{max_rows})",
    )


def check_no_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    overlap = train.index.intersection(test.index)
    record(
        "Train / test date overlap",
        len(overlap) == 0,
        "No overlap" if len(overlap) == 0
        else f"{len(overlap)} overlapping dates: {list(overlap[:5])}",
    )


def check_schema(df: pd.DataFrame, label: str) -> None:
    present_cols = set(df.columns)

    # Required fields for every asset
    missing_required: list[str] = []
    for asset in ASSET_UNIVERSE:
        for field in REQUIRED_FIELDS_ALL:
            col = f"{asset}_{field}"
            if col not in present_cols:
                missing_required.append(col)

    record(
        f"{label} all required columns present",
        len(missing_required) == 0,
        "All required columns found"
        if not missing_required
        else f"MISSING: {missing_required}",
    )

    # Optional fields — warn only
    missing_optional: list[str] = []
    for asset in ASSET_UNIVERSE:
        for field in OPTIONAL_FIELDS:
            col = f"{asset}_{field}"
            if col not in present_cols:
                missing_optional.append(col)

    if missing_optional:
        logger.warning(
            "  [~] %-45s %s",
            f"{label} optional columns",
            f"Missing (non-fatal): {missing_optional}",
        )


def check_dtypes(df: pd.DataFrame, label: str) -> None:
    non_numeric = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    record(
        f"{label} all columns numeric",
        len(non_numeric) == 0,
        "All numeric"
        if not non_numeric
        else f"Non-numeric cols: {non_numeric}",
    )


def check_nan_audit(df: pd.DataFrame, label: str) -> None:
    total_cells    = df.size
    total_nan      = int(df.isna().sum().sum())
    nan_pct        = (total_nan / total_cells * 100) if total_cells else 0.0

    # Overall NaN rate — PASS <10%, WARN 10-20%, FAIL >20%
    if nan_pct < 10.0:
        passed = True
        msg = f"PASS: {nan_pct:.2f}% NaN"
    elif nan_pct <= 20.0:
        passed = True  # Still pass but warn
        msg = f"WARN: {nan_pct:.2f}% NaN (10-20% range)"
        logger.warning(
            "  [~] %-45s NaN rate %.2f%% in warning range — monitor closely.",
            f"{label} NaN advisory",
            nan_pct,
        )
    else:
        passed = False
        msg = f"FAIL: {nan_pct:.2f}% NaN (>20%)"

    record(
        f"{label} NaN rate validation",
        passed,
        f"{msg}  ({total_nan:,} / {total_cells:,} cells)",
    )

    # Columns with > 50% NaN
    high_nan_cols = df.columns[df.isna().mean() > 0.50].tolist()
    if high_nan_cols:
        logger.warning(
            "  [~] %-45s %s",
            f"{label} high-NaN columns (>50%%)",
            high_nan_cols,
        )

    # Rows that are entirely NaN
    all_nan_rows = int(df.isna().all(axis=1).sum())
    record(
        f"{label} no all-NaN rows",
        all_nan_rows == 0,
        "None" if all_nan_rows == 0 else f"{all_nan_rows} fully-NaN rows found",
    )


def check_ohlcv_sanity(df: pd.DataFrame, label: str) -> None:
    for asset in ASSET_UNIVERSE:
        h_col = f"{asset}_High"
        l_col = f"{asset}_Low"
        c_col = f"{asset}_Close"
        v_col = f"{asset}_Volume"

        if h_col not in df.columns or l_col not in df.columns:
            continue

        valid_mask = df[h_col].notna() & df[l_col].notna()
        h = df.loc[valid_mask, h_col]
        l = df.loc[valid_mask, l_col]

        # High >= Low
        hl_violations = int((h < l).sum())
        record(
            f"{label} {asset} High ≥ Low",
            hl_violations == 0,
            "OK" if hl_violations == 0
            else f"{hl_violations} rows where High < Low",
        )

        # Close within [Low, High]
        if c_col in df.columns:
            c = df.loc[valid_mask & df[c_col].notna(), c_col]
            h2 = df.loc[c.index, h_col]
            l2 = df.loc[c.index, l_col]
            close_violations = int(((c > h2) | (c < l2)).sum())
            record(
                f"{label} {asset} Close in [Low, High]",
                close_violations == 0,
                "OK" if close_violations == 0
                else f"{close_violations} rows where Close outside [Low,High]",
            )

        # Volume >= 0
        if v_col in df.columns:
            neg_vol = int((df[v_col].dropna() < 0).sum())
            record(
                f"{label} {asset} Volume ≥ 0",
                neg_vol == 0,
                "OK" if neg_vol == 0 else f"{neg_vol} negative volume rows",
            )


def check_price_sanity(df: pd.DataFrame, label: str) -> None:
    for asset in ASSET_UNIVERSE:
        c_col = f"{asset}_Close"
        if c_col not in df.columns:
            continue

        invalid_prices = int((df[c_col].dropna() <= 0).sum())
        record(
            f"{label} {asset} Close > 0",
            invalid_prices == 0,
            "OK" if invalid_prices == 0 else f"{invalid_prices} non-positive close prices",
        )


def check_returns(df: pd.DataFrame, label: str) -> None:
    for asset in ASSET_UNIVERSE:
        ret_col = f"{asset}_Return"
        if ret_col not in df.columns:
            continue

        returns = df[ret_col].dropna()
        if returns.empty:
            continue

        # Extreme outliers
        extreme = int((returns.abs() > MAX_DAILY_RETURN).sum())
        record(
            f"{label} {asset} Return within ±150%",
            extreme == 0,
            "OK" if extreme == 0
            else f"{extreme} rows with |return| > 150% (verify data)",
        )

        # Returns should not be all-zero or constant
        std = returns.std()
        record(
            f"{label} {asset} Return has variance",
            std > 1e-10,
            f"std={std:.6f}" if std > 1e-10 else "Returns are constant — possible error",
        )


def check_rsi_bounds(df: pd.DataFrame, label: str) -> None:
    for asset in ASSET_UNIVERSE:
        rsi_col = f"{asset}_RSI"
        if rsi_col not in df.columns:
            continue

        rsi = df[rsi_col].dropna()
        if rsi.empty:
            continue

        out_of_bounds = int(((rsi < 0) | (rsi > 100)).sum())
        record(
            f"{label} {asset} RSI in [0, 100]",
            out_of_bounds == 0,
            "OK" if out_of_bounds == 0
            else f"{out_of_bounds} RSI values outside [0,100]",
        )


def check_ma_and_volatility(df: pd.DataFrame, label: str) -> None:
    for asset in ASSET_UNIVERSE:
        ma20_col = f"{asset}_MA20"
        ma50_col = f"{asset}_MA50"
        ma200_col = f"{asset}_MA200"
        vol_col = f"{asset}_Volatility"

        for ma_col, ma_name in [(ma20_col, "MA20"), (ma50_col, "MA50"), (ma200_col, "MA200")]:
            if ma_col in df.columns:
                neg_ma = int((df[ma_col].dropna() <= 0).sum())
                record(
                    f"{label} {asset} {ma_name} > 0",
                    neg_ma == 0,
                    "OK" if neg_ma == 0 else f"{neg_ma} non-positive {ma_name} values",
                )

        if vol_col in df.columns:
            neg_vol = int((df[vol_col].dropna() < 0).sum())
            record(
                f"{label} {asset} Volatility ≥ 0",
                neg_vol == 0,
                "OK" if neg_vol == 0 else f"{neg_vol} negative volatility values",
            )


def check_regime_features(df: pd.DataFrame, label: str) -> None:
    for asset in ASSET_UNIVERSE:
        trend_col = f"{asset}_trend_strength"
        vol_col   = f"{asset}_volatility"
        reg_col   = f"{asset}_regime"
        bull_col  = f"{asset}_bull_prob"
        neut_col  = f"{asset}_neutral_prob"
        bear_col  = f"{asset}_bear_prob"

        # Check trend_strength is finite
        if trend_col in df.columns:
            invalid_trend = int(~np.isfinite(df[trend_col].dropna()).all())
            record(
                f"{label} {asset} trend_strength finite",
                invalid_trend == 0,
                "OK" if invalid_trend == 0 else f"Non-finite trend_strength values",
            )

        # Check volatility >= 0
        if vol_col in df.columns:
            neg_vol = int((df[vol_col].dropna() < 0).sum())
            record(
                f"{label} {asset} volatility ≥ 0",
                neg_vol == 0,
                "OK" if neg_vol == 0 else f"{neg_vol} negative volatility values",
            )

        # Check regime in {-1, 0, 1}
        if reg_col in df.columns:
            invalid_regime = int(~df[reg_col].dropna().isin([-1, 0, 1]).all())
            record(
                f"{label} {asset} regime in {{-1,0,1}}",
                invalid_regime == 0,
                "OK" if invalid_regime == 0 else f"Invalid regime values",
            )

        # Check probabilities in [0,1] and sum ≈ 1
        if bull_col in df.columns and neut_col in df.columns and bear_col in df.columns:
            probs = df[[bull_col, neut_col, bear_col]].dropna()
            if not probs.empty:
                # Check each prob in [0,1]
                out_of_range = int(((probs < 0) | (probs > 1)).any().any())
                record(
                    f"{label} {asset} probs in [0,1]",
                    out_of_range == 0,
                    "OK" if out_of_range == 0 else f"Probabilities outside [0,1]",
                )

                # Check sum ≈ 1
                prob_sums = probs.sum(axis=1)
                invalid_sums = int((prob_sums.abs() - 1 > 1e-3).sum())
                record(
                    f"{label} {asset} prob sum ≈ 1",
                    invalid_sums == 0,
                    "OK" if invalid_sums == 0 else f"{invalid_sums} probability sums not ≈ 1",
                )


def check_feature_variance(df: pd.DataFrame, label: str) -> None:
    for col in df.columns:
        if col.endswith("Return") or col.endswith("trend_strength") or col.endswith("volatility"):
            std_val = df[col].std()
            has_variance = std_val > 1e-8
            record(
                f"{label} {col} has variance",
                has_variance,
                f"std={std_val:.2e}" if has_variance else f"Constant feature (std={std_val:.2e})",
            )


def check_cross_split_consistency(train: pd.DataFrame, test: pd.DataFrame) -> None:
    # Same columns in both splits
    train_cols = set(train.columns)
    test_cols  = set(test.columns)
    only_train = train_cols - test_cols
    only_test  = test_cols  - train_cols

    record(
        "Train / test same column set",
        not only_train and not only_test,
        "Identical columns"
        if not only_train and not only_test
        else f"Only in train: {only_train}  |  Only in test: {only_test}",
    )

    # Close price ranges should have some overlap (same assets, different periods)
    for asset in ASSET_UNIVERSE:
        close_col = f"{asset}_Close"
        if close_col not in train.columns or close_col not in test.columns:
            continue

        tr_max = train[close_col].max()
        te_min = test[close_col].min()
        # Very loosely: test min should be < 10× train max (gross sanity only)
        plausible = te_min < tr_max * 10
        record(
            f"Cross-split {asset} Close price plausible",
            plausible,
            f"Train max={tr_max:.2f}  Test min={te_min:.2f}"
            if plausible
            else f"Test prices seem implausible vs train ({tr_max:.2f} → {te_min:.2f})",
        )


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary() -> int:
    """Print final pass/fail summary. Returns exit code (0=all pass, 1=any fail)."""
    passed  = [r for r in results if r.passed]
    failed  = [r for r in results if not r.passed]

    sep = "═" * 65
    print(f"\n{sep}")
    print("  VALIDATION SUMMARY")
    print(sep)
    print(f"  Total checks : {len(results)}")
    print(f"  Passed       : {len(passed)}")
    print(f"  Failed       : {len(failed)}")
    print(sep)

    if failed:
        print("\n  FAILED CHECKS:")
        for r in failed:
            print(f"    ✗  {r.name}")
            print(f"       → {r.message}")
    else:
        print("\n  ✓ All checks passed — datasets look good!")

    print(f"\n{sep}\n")
    return 0 if not failed else 1


# ── Per-dataset quick stats ───────────────────────────────────────────────────

def print_dataset_stats(df: pd.DataFrame, label: str) -> None:
    nan_pct = df.isna().mean().mean() * 100
    print(f"\n  {'─'*55}")
    print(f"  {label}")
    print(f"  {'─'*55}")
    print(f"  Rows         : {len(df)}")
    print(f"  Columns      : {len(df.columns)}")
    if not df.empty:
        print(f"  Date range   : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  NaN rate     : {nan_pct:.2f}%")

    # Per-asset NaN breakdown
    print(f"  NaN by asset :")
    for asset in ASSET_UNIVERSE:
        asset_cols = [c for c in df.columns if c.startswith(f"{asset}_")]
        if not asset_cols:
            continue
        asset_nan = df[asset_cols].isna().mean().mean() * 100
        bar = "█" * int(asset_nan / 5)
        print(f"    {asset:<10} {asset_nan:5.1f}%  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_validation(train_path: str, test_path: str) -> int:
    """
    Run all validation checks on the train and test CSVs.

    Parameters
    ----------
    train_path : Path to train_dataset.csv
    test_path  : Path to test_dataset.csv

    Returns
    -------
    Exit code: 0 if all checks pass, 1 if any fail.
    """
    sep = "═" * 65
    print(f"\n{sep}")
    print("  SAFE RL PORTFOLIO — DATASET VALIDATION")
    print(f"{sep}\n")

    # ── File checks ────────────────────────────────────────────────────────────
    logger.info("── FILE EXISTENCE ──────────────────────────────────────────")
    train_exists = check_file_exists(train_path, "Train")
    test_exists  = check_file_exists(test_path,  "Test")

    if not train_exists or not test_exists:
        logger.error("Cannot continue — one or more files not found.")
        return print_summary()

    # ── Load ───────────────────────────────────────────────────────────────────
    logger.info("── LOADING ─────────────────────────────────────────────────")
    train = load_csv(train_path, "Train")
    test  = load_csv(test_path,  "Test")

    if train is None or test is None:
        logger.error("Cannot continue — failed to load datasets.")
        return print_summary()

    # ── Quick stats ────────────────────────────────────────────────────────────
    print_dataset_stats(train, "TRAINING SET")
    print_dataset_stats(test,  "TEST SET")
    print()

    # ── Index ──────────────────────────────────────────────────────────────────
    logger.info("── INDEX INTEGRITY ─────────────────────────────────────────")
    check_index(train, "Train")
    check_index(test,  "Test")

    # ── Date ranges ────────────────────────────────────────────────────────────
    logger.info("── DATE RANGE COVERAGE ─────────────────────────────────────")
    check_date_range(train, "Train", TRAIN_EXPECTED_START, TRAIN_EXPECTED_END)
    check_date_range(test,  "Test",  TEST_EXPECTED_START,  TEST_EXPECTED_END)

    # ── Row counts ─────────────────────────────────────────────────────────────
    logger.info("── ROW COUNT PLAUSIBILITY ──────────────────────────────────")
    check_row_count(train, "Train", TRAIN_MIN_ROWS, TRAIN_MAX_ROWS)
    check_row_count(test,  "Test",  TEST_MIN_ROWS,  TEST_MAX_ROWS)

    # ── Overlap ────────────────────────────────────────────────────────────────
    logger.info("── TRAIN/TEST SPLIT ────────────────────────────────────────")
    check_no_overlap(train, test)

    # ── Schema ─────────────────────────────────────────────────────────────────
    logger.info("── COLUMN SCHEMA ───────────────────────────────────────────")
    check_schema(train, "Train")
    check_schema(test,  "Test")

    # ── Dtypes ─────────────────────────────────────────────────────────────────
    logger.info("── NUMERIC TYPES ───────────────────────────────────────────")
    check_dtypes(train, "Train")
    check_dtypes(test,  "Test")

    # ── NaN audit ──────────────────────────────────────────────────────────────
    logger.info("── NaN AUDIT ───────────────────────────────────────────────")
    check_nan_audit(train, "Train")
    check_nan_audit(test,  "Test")

    # ── OHLCV sanity ───────────────────────────────────────────────────────────
    logger.info("── OHLCV SANITY ────────────────────────────────────────────")
    check_ohlcv_sanity(train, "Train")
    check_ohlcv_sanity(test,  "Test")

    # ── Price sanity ───────────────────────────────────────────────────────────
    logger.info("── PRICE SANITY ────────────────────────────────────────────")
    check_price_sanity(train, "Train")
    check_price_sanity(test,  "Test")

    # ── Returns ────────────────────────────────────────────────────────────────
    logger.info("── RETURN SANITY ───────────────────────────────────────────")
    check_returns(train, "Train")
    check_returns(test,  "Test")

    # ── RSI bounds ─────────────────────────────────────────────────────────────
    logger.info("── RSI BOUNDS ──────────────────────────────────────────────")
    check_rsi_bounds(train, "Train")
    check_rsi_bounds(test,  "Test")

    # ── MA / Volatility ────────────────────────────────────────────────────────
    logger.info("── MA & VOLATILITY ────────────────────────────────────────")
    check_ma_and_volatility(train, "Train")
    check_ma_and_volatility(test,  "Test")

    # ── Regime features ────────────────────────────────────────────────────────
    logger.info("── REGIME FEATURES ────────────────────────────────────────")
    check_regime_features(train, "Train")
    check_regime_features(test,  "Test")

    # ── Feature variance ───────────────────────────────────────────────────────
    logger.info("── FEATURE VARIANCE ───────────────────────────────────────")
    check_feature_variance(train, "Train")
    check_feature_variance(test,  "Test")

    # ── Cross-split consistency ────────────────────────────────────────────────
    logger.info("── CROSS-SPLIT CONSISTENCY ─────────────────────────────────")
    check_cross_split_consistency(train, test)

    return print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Safe RL portfolio management datasets."
    )
    parser.add_argument(
        "--train",
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to training CSV  (default: {DEFAULT_TRAIN_PATH})",
    )
    parser.add_argument(
        "--test",
        default=DEFAULT_TEST_PATH,
        help=f"Path to test CSV  (default: {DEFAULT_TEST_PATH})",
    )
    args = parser.parse_args()

    exit_code = run_validation(args.train, args.test)
    sys.exit(exit_code)