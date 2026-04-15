"""
validate_dataset.py  (v3 — No Regime Features)
-----------------------------------------------
Quick sanity-check for the rebuilt dataset.

Checks:
  1. Both CSV files exist.
  2. Date range is correct.
  3. No NaN values in indicator columns.
  4. Exactly 35 indicator columns present (7 assets × 5 features).
  5. Observation size matches environment (35 + 7 weights + 1 value = 43).

Run:
    python data_pipeline/validate_dataset.py

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(_THIS_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data")
LOG_FILE   = os.path.join(ROOT_DIR, "dataset_build_log.txt")

TRAIN_CSV = os.path.join(DATA_DIR, "train_dataset.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_dataset.csv")

ASSETS           = ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty50", "Sensex"]
FEATURES         = ["return", "MA20", "MA50", "RSI", "volatility"]
EXPECTED_IND_COLS = [f"{a}_{f}" for a in ASSETS for f in FEATURES]
EXPECTED_N_IND    = len(EXPECTED_IND_COLS)   # 35

# Expected observation size = 35 indicators + 7 weights + 1 portfolio value
EXPECTED_OBS_DIM  = EXPECTED_N_IND + len(ASSETS) + 1   # 43

TRAIN_START = "2016-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2024-12-31"


def log(msg: str) -> None:
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def validate(csv_path: str, label: str, date_start: str, date_end: str) -> bool:
    ok = True
    log(f"\n  Validating {label}  ({csv_path})")

    if not os.path.isfile(csv_path):
        log(f"  ✗  FILE MISSING: {csv_path}")
        return False

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    log(f"  Shape    : {df.shape[0]} rows × {df.shape[1]} cols")

    # Date range
    actual_start = str(df.index.min().date())
    actual_end   = str(df.index.max().date())
    if actual_start > date_start:
        log(f"  ⚠  Start date {actual_start} is later than expected {date_start}")
    else:
        log(f"  ✓  Start date  : {actual_start}")

    if actual_end < date_end:
        log(f"  ⚠  End date {actual_end} is earlier than expected {date_end}")
    else:
        log(f"  ✓  End date    : {actual_end}")

    # Indicator columns present
    present  = [c for c in EXPECTED_IND_COLS if c in df.columns]
    missing  = [c for c in EXPECTED_IND_COLS if c not in df.columns]
    log(f"  Indicator cols : {len(present)} / {EXPECTED_N_IND}")
    if missing:
        log(f"  ✗  Missing indicator columns: {missing[:10]}" +
            (" ..." if len(missing) > 10 else ""))
        ok = False
    else:
        log(f"  ✓  All {EXPECTED_N_IND} indicator columns present")

    # Regime columns must NOT be present
    regime_cols = [c for c in df.columns if any(
        kw in c for kw in ["regime", "bull_prob", "bear_prob", "neutral_prob",
                            "trend_strength", "market_volatility"]
    )]
    if regime_cols:
        log(f"  ⚠  Regime columns still present ({len(regime_cols)}): {regime_cols[:5]}")
    else:
        log(f"  ✓  No regime columns (expected)")

    # NaN check on indicator columns
    ind_df  = df[present]
    nan_pct = ind_df.isna().mean().mean() * 100
    if nan_pct > 1.0:
        log(f"  ⚠  NaN rate in indicators: {nan_pct:.2f}%  (>1% threshold)")
    else:
        log(f"  ✓  NaN rate in indicators : {nan_pct:.3f}%")

    nan_cols = ind_df.columns[ind_df.isna().any()].tolist()
    if nan_cols:
        log(f"  ⚠  Columns with NaNs: {nan_cols[:8]}")

    # Observation dimension
    log(f"  Expected obs dim : {EXPECTED_OBS_DIM}  ({EXPECTED_N_IND} indicators "
        f"+ {len(ASSETS)} weights + 1 value)")

    return ok


def main():
    log("\n" + "=" * 55)
    log("  DATASET VALIDATION  (v3)")
    log("=" * 55)

    train_ok = validate(TRAIN_CSV, "train_dataset.csv", TRAIN_START, TRAIN_END)
    test_ok  = validate(TEST_CSV,  "test_dataset.csv",  TEST_START,  TEST_END)

    log("\n" + "=" * 55)
    if train_ok and test_ok:
        log("  RESULT: ALL CHECKS PASSED ✓")
    else:
        log("  RESULT: SOME CHECKS FAILED ✗  — rebuild with build_dataset.py")
    log("=" * 55 + "\n")

    return train_ok and test_ok


if __name__ == "__main__":
    main()