"""
train_all.py  (v2 — Full Pipeline, Simple Logging)
---------------------------------------------------
Master orchestration script.

Pipeline order:
  1. data_pipeline/build_dataset.py  — build / rebuild dataset
  2. data_pipeline/validate_dataset.py — sanity check
  3. agents/train_dqn.py
  4. agents/train_ppo.py
  5. agents/train_safe_ppo.py
  6. evaluation/evaluate_agent.py

Usage
-----
    python train_all.py                       # full pipeline
    python train_all.py --skip-build          # skip dataset rebuild
    python train_all.py --agents ppo          # train only PPO
    python train_all.py --skip-existing       # skip already-trained models
    python train_all.py --no-eval             # skip final evaluation

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
import time
import argparse
import importlib
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "training_log.txt")

def log(msg: str) -> None:
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ── Agent registry ────────────────────────────────────────────────────────────
AGENTS = {
    "dqn": {
        "label":      "DQN (Baseline)",
        "module":     "agents.train_dqn",
        "fn":         "train_dqn",
        "fn_kwargs":  {},
        "model_file": "models/dqn_portfolio.zip",
    },
    "ppo": {
        "label":      "PPO (Alpha)",
        "module":     "agents.train_ppo",
        "fn":         "train_ppo",
        "fn_kwargs":  {"safe_reward": False},
        "model_file": "models/ppo_portfolio.zip",
    },
    "safe_ppo": {
        "label":      "Safe PPO (Risk-Aware)",
        "module":     "agents.train_safe_ppo",
        "fn":         "train_safe_ppo",
        "fn_kwargs":  {},
        "model_file": "models/safe_ppo_portfolio.zip",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(s: float) -> str:
    return str(timedelta(seconds=int(s)))


def _model_exists(key: str) -> bool:
    return os.path.isfile(os.path.join(PROJECT_ROOT, AGENTS[key]["model_file"]))


def _sep(char="═", n=62) -> str:
    return char * n


# ── Step runners ──────────────────────────────────────────────────────────────

def run_build_dataset() -> bool:
    log(f"\n{_sep()}")
    log("  Step 1/2: Building dataset …")
    log(_sep())
    t0 = time.time()
    try:
        m  = importlib.import_module("data_pipeline.build_dataset")
        m.build_full_pipeline(use_cached_raw=False)
        log(f"  ✓  Dataset built  ({_fmt(time.time()-t0)})")
        return True
    except Exception as exc:
        log(f"  ✗  Dataset build FAILED: {exc}")
        import traceback; traceback.print_exc()
        return False


def run_validate_dataset() -> bool:
    log(f"\n{_sep()}")
    log("  Step 2/2: Validating dataset …")
    log(_sep())
    try:
        m  = importlib.import_module("data_pipeline.validate_dataset")
        ok = m.main()
        status = "✓  Validation passed" if ok else "⚠  Validation warnings (check log)"
        log(f"  {status}")
        return True   # don't abort on warnings
    except Exception as exc:
        log(f"  ⚠  Validation error: {exc}")
        return True   # non-fatal


def run_agent(key: str) -> tuple:
    cfg = AGENTS[key]
    log(f"\n{_sep('-')}")
    log(f"  Training: {cfg['label']}")
    log(_sep("-"))
    t0 = time.time()
    try:
        m      = importlib.import_module(cfg["module"])
        fn     = getattr(m, cfg["fn"])
        fn(**cfg.get("fn_kwargs", {}))
        elapsed = time.time() - t0
        log(f"  ✓  {cfg['label']} finished  ({_fmt(elapsed)})")
        return True, elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        log(f"  ✗  {cfg['label']} FAILED: {exc}  ({_fmt(elapsed)})")
        import traceback; traceback.print_exc()
        return False, elapsed


def run_evaluation(agent_keys: list) -> None:
    log(f"\n{_sep()}")
    log("  Final step: Evaluating agents …")
    log(_sep())
    try:
        m = importlib.import_module("evaluation.evaluate_agent")
        m.evaluate_all(agent_keys=agent_keys)
    except Exception as exc:
        log(f"  ⚠  Evaluation error: {exc}")
        import traceback; traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    agent_keys: list,
    skip_build: bool = False,
    skip_existing: bool = False,
    run_eval: bool = True,
) -> None:
    # Start fresh log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    wall_start = time.time()

    log(_sep())
    log("  SAFE RL PORTFOLIO — FULL PIPELINE  (v2)")
    log(_sep())

    # ── Device check ─────────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            log(f"  ⚡  GPU: {torch.cuda.get_device_name(0)}")
        else:
            log("  💻  CPU training (no GPU detected)")
    except ImportError:
        log("  ⚠   torch not importable — SB3 will manage device")

    # ── Step 1-2: Dataset ─────────────────────────────────────────────────────
    if skip_build:
        log("\n  Skipping dataset build (--skip-build).")
        train_csv = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")
        test_csv  = os.path.join(PROJECT_ROOT, "data", "test_dataset.csv")
        for p, lbl in [(train_csv, "train_dataset.csv"),
                       (test_csv,  "test_dataset.csv")]:
            if not os.path.isfile(p):
                log(f"  ✗  MISSING: {p} — run without --skip-build first.")
                sys.exit(1)
    else:
        if not run_build_dataset():
            log("Aborting: dataset build failed.")
            sys.exit(1)
        run_validate_dataset()

    # ── Steps 3-5: Train agents ───────────────────────────────────────────────
    to_train = []
    skipped  = []
    for key in agent_keys:
        if key not in AGENTS:
            log(f"  ⚠  Unknown agent '{key}' — skipping.")
            continue
        if skip_existing and _model_exists(key):
            skipped.append(key)
            log(f"  ~  {AGENTS[key]['label']:<30} already trained — skipping")
        else:
            to_train.append(key)

    results = {}
    for key in to_train:
        ok, elapsed = run_agent(key)
        results[key] = (ok, elapsed)

    # ── Summary ───────────────────────────────────────────────────────────────
    log(f"\n{_sep()}")
    log("  TRAINING SUMMARY")
    log(_sep())
    all_ok = True
    for key in to_train:
        ok, elapsed = results[key]
        status = "✓ Done  " if ok else "✗ FAILED"
        if not ok:
            all_ok = False
        log(f"  {status}  {AGENTS[key]['label']:<30}  {_fmt(elapsed)}")
    for key in skipped:
        log(f"  ~ Skip   {AGENTS[key]['label']:<30}  (model existed)")
    log(f"\n  Total wall time: {_fmt(time.time() - wall_start)}")

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    if run_eval:
        if all_ok:
            run_evaluation(to_train + skipped)
        else:
            log("  ⚠   Skipping evaluation — one or more agents failed.")

    if all_ok:
        log("\n  All steps completed successfully.\n")
    else:
        failed = [k for k in to_train if not results[k][0]]
        log(f"\n  ✗  Failed: {failed}\n")
        sys.exit(1)

    log(f"  Log saved → {LOG_FILE}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Safe RL portfolio pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENTS.keys()),
        choices=list(AGENTS.keys()),
        metavar="AGENT",
        help="Agents to train (default: dqn ppo safe_ppo).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        default=False,
        help="Rebuild dataset before training agents (default: skip dataset).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip agents whose model file already exists.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip final evaluation step.",
    )
    args = parser.parse_args()

    main(
        agent_keys    = args.agents,
        skip_build    = not args.build,
        skip_existing = args.skip_existing,
        run_eval      = not args.no_eval,
    )