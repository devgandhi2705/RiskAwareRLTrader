"""
train_all.py
------------
Master training script — trains DQN, PPO, and Safe PPO in one command.

Usage
-----
    # Train all three agents sequentially (safest, works on any machine)
    python train_all.py

    # Train sequentially but skip already-trained models
    python train_all.py --skip-existing

    # Train only specific agents
    python train_all.py --skip-existing

    # Run evaluate_agent.py automatically after training completes
    python train_all.py --evaluate

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
import time
import argparse
import subprocess
import importlib
from datetime import timedelta

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Agent registry ────────────────────────────────────────────────────────────
# Each entry: display name, module path, entry function, saved model filename
AGENTS = {
    "dqn": {
        "label":      "DQN (Baseline)",
        "module":     "agents.train_dqn",
        "fn":         "train_dqn",
        "model_file": "models/dqn_portfolio.zip",
    },
    "ppo": {
        "label":      "PPO (Baseline)",
        "module":     "agents.train_ppo",
        "fn":         "train_ppo",
        "fn_kwargs":  {"safe_reward": False},
        "model_file": "models/ppo_portfolio.zip",
    },
    "safe_ppo": {
        "label":      "Safe PPO (Risk-Aware)",
        "module":     "agents.train_safe_ppo",
        "fn":         "train_safe_ppo",
        "model_file": "models/safe_ppo_portfolio.zip",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    return str(timedelta(seconds=int(seconds)))


def _model_exists(agent_key: str) -> bool:
    path = os.path.join(PROJECT_ROOT, AGENTS[agent_key]["model_file"])
    return os.path.isfile(path)


def _print_header(text: str) -> None:
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  {text}")
    print(f"{sep}\n")


def _print_section(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}\n")


# ── Training runner ───────────────────────────────────────────────────────────

def run_agent(agent_key: str, cfg: dict) -> tuple[bool, float]:
    """
    Import and call the training function for one agent in-process.

    Returns (success: bool, elapsed_seconds: float).
    """
    _print_section(f"Training: {cfg['label']}")

    t_start = time.time()
    try:
        module   = importlib.import_module(cfg["module"])
        fn       = getattr(module, cfg["fn"])
        kwargs   = cfg.get("fn_kwargs", {})
        fn(**kwargs)
        elapsed  = time.time() - t_start
        print(f"\n  ✓  {cfg['label']} finished in {_fmt_duration(elapsed)}")
        return True, elapsed

    except Exception as exc:
        elapsed = time.time() - t_start
        print(f"\n  ✗  {cfg['label']} FAILED after {_fmt_duration(elapsed)}")
        print(f"     Error: {exc}")
        import traceback
        traceback.print_exc()
        return False, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def train_all(
    agent_keys: list[str],
    skip_existing: bool = False,
    run_eval: bool = False,
) -> None:
    """
    Train the specified agents one after another.

    Parameters
    ----------
    agent_keys    : Which agents to train (ordered list).
    skip_existing : Skip agents whose model file already exists on disk.
    run_eval      : If True, run evaluate_agent.py after all training completes.
    """
    _print_header("SAFE RL PORTFOLIO MANAGEMENT — MASTER TRAINING SCRIPT")

    # ── Pre-flight: check data files exist ───────────────────────────────────
    train_csv = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")
    test_csv  = os.path.join(PROJECT_ROOT, "data", "test_dataset.csv")

    for path, label in [(train_csv, "train_dataset.csv"), (test_csv, "test_dataset.csv")]:
        if not os.path.isfile(path):
            print(f"  ✗  MISSING: {path}")
            print(f"     Run 'python data_pipeline/build_dataset.py' first.\n")
            sys.exit(1)
    print(f"  ✓  data/train_dataset.csv found")
    print(f"  ✓  data/test_dataset.csv  found\n")

    # ── Detect device ─────────────────────────────────────────────────────────
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ⚡  GPU detected: {gpu_name}")
        else:
            print(f"  💻  No GPU detected — training on CPU")
            print(f"      (Install CUDA PyTorch for ~5x speedup on RTX 4050)")
    except ImportError:
        device = "cpu"
        print("  ⚠   PyTorch not importable directly — SB3 will handle device selection")

    print()

    # ── Plan which agents to run ──────────────────────────────────────────────
    to_train: list[str] = []
    skipped:  list[str] = []

    for key in agent_keys:
        if key not in AGENTS:
            print(f"  ⚠   Unknown agent '{key}' — skipping.")
            continue
        if skip_existing and _model_exists(key):
            skipped.append(key)
            print(f"  ~   {AGENTS[key]['label']:<30} model already exists — skipping (--skip-existing)")
        else:
            to_train.append(key)

    if not to_train:
        print("\n  Nothing to train. All models already exist.")
        print("  Remove --skip-existing or delete model files to retrain.\n")
        return

    print(f"\n  Agents to train : {[AGENTS[k]['label'] for k in to_train]}")
    if skipped:
        print(f"  Agents skipped  : {[AGENTS[k]['label'] for k in skipped]}")
    print()

    # ── Train sequentially ────────────────────────────────────────────────────
    wall_start    = time.time()
    results:  dict[str, tuple[bool, float]] = {}

    for key in to_train:
        success, elapsed = run_agent(key, AGENTS[key])
        results[key] = (success, elapsed)

    total_elapsed = time.time() - wall_start

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_header("TRAINING SUMMARY")

    all_passed = True
    for key in to_train:
        success, elapsed = results[key]
        status = "✓  Done   " if success else "✗  FAILED "
        if not success:
            all_passed = False
        print(f"  {status}  {AGENTS[key]['label']:<30}  {_fmt_duration(elapsed)}")

    for key in skipped:
        print(f"  ~  Skipped  {AGENTS[key]['label']:<30}  (model existed)")

    print(f"\n  Total wall time : {_fmt_duration(total_elapsed)}")
    print()

    # ── Optional evaluation ───────────────────────────────────────────────────
    if run_eval:
        if all_passed:
            _print_section("Running evaluation …")
            eval_module = importlib.import_module("evaluation.evaluate_agent")
            eval_module.evaluate_all()
        else:
            print("  ⚠   Skipping evaluation — one or more agents failed to train.")

    if all_passed:
        print("  All agents trained successfully.\n")
    else:
        failed = [k for k in to_train if not results[k][0]]
        print(f"  ✗  Failed agents: {failed}\n")
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all Safe RL portfolio agents in one command.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENTS.keys()),
        choices=list(AGENTS.keys()),
        metavar="AGENT",
        help=(
            "Which agents to train. Choose from: dqn, ppo, safe_ppo\n"
            "Default: all three.\n"
            "Example: --agents ppo safe_ppo"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip agents whose .zip model file already exists in models/.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Run evaluate_agent.py automatically after training completes.",
    )

    args = parser.parse_args()

    train_all(
        agent_keys    = args.agents,
        skip_existing = args.skip_existing,
        run_eval      = args.evaluate,
    )