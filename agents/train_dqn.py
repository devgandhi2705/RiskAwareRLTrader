"""
train_dqn.py  (v5 — Alpha Reward, 43-dim Obs)
----------------------------------------------
DQN baseline agent.  Uses the same environment (v8) and reward function
as PPO/Safe PPO (alpha reward, no regime features).

Changes from v4
---------------
  D1 : OBS_DIM updated to 43 (matches v8 environment).
  D2 : Reward = portfolio_return + 0.5 × excess − 0.0025 × turnover
       (same alpha formula as PPO, implemented in environment).
  D3 : Random-episode training (same as PPO v5).
  D4 : Log via log() → terminal + training_log.txt.
  D5 : Device = auto.

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.trading_environment import PortfolioTradingEnv, ASSETS, N_ASSETS, OBS_DIM

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "training_log.txt")

def log(msg: str) -> None:
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")
MODEL_DIR         = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR       = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH        = os.path.join(MODEL_DIR, "dqn_portfolio")
ACTION_TABLE_PATH = os.path.join(MODEL_DIR, "dqn_action_table.npy")

TOTAL_TIMESTEPS      = 1_000_000
LEARNING_RATE        = 1e-4
BATCH_SIZE           = 64
BUFFER_SIZE          = 50_000
GAMMA                = 0.99
EXPLORATION_FRACTION = 0.15
N_DISCRETE_LEVELS    = 8
MAX_WEIGHT           = 0.90   # matches env v8 max_weight


# ── Action table ───────────────────────────────────────────────────────────────

def _build_action_table(n_assets: int, n_levels: int, max_w: float) -> np.ndarray:
    """Build a diverse set of discrete portfolio weight vectors."""
    rng     = np.random.default_rng(42)
    actions = set()

    for _ in range(600):
        w = rng.dirichlet(np.ones(n_assets))
        w = np.round(w * n_levels) / n_levels
        s = w.sum()
        if s > 0:
            w /= s
            w  = np.minimum(w, max_w)
            s  = w.sum()
            if s > 0:
                w /= s
            actions.add(tuple(np.round(w, 4)))

    # Equal weight (benchmark)
    actions.add(tuple(np.round(np.ones(n_assets) / n_assets, 4)))

    # Single-asset concentrated portfolios
    for i in range(n_assets):
        w    = np.ones(n_assets) * (1.0 - max_w) / (n_assets - 1)
        w[i] = max_w
        actions.add(tuple(np.round(w / w.sum(), 4)))

    table = np.array(list(actions), dtype=np.float32)
    table = table / table.sum(axis=1, keepdims=True)
    return table


# ── Discrete wrapper ───────────────────────────────────────────────────────────

class DiscretePortfolioWrapper(gym.Wrapper):
    """Maps integer actions → portfolio weight vectors."""

    def __init__(self, env: gym.Env, action_table: np.ndarray):
        super().__init__(env)
        self.action_table = action_table
        self.action_space = spaces.Discrete(len(action_table))

    def step(self, action: int):
        return self.env.step(self.action_table[int(action)])

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ── Callback ───────────────────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._buf = 0.0

    def _on_step(self):
        self._buf += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._buf)
            self._buf = 0.0
        return True


# ── Training ───────────────────────────────────────────────────────────────────

def train_dqn():
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log("=" * 60)
    log("  DQN — ALPHA REWARD TRAINING  (v5)")
    log("=" * 60)

    df = pd.read_csv(TRAIN_DATA_PATH, index_col="Date", parse_dates=True)
    log(f"Dataset: {df.shape}  "
        f"({df.index.min().date()} → {df.index.max().date()})")
    log(f"OBS_DIM={OBS_DIM}  total_timesteps={TOTAL_TIMESTEPS:,}")

    base_env     = PortfolioTradingEnv(df, safe_reward=False, random_start=True)
    action_table = _build_action_table(N_ASSETS, N_DISCRETE_LEVELS, MAX_WEIGHT)
    log(f"Action space: {len(action_table)} discrete portfolio vectors")

    env = Monitor(DiscretePortfolioWrapper(base_env, action_table))

    model = DQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = LEARNING_RATE,
        batch_size             = BATCH_SIZE,
        buffer_size            = BUFFER_SIZE,
        gamma                  = GAMMA,
        exploration_fraction   = EXPLORATION_FRACTION,
        exploration_final_eps  = 0.02,
        train_freq             = 4,
        target_update_interval = 1000,
        policy_kwargs          = dict(net_arch=[256, 256, 128]),
        device                 = "auto",
        verbose                = 0,
    )

    cb = RewardLogger()
    log(f"Training for {TOTAL_TIMESTEPS:,} timesteps …")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=True)
    elapsed = time.time() - t0
    log(f"DQN training complete in {elapsed/60:.1f} min.")

    model.save(MODEL_PATH)
    np.save(ACTION_TABLE_PATH, action_table)
    log(f"Model saved        → {MODEL_PATH}.zip")
    log(f"Action table saved → {ACTION_TABLE_PATH}  ({len(action_table)} vectors)")

    if cb.episode_rewards:
        _plot_curve(
            cb.episode_rewards,
            "DQN Training — Episode Rewards",
            os.path.join(RESULTS_DIR, "dqn_train_curve.png"),
        )


def _plot_curve(rewards, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
    if len(rewards) >= 10:
        w      = max(10, len(rewards) // 20)
        smooth = pd.Series(rewards).rolling(w, min_periods=1).mean()
        ax.plot(smooth, color="steelblue", linewidth=2,
                label=f"Rolling mean ({w} ep)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log(f"Training curve saved → {save_path}")


if __name__ == "__main__":
    train_dqn()