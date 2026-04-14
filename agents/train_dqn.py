"""
train_dqn.py  (v4 — Alpha-Aware, Tighter Config)
-------------------------------------------------
DQN baseline — uses the v6 continuous trading environment via a discrete
action wrapper.  Incorporates same data-loading guard as train_ppo.py v4.

Changes from v3
---------------
  D1 : _load_and_prepare() guard: skip detect_market_regime() when columns
       already exist in the dataset CSV (avoids double computation).
  D2 : EXPLORATION_FRACTION reduced 0.2 → 0.15 — agent exploits sooner.
  D3 : device changed "cpu" → "auto" (use GPU when available).
  D4 : Action table generation tightened: 500 samples with max_weight=0.40
       consistent with environment constraint.
  D5 : ACTION_TABLE_PATH always saved so evaluate_agent.py can load it.

Retained from v3
----------------
  DiscretePortfolioWrapper, RewardLogger, _plot_curve
  400 000 total timesteps, lr=1e-4, batch=64, buffer=50 000

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
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

from env.trading_environment import PortfolioTradingEnv, ASSETS, N_ASSETS
from data_pipeline.regime_detection import detect_market_regime

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")
MODEL_DIR         = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR       = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH        = os.path.join(MODEL_DIR, "dqn_portfolio")
ACTION_TABLE_PATH = os.path.join(MODEL_DIR, "dqn_action_table.npy")

TOTAL_TIMESTEPS      = 400_000
LEARNING_RATE        = 1e-4
BATCH_SIZE           = 64
BUFFER_SIZE          = 50_000
GAMMA                = 0.99
EXPLORATION_FRACTION = 0.15   # D2: reduced from 0.2
N_DISCRETE_LEVELS    = 8
MAX_WEIGHT           = 0.40   # D4: consistent with env constraint


# ── Action table ──────────────────────────────────────────────────────────────

def _build_action_table(n_assets: int, n_levels: int, max_w: float = MAX_WEIGHT) -> np.ndarray:
    """
    Build a diverse set of discrete portfolio weight vectors.

    Includes:
      - Dirichlet-sampled vectors (broad coverage)
      - Equal-weight allocation  (benchmark)
      - Single-asset dominated allocations capped at max_w
    """
    rng     = np.random.default_rng(42)
    actions = set()

    for _ in range(500):
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

    # Equal-weight benchmark
    actions.add(tuple(np.round(np.ones(n_assets) / n_assets, 4)))

    # Single-asset capped portfolios
    for i in range(n_assets):
        w       = np.ones(n_assets) * (1.0 - max_w) / (n_assets - 1)
        w[i]    = max_w
        actions.add(tuple(np.round(w / w.sum(), 4)))

    table = np.array(list(actions), dtype=np.float32)
    table = table / table.sum(axis=1, keepdims=True)
    return table


# ── Discrete wrapper ──────────────────────────────────────────────────────────

class DiscretePortfolioWrapper(gym.Wrapper):
    """
    Wraps the continuous PortfolioTradingEnv with a Discrete action space.
    Each integer action maps to a pre-built portfolio weight vector.
    PPO / Safe PPO use the environment directly; only DQN needs this wrapper.
    """

    def __init__(self, env: gym.Env, action_table: np.ndarray):
        super().__init__(env)
        self.action_table = action_table
        self.action_space = spaces.Discrete(len(action_table))

    def step(self, action: int):
        weights = self.action_table[int(action)]
        return self.env.step(weights)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ── Callback ──────────────────────────────────────────────────────────────────

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


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_and_prepare(path: str) -> pd.DataFrame:
    """
    D1: Skip detect_market_regime() when columns already present in CSV.
    This avoids redundant recomputation when build_dataset.py has already
    produced the full feature set.
    """
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if "BTC_bull_prob" in df.columns:
        return df
    return detect_market_regime(df)


# ── Training ──────────────────────────────────────────────────────────────────

def train_dqn():
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  DQN BASELINE — PORTFOLIO MANAGEMENT (v4)")
    print("=" * 60)

    df = _load_and_prepare(TRAIN_DATA_PATH)
    print(f"Dataset: {df.shape}  ({df.index.min().date()} -> {df.index.max().date()})")

    base_env     = PortfolioTradingEnv(df, safe_reward=False, random_start=True)
    action_table = _build_action_table(N_ASSETS, N_DISCRETE_LEVELS)
    print(f"Action space: {len(action_table)} discrete portfolio vectors")

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
        device                 = "auto",   # D3
        verbose                = 0,
    )

    cb = RewardLogger()
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps …")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=True)

    # D5: always save both model and action table
    model.save(MODEL_PATH)
    np.save(ACTION_TABLE_PATH, action_table)
    print(f"Model saved        → {MODEL_PATH}.zip")
    print(f"Action table saved → {ACTION_TABLE_PATH}  ({len(action_table)} vectors)")

    if cb.episode_rewards:
        _plot_curve(
            cb.episode_rewards,
            "DQN Training — Episode Rewards",
            os.path.join(RESULTS_DIR, "dqn_train_curve.png"),
        )

    print("DQN training complete.\n")


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_curve(rewards: list, title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
    if len(rewards) >= 10:
        w = max(10, len(rewards) // 20)
        smooth = pd.Series(rewards).rolling(w, min_periods=1).mean()
        ax.plot(smooth, color="steelblue", linewidth=2,
                label=f"Rolling mean ({w} ep)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curve saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_dqn()