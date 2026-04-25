"""
train_dqn.py  (v3 — Regime-Aware)
------------------------------------
DQN baseline — now injects regime features into DataFrame before training.
Action table saved to disk to prevent IndexError at evaluation.
"""

import os, sys
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
EXPLORATION_FRACTION = 0.5
N_DISCRETE_LEVELS    = 8


def _build_action_table(n_assets, n_levels):
    rng     = np.random.default_rng(42)
    actions = set()

    for _ in range(1000):   # S6: reduced from 5000
        w = rng.dirichlet(np.ones(n_assets))
        w = np.round(w * n_levels) / n_levels
        s = w.sum()
        if s > 0:
            w = w / s
            w = np.minimum(w, 0.40)
            s = w.sum()
            if s > 0:
                w = w / s
            actions.add(tuple(np.round(w, 4)))

    actions.add(tuple(np.round(np.ones(n_assets) / n_assets, 4)))
    for i in range(n_assets):
        single    = np.zeros(n_assets)
        single[i] = 0.40
        rest      = np.ones(n_assets) * (0.60 / (n_assets - 1))
        rest[i]   = 0.0
        single   += rest
        actions.add(tuple(np.round(single, 4)))

    table = np.array(list(actions), dtype=np.float32)
    table = table / table.sum(axis=1, keepdims=True)
    return table


class DiscretePortfolioWrapper(gym.Wrapper):
    def __init__(self, env, action_table):
        super().__init__(env)
        self.action_table = action_table
        self.action_space = spaces.Discrete(len(action_table))

    def step(self, action):
        weights = self.action_table[int(action)]
        return self.env.step(weights)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


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


def train_dqn():
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  DQN BASELINE — PORTFOLIO MANAGEMENT (Regime-Aware)")
    print("=" * 60)

    # R1: inject regime features
    raw_df = pd.read_csv(TRAIN_DATA_PATH, index_col="Date", parse_dates=True)
    df     = detect_market_regime(raw_df)
    print(f"Dataset: {df.shape}  ({df.index.min().date()} -> {df.index.max().date()})")

    base_env     = PortfolioTradingEnv(df, safe_reward=False, random_start=True)
    action_table = _build_action_table(N_ASSETS, N_DISCRETE_LEVELS)
    print(f"Action space: {len(action_table)} portfolio vectors")

    env = Monitor(DiscretePortfolioWrapper(base_env, action_table))

    model = DQN(
        policy               = "MlpPolicy",
        env                  = env,
        learning_rate        = LEARNING_RATE,
        batch_size           = BATCH_SIZE,
        buffer_size          = BUFFER_SIZE,
        gamma                = GAMMA,
        exploration_fraction = EXPLORATION_FRACTION,
        exploration_final_eps= 0.02,
        train_freq           = 4,
        target_update_interval = 1000,
        policy_kwargs        = dict(net_arch=[256, 256, 128]),
        device               = "cpu",
        verbose              = 0,
    )

    cb = RewardLogger()
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps ...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=True)

    model.save(MODEL_PATH)
    np.save(ACTION_TABLE_PATH, action_table)
    print(f"Model saved        -> {MODEL_PATH}.zip")
    print(f"Action table saved -> {ACTION_TABLE_PATH}  ({len(action_table)} vectors)")

    if cb.episode_rewards:
        _plot_curve(cb.episode_rewards, "DQN Training — Episode Rewards",
                    os.path.join(RESULTS_DIR, "dqn_train_curve.png"))

    print("DQN training complete.\n")


def _plot_curve(rewards, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
    if len(rewards) >= 10:
        w = max(10, len(rewards) // 20)
        smooth = pd.Series(rewards).rolling(w, min_periods=1).mean()
        ax.plot(smooth, color="steelblue", linewidth=2, label=f"Rolling mean ({w} ep)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Curve saved -> {save_path}")


if __name__ == "__main__":
    train_dqn()