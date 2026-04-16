"""
train_ppo.py  (v5 — Random-Episode Training)
---------------------------------------------
Trains PPO (alpha agent) and Safe PPO (risk-aware agent).

Changes from v4
---------------
  P1 : Walk-forward training REMOVED.
       Random-episode training: each episode starts at a random index.
  P2 : Total timesteps = 1,000,000.
  P3 : Validation set 2021-01-01 → 2022-12-31.
       Checkpoint saved based on best validation Sharpe.
  P4 : Logging via log() → terminal + training_log.txt.
  P5 : No regime features in data (compatible with v3 dataset).
  P6 : ent_coef = 0.003 (from v4).

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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.trading_environment import PortfolioTradingEnv, OBS_DIM

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(PROJECT_ROOT, "logs","training_log.txt")

def log(msg: str) -> None:
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")
MODEL_DIR       = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")

TOTAL_TIMESTEPS   = 1_000_000
LEARNING_RATE     = 1e-4
N_STEPS           = 2048
BATCH_SIZE        = 64
N_EPOCHS          = 10
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_RANGE        = 0.1
ENT_COEF          = 0.003

# Validation window (held out from random-episode training for checkpointing)
VAL_START = "2021-01-01"
VAL_END   = "2022-12-31"

# How often to run validation (in timesteps)
VAL_INTERVAL = 100_000


# ── Callbacks ─────────────────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Accumulates per-episode total rewards."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._buf = []

    def _on_step(self):
        self._buf.append(float(self.locals["rewards"][0]))
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self._buf))
            self._buf = []
        return True


class ValidationCallback(BaseCallback):
    """
    Runs one deterministic validation episode every VAL_INTERVAL steps.
    Saves a checkpoint whenever validation Sharpe improves.
    """
    def __init__(self, val_df, model_path, vecnorm_path, safe, interval, verbose=0):
        super().__init__(verbose)
        self.val_df       = val_df
        self.model_path   = model_path
        self.vecnorm_path = vecnorm_path
        self.safe         = safe
        self.interval     = interval
        self.best_sharpe  = -np.inf
        self._last_check  = 0

    def _on_step(self):
        if self.num_timesteps - self._last_check < self.interval:
            return True
        self._last_check = self.num_timesteps

        env_val  = PortfolioTradingEnv(self.val_df, safe_reward=self.safe,
                                       random_start=False)
        obs, _   = env_val.reset()
        done     = False
        vals     = [env_val.initial_value]
        rets     = []

        while not done:
            obs_n     = self.training_env.normalize_obs(obs.reshape(1, -1))[0]
            action, _ = self.model.predict(obs_n, deterministic=True)
            obs, _, done, trunc, info = env_val.step(action)
            done = done or trunc
            vals.append(info["portfolio_value"])
            rets.append(info["net_return"])

        if len(rets) < 5:
            return True

        r   = np.array(rets)
        std = r.std()
        sh  = float(r.mean() / std * np.sqrt(252)) if std > 1e-10 else 0.0
        ret = (vals[-1] - vals[0]) / vals[0] * 100

        dd_arr = np.maximum.accumulate(vals)
        dd_max = float(((dd_arr - vals) / np.where(dd_arr > 0, dd_arr, 1e-9)).max()) * 100

        log(f"  [Val @ {self.num_timesteps:,}]  "
            f"Return={ret:+.2f}%  Sharpe={sh:.3f}  MaxDD={dd_max:.2f}%")

        if sh > self.best_sharpe:
            self.best_sharpe = sh
            self.model.save(self.model_path + "_best")
            self.training_env.save(self.vecnorm_path + "_best")
            log(f"  → New best Sharpe={sh:.3f}  checkpoint saved.")

        return True


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._buf = []

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if info:
            self._buf.append({
                "turnover":   info.get("turnover",        0.0),
                "drawdown":   info.get("drawdown",        0.0),
                "volatility": info.get("volatility",      0.0),
            })
        if self.locals["dones"][0] and self._buf:
            avg_turn = float(np.mean([s["turnover"]   for s in self._buf]))
            avg_dd   = float(np.mean([s["drawdown"]   for s in self._buf]))
            avg_vol  = float(np.mean([s["volatility"] for s in self._buf]))
            if self.logger:
                self.logger.record("env/avg_turnover",  avg_turn)
                self.logger.record("env/avg_drawdown",  avg_dd)
                self.logger.record("env/avg_volatility",avg_vol)
            self._buf = []
        return True


# ── Data loading ───────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="Date", parse_dates=True)


# ── Environment factory ────────────────────────────────────────────────────────

def _make_env_fn(df, safe, random_start=True):
    def _fn():
        env = PortfolioTradingEnv(df, safe_reward=safe, random_start=random_start)
        return Monitor(env)
    return _fn


def _build_vecenv(df, safe, random_start=True):
    vec = DummyVecEnv([_make_env_fn(df, safe, random_start)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False,
                       clip_obs=10.0, gamma=GAMMA)
    return vec


# ── Main training function ─────────────────────────────────────────────────────

def train_ppo(safe_reward: bool = False):
    """
    Train PPO / Safe PPO using random-episode training.

    No walk-forward windows. Validation checkpointing based on Sharpe.

    Returns (model, vecnorm) or (None, None).
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    label        = "SAFE PPO" if safe_reward else "PPO"
    model_name   = "safe_ppo_portfolio" if safe_reward else "ppo_portfolio"
    model_path   = os.path.join(MODEL_DIR, model_name)
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnorm.pkl")
    curve_path   = os.path.join(RESULTS_DIR,
                                "safe_ppo_train_curve.png" if safe_reward
                                else "ppo_train_curve.png")

    log("=" * 60)
    log(f"  {label} — RANDOM-EPISODE TRAINING  (v5)")
    log("=" * 60)

    full_df = _load(TRAIN_DATA_PATH)
    log(f"Training data: {full_df.shape}  "
        f"({full_df.index.min().date()} → {full_df.index.max().date()})")
    log(f"OBS_DIM={OBS_DIM}  total_timesteps={TOTAL_TIMESTEPS:,}  "
        f"ent_coef={ENT_COEF}")

    # Training data = everything except validation window
    train_df = full_df[
        (full_df.index < VAL_START) | (full_df.index > VAL_END)
    ].copy()
    val_df   = full_df.loc[VAL_START:VAL_END].copy()

    log(f"Train rows : {len(train_df)}  "
        f"Val rows   : {len(val_df)}  [{VAL_START} → {VAL_END}]")

    if len(train_df) < 300:
        log("ERROR: insufficient training data.")
        return None, None

    vec_env = _build_vecenv(train_df, safe=safe_reward, random_start=True)

    model = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        learning_rate = LEARNING_RATE,
        n_steps       = N_STEPS,
        batch_size    = BATCH_SIZE,
        n_epochs      = N_EPOCHS,
        gamma         = GAMMA,
        gae_lambda    = GAE_LAMBDA,
        clip_range    = CLIP_RANGE,
        ent_coef      = ENT_COEF,
        policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        device        = "auto",
        verbose       = 0,
    )

    reward_cb = RewardLogger()
    stats_cb  = EpisodeStatsCallback()
    val_cb    = ValidationCallback(
        val_df=val_df,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        safe=safe_reward,
        interval=VAL_INTERVAL,
    )

    log(f"\nTraining {label} for {TOTAL_TIMESTEPS:,} timesteps …")
    t0 = time.time()

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = CallbackList([reward_cb, stats_cb, val_cb]),
        progress_bar    = True,
    )

    elapsed = time.time() - t0
    log(f"\n{label} training complete in {elapsed/60:.1f} min.")
    log(f"Best validation Sharpe: {val_cb.best_sharpe:.3f}")

    # Load best checkpoint if it improved; else use final weights
    best_zip = model_path + "_best.zip"
    if os.path.isfile(best_zip):
        log(f"Loading best checkpoint from {best_zip}")
        model = PPO.load(model_path + "_best", env=vec_env)

    model.save(model_path)
    vec_env.save(vecnorm_path)
    log(f"Model saved        → {model_path}.zip")
    log(f"VecNormalize saved → {vecnorm_path}")

    if reward_cb.episode_rewards:
        _plot_curve(reward_cb.episode_rewards,
                    f"{label} Training — Episode Rewards", curve_path)

    return model, vec_env


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_curve(rewards, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.25, color="darkorange", label="Episode reward")
    if len(rewards) >= 10:
        w      = max(10, len(rewards) // 20)
        smooth = pd.Series(rewards).rolling(w, min_periods=1).mean()
        ax.plot(smooth, color="darkorange", linewidth=2,
                label=f"Rolling mean ({w} ep)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log(f"Training curve saved → {save_path}")


if __name__ == "__main__":
    train_ppo(safe_reward=False)