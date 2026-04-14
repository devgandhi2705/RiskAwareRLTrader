"""
train_ppo.py  (v3 — Regime-Aware)
-----------------------------------
Trains PPO (baseline) and Safe PPO agents with regime-aware environment.

Changes from v2
---------------
  R1 : regime features injected into DataFrame before environment creation
       via detect_market_regime() — all windows share the same enriched df.
  R7 : EpisodeStatsCallback now logs regime counts to TensorBoard.
  Backward-compatible: safe_reward=False → standard PPO reward.

Retained from v2
----------------
  S6  : norm_obs=True, norm_reward=False
  S12 : lr=1e-4, ent=0.02, clip=0.1
  S13 : walk-forward training (5 windows) + full fine-tune
  S17 : random_start=True during training

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys

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

from env.trading_environment import PortfolioTradingEnv
from data_pipeline.regime_detection import detect_market_regime, get_regime_stats

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")
MODEL_DIR       = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")

TOTAL_TIMESTEPS      = 500_000
LEARNING_RATE        = 1e-4
N_STEPS              = 2048
BATCH_SIZE           = 64
N_EPOCHS             = 10
GAMMA                = 0.99
GAE_LAMBDA           = 0.95
CLIP_RANGE           = 0.1
ENT_COEF             = 0.02
TIMESTEPS_PER_WINDOW = 150_000

WALK_FORWARD_WINDOWS = [
    ("2016-01-01", "2017-12-31", "2018-12-31"),
    ("2017-01-01", "2018-12-31", "2019-12-31"),
    ("2018-01-01", "2019-12-31", "2020-12-31"),
    ("2019-01-01", "2020-12-31", "2021-12-31"),
    ("2020-01-01", "2021-12-31", "2022-12-31"),
]


# ── Callbacks ─────────────────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Accumulates per-episode total rewards for plotting."""

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


class DebugRewardCallback(BaseCallback):
    """
    R7: Logs full reward component breakdown for the first N_DEBUG_EPISODES.
    Uses env.get_episode_debug_log() which includes all regime + metric info.
    Active for both PPO and Safe PPO when safe_reward=True.
    """
    N_DEBUG_EPISODES = 5

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_count = 0

    def _on_step(self):
        if self.locals["dones"][0] and self._ep_count < self.N_DEBUG_EPISODES:
            try:
                # Unwrap VecEnv → Monitor → PortfolioTradingEnv
                env = self.training_env.envs[0].env
                log = env.get_episode_debug_log()
                if log:
                    print(f"  [DEBUG ep {self._ep_count + 1}] {log}")
            except Exception:
                pass   # silently skip if env unwrapping fails
            self._ep_count += 1
        return True


class EpisodeStatsCallback(BaseCallback):
    """
    R7: Logs per-episode environment stats to TensorBoard.
    Tracks: avg_turnover, avg_drawdown, avg_volatility, final_portfolio_value,
            and per-regime step counts.
    """

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
                "port_value": info.get("portfolio_value", 0.0),
                "regime":     info.get("regime",          0),
            })

        if self.locals["dones"][0] and self._buf:
            avg_turn = float(np.mean([s["turnover"]   for s in self._buf]))
            avg_dd   = float(np.mean([s["drawdown"]   for s in self._buf]))
            avg_vol  = float(np.mean([s["volatility"] for s in self._buf]))
            fin_val  = self._buf[-1]["port_value"]
            bull_n   = sum(1 for s in self._buf if s["regime"] ==  1)
            bear_n   = sum(1 for s in self._buf if s["regime"] == -1)
            neut_n   = sum(1 for s in self._buf if s["regime"] ==  0)

            if self.logger:
                self.logger.record("env/avg_turnover",      avg_turn)
                self.logger.record("env/avg_drawdown",      avg_dd)
                self.logger.record("env/avg_volatility",    avg_vol)
                self.logger.record("env/final_portfolio",   fin_val)
                self.logger.record("env/regime_bull_steps", bull_n)
                self.logger.record("env/regime_bear_steps", bear_n)
                self.logger.record("env/regime_neut_steps", neut_n)
            self._buf = []
        return True


# ── Environment factory ───────────────────────────────────────────────────────

def _make_env_fn(df: pd.DataFrame, safe: bool, random_start: bool = True):
    """Return a thunk that creates a monitored PortfolioTradingEnv."""
    def _fn():
        env = PortfolioTradingEnv(df, safe_reward=safe, random_start=random_start)
        return Monitor(env)
    return _fn


def _build_vecenv(
    df: pd.DataFrame,
    safe: bool,
    random_start: bool = True,
) -> VecNormalize:
    """Wrap df in DummyVecEnv + VecNormalize (obs normalised, reward not)."""
    vec = DummyVecEnv([_make_env_fn(df, safe, random_start)])
    vec = VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=GAMMA,
    )
    return vec


def _evaluate_on_window(
    model: PPO,
    vecnorm: VecNormalize,
    df_val: pd.DataFrame,
) -> float:
    """
    Run one deterministic episode on a validation slice.

    Returns
    -------
    Total net return of the episode as a fraction of initial value.
    """
    env_val = PortfolioTradingEnv(df_val, safe_reward=False, random_start=False)
    obs, _  = env_val.reset()
    done    = False
    while not done:
        obs_n = vecnorm.normalize_obs(obs.reshape(1, -1))[0]
        action, _ = model.predict(obs_n, deterministic=True)
        obs, _, done, trunc, _ = env_val.step(action)
        done = done or trunc
    return (env_val.portfolio_value - env_val.initial_value) / env_val.initial_value


# ── Training ──────────────────────────────────────────────────────────────────

def train_ppo(safe_reward: bool = False):
    """
    Train PPO / Safe PPO with walk-forward validation + full fine-tune.

    Regime features are attached to the DataFrame once before any environment
    is created, so every window and the final fine-tune share the same enriched
    multi-asset regime columns.

    Parameters
    ----------
    safe_reward : If True, trains Safe PPO (CVaR / tail penalties active).
                  If False, trains standard PPO baseline.

    Returns
    -------
    (best_model, full_vecnorm)  or  (None, None) if no window had data.
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    label        = "SAFE PPO" if safe_reward else "PPO BASELINE"
    model_name   = "safe_ppo_portfolio" if safe_reward else "ppo_portfolio"
    curve_name   = "safe_ppo_train_curve.png" if safe_reward else "ppo_train_curve.png"
    model_path   = os.path.join(MODEL_DIR, model_name)
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnorm.pkl")

    print("=" * 60)
    print(f"  {label} — WALK-FORWARD TRAINING (Regime-Aware v3)")
    print("=" * 60)

    # R1: Load CSV and attach regime features ONCE to the full dataset
    raw_df  = pd.read_csv(TRAIN_DATA_PATH, index_col="Date", parse_dates=True)
    full_df = detect_market_regime(raw_df)

    stats = get_regime_stats(full_df)
    print(
        f"Training data: {full_df.shape}  "
        f"({full_df.index.min().date()} -> {full_df.index.max().date()})"
    )
    print(
        f"Regime distribution — "
        f"Bull: {stats.get('bull_pct', 0):.1f}%  "
        f"Neutral: {stats.get('neutral_pct', 0):.1f}%  "
        f"Bear: {stats.get('bear_pct', 0):.1f}%"
    )

    best_val_return = -np.inf
    best_model      = None
    best_vecnorm    = None
    all_rewards     = []

    # ── Walk-forward loop (S13) ───────────────────────────────────────────────
    for i, (tr_start, tr_end, val_end) in enumerate(WALK_FORWARD_WINDOWS):
        print(
            f"\n  Window {i+1}/{len(WALK_FORWARD_WINDOWS)} | "
            f"train: {tr_start} → {tr_end}  |  val: {tr_end} → {val_end}"
        )

        df_train = full_df.loc[tr_start:tr_end].copy()
        df_val   = full_df.loc[tr_end:val_end].copy()

        if len(df_train) < 100 or len(df_val) < 20:
            print("    Skipping — insufficient data for this window.")
            continue

        # R7: per-window regime distribution
        ws = get_regime_stats(df_train)
        print(
            f"    Window regimes — "
            f"Bull: {ws.get('bull_pct', 0):.0f}%  "
            f"Neutral: {ws.get('neutral_pct', 0):.0f}%  "
            f"Bear: {ws.get('bear_pct', 0):.0f}%"
        )

        vec_env = _build_vecenv(df_train, safe=safe_reward, random_start=True)

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
            device        = "cpu",
            verbose       = 0,
        )

        reward_cb = RewardLogger()
        stats_cb  = EpisodeStatsCallback()
        cb_list   = [reward_cb, stats_cb]
        if safe_reward:
            cb_list.append(DebugRewardCallback())
            print("  [Debug ON: reward components logged for first 5 episodes]")

        model.learn(
            total_timesteps = TIMESTEPS_PER_WINDOW,
            callback        = CallbackList(cb_list),
            progress_bar    = True,
        )
        all_rewards.extend(reward_cb.episode_rewards)

        # Evaluate on validation window
        vec_env.training    = False
        vec_env.norm_reward = False
        val_ret = _evaluate_on_window(model, vec_env, df_val)
        print(f"    Validation return: {val_ret * 100:+.2f}%")

        if val_ret > best_val_return:
            best_val_return = val_ret
            best_model      = model
            best_vecnorm    = vec_env
            print("    → New best model saved.")

    if best_model is None:
        print("No model trained successfully — all windows had insufficient data.")
        return None, None

    print(f"\n  Best validation return across all windows: {best_val_return * 100:+.2f}%")

    # ── Fine-tune best model on the full training dataset ─────────────────────
    print("\n  Fine-tuning best model on full training data …")
    full_vec = _build_vecenv(full_df, safe=safe_reward, random_start=True)
    best_model.set_env(full_vec)

    fine_reward_cb = RewardLogger()
    fine_stats_cb  = EpisodeStatsCallback()

    remaining_steps = TOTAL_TIMESTEPS - (TIMESTEPS_PER_WINDOW * len(WALK_FORWARD_WINDOWS))
    best_model.learn(
        total_timesteps   = max(remaining_steps, 10_000),
        callback          = CallbackList([fine_reward_cb, fine_stats_cb]),
        progress_bar      = True,
        reset_num_timesteps = False,
    )
    all_rewards.extend(fine_reward_cb.episode_rewards)

    # ── Persist model and normaliser ──────────────────────────────────────────
    best_model.save(model_path)
    full_vec.save(vecnorm_path)
    print(f"Model saved          → {model_path}.zip")
    print(f"VecNormalize saved   → {vecnorm_path}")

    # ── Training curve plot ────────────────────────────────────────────────────
    if all_rewards:
        _plot_curve(
            all_rewards,
            f"{label} Training — Episode Rewards",
            os.path.join(RESULTS_DIR, curve_name),
        )

    print(f"\n{label} training complete.\n")
    return best_model, full_vec


# ── Plotting helper ────────────────────────────────────────────────────────────

def _plot_curve(rewards: list, title: str, save_path: str) -> None:
    """Save a smoothed episode-reward training curve."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.25, color="darkorange", label="Episode reward")
    if len(rewards) >= 10:
        window = max(10, len(rewards) // 20)
        smooth = pd.Series(rewards).rolling(window, min_periods=1).mean()
        ax.plot(
            smooth,
            color="darkorange",
            linewidth=2,
            label=f"Rolling mean ({window} ep)",
        )
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
    train_ppo(safe_reward=False)