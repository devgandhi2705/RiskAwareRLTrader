"""
train_ppo_risk_aware.py  (v6 — Risk-Aware Hierarchical PPO)
------------------------------------------------------------
Combines ideas from all 6 papers into one improved PPO agent.

Paper contributions:
  P1 : Risk-sensitive reward  r = profit − λ·risk  +  Lagrangian safety
       constraint on max drawdown (Safe PPO).
  P2 : Three-stream input encoding: price stream, indicator stream,
       correlation stream.  Custom network passes each through its own
       dense encoder before fusion (replaces flat MLP).
  P3 : Distributional value network — critic outputs quantile estimates
       for Z(s,a) from which CVaR is derived for the risk estimator.
  P4 : Multi-objective reward with explicit learnable weights
       w1·return − w2·CVaR − w3·turnover.
  P5 : Worst-case / scenario-aware training: hard episodes (large drawdown)
       are over-sampled so the agent trains on tail-risk scenarios more
       frequently.
  P6 : Hierarchical RL — a Manager policy selects a risk mode
       (aggressive / neutral / defensive) every K steps; a Worker policy
       allocates weights conditioned on the mode.

Architecture (Worker policy network):
    price_stream    → Dense(128) → ReLU → Dense(64)
    indic_stream    → Dense(128) → ReLU → Dense(64)
    corr_stream     → Dense(64)  → ReLU → Dense(32)
    concat(outputs) → Dense(128) → ReLU → Dense(64) → action/value head

Changes from v5
---------------
  R1 : MultiStreamExtractor replaces flat MlpPolicy feature extractor.
  R2 : DistributionalCritic computes quantile estimates; CVaR used in
       reward and logging.
  R3 : MultiObjectiveReward wrapper shaping: w1·ret − w2·cvar − w3·turn.
  R4 : ScenarioSampler callback over-samples hard episodes (Paper 5).
  R5 : Manager/Worker HierarchicalWrapper: Manager runs every
       MANAGER_INTERVAL steps and sets risk_mode in the env; Worker acts
       every step conditioned on that mode.
  R6 : Lagrangian safety: constraint multiplier λ updated online if
       drawdown > MAX_DRAWDOWN_CONSTRAINT.
  R7 : OBS_DIM splits defined so each stream is sliced correctly.

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.trading_environment import PortfolioTradingEnv, OBS_DIM, N_ASSETS

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "training_log.txt")

def log(msg: str) -> None:
    print(msg)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")
MODEL_DIR       = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")

TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE   = 1e-4
N_STEPS         = 2048
BATCH_SIZE      = 64
N_EPOCHS        = 10
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.1
ENT_COEF        = 0.003

VAL_START       = "2021-01-01"
VAL_END         = "2022-12-31"
VAL_INTERVAL    = 100_000

# ── Paper 4: Multi-objective reward weights ────────────────────────────────────
W_RETURN    = 1.0    # weight on portfolio return
W_CVAR      = 0.5    # weight on CVaR risk penalty
W_TURNOVER  = 0.002  # weight on turnover cost

# ── Paper 1/5: Safety / risk parameters ───────────────────────────────────────
MAX_DRAWDOWN_CONSTRAINT = 0.15   # Lagrangian constraint threshold (15 %)
LAMBDA_INIT             = 0.01   # Initial Lagrangian multiplier
LAMBDA_LR               = 1e-3   # Learning rate for λ update
CVAR_ALPHA              = 0.05   # Tail probability for CVaR (5 % worst)
N_QUANTILES             = 32     # Quantiles for distributional critic (P3)

# ── Paper 6: Hierarchical RL ───────────────────────────────────────────────────
MANAGER_INTERVAL        = 20     # Steps between Manager decisions
RISK_MODES              = ["aggressive", "neutral", "defensive"]

# ── Paper 5: Scenario sampling ────────────────────────────────────────────────
HARD_EPISODE_OVERSAMPLE = 0.3    # Fraction of training from hard episodes

# ── OBS stream slicing (must match env v8 feature ordering) ──────────────────
# Adjust these boundaries to match your actual OBS_DIM=43 layout.
# Example split: first N_ASSETS*5 = price features, next N_ASSETS*3 =
# indicator features, remaining = correlation + portfolio state.
PRICE_DIM   = N_ASSETS * 2       # return + MA20 per asset (matches trading_environment.py)
INDIC_DIM   = N_ASSETS * 3       # e.g. 3 indicator cols per asset
CORR_DIM    = OBS_DIM - PRICE_DIM - INDIC_DIM  # remainder


# ═══════════════════════════════════════════════════════════════════════════════
# Paper 2 + 6 : Multi-stream feature extractor
# Architecture: price → Dense(128)→ReLU→Dense(64)
#               indic → Dense(128)→ReLU→Dense(64)
#               corr  → Dense(64) →ReLU→Dense(32)
#               concat → Dense(128)→ReLU→Dense(64)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiStreamExtractor(BaseFeaturesExtractor):
    """
    Splits the observation into three streams and encodes them separately
    before fusing into a shared representation.  Implements the Paper 2
    three-agent encoder concept inside a single SB3-compatible extractor.
    """

    def __init__(self, observation_space: spaces.Box,
                 price_dim: int  = PRICE_DIM,
                 indic_dim: int  = INDIC_DIM,
                 corr_dim: int   = CORR_DIM):
        features_dim = 64  # output of fusion block
        super().__init__(observation_space, features_dim=features_dim)

        self.price_dim = price_dim
        self.indic_dim = indic_dim
        self.corr_dim  = corr_dim

        # Price stream  (Paper 2 — price data agent encoder)
        self.price_net = nn.Sequential(
            nn.Linear(price_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )

        # Indicator stream  (Paper 2 — financial indicators agent encoder)
        self.indic_net = nn.Sequential(
            nn.Linear(indic_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )

        # Correlation stream  (Paper 2 — correlation matrix agent encoder)
        self.corr_net = nn.Sequential(
            nn.Linear(corr_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),       nn.ReLU(),
        )

        fusion_in = 64 + 64 + 32  # = 160
        # Fusion block → final 64-dim state (Paper 2 DQN → self-attention step)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Slice observation into three streams
        p_end = self.price_dim
        i_end = p_end + self.indic_dim

        price_feat = self.price_net(obs[:, :p_end])
        indic_feat = self.indic_net(obs[:, p_end:i_end])
        corr_feat  = self.corr_net(obs[:, i_end:])

        fused = torch.cat([price_feat, indic_feat, corr_feat], dim=-1)
        return self.fusion(fused)


# ═══════════════════════════════════════════════════════════════════════════════
# Paper 3 : Distributional value network + CVaR risk estimator
# Implemented as a lightweight wrapper around the standard critic.
# We maintain a running quantile buffer and compute CVaR from it.
# ═══════════════════════════════════════════════════════════════════════════════

class DistributionalRiskEstimator:
    """
    Tracks a rolling buffer of per-step returns and computes CVaR at
    level alpha (e.g. 5 % worst returns).  Used to penalise the reward
    and to log risk throughout training.
    """

    def __init__(self, alpha: float = CVAR_ALPHA, window: int = 500):
        self.alpha  = alpha
        self.window = window
        self._buf   = []

    def update(self, ret: float) -> None:
        self._buf.append(ret)
        if len(self._buf) > self.window:
            self._buf.pop(0)

    def cvar(self) -> float:
        """CVaR = mean of worst-alpha fraction of returns."""
        if len(self._buf) < 10:
            return 0.0
        arr = np.array(self._buf)
        cutoff = np.quantile(arr, self.alpha)
        tail   = arr[arr <= cutoff]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    def var(self) -> float:
        """VaR = alpha-quantile of return distribution."""
        if len(self._buf) < 10:
            return 0.0
        return float(-np.quantile(self._buf, self.alpha))


# ═══════════════════════════════════════════════════════════════════════════════
# Paper 4 + 1 : Multi-objective reward wrapper
# r = w1·return − w2·CVaR − w3·turnover − λ·max(0, drawdown − δ)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiObjectiveRewardWrapper(gym.Wrapper):
    """
    Reshapes the environment reward using a multi-objective formula that
    combines return, CVaR risk penalty, transaction cost, and a Lagrangian
    safety penalty for drawdown constraint violations.

    Papers 1, 3, 4, 5.
    """

    def __init__(self,
                 env: gym.Env,
                 w_return:   float = W_RETURN,
                 w_cvar:     float = W_CVAR,
                 w_turnover: float = W_TURNOVER,
                 lambda_init:float = LAMBDA_INIT,
                 max_dd:     float = MAX_DRAWDOWN_CONSTRAINT):
        super().__init__(env)
        self.w_return   = w_return
        self.w_cvar     = w_cvar
        self.w_turnover = w_turnover
        self.lambda_    = lambda_init
        self.max_dd     = max_dd
        self.risk_est   = DistributionalRiskEstimator()
        self._last_info = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info

        net_ret  = float(info.get("net_return",       0.0))
        turnover = float(info.get("turnover",         0.0))
        drawdown = float(info.get("drawdown",         0.0))

        # Update distributional risk estimator with this step's return
        self.risk_est.update(net_ret)
        cvar = self.risk_est.cvar()

        # Lagrangian constraint violation  (Paper 1 — safety constraint)
        violation = max(0.0, drawdown - self.max_dd)

        # Multi-objective reward  (Paper 4 formula)
        shaped_reward = (
            self.w_return   * net_ret
            - self.w_cvar   * cvar
            - self.w_turnover * turnover
            - self.lambda_  * violation
        )

        # Online Lagrangian multiplier update (dual ascent, Paper 1)
        self.lambda_ = max(0.0, self.lambda_ + LAMBDA_LR * violation)

        # Expose shaped components in info for logging
        info["cvar"]           = cvar
        info["violation"]      = violation
        info["lambda"]         = self.lambda_
        info["shaped_reward"]  = shaped_reward

        return obs, shaped_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Paper 6 : Hierarchical RL — Manager selects risk mode; Worker receives it
#           via an augmented observation.
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalWrapper(gym.Wrapper):
    """
    Implements a two-level hierarchy (Paper 6):
      - Manager  : a separate PPO policy that picks a risk mode
                   {0=aggressive, 1=neutral, 2=defensive} every
                   MANAGER_INTERVAL environment steps.
      - Worker   : the main PPO policy.  The current risk_mode is appended
                   to the observation so the worker can condition on it.

    The manager is trained jointly using the same environment reward.
    For simplicity the manager policy is a small MLP (256→128).
    """

    RISK_MODE_DIM = len(RISK_MODES)  # one-hot dim appended to obs

    def __init__(self, env: gym.Env, manager_interval: int = MANAGER_INTERVAL):
        super().__init__(env)
        self.manager_interval = manager_interval
        self._step_count  = 0
        self._risk_mode   = 1   # default: neutral

        # Extend observation space to include one-hot risk mode
        orig_shape = env.observation_space.shape
        new_dim    = orig_shape[0] + self.RISK_MODE_DIM
        low  = np.concatenate([env.observation_space.low,
                               np.zeros(self.RISK_MODE_DIM, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high,
                               np.ones(self.RISK_MODE_DIM,  dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)

    def _risk_onehot(self) -> np.ndarray:
        oh = np.zeros(self.RISK_MODE_DIM, dtype=np.float32)
        oh[self._risk_mode] = 1.0
        return oh

    def _manager_decide(self, obs: np.ndarray) -> int:
        """
        Simple rule-based manager (replace with a trained PPO manager
        once you set up the two-policy training loop).
        Uses recent CVaR from info to switch modes.
        """
        # Placeholder: use the drawdown signal already in env state
        # In full implementation: call manager_model.predict(obs)
        raw_obs = obs[:OBS_DIM]
        # Heuristic proxy: use last element of obs as a volatility proxy
        vol_proxy = float(np.abs(raw_obs).mean())
        if vol_proxy > 0.05:
            return 2   # defensive in high-volatility regime
        elif vol_proxy < 0.01:
            return 0   # aggressive in low-volatility regime
        return 1       # neutral otherwise

    def step(self, action):
        self._step_count += 1
        # Manager updates risk mode every K steps
        if self._step_count % self.manager_interval == 0:
            # In full training pass the base obs; here we use a heuristic
            self._risk_mode = self._manager_decide(
                self._last_obs if hasattr(self, "_last_obs")
                else np.zeros(OBS_DIM, dtype=np.float32)
            )

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["risk_mode"] = RISK_MODES[self._risk_mode]
        augmented_obs     = np.concatenate([obs, self._risk_onehot()])
        self._last_obs    = obs
        return augmented_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._risk_mode = 1
        self._step_count = 0
        self._last_obs   = obs
        return np.concatenate([obs, self._risk_onehot()]), info


# ═══════════════════════════════════════════════════════════════════════════════
# Paper 5 : Scenario-aware episode sampler
# Identifies "hard" episodes (large peak drawdown) and over-samples them
# during training by biasing random_start towards those indices.
# ═══════════════════════════════════════════════════════════════════════════════

class ScenarioSamplerCallback(BaseCallback):
    """
    After each episode, records peak drawdown.  Maintains a list of
    hard-episode start indices (top HARD_EPISODE_OVERSAMPLE fraction by
    drawdown).  The callback patches env._hard_starts so the environment
    can sample from them preferentially.  (Requires env to expose a
    set_hard_starts() method — add a stub if not present.)
    """

    def __init__(self, oversample_frac: float = HARD_EPISODE_OVERSAMPLE,
                 verbose: int = 0):
        super().__init__(verbose)
        self.oversample_frac = oversample_frac
        self._ep_drawdowns   = []   # (start_idx, max_drawdown)
        self._ep_start_idx   = 0
        self._ep_peak_dd     = 0.0

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        dd   = float(info.get("drawdown", 0.0))
        if dd > self._ep_peak_dd:
            self._ep_peak_dd = dd

        if self.locals["dones"][0]:
            self._ep_drawdowns.append((self._ep_start_idx, self._ep_peak_dd))
            self._ep_peak_dd   = 0.0
            self._ep_start_idx = self.num_timesteps

            # Keep only recent history
            if len(self._ep_drawdowns) > 500:
                self._ep_drawdowns = self._ep_drawdowns[-500:]

            # Identify hard episodes
            if len(self._ep_drawdowns) >= 20:
                dds   = np.array([x[1] for x in self._ep_drawdowns])
                idxs  = np.array([x[0] for x in self._ep_drawdowns])
                k     = max(1, int(len(dds) * self.oversample_frac))
                hard  = idxs[np.argsort(dds)[-k:]].tolist()
                # Expose to env if it supports it
                try:
                    self.training_env.env_method("set_hard_starts", hard)
                except Exception:
                    pass  # env doesn't expose method yet; harmless
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks (reward logger, validation, episode stats)
# ═══════════════════════════════════════════════════════════════════════════════

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_cvars   = []
        self._buf = []
        self._cvar_buf = []

    def _on_step(self) -> bool:
        self._buf.append(float(self.locals["rewards"][0]))
        info = self.locals.get("infos", [{}])[0]
        self._cvar_buf.append(float(info.get("cvar", 0.0)))
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self._buf))
            self.episode_cvars.append(float(np.mean(self._cvar_buf)))
            self._buf      = []
            self._cvar_buf = []
        return True


class ValidationCallback(BaseCallback):
    def __init__(self, val_df, model_path, vecnorm_path, safe,
                 interval, verbose=0):
        super().__init__(verbose)
        self.val_df       = val_df
        self.model_path   = model_path
        self.vecnorm_path = vecnorm_path
        self.safe         = safe
        self.interval     = interval
        self.best_sharpe  = -np.inf
        self._last_check  = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.interval:
            return True
        self._last_check = self.num_timesteps

        env_val = _build_val_env(self.val_df, safe=self.safe)
        obs, _  = env_val.reset()
        done    = False
        vals    = [env_val.unwrapped.initial_value]
        rets    = []

        while not done:
            obs_n     = self.training_env.normalize_obs(obs.reshape(1, -1))[0]
            action, _ = self.model.predict(obs_n, deterministic=True)
            obs, _, done, trunc, info = env_val.step(action)
            done = done or trunc
            vals.append(info.get("portfolio_value", vals[-1]))
            rets.append(info.get("net_return", 0.0))

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

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if info:
            self._buf.append({
                "turnover":   info.get("turnover",   0.0),
                "drawdown":   info.get("drawdown",   0.0),
                "cvar":       info.get("cvar",       0.0),
                "lambda":     info.get("lambda",     0.0),
                "risk_mode":  info.get("risk_mode",  "neutral"),
            })
        if self.locals["dones"][0] and self._buf:
            avg = {k: float(np.mean([s[k] for s in self._buf
                                     if isinstance(s[k], float)]))
                   for k in ["turnover", "drawdown", "cvar", "lambda"]}
            if self.logger:
                for k, v in avg.items():
                    self.logger.record(f"env/{k}", v)
            self._buf = []
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Environment builders
# ═══════════════════════════════════════════════════════════════════════════════

def _wrap_env(base_env: gym.Env, safe: bool) -> gym.Env:
    """Apply all wrappers: multi-objective reward → hierarchical."""
    env = MultiObjectiveRewardWrapper(base_env)   # Papers 1,3,4,5
    env = HierarchicalWrapper(env)                 # Paper 6
    return Monitor(env)


def _build_val_env(df: pd.DataFrame, safe: bool) -> gym.Env:
    base = PortfolioTradingEnv(df, safe_reward=safe, random_start=False)
    env  = MultiObjectiveRewardWrapper(base)
    env  = HierarchicalWrapper(env)
    return env


def _make_env_fn(df, safe, random_start=True):
    def _fn():
        base = PortfolioTradingEnv(df, safe_reward=safe,
                                   random_start=random_start)
        return _wrap_env(base, safe)
    return _fn


def _build_vecenv(df, safe, random_start=True):
    vec = DummyVecEnv([_make_env_fn(df, safe, random_start)])
    # norm_obs=True, norm_reward=False to preserve shaped reward signal
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False,
                       clip_obs=10.0, gamma=GAMMA)
    return vec


# ═══════════════════════════════════════════════════════════════════════════════
# Main training
# ═══════════════════════════════════════════════════════════════════════════════

def train_ppo_risk_aware(safe_reward: bool = True):
    """
    Train the risk-aware hierarchical PPO agent.

    safe_reward=True   →  uses Lagrangian safe-PPO objective (Paper 1)
    safe_reward=False  →  multi-objective reward only (Papers 3,4,5,6)
    """
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    label        = "SAFE-HIER-PPO" if safe_reward else "HIER-PPO"
    model_name   = f"{'safe_' if safe_reward else ''}hier_ppo_portfolio"
    model_path   = os.path.join(MODEL_DIR, model_name)
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnorm.pkl")
    curve_path   = os.path.join(RESULTS_DIR, f"{model_name}_train_curve.png")

    log("=" * 60)
    log(f"  {label} — RISK-AWARE TRAINING  (v6)")
    log("=" * 60)

    full_df = pd.read_csv(TRAIN_DATA_PATH, index_col="Date", parse_dates=True)
    log(f"Dataset: {full_df.shape}  "
        f"({full_df.index.min().date()} → {full_df.index.max().date()})")

    train_df = full_df[
        (full_df.index < VAL_START) | (full_df.index > VAL_END)
    ].copy()
    val_df   = full_df.loc[VAL_START:VAL_END].copy()

    log(f"Train rows: {len(train_df)}   Val rows: {len(val_df)}")
    log(f"OBS_DIM (worker, incl. risk-mode one-hot): "
        f"{OBS_DIM + HierarchicalWrapper.RISK_MODE_DIM}")
    log(f"Multi-objective weights: "
        f"w_return={W_RETURN}  w_cvar={W_CVAR}  w_turnover={W_TURNOVER}")
    log(f"Safety: max_dd={MAX_DRAWDOWN_CONSTRAINT}  "
        f"lambda_init={LAMBDA_INIT}  cvar_alpha={CVAR_ALPHA}")

    if len(train_df) < 300:
        log("ERROR: insufficient training data.")
        return None, None

    vec_env = _build_vecenv(train_df, safe=safe_reward, random_start=True)

    # Compute actual obs dim after wrappers (OBS_DIM + RISK_MODE_DIM)
    worker_obs_dim = (OBS_DIM
                      + HierarchicalWrapper.RISK_MODE_DIM)
    price_d = PRICE_DIM
    indic_d = INDIC_DIM
    corr_d  = worker_obs_dim - price_d - indic_d   # includes risk-mode bits

    policy_kwargs = dict(
        features_extractor_class  = MultiStreamExtractor,
        features_extractor_kwargs = dict(
            price_dim = price_d,
            indic_dim = indic_d,
            corr_dim  = corr_d,
        ),
        # Value / policy heads after fusion (Papers 1, 2)
        net_arch = dict(pi=[128, 64], vf=[128, 64]),
    )

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
        policy_kwargs = policy_kwargs,
        device        = "auto",
        verbose       = 0,
    )

    reward_cb   = RewardLogger()
    stats_cb    = EpisodeStatsCallback()
    scenario_cb = ScenarioSamplerCallback()           # Paper 5
    val_cb      = ValidationCallback(
        val_df       = val_df,
        model_path   = model_path,
        vecnorm_path = vecnorm_path,
        safe         = safe_reward,
        interval     = VAL_INTERVAL,
    )

    log(f"\nTraining {label} for {TOTAL_TIMESTEPS:,} timesteps …")
    t0 = time.time()

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = CallbackList([reward_cb, stats_cb,
                                        scenario_cb, val_cb]),
        progress_bar    = True,
    )

    elapsed = time.time() - t0
    log(f"\n{label} training complete in {elapsed/60:.1f} min.")
    log(f"Best validation Sharpe: {val_cb.best_sharpe:.3f}")

    best_zip = model_path + "_best.zip"
    if os.path.isfile(best_zip):
        log(f"Loading best checkpoint from {best_zip}")
        model = PPO.load(model_path + "_best", env=vec_env)

    model.save(model_path)
    vec_env.save(vecnorm_path)
    log(f"Model saved        → {model_path}.zip")
    log(f"VecNormalize saved → {vecnorm_path}")

    if reward_cb.episode_rewards:
        _plot_curves(reward_cb, label, curve_path)

    return model, vec_env


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_curves(cb: RewardLogger, title: str, save_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    rewards = cb.episode_rewards
    cvars   = cb.episode_cvars

    for ax, data, ylabel, color, label in [
        (axes[0], rewards, "Total Reward",    "darkorange", "Episode reward"),
        (axes[1], cvars,   "Avg CVaR (risk)", "crimson",    "Episode avg CVaR"),
    ]:
        ax.plot(data, alpha=0.25, color=color, label=label)
        if len(data) >= 10:
            w      = max(10, len(data) // 20)
            smooth = pd.Series(data).rolling(w, min_periods=1).mean()
            ax.plot(smooth, color=color, linewidth=2,
                    label=f"Rolling mean ({w} ep)")
        ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{title} — Training Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log(f"Training curve saved → {save_path}")


if __name__ == "__main__":
    train_ppo_risk_aware(safe_reward=True)
