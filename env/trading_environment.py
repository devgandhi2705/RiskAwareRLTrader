"""
trading_environment.py  (v3 — Regime-Aware)
--------------------------------------------
Regime-Aware Safe RL environment for multi-asset portfolio management.

New in v3
---------
  R1 : market_regime, trend_strength, market_volatility added to state space
  R2 : observation_space updated to reflect 3 extra features
  R3 : regime-conditional risk_multiplier scales safe penalties
  R4 : bull-market exposure bonus in safe reward
  R5 : CVaR tail-risk penalty using rolling 50-step deque
  R7 : regime debug logging via debug_reward flag
  Backward-compatible: safe_reward=False reverts to plain PPO reward

Previous features retained
--------------------------
  S1  : clip+renorm weight constraints
  S2  : transaction cost on turnover
  S3  : turnover penalty + gradual penalty warm-up
  S4  : max position size cap (40%)
  S7  : drawdown + CVaR penalties with max() guards
  S8  : volatility penalty
  S10 : equal-weight init
  S11 : rebalance frequency gate
  S16 : action smoothing
  S17 : randomised episode start

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque


# ── Asset / Feature Configuration ────────────────────────────────────────────
ASSETS   = ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty50", "Sensex"]
N_ASSETS = len(ASSETS)

BASE_FEATURES  = ["Close", "Return", "RSI"]
EXTRA_FEATURES = ["MA20", "Volatility"]   # only BTC, ETH in this dataset

N_PORTFOLIO_FEATURES = N_ASSETS + 1       # weights + normalised value
N_REGIME_FEATURES    = 3                  # R1: regime, trend_strength, market_vol

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INITIAL_VALUE      = 100_000.0
DEFAULT_TRANSACTION_COST   = 0.001
DEFAULT_TURNOVER_LAMBDA    = 0.001
DEFAULT_MAX_WEIGHT         = 0.40
DEFAULT_CVAR_WINDOW        = 30
DEFAULT_CVAR_ALPHA         = 0.05
DEFAULT_MAX_DD_THRESHOLD   = 0.20
DEFAULT_CVAR_THRESHOLD     = -0.05
DEFAULT_LAMBDA_DD          = 0.5
DEFAULT_LAMBDA_CVAR        = 0.5
DEFAULT_LAMBDA_VOL         = 0.1
DEFAULT_REBALANCE_FREQ     = 5
DEFAULT_ACTION_SMOOTH      = 0.3
DEFAULT_RANDOM_START       = True
DEFAULT_RISK_PENALTY_SCALE = 0.1
MIN_EPISODE_STEPS          = 252

# R3: regime → risk multiplier mapping
REGIME_RISK_MULTIPLIER = {
     1: 0.1,   # Bull  — nearly no penalties, let agent be aggressive
     0: 0.5,   # Neutral — moderate constraints
    -1: 1.0,   # Bear  — full penalties, strong risk control
}

# R4: bull exposure bonus weight
BULL_EXPOSURE_BONUS = 0.05

# R5: tail-risk CVaR window
TAIL_RISK_WINDOW = 50


class PortfolioTradingEnv(gym.Env):
    """
    Regime-Aware Portfolio Management MDP.

    Observation layout
    ------------------
    [ market_features (N_ASSETS * 5)
    | portfolio_weights (N_ASSETS)
    | norm_portfolio_value (1)
    | market_regime (1)           <- R1
    | trend_strength (1)          <- R1
    | market_volatility (1) ]     <- R1

    Total dims = N_ASSETS*5 + N_ASSETS + 1 + 3

    Reward (safe=False)
    -------------------
    net_return - turnover_penalty

    Reward (safe=True)
    ------------------
    net_return
      + exposure_bonus            (R4 — bull only)
      - turnover_penalty
      - risk_mult * dd_penalty    (R3 — regime-scaled)
      - risk_mult * cvar_penalty  (R3)
      - risk_mult * vol_penalty   (R3)
      - tail_cvar_penalty         (R5)
    All scaled by gradual warm-up factor (over 200k steps).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        safe_reward=False,
        initial_value=DEFAULT_INITIAL_VALUE,
        transaction_cost=DEFAULT_TRANSACTION_COST,
        turnover_lambda=DEFAULT_TURNOVER_LAMBDA,
        max_weight=DEFAULT_MAX_WEIGHT,
        cvar_window=DEFAULT_CVAR_WINDOW,
        cvar_alpha=DEFAULT_CVAR_ALPHA,
        dd_threshold=DEFAULT_MAX_DD_THRESHOLD,
        cvar_threshold=DEFAULT_CVAR_THRESHOLD,
        lambda_dd=DEFAULT_LAMBDA_DD,
        lambda_cvar=DEFAULT_LAMBDA_CVAR,
        lambda_vol=DEFAULT_LAMBDA_VOL,
        rebalance_freq=DEFAULT_REBALANCE_FREQ,
        action_smooth=DEFAULT_ACTION_SMOOTH,
        random_start=DEFAULT_RANDOM_START,
        risk_penalty_scale=DEFAULT_RISK_PENALTY_SCALE,
        debug_reward=False,
    ):
        super().__init__()

        self.df                 = df.copy().reset_index(drop=True)
        self.safe_reward        = safe_reward
        self.initial_value      = initial_value
        self.transaction_cost   = transaction_cost
        self.turnover_lambda    = turnover_lambda
        self.max_weight         = max_weight
        self.cvar_window        = cvar_window
        self.cvar_alpha         = cvar_alpha
        self.dd_threshold       = dd_threshold
        self.cvar_threshold     = cvar_threshold
        self.lambda_dd          = lambda_dd
        self.lambda_cvar        = lambda_cvar
        self.lambda_vol         = lambda_vol
        self.rebalance_freq     = rebalance_freq
        self.action_smooth      = action_smooth
        self.random_start       = random_start
        self.risk_penalty_scale = risk_penalty_scale
        self.debug_reward       = debug_reward

        # ── Feature columns ───────────────────────────────────────────────────
        self.obs_feature_cols          = []
        self.obs_feature_dim_per_asset = len(BASE_FEATURES) + len(EXTRA_FEATURES)

        for asset in ASSETS:
            for feat in BASE_FEATURES + EXTRA_FEATURES:
                self.obs_feature_cols.append(f"{asset}_{feat}")

        self.return_cols = [f"{asset}_Return" for asset in ASSETS]

        # R1: verify regime columns present (added by detect_market_regime)
        self._has_regime = all(
            c in self.df.columns
            for c in ["regime", "trend_strength", "market_volatility"]
        )

        # R2: observation space = market + portfolio + 3 regime features
        n_market   = N_ASSETS * self.obs_feature_dim_per_asset
        n_obs      = n_market + N_PORTFOLIO_FEATURES + N_REGIME_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_ASSETS,), dtype=np.float32
        )

        self._compute_norm_stats()

        # Episode state
        self.current_step    = 0
        self.episode_start   = 0
        self.portfolio_value = initial_value
        self.peak_value      = initial_value
        self.weights         = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
        self.return_history  = []
        self.tail_risk_buf   = deque(maxlen=TAIL_RISK_WINDOW)  # R5
        self.total_rows      = len(self.df)
        self._total_steps    = 0

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _compute_norm_stats(self):
        all_cols = self.obs_feature_cols + ["trend_strength", "market_volatility"]
        self._col_mean = {}
        self._col_std  = {}
        for col in all_cols:
            if col in self.df.columns:
                vals = self.df[col].dropna()
                self._col_mean[col] = float(vals.mean())
                std = float(vals.std())
                self._col_std[col]  = std if std > 1e-8 else 1.0
            else:
                self._col_mean[col] = 0.0
                self._col_std[col]  = 1.0

    def _normalise(self, col, value):
        return (value - self._col_mean[col]) / self._col_std[col]

    # ── Weight processing ─────────────────────────────────────────────────────

    def _process_weights(self, raw_action, prev_weights, apply_smoothing=True):
        w = np.clip(raw_action, 0.0, 1.0).astype(np.float64)
        s = w.sum()
        w = w / s if s > 1e-8 else np.ones(N_ASSETS) / N_ASSETS

        w = np.minimum(w, self.max_weight)
        s = w.sum()
        w = w / s if s > 1e-8 else np.ones(N_ASSETS) / N_ASSETS

        if apply_smoothing:
            w = (1.0 - self.action_smooth) * prev_weights.astype(np.float64) + self.action_smooth * w
            w = w / w.sum()

        return w.astype(np.float32)

    # ── Observation builder (R1 — adds regime features) ──────────────────────

    def _get_obs(self, step):
        row = self.df.iloc[step]
        market_obs = []

        for col in self.obs_feature_cols:
            if col in self.df.columns:
                raw = row[col]
                val = (0.0 if (pd.isna(raw) or not np.isfinite(raw))
                       else self._normalise(col, float(raw)))
            else:
                val = 0.0
            market_obs.append(val)

        portfolio_obs = list(self.weights) + [self.portfolio_value / self.initial_value]

        # R1: regime features — regime already in {-1,0,1}, trend & vol z-scored
        if self._has_regime:
            regime_raw  = float(row.get("regime", 0))
            trend_raw   = float(row.get("trend_strength", 0.0))
            vol_raw     = float(row.get("market_volatility", 0.0))
            trend_norm  = self._normalise("trend_strength", trend_raw)
            vol_norm    = self._normalise("market_volatility", vol_raw)
            regime_obs  = [regime_raw, trend_norm, vol_norm]
        else:
            regime_obs  = [0.0, 0.0, 0.0]

        obs = np.array(market_obs + portfolio_obs + regime_obs, dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    # ── Risk metrics ──────────────────────────────────────────────────────────

    def _compute_drawdown(self):
        if self.peak_value <= 0:
            return 0.0
        return float((self.peak_value - self.portfolio_value) / self.peak_value)

    def _compute_cvar(self):
        if len(self.return_history) < max(self.cvar_window, 5):
            return 0.0
        recent = np.array(self.return_history[-self.cvar_window:])
        var    = np.percentile(recent, self.cvar_alpha * 100)
        tail   = recent[recent <= var]
        return float(tail.mean()) if len(tail) > 0 else float(var)

    def _compute_portfolio_vol(self):
        if len(self.return_history) < 5:
            return 0.0
        return float(np.array(self.return_history[-self.cvar_window:]).std())

    def _compute_tail_cvar(self):
        """R5: CVaR over rolling 50-step return buffer (10th percentile tail)."""
        if len(self.tail_risk_buf) < 10:
            return 0.0
        arr  = np.array(self.tail_risk_buf)
        n_tail = max(1, int(0.10 * len(arr)))
        worst  = np.sort(arr)[:n_tail]
        return float(np.abs(worst.mean()))

    def _get_current_regime(self, step):
        """Return integer regime {-1, 0, 1} at the current step."""
        if self._has_regime:
            val = self.df.iloc[step].get("regime", 0)
            return int(val) if not pd.isna(val) else 0
        return 0

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = max(0, self.total_rows - MIN_EPISODE_STEPS - 1)
        if self.random_start and max_start > 0:
            self.episode_start = int(self.np_random.integers(0, max_start))
        else:
            self.episode_start = 0

        self.current_step    = self.episode_start
        self.portfolio_value = self.initial_value
        self.peak_value      = self.initial_value
        self.weights         = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
        self.return_history  = []
        self.tail_risk_buf   = deque(maxlen=TAIL_RISK_WINDOW)

        if not hasattr(self, "_total_steps"):
            self._total_steps = 0

        # R7: episode regime counter for debug logging
        self._ep_regime_counts = {-1: 0, 0: 0, 1: 0}

        return self._get_obs(self.current_step), {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        self._total_steps += 1

        # ── Rebalance gate ────────────────────────────────────────────────────
        steps_since_start = self.current_step - self.episode_start
        if steps_since_start % self.rebalance_freq == 0:
            new_weights = self._process_weights(action, self.weights, apply_smoothing=True)
        else:
            new_weights = self.weights.copy()

        # ── Transaction cost ──────────────────────────────────────────────────
        turnover = float(np.abs(new_weights - self.weights).sum())
        txn_cost = self.transaction_cost * turnover
        self.weights = new_weights

        # ── Asset returns ─────────────────────────────────────────────────────
        row = self.df.iloc[self.current_step]
        asset_returns = np.array(
            [float(row.get(col, 0.0) or 0.0) for col in self.return_cols],
            dtype=np.float64,
        )
        asset_returns    = np.where(np.isfinite(asset_returns), asset_returns, 0.0)
        portfolio_return = float(np.dot(self.weights, asset_returns))
        net_return       = portfolio_return - txn_cost

        # ── Portfolio update ──────────────────────────────────────────────────
        self.portfolio_value = max(self.portfolio_value * (1.0 + net_return), 1e-6)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        self.return_history.append(net_return)
        self.tail_risk_buf.append(net_return)

        # ── Risk metrics ──────────────────────────────────────────────────────
        drawdown    = self._compute_drawdown()
        cvar        = self._compute_cvar()
        port_vol    = self._compute_portfolio_vol()
        tail_cvar   = self._compute_tail_cvar()     # R5
        regime      = self._get_current_regime(self.current_step)

        # R7: track regime counts for debug
        if hasattr(self, "_ep_regime_counts"):
            self._ep_regime_counts[regime] = self._ep_regime_counts.get(regime, 0) + 1

        # ── Reward ────────────────────────────────────────────────────────────
        turn_pen = self.turnover_lambda * turnover
        reward   = net_return - turn_pen

        if self.safe_reward:
            # R3: regime-conditional risk multiplier
            risk_mult = REGIME_RISK_MULTIPLIER.get(regime, 0.5)

            # Penalties with max() guards
            dd_penalty   = self.lambda_dd   * max(0.0, drawdown - self.dd_threshold)
            cvar_penalty = self.lambda_cvar * max(0.0, -(cvar + self.cvar_threshold))
            vol_penalty  = self.lambda_vol  * port_vol

            # Gradual warm-up over 200k steps
            warmup_scale  = min(1.0, self._total_steps / 200_000)
            penalty_scale = warmup_scale * self.risk_penalty_scale * risk_mult

            # R5: tail CVaR penalty (not regime-gated — always small)
            tail_pen = 0.05 * tail_cvar

            # R4: exposure bonus during bull markets
            portfolio_exposure = float(np.sum(np.abs(self.weights)))
            if regime == 1:
                exposure_bonus = BULL_EXPOSURE_BONUS * portfolio_exposure
            else:
                exposure_bonus = 0.0

            risk_penalty = penalty_scale * (dd_penalty + cvar_penalty + vol_penalty)
            reward       = (net_return
                            + exposure_bonus
                            - turn_pen
                            - risk_penalty
                            - tail_pen)

            # R7: optional debug print
            if self.debug_reward:
                print(
                    f"[DBG] ret={net_return:+.5f}  "
                    f"regime={regime:+d}  risk_mult={risk_mult:.1f}  "
                    f"turn={turn_pen:.5f}  dd={dd_penalty:.5f}  "
                    f"cvar={cvar_penalty:.5f}  vol={vol_penalty:.5f}  "
                    f"tail={tail_pen:.5f}  exp_bonus={exposure_bonus:.5f}  "
                    f"scale={penalty_scale:.4f}"
                )

        # Scale *100 — no clipping (preserves gradient signal)
        reward = float(reward * 100.0)

        # ── Advance ───────────────────────────────────────────────────────────
        self.current_step += 1
        end_step = self.episode_start + MIN_EPISODE_STEPS
        done     = self.current_step >= min(end_step, self.total_rows - 1)

        next_step = min(self.current_step, self.total_rows - 1)
        next_obs  = self._get_obs(next_step)

        info = {
            "portfolio_value":  self.portfolio_value,
            "portfolio_return": portfolio_return,
            "net_return":       net_return,
            "drawdown":         drawdown,
            "cvar":             cvar,
            "tail_cvar":        tail_cvar,
            "volatility":       port_vol,
            "turnover":         turnover,
            "transaction_cost": txn_cost,
            "weights":          self.weights.tolist(),
            "regime":           regime,
        }

        return next_obs, reward, done, False, info

    def render(self):
        dd      = self._compute_drawdown()
        cvar    = self._compute_cvar()
        regime  = self._get_current_regime(max(0, self.current_step - 1))
        labels  = {1: "BULL", 0: "NEUT", -1: "BEAR"}
        print(
            f"Step {self.current_step:4d} | "
            f"Value: ${self.portfolio_value:>12,.2f} | "
            f"DD: {dd*100:5.2f}% | CVaR: {cvar*100:6.2f}% | "
            f"Regime: {labels.get(regime,'?')} | "
            f"W: {[f'{w:.2f}' for w in self.weights]}"
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def make_env(data_path, safe_reward=False, **kwargs):
    """Load CSV, attach regime features, return configured env."""
    from data_pipeline.regime_detection import detect_market_regime
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    df = detect_market_regime(df)
    return PortfolioTradingEnv(df, safe_reward=safe_reward, **kwargs)