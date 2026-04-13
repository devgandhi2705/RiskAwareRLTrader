"""
trading_environment.py  (v4 — Regime-Aware Reward Scaling)
-----------------------------------------------------------
Regime-Aware Safe RL environment for multi-asset portfolio management.

New in v4
---------
  V4-1 : Regime-dependent return scaling
           Bull   → 1.5x return bonus, low penalties
           Neutral → 1.0x return, balanced penalties
           Bear   → 1.0x return, heavy penalties
  V4-2 : Trend-alignment reward component (+0.1 * alignment)
  V4-3 : Exposure incentive in bull regime (+0.1 * exposure)
  V4-4 : Volatility metric uses explicit 20-day window (spec §4)
  V4-5 : Extended debug log — regime counts + all reward components
  Backward-compatible: safe_reward=False uses same regime-reward
         but without risk penalties (plain PPO still benefits from
         regime scaling and trend alignment)

Retained from v3
----------------
  R1 : regime features in observation
  R2 : updated obs space
  R5 : tail CVaR deque
  S1-S17: all weight, cost, smoothing constraints unchanged
  Evaluation scripts untouched

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
N_REGIME_FEATURES    = 3                  # regime, trend_strength, market_vol

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
TAIL_RISK_WINDOW           = 50
TREND_WINDOW               = 20    # V4-2: rolling window for momentum signal

# ── V4-1: Regime reward parameters ───────────────────────────────────────────
# Return multipliers per regime
REGIME_RETURN_SCALE = {
     1:  1.5,   # Bull  — amplify gains
     0:  1.0,   # Neutral — no change
    -1:  1.0,   # Bear  — no amplification
}

# Volatility penalty weight per regime
REGIME_VOL_LAMBDA = {
     1: 0.2,    # Bull  — tolerate volatility
     0: 0.5,    # Neutral
    -1: 1.2,    # Bear  — punish volatility hard
}

# Drawdown penalty weight per regime
REGIME_DD_LAMBDA = {
     1: 0.1,    # Bull  — light drawdown control
     0: 0.5,    # Neutral
    -1: 1.5,    # Bear  — heavy protection
}

# Turnover penalty weight per regime (Bear discourages churning)
REGIME_TURN_LAMBDA = {
     1: 0.0,    # Bull  — free rebalancing
     0: 0.001,  # Neutral
    -1: 0.3,    # Bear  — penalise excessive trading
}

# V4-3: Exposure incentive (bull only)
EXPOSURE_BONUS_SCALE = 0.1   # +0.1 * sum(weights) in bull

# V4-2: Trend alignment bonus
TREND_ALIGN_SCALE    = 0.1   # +0.1 * alignment


class PortfolioTradingEnv(gym.Env):
    """
    Regime-Aware Portfolio Management MDP  (v4).

    Observation layout
    ------------------
    [ market_features (N_ASSETS * 5)
    | portfolio_weights (N_ASSETS)
    | norm_portfolio_value (1)
    | market_regime (1)
    | trend_strength (1)
    | market_volatility (1) ]

    Total dims = N_ASSETS*5 + N_ASSETS + 1 + 3

    Reward structure (both safe and baseline)
    -----------------------------------------
    All agents now use regime-dependent reward scaling.

    reward =
        return_scale[regime] * portfolio_return    # V4-1
      + exposure_bonus                             # V4-3 (bull only)
      + trend_alignment_bonus                      # V4-2
      - turn_lambda[regime] * turnover             # V4-1
      - dd_lambda[regime] * drawdown               # V4-1
      - vol_lambda[regime] * volatility            # V4-1

    When safe_reward=True, additional penalties are added:
      - cvar_penalty  (regime-gated warm-up scale)
      - tail_cvar_penalty

    All multiplied by *100 at the end (no clipping).
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

        # V4-2: pre-compute 20-day rolling mean returns for trend signal
        # sign(rolling_return_20d) per asset → {-1, 0, +1}
        self._trend_signals = self._precompute_trend_signals()

        # Regime columns check
        self._has_regime = all(
            c in self.df.columns
            for c in ["regime", "trend_strength", "market_volatility"]
        )

        # Observation space: market + portfolio + regime features
        n_market = N_ASSETS * self.obs_feature_dim_per_asset
        n_obs    = n_market + N_PORTFOLIO_FEATURES + N_REGIME_FEATURES
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
        self.tail_risk_buf   = deque(maxlen=TAIL_RISK_WINDOW)
        self.total_rows      = len(self.df)
        self._total_steps    = 0

    # ── V4-2: Trend signal precomputation ─────────────────────────────────────

    def _precompute_trend_signals(self):
        """
        Pre-compute sign(20-day rolling mean return) for every asset at every
        row.  Stored as a (total_rows, N_ASSETS) int8 array.

        sign > 0  → asset trending up
        sign < 0  → asset trending down
        sign = 0  → flat

        Using rolling mean (not sum) keeps the value in [-1, +1] regardless
        of window length, which makes the alignment score magnitude stable.
        Uses only past data (min_periods=1 ensures no NaN in early rows).
        """
        signals = np.zeros((len(self.df), N_ASSETS), dtype=np.float32)
        for j, col in enumerate(self.return_cols):
            if col in self.df.columns:
                roll = (
                    self.df[col]
                    .fillna(0.0)
                    .rolling(window=TREND_WINDOW, min_periods=1)
                    .mean()
                )
                signals[:, j] = np.sign(roll.values).astype(np.float32)
        return signals   # shape: (total_rows, N_ASSETS)

    def _get_trend_alignment(self, step):
        """
        V4-2: Compute portfolio–trend alignment at the given step.

        alignment = sum(weights * trend_sign)  in [-1, +1]

        Positive  → portfolio tilted toward trending assets (good)
        Negative  → portfolio fighting the trend (bad)
        """
        if step >= len(self._trend_signals):
            return 0.0
        trend = self._trend_signals[step]            # shape (N_ASSETS,)
        return float(np.dot(self.weights, trend))

    # ── V4-1: Regime reward computation ──────────────────────────────────────

    def _compute_regime_reward(
        self,
        regime,
        portfolio_return,
        drawdown,
        port_vol,
        turnover,
        trend_alignment,
    ):
        """
        Compute the full regime-scaled reward signal.

        Parameters
        ----------
        regime           : int  {-1, 0, 1}
        portfolio_return : float  gross return before transaction cost
        drawdown         : float  current drawdown from peak
        port_vol         : float  20-day rolling std of returns
        turnover         : float  sum(|delta_weights|)
        trend_alignment  : float  dot(weights, trend_sign)

        Returns
        -------
        reward           : float  (before *100 scaling)
        components       : dict   breakdown for debugging
        """
        # V4-1: regime-dependent coefficients
        ret_scale  = REGIME_RETURN_SCALE.get(regime, 1.0)
        vol_lam    = REGIME_VOL_LAMBDA.get(regime, 0.5)
        dd_lam     = REGIME_DD_LAMBDA.get(regime, 0.5)
        turn_lam   = REGIME_TURN_LAMBDA.get(regime, 0.001)

        # Core reward
        scaled_return = ret_scale * portfolio_return
        vol_pen       = vol_lam   * port_vol
        dd_pen        = dd_lam    * drawdown
        turn_pen      = turn_lam  * turnover

        # V4-3: exposure incentive in bull only
        exposure = float(np.sum(np.abs(self.weights)))  # = 1.0 for long-only
        if regime == 1:
            exp_bonus = EXPOSURE_BONUS_SCALE * exposure
        else:
            exp_bonus = 0.0

        # V4-2: trend alignment bonus (all regimes, but most meaningful in bull)
        align_bonus = TREND_ALIGN_SCALE * trend_alignment

        reward = (
            scaled_return
            + exp_bonus
            + align_bonus
            - vol_pen
            - dd_pen
            - turn_pen
        )

        components = {
            "scaled_return": scaled_return,
            "exp_bonus":     exp_bonus,
            "align_bonus":   align_bonus,
            "vol_pen":       vol_pen,
            "dd_pen":        dd_pen,
            "turn_pen":      turn_pen,
        }
        return reward, components

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

    # ── Observation builder ───────────────────────────────────────────────────

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
        """V4-4: Use explicit 20-day window as specified in §4."""
        if len(self.return_history) < 5:
            return 0.0
        window = min(TREND_WINDOW, len(self.return_history))
        return float(np.array(self.return_history[-window:]).std())

    def _compute_tail_cvar(self):
        if len(self.tail_risk_buf) < 10:
            return 0.0
        arr    = np.array(self.tail_risk_buf)
        n_tail = max(1, int(0.10 * len(arr)))
        worst  = np.sort(arr)[:n_tail]
        return float(np.abs(worst.mean()))

    def _get_current_regime(self, step):
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

        # V4-5: episode-level accumulators for debug logging
        self._ep_regime_counts = {-1: 0, 0: 0, 1: 0}
        self._ep_ret_buf   = []
        self._ep_dd_buf    = []
        self._ep_vol_buf   = []
        self._ep_turn_buf  = []

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

        # ── Transaction cost (always deducted from portfolio, regime-agnostic) ─
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
        net_return       = portfolio_return - txn_cost   # actual P&L after cost

        # ── Portfolio update ──────────────────────────────────────────────────
        self.portfolio_value = max(self.portfolio_value * (1.0 + net_return), 1e-6)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        self.return_history.append(net_return)
        self.tail_risk_buf.append(net_return)

        # ── Risk metrics (§4) ─────────────────────────────────────────────────
        drawdown  = self._compute_drawdown()
        cvar      = self._compute_cvar()
        port_vol  = self._compute_portfolio_vol()   # V4-4: 20-day window
        tail_cvar = self._compute_tail_cvar()
        regime    = self._get_current_regime(self.current_step)

        # V4-2: trend alignment
        trend_alignment = self._get_trend_alignment(self.current_step)

        # V4-5: track for debug
        if hasattr(self, "_ep_regime_counts"):
            self._ep_regime_counts[regime] = self._ep_regime_counts.get(regime, 0) + 1
        if hasattr(self, "_ep_ret_buf"):
            self._ep_ret_buf.append(net_return)
            self._ep_dd_buf.append(drawdown)
            self._ep_vol_buf.append(port_vol)
            self._ep_turn_buf.append(turnover)

        # ── Reward (V4-1 regime-dependent scaling) ────────────────────────────
        # Both safe and baseline agents use regime-aware reward.
        # safe_reward=True adds CVaR penalty on top.
        reward, comps = self._compute_regime_reward(
            regime          = regime,
            portfolio_return = portfolio_return,   # gross (before txn cost)
            drawdown        = drawdown,
            port_vol        = port_vol,
            turnover        = turnover,
            trend_alignment = trend_alignment,
        )

        if self.safe_reward:
            # Additional CVaR penalty with gradual warm-up
            cvar_penalty  = self.lambda_cvar * max(0.0, -(cvar + self.cvar_threshold))
            tail_pen      = 0.05 * tail_cvar

            # Warm-up: penalties ramp from 0 → full over 200k steps
            warmup_scale  = min(1.0, self._total_steps / 200_000)
            penalty_scale = warmup_scale * self.risk_penalty_scale

            reward -= penalty_scale * cvar_penalty
            reward -= tail_pen

            comps["cvar_penalty"] = penalty_scale * cvar_penalty
            comps["tail_pen"]     = tail_pen

        # V4-5: debug logging (per-step component print)
        if self.debug_reward:
            print(
                f"[DBG] regime={regime:+d}  "
                f"ret={portfolio_return:+.5f}  "
                f"scaled={comps['scaled_return']:+.5f}  "
                f"align={comps['align_bonus']:+.5f}  "
                f"exp={comps['exp_bonus']:+.5f}  "
                f"dd_pen={comps['dd_pen']:.5f}  "
                f"vol_pen={comps['vol_pen']:.5f}  "
                f"turn_pen={comps['turn_pen']:.5f}  "
                f"reward_raw={reward:+.5f}"
            )

        # Scale *100 — no clipping
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
            "trend_alignment":  trend_alignment,
        }

        return next_obs, reward, done, False, info

    def get_episode_debug_log(self):
        """
        V4-5: Return formatted debug string for the completed episode.
        Called by training callbacks after each episode.

        Example output
        --------------
        [DEBUG ep N] avg_ret=+0.00082 avg_dd=0.0031 avg_vol=0.00610
                     avg_turn=0.0124 regimes bull=180 neutral=40 bear=32
        """
        if not hasattr(self, "_ep_ret_buf") or not self._ep_ret_buf:
            return ""
        avg_ret  = float(np.mean(self._ep_ret_buf))
        avg_dd   = float(np.mean(self._ep_dd_buf))
        avg_vol  = float(np.mean(self._ep_vol_buf))
        avg_turn = float(np.mean(self._ep_turn_buf))
        bull  = self._ep_regime_counts.get( 1, 0)
        neut  = self._ep_regime_counts.get( 0, 0)
        bear  = self._ep_regime_counts.get(-1, 0)
        return (
            f"avg_ret={avg_ret:+.5f}  avg_dd={avg_dd:.5f}  "
            f"avg_vol={avg_vol:.5f}  avg_turn={avg_turn:.5f}  "
            f"regimes bull={bull} neutral={neut} bear={bear}"
        )

    def render(self):
        dd     = self._compute_drawdown()
        cvar   = self._compute_cvar()
        regime = self._get_current_regime(max(0, self.current_step - 1))
        labels = {1: "BULL", 0: "NEUT", -1: "BEAR"}
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