"""
trading_environment.py  (v5 — Multi-Asset Regime-Aware, Probability Rewards)
------------------------------------------------------------------------------
Regime-Aware Safe RL environment for multi-asset portfolio management.

New in v5
---------
  M1 : Extended observation vector — per-asset returns, trend_strength,
       volatility, and full regime probabilities (bull/neutral/bear per asset).
  M2 : Normalized base reward:  base = 100 × daily_net_return  → range ≈ −5..+5
  M3 : Momentum alignment bonus: 0.5 × Σ(w_i × trend_strength_i)
  M4 : Diversification bonus:    0.1 × entropy(weights)
  M5 : Regime-weighted risk penalty:  bear_prob_avg × (λ_dd × drawdown + λ_vol × vol)
  M6 : Safe PPO additional CVaR / tail CVaR penalties (regime-gated)
  M7 : Observation layout documented below.

Backward-compatible with
------------------------
  train_ppo.py, train_safe_ppo.py, train_dqn.py, evaluate_agent.py
  — none of those import obs-space constants directly; they rely on env.

Observation layout (v5)
-----------------------
  Section                    Dim
  ─────────────────────────────────────────
  Per-asset returns           N_ASSETS  (7)
  Per-asset trend_strength    N_ASSETS  (7)
  Per-asset volatility        N_ASSETS  (7)
  Per-asset bull_prob         N_ASSETS  (7)
  Per-asset neutral_prob      N_ASSETS  (7)
  Per-asset bear_prob         N_ASSETS  (7)
  Current portfolio weights   N_ASSETS  (7)
  Normalised portfolio value  1
  Global regime               1
  Global trend_strength       1
  Global market_volatility    1
  ─────────────────────────────────────────
  Total                       7×6 + 7 + 4 = 53

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque


# ── Asset / Feature Configuration ─────────────────────────────────────────────
ASSETS   = ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty50", "Sensex"]
N_ASSETS = len(ASSETS)

# Per-asset feature groups used in observation
_PER_ASSET_OBS_GROUPS = [
    "Return",
    "trend_strength",
    "volatility",
    "bull_prob",
    "neutral_prob",
    "bear_prob",
]
N_PER_ASSET_FEATURES = len(_PER_ASSET_OBS_GROUPS)   # 6

# Additional global / portfolio features
N_PORTFOLIO_FEATURES = N_ASSETS + 1   # weights (7) + norm_value (1)
N_GLOBAL_FEATURES    = 3              # regime, trend_strength, market_volatility

# Total obs dim:  7 * 6 + 8 + 3 = 53
OBS_DIM = N_ASSETS * N_PER_ASSET_FEATURES + N_PORTFOLIO_FEATURES + N_GLOBAL_FEATURES  # 53

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

# ── M2: Reward normalization ──────────────────────────────────────────────────
BASE_REWARD_SCALE   = 100.0   # multiplies daily_net_return  → range ≈ −5..+5
MOMENTUM_SCALE      = 0.5     # M3: Σ(w_i × trend_i) bonus weight
DIVERSIF_SCALE      = 0.1     # M4: entropy bonus weight

# ── Regime-dependent scales ───────────────────────────────────────────────────
REGIME_RETURN_SCALE = {1: 1.5, 0: 1.0, -1: 1.0}

# ── Safe PPO penalty lambdas ───────────────────────────────────────────────────
SAFE_LAMBDA_DD   = 0.5
SAFE_LAMBDA_VOL  = 0.1
SAFE_LAMBDA_CVAR = 0.5
SAFE_TAIL_SCALE  = 0.05


class PortfolioTradingEnv(gym.Env):
    """
    Multi-Asset Regime-Aware Portfolio Management MDP  (v5).

    Reward structure
    ----------------
    reward =
        BASE_REWARD_SCALE × net_return               # M2 — normalised base
      + MOMENTUM_SCALE × Σ(w_i × trend_i)           # M3 — momentum alignment
      + DIVERSIF_SCALE × entropy(weights)            # M4 — diversification
      − bear_prob_avg × (λ_dd × drawdown +           # M5 — regime-weighted risk
                         λ_vol × volatility)
      − turnover_lambda × turnover                   # turnover cost
      [ − safe CVaR penalties  if safe_reward=True ] # M6

    Expected range: approximately −5 to +5 per step.
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

        self.total_rows = len(self.df)

        # ── Return columns ─────────────────────────────────────────────────────
        self.return_cols = [f"{a}_Return" for a in ASSETS]

        # ── Per-asset feature column lookup (tolerant — fills 0 if absent) ────
        # Keys: group name → list of 7 column names in ASSETS order
        self._obs_cols = {}
        for group in _PER_ASSET_OBS_GROUPS:
            self._obs_cols[group] = [f"{a}_{group}" for a in ASSETS]

        # ── Global regime columns presence flag ────────────────────────────────
        self._has_global_regime = all(
            c in self.df.columns
            for c in ["regime", "trend_strength", "market_volatility"]
        )

        # ── Per-asset bear_prob columns (M5 penalty) ──────────────────────────
        self._bear_prob_cols = [f"{a}_bear_prob"      for a in ASSETS]
        self._trend_cols     = [f"{a}_trend_strength" for a in ASSETS]

        # ── Observation / action spaces ────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_ASSETS,), dtype=np.float32
        )

        # ── Initial state ──────────────────────────────────────────────────────
        self.portfolio_value = initial_value
        self.peak_value      = initial_value
        self.weights         = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
        self.return_history  = []
        self.tail_risk_buf   = deque(maxlen=TAIL_RISK_WINDOW)
        self.episode_start   = 0
        self.current_step    = 0
        self._total_steps    = 0

        # Episode-level debug accumulators (initialised here for safety)
        self._ep_regime_counts = {-1: 0, 0: 0, 1: 0}
        self._ep_ret_buf   = []
        self._ep_dd_buf    = []
        self._ep_vol_buf   = []
        self._ep_turn_buf  = []

    # ── Observation builder ────────────────────────────────────────────────────

    def _safe_row_values(self, row: pd.Series, cols: list) -> np.ndarray:
        """
        Extract named columns from a DataFrame row.
        Returns 0.0 for any missing or non-finite column.
        """
        vals = []
        for c in cols:
            v = row.get(c, 0.0)
            try:
                v = float(v)
                if not np.isfinite(v):
                    v = 0.0
            except (TypeError, ValueError):
                v = 0.0
            vals.append(v)
        return np.array(vals, dtype=np.float32)

    def _get_obs(self, row: pd.Series) -> np.ndarray:
        """
        Build the observation vector for a given step index.

        Layout (total = OBS_DIM = 53):
          returns (7) | trend_strength (7) | volatility (7)
          | bull_prob (7) | neutral_prob (7) | bear_prob (7)
          | weights (7) | norm_value (1)
          | global_regime (1) | global_trend (1) | global_vol (1)
        """
        # row is passed as parameter

        # Per-asset feature blocks: 6 groups × 7 assets = 42 dims
        per_asset_feats = []
        for group in _PER_ASSET_OBS_GROUPS:
            per_asset_feats.append(
                self._safe_row_values(row, self._obs_cols[group])
            )
        per_asset_vec = np.concatenate(per_asset_feats)   # shape (42,)

        # Portfolio state: weights (7) + normalised value (1) = 8 dims
        norm_value = float(self.portfolio_value / self.initial_value) - 1.0
        portfolio_vec = np.concatenate(
            [self.weights, np.array([norm_value], dtype=np.float32)]
        )   # shape (8,)

        # Global regime context: 3 dims
        if self._has_global_regime:
            g_regime = float(row.get("regime",            0) or 0)
            g_trend  = float(row.get("trend_strength",    0) or 0)
            g_vol    = float(row.get("market_volatility", 0) or 0)
        else:
            g_regime = g_trend = g_vol = 0.0
        global_vec = np.array([g_regime, g_trend, g_vol], dtype=np.float32)

        # Concatenate all sections
        obs = np.concatenate([per_asset_vec, portfolio_vec, global_vec])

        # Clip observation to finite range to prevent NaN propagation
        obs = np.clip(obs, -1e6, 1e6)
        return obs.astype(np.float32)

    # ── Action processing ──────────────────────────────────────────────────────

    def _process_weights(
        self,
        raw_action: np.ndarray,
        prev_weights: np.ndarray,
        apply_smoothing: bool = True,
    ) -> np.ndarray:
        """
        Project raw action → valid portfolio weights.

        Steps:
          1. Take absolute values (weights are non-negative).
          2. Clip each weight to max_weight (concentration limit).
          3. L1-normalise so weights sum to 1.
          4. Exponential smoothing with previous weights (reduces turnover).
        """
        w = np.abs(raw_action).astype(np.float64)
        w = np.clip(w, 0.0, self.max_weight)
        total = w.sum()
        if total < 1e-8:
            w = np.ones(N_ASSETS) / N_ASSETS
        else:
            w /= total

        if apply_smoothing:
            w = (1.0 - self.action_smooth) * w + self.action_smooth * prev_weights.astype(np.float64)
            total = w.sum()
            if total > 1e-8:
                w /= total

        return w.astype(np.float32)

    # ── Risk metrics ───────────────────────────────────────────────────────────

    def _compute_drawdown(self) -> float:
        """Current drawdown from peak portfolio value."""
        if self.peak_value < 1e-9:
            return 0.0
        dd = (self.peak_value - self.portfolio_value) / self.peak_value
        return max(float(dd), 0.0)

    def _compute_cvar(self) -> float:
        """CVaR (Conditional Value at Risk) over recent return history."""
        hist = self.return_history[-self.cvar_window:]
        if len(hist) < 5:
            return 0.0
        r      = np.array(hist)
        cutoff = np.percentile(r, self.cvar_alpha * 100)
        tail   = r[r <= cutoff]
        return float(tail.mean()) if len(tail) > 0 else float(cutoff)

    def _compute_portfolio_vol(self) -> float:
        """Rolling 20-day portfolio return volatility."""
        hist = self.return_history[-20:]
        if len(hist) < 5:
            return 0.0
        return float(np.std(hist))

    def _compute_tail_cvar(self) -> float:
        """Tail CVaR over recent TAIL_RISK_WINDOW steps (worst 10%)."""
        buf = list(self.tail_risk_buf)
        if len(buf) < 10:
            return 0.0
        r     = np.array(buf)
        worst = r[r <= np.percentile(r, 10)]
        return float(np.abs(worst.mean())) if len(worst) > 0 else 0.0

    # ── Reward computation (M2–M5) ─────────────────────────────────────────────

    def _compute_reward(
        self,
        net_return: float,
        drawdown: float,
        port_vol: float,
        turnover: float,
        row: pd.Series,
    ) -> tuple:
        """
        Compute the full regime-aware reward.

        Components
        ----------
        M2  base            = 100 × net_return           (normalised, ≈ −5..+5)
        M3  momentum_bonus  = 0.5 × Σ(w_i × trend_i)
        M4  diversif_bonus  = 0.1 × entropy(weights)
        M5  risk_pen        = bear_prob_avg × (λ_dd × drawdown + λ_vol × vol)
            turn_pen        = turnover_lambda × turnover  (always active, small)

        Returns
        -------
        (reward_float, components_dict)
        """
        # M2: normalised base return
        base = BASE_REWARD_SCALE * net_return

        # M3: momentum alignment bonus — reward allocating to trending assets
        trend_vals     = self._safe_row_values(row, self._trend_cols)
        momentum       = float(np.dot(self.weights, trend_vals))
        momentum_bonus = MOMENTUM_SCALE * momentum

        # M4: diversification bonus — Shannon entropy of portfolio weights
        w_safe         = np.clip(self.weights, 1e-9, 1.0)
        entropy        = float(-np.sum(w_safe * np.log(w_safe)))
        diversif_bonus = DIVERSIF_SCALE * entropy

        # M5: regime-weighted risk penalty
        # Penalty is scaled by portfolio-average bear probability
        # → high bear probability amplifies drawdown and volatility penalties
        bear_vals   = self._safe_row_values(row, self._bear_prob_cols)
        bear_avg    = float(np.clip(bear_vals.mean(), 0.0, 1.0))

        dd_penalty  = self.lambda_dd  * drawdown
        vol_penalty = self.lambda_vol * port_vol
        risk_pen    = bear_avg * (dd_penalty + vol_penalty)

        # Turnover penalty (always active, small, keeps transaction costs in)
        turn_pen = self.turnover_lambda * turnover

        reward = base + momentum_bonus + diversif_bonus - risk_pen - turn_pen

        components = {
            "base":       base,
            "momentum":   momentum_bonus,
            "diversif":   diversif_bonus,
            "risk_pen":   risk_pen,
            "bear_avg":   bear_avg,
            "turn_pen":   turn_pen,
        }
        return reward, components

    # ── Gym API ────────────────────────────────────────────────────────────────

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

        # Reset episode-level debug accumulators
        self._ep_regime_counts = {-1: 0, 0: 0, 1: 0}
        self._ep_ret_buf   = []
        self._ep_dd_buf    = []
        self._ep_vol_buf   = []
        self._ep_turn_buf  = []

        return self._get_obs(self.df.iloc[self.current_step]), {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        self._total_steps += 1

        # ── Rebalance gate ─────────────────────────────────────────────────────
        # Only apply new weights every rebalance_freq steps to reduce turnover
        steps_since_start = self.current_step - self.episode_start
        if steps_since_start % self.rebalance_freq == 0:
            new_weights = self._process_weights(action, self.weights, apply_smoothing=True)
        else:
            new_weights = self.weights.copy()

        # ── Transaction cost ───────────────────────────────────────────────────
        turnover = float(np.abs(new_weights - self.weights).sum())
        txn_cost = self.transaction_cost * turnover
        self.weights = new_weights

        # ── Asset returns ──────────────────────────────────────────────────────
        next_index = min(self.current_step + 1, self.total_rows - 1)
        row_obs    = self.df.iloc[self.current_step]
        row_return = self.df.iloc[next_index]
        asset_returns = np.array(
            [float(row_return.get(col, 0.0) or 0.0) for col in self.return_cols],
            dtype=np.float64,
        )
        asset_returns    = np.where(np.isfinite(asset_returns), asset_returns, 0.0)
        portfolio_return = float(np.dot(self.weights, asset_returns))
        net_return       = portfolio_return - txn_cost

        # ── Portfolio update ───────────────────────────────────────────────────
        self.portfolio_value = max(self.portfolio_value * (1.0 + net_return), 1e-6)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        self.return_history.append(net_return)
        self.tail_risk_buf.append(net_return)

        # ── Risk metrics ───────────────────────────────────────────────────────
        drawdown  = self._compute_drawdown()
        port_vol  = self._compute_portfolio_vol()
        cvar      = self._compute_cvar()
        tail_cvar = self._compute_tail_cvar()

        # Global regime (backward-compat info field)
        regime = int(row_obs.get("regime", 0) or 0)

        # ── Reward (M2–M5) ─────────────────────────────────────────────────────
        reward, comps = self._compute_reward(
            net_return=net_return,
            drawdown=drawdown,
            port_vol=port_vol,
            turnover=turnover,
            row=row_obs,
        )

        # ── M6: Safe PPO additional CVaR / tail penalties ─────────────────────
        if self.safe_reward:
            # CVaR penalty: penalise only when CVaR exceeds cvar_threshold
            cvar_pen  = SAFE_LAMBDA_CVAR * max(0.0, -(cvar + self.cvar_threshold))
            tail_pen  = SAFE_TAIL_SCALE  * tail_cvar

            # Warm-up: ramp penalty scale 0 → 1 over first 200 k training steps
            # This prevents safe penalties from disrupting early learning
            warmup_scale  = min(1.0, self._total_steps / 200_000)
            penalty_scale = warmup_scale * self.risk_penalty_scale

            reward -= penalty_scale * cvar_pen
            reward -= tail_pen

            comps["cvar_pen"] = penalty_scale * cvar_pen
            comps["tail_pen"] = tail_pen

        # ── Episode debug tracking ─────────────────────────────────────────────
        self._ep_regime_counts[regime] = self._ep_regime_counts.get(regime, 0) + 1
        self._ep_ret_buf.append(net_return)
        self._ep_dd_buf.append(drawdown)
        self._ep_vol_buf.append(port_vol)
        self._ep_turn_buf.append(turnover)

        # ── Per-step debug print ────────────────────────────────────────────────
        if self.debug_reward:
            print(
                f"[DBG] regime={regime:+d}  ret={net_return:+.5f}  "
                f"base={comps['base']:+.4f}  mom={comps['momentum']:+.4f}  "
                f"div={comps['diversif']:+.4f}  risk_pen={comps['risk_pen']:.4f}  "
                f"bear_avg={comps['bear_avg']:.3f}  reward={reward:+.4f}"
            )

        # ── Advance step ────────────────────────────────────────────────────────
        self.current_step += 1
        end_step = self.episode_start + MIN_EPISODE_STEPS
        done     = self.current_step >= min(end_step, self.total_rows - 1)

        next_step = min(self.current_step, self.total_rows - 1)
        next_obs  = self._get_obs(self.df.iloc[next_index])

        info = {
            "portfolio_value":   self.portfolio_value,
            "portfolio_return":  portfolio_return,
            "net_return":        net_return,
            "drawdown":          drawdown,
            "cvar":              cvar,
            "tail_cvar":         tail_cvar,
            "volatility":        port_vol,
            "turnover":          turnover,
            "transaction_cost":  txn_cost,
            "weights":           self.weights.tolist(),
            "regime":            regime,
            "reward_components": comps,
        }

        return next_obs, float(reward), done, False, info

    # ── Debug log ──────────────────────────────────────────────────────────────

    def get_episode_debug_log(self) -> str:
        """Return a formatted one-line summary string for the completed episode."""
        if not self._ep_ret_buf:
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

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self):
        dd     = self._compute_drawdown()
        cvar   = self._compute_cvar()
        step   = max(0, self.current_step - 1)
        regime = int(self.df.iloc[step].get("regime", 0) or 0)
        labels = {1: "BULL", 0: "NEUT", -1: "BEAR"}
        print(
            f"Step {self.current_step:4d} | "
            f"Value: ${self.portfolio_value:>12,.2f} | "
            f"DD: {dd*100:5.2f}% | CVaR: {cvar*100:6.2f}% | "
            f"Regime: {labels.get(regime, '?')} | "
            f"W: {[f'{w:.2f}' for w in self.weights]}"
        )


# ── Factory ────────────────────────────────────────────────────────────────────

def make_env(data_path: str, safe_reward: bool = False, **kwargs) -> PortfolioTradingEnv:
    """
    Convenience factory: load CSV, attach multi-asset regime features, return env.

    Parameters
    ----------
    data_path   : Path to a CSV with Date index.
    safe_reward : Whether to use Safe PPO reward (CVaR / tail penalties).
    **kwargs    : Any PortfolioTradingEnv constructor keyword arguments.
    """
    from data_pipeline.regime_detection import detect_market_regime
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    df = detect_market_regime(df)
    return PortfolioTradingEnv(df, safe_reward=safe_reward, **kwargs)