"""
trading_environment.py  (v8 — Simplified, No Regime Features)
--------------------------------------------------------------
RL trading environment using only technical indicators.

Architecture change from v7
---------------------------
  V8-1 : Observation vector reduced from 53 → 43 dims.
         No regime probability or global regime features.
         Layout:
           7 × 5 indicator features  = 35
           7 portfolio weights        =  7
           1 normalised portfolio val =  1
           ─────────────────────────────
           Total                     = 43

  V8-2 : max_weight raised 0.40 → 0.90 for aggressive allocation.

  V8-3 : Action smoothing removed (ACTION_SMOOTH = 0).

  V8-4 : Turnover penalty = 0.0025 × turnover.

  V8-5 : PPO reward (safe_reward=False):
           portfolio_return
         + 0.5 × (portfolio_return − momentum_return)
         − 0.0025 × turnover
         where momentum_return = return of 60d-momentum weighted portfolio
         based on past asset returns in the current row.

  V8-6 : Safe PPO reward (safe_reward=True) — market-return switched:
         market_return >= 0  (bull/neutral):
             same formula as PPO
         market_return < 0   (bear):
             0.5 × portfolio_return − 2 × volatility − 3 × drawdown

  V8-7 : market_return = SPY_return (proxy for broad market).
         Falls back to average of all asset returns if SPY absent.

  V8-8 : Episode length = 255 steps (≈1 trading year).
         random_start = True during training.

Backward-compat
---------------
  Constructor accepts (and silently ignores) legacy kwargs from v7:
  bear_prob, cvar_threshold, lambda_dd, lambda_cvar, lambda_vol, etc.

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

# 5 indicator features per asset, in order
_ASSET_FEATURES = ["return", "MA20", "MA50", "RSI", "volatility"]
N_ASSET_FEAT    = len(_ASSET_FEATURES)   # 5

# Observation: 35 indicators + 7 weights + 1 value = 43
OBS_DIM = N_ASSETS * N_ASSET_FEAT + N_ASSETS + 1   # 43

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INITIAL_VALUE    = 100_000.0
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_MAX_WEIGHT       = 0.90          # V8-2: raised from 0.40
DEFAULT_REBALANCE_FREQ   = 1             # rebalance every step (no gate)
DEFAULT_RANDOM_START     = True
MIN_EPISODE_STEPS        = 255           # V8-8: ≈1 trading year
TAIL_RISK_WINDOW         = 50

# ── V8 Reward constants ────────────────────────────────────────────────────────
TURNOVER_PEN     = 0.0025    # V8-4
MOM_ALPHA        = 0.5       # V8-5: momentum excess return weight
BEAR_BASE_SCALE  = 0.5       # V8-6: bear-mode base return weight
BEAR_VOL_PEN     = 2.0       # V8-6
BEAR_DD_PEN      = 3.0       # V8-6

# SPY column used as market proxy for bear detection
SPY_RETURN_COL   = "SPY_return"

# Lookback window for 60-day momentum weights (in steps)
MOM_WINDOW       = 60


class PortfolioTradingEnv(gym.Env):
    """
    Multi-Asset RL Portfolio Environment  (v8).

    Observation (43 dims)
    ---------------------
    [BTC_return … Sensex_volatility]  (35)
    [w_BTC … w_Sensex]               (7)
    [norm_portfolio_value]            (1)

    Reward
    ------
    PPO (safe_reward=False):
        R = portfolio_return
          + 0.5 × (portfolio_return − momentum_return)
          − 0.0025 × turnover

    Safe PPO (safe_reward=True):
      market_return >= 0  → same as PPO
      market_return <  0  → 0.5 × portfolio_return − 2 × vol − 3 × drawdown
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        safe_reward=False,
        initial_value=DEFAULT_INITIAL_VALUE,
        transaction_cost=DEFAULT_TRANSACTION_COST,
        max_weight=DEFAULT_MAX_WEIGHT,
        rebalance_freq=DEFAULT_REBALANCE_FREQ,
        random_start=DEFAULT_RANDOM_START,
        debug_reward=False,
        # Legacy kwargs silently ignored (backward compat)
        turnover_lambda=None,
        lambda_dd=None,
        lambda_cvar=None,
        lambda_vol=None,
        cvar_threshold=None,
        dd_threshold=None,
        risk_penalty_scale=None,
        action_smooth=None,
        cvar_window=None,
        cvar_alpha=None,
    ):
        super().__init__()

        self.df               = df.copy().reset_index(drop=True)
        self.safe_reward      = safe_reward
        self.initial_value    = initial_value
        self.transaction_cost = transaction_cost
        self.max_weight       = max_weight
        self.rebalance_freq   = rebalance_freq
        self.random_start     = random_start
        self.debug_reward     = debug_reward

        self.total_rows = len(self.df)

        # ── Column lookup ──────────────────────────────────────────────────────
        # Build column name list for observation: 5 features × 7 assets = 35
        self._obs_feature_cols = [
            f"{asset}_{feat}"
            for asset in ASSETS
            for feat in _ASSET_FEATURES
        ]
        self.return_cols = [f"{a}_return" for a in ASSETS]

        # ── Spaces ─────────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_ASSETS,), dtype=np.float32
        )

        # ── State ──────────────────────────────────────────────────────────────
        self.portfolio_value = initial_value
        self.peak_value      = initial_value
        self.weights         = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
        self.return_history  = []
        self.tail_risk_buf   = deque(maxlen=TAIL_RISK_WINDOW)
        self.episode_start   = 0
        self.current_step    = 0
        self._total_steps    = 0

        # Episode debug accumulators
        self._ep_ret_buf  = []
        self._ep_dd_buf   = []
        self._ep_vol_buf  = []
        self._ep_turn_buf = []

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _safe_val(self, row: pd.Series, col: str, default: float = 0.0) -> float:
        if col not in row.index:
            return default
        try:
            v = float(row[col])
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self, row: pd.Series) -> np.ndarray:
        """
        Build the 43-dim observation vector.

        Layout:
          [indicator features (35)] | [weights (7)] | [norm_value (1)]
        """
        # 35 indicator features
        indicators = np.array(
            [self._safe_val(row, c) for c in self._obs_feature_cols],
            dtype=np.float32,
        )

        # Portfolio state
        norm_val = float(self.portfolio_value / self.initial_value) - 1.0
        port_vec = np.append(self.weights, np.float32(norm_val))   # (8,)

        obs = np.concatenate([indicators, port_vec])
        obs = np.clip(obs, -1e6, 1e6)
        return obs.astype(np.float32)

    # ── Action processing ──────────────────────────────────────────────────────

    def _process_weights(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Project continuous action → valid portfolio weights.

        1. Clip to [1e-6, max_weight]
        2. L1-normalise → sum to 1
        No smoothing (V8-3).
        """
        w = np.clip(raw_action.astype(np.float64), 1e-6, self.max_weight)
        total = w.sum()
        if total < 1e-8:
            w = np.ones(N_ASSETS, dtype=np.float64) / N_ASSETS
        else:
            w /= total
        return w.astype(np.float32)

    # ── Risk metrics ───────────────────────────────────────────────────────────

    def _compute_drawdown(self) -> float:
        if self.peak_value < 1e-9:
            return 0.0
        return max(float((self.peak_value - self.portfolio_value) / self.peak_value), 0.0)

    def _compute_portfolio_vol(self) -> float:
        hist = self.return_history[-20:]
        return float(np.std(hist)) if len(hist) >= 5 else 0.0

    def _compute_cvar(self) -> float:
        hist = self.return_history[-30:]
        if len(hist) < 5:
            return 0.0
        r      = np.array(hist)
        cutoff = np.percentile(r, 5)
        tail   = r[r <= cutoff]
        return float(tail.mean()) if len(tail) > 0 else float(cutoff)

    def _compute_tail_cvar(self) -> float:
        buf = list(self.tail_risk_buf)
        if len(buf) < 10:
            return 0.0
        r     = np.array(buf)
        worst = r[r <= np.percentile(r, 10)]
        return float(np.abs(worst.mean())) if len(worst) > 0 else 0.0

    # ── Momentum return helper ─────────────────────────────────────────────────

    def _momentum_return(self, row: pd.Series) -> float:
        """
        Compute the 60-day momentum strategy return for this step.

        Uses 60-day cumulative returns per asset to compute weights,
        then dots with current step's asset returns.
        The return cols are read from self.df[self.return_cols].
        """
        start = max(0, self.current_step - MOM_WINDOW)
        end   = self.current_step  # exclusive; current row not yet consumed

        if end - start < 10:
            # Not enough history — use equal weight
            mom_w = np.ones(N_ASSETS, dtype=np.float64) / N_ASSETS
        else:
            window_rets = self.df.iloc[start:end][self.return_cols].fillna(0.0).values
            cum_rets    = (1 + window_rets).prod(axis=0) - 1.0  # shape (N_ASSETS,)
            pos_rets    = np.maximum(cum_rets, 0.0)
            total       = pos_rets.sum()
            if total < 1e-8:
                mom_w = np.ones(N_ASSETS, dtype=np.float64) / N_ASSETS
            else:
                mom_w = pos_rets / total
                mom_w = np.minimum(mom_w, 0.40)
                mom_w = mom_w / mom_w.sum()

        asset_rets = np.array(
            [self._safe_val(row, c) for c in self.return_cols],
            dtype=np.float64,
        )
        return float(np.dot(mom_w, asset_rets))

    # ── Reward (V8) ────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        portfolio_return: float,
        asset_returns: np.ndarray,
        turnover: float,
        drawdown: float,
        port_vol: float,
        row: pd.Series,
    ) -> tuple:
        """
        V8-5 PPO reward (safe_reward=False):
            portfolio_return
          + 0.5 × (portfolio_return − momentum_return)
          − 0.0025 × turnover

        V8-6 Safe PPO reward (safe_reward=True):
          market_return >= 0  → same as PPO
          market_return <  0  → 0.5 × portfolio_return − 2 × vol − 3 × drawdown

        Returns (reward_float, components_dict)
        """
        momentum_ret = self._momentum_return(row)
        excess       = portfolio_return - momentum_ret
        turn_pen     = TURNOVER_PEN * turnover

        # Determine market direction (SPY return, fallback to avg)
        market_ret = self._safe_val(row, SPY_RETURN_COL,
                                    default=float(np.mean(asset_returns)))

        in_bear_mode = self.safe_reward and (market_ret < 0)

        if in_bear_mode:
            reward = (
                BEAR_BASE_SCALE * portfolio_return
                - BEAR_VOL_PEN  * port_vol
                - BEAR_DD_PEN   * drawdown
            )
            comps = {
                "mode":       "bear",
                "base":       BEAR_BASE_SCALE * portfolio_return,
                "vol_pen":    BEAR_VOL_PEN * port_vol,
                "dd_pen":     BEAR_DD_PEN  * drawdown,
                "market_ret": market_ret,
            }
        else:
            reward = portfolio_return + MOM_ALPHA * excess - turn_pen
            comps  = {
                "mode":        "alpha",
                "base":        portfolio_return,
                "excess":      MOM_ALPHA * excess,
                "turn_pen":    turn_pen,
                "market_ret":  market_ret,
            }

        return float(reward), comps

    # ── Gym API ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = max(0, self.total_rows - MIN_EPISODE_STEPS - 1)
        self.episode_start = (
            int(self.np_random.integers(0, max_start))
            if self.random_start and max_start > 0
            else 0
        )

        self.current_step    = self.episode_start
        self.portfolio_value = self.initial_value
        self.peak_value      = self.initial_value
        self.weights         = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
        self.return_history  = []
        self.tail_risk_buf   = deque(maxlen=TAIL_RISK_WINDOW)

        if not hasattr(self, "_total_steps"):
            self._total_steps = 0

        self._ep_ret_buf  = []
        self._ep_dd_buf   = []
        self._ep_vol_buf  = []
        self._ep_turn_buf = []

        return self._get_obs(self.df.iloc[self.current_step]), {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        self._total_steps += 1

        # ── New weights (no smoothing, V8-3) ───────────────────────────────────
        new_weights = self._process_weights(action)
        turnover    = float(np.abs(new_weights - self.weights).sum())
        txn_cost    = self.transaction_cost * turnover
        self.weights = new_weights

        # ── Asset returns (look ahead: next row materialises) ──────────────────
        next_index   = min(self.current_step + 1, self.total_rows - 1)
        row_obs      = self.df.iloc[self.current_step]
        row_return   = self.df.iloc[next_index]

        asset_returns = np.array(
            [self._safe_val(row_return, c) for c in self.return_cols],
            dtype=np.float64,
        )

        portfolio_return = float(np.dot(self.weights, asset_returns))
        net_return       = portfolio_return - txn_cost

        # ── Portfolio update ───────────────────────────────────────────────────
        self.portfolio_value = max(
            self.portfolio_value * (1.0 + net_return), 1e-6
        )
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        self.return_history.append(net_return)
        self.tail_risk_buf.append(net_return)

        # ── Risk metrics ───────────────────────────────────────────────────────
        drawdown  = self._compute_drawdown()
        port_vol  = self._compute_portfolio_vol()
        cvar      = self._compute_cvar()
        tail_cvar = self._compute_tail_cvar()

        # ── Reward (V8) ────────────────────────────────────────────────────────
        reward, comps = self._compute_reward(
            portfolio_return=portfolio_return,
            asset_returns=asset_returns,
            turnover=turnover,
            drawdown=drawdown,
            port_vol=port_vol,
            row=row_obs,
        )

        # ── Debug tracking ─────────────────────────────────────────────────────
        self._ep_ret_buf.append(net_return)
        self._ep_dd_buf.append(drawdown)
        self._ep_vol_buf.append(port_vol)
        self._ep_turn_buf.append(turnover)

        if self.debug_reward:
            mode = comps.get("mode", "?")
            if mode == "bear":
                print(
                    f"[DBG] BEAR  ret={net_return:+.5f}  "
                    f"base={comps['base']:+.5f}  "
                    f"vol={comps['vol_pen']:.5f}  "
                    f"dd={comps['dd_pen']:.5f}  "
                    f"mkt={comps['market_ret']:+.5f}  r={reward:+.5f}"
                )
            else:
                print(
                    f"[DBG] ALPHA  ret={net_return:+.5f}  "
                    f"base={comps['base']:+.5f}  "
                    f"exc={comps['excess']:+.5f}  "
                    f"turn={comps['turn_pen']:.5f}  "
                    f"mkt={comps['market_ret']:+.5f}  r={reward:+.5f}"
                )

        # ── Advance ────────────────────────────────────────────────────────────
        self.current_step += 1
        end_step = self.episode_start + MIN_EPISODE_STEPS
        done     = self.current_step >= min(end_step, self.total_rows - 1)

        next_obs = self._get_obs(self.df.iloc[next_index])

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
            "reward_mode":       comps.get("mode"),
            "reward_components": comps,
            # Provide regime=0 always so evaluate_agent.py doesn't break
            "regime":            0,
        }

        return next_obs, float(reward), done, False, info

    # ── Debug log ──────────────────────────────────────────────────────────────

    def get_episode_debug_log(self) -> str:
        if not self._ep_ret_buf:
            return ""
        return (
            f"avg_ret={np.mean(self._ep_ret_buf):+.5f}  "
            f"avg_dd={np.mean(self._ep_dd_buf):.5f}  "
            f"avg_vol={np.mean(self._ep_vol_buf):.5f}  "
            f"avg_turn={np.mean(self._ep_turn_buf):.5f}"
        )

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self):
        dd   = self._compute_drawdown()
        cvar = self._compute_cvar()
        mode = "SAFE" if self.safe_reward else "PPO"
        print(
            f"Step {self.current_step:4d} | "
            f"Value: ${self.portfolio_value:>12,.2f} | "
            f"DD: {dd*100:5.2f}% | CVaR: {cvar*100:6.2f}% | "
            f"Mode: {mode} | "
            f"W: {[f'{w:.2f}' for w in self.weights]}"
        )


# ── Factory ────────────────────────────────────────────────────────────────────

def make_env(data_path: str, safe_reward: bool = False, **kwargs) -> PortfolioTradingEnv:
    """Load CSV and return configured env. No regime features needed."""
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    return PortfolioTradingEnv(df, safe_reward=safe_reward, **kwargs)