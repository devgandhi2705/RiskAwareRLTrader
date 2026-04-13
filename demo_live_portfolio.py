"""
demo_live_portfolio.py
----------------------
Simulates live portfolio management using trained RL agents.
Reads 365 rows from data/demo_market_data.csv, stepping every 3 seconds.

Usage:
    python demo_live_portfolio.py --agent ppo
    python demo_live_portfolio.py --agent safe_ppo
    python demo_live_portfolio.py --agent dqn
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS          = ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty50", "Sensex"]
N_ASSETS        = len(ASSETS)
INITIAL_CAPITAL = 100_000.0
STEP_SLEEP      = 3          # seconds between steps
STATE_FILE      = os.path.join(PROJECT_ROOT, "demo_state.json")

MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "demo_market_data.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def detect_regime(df: pd.DataFrame, idx: int) -> int:
    """Return regime integer from pre-computed column, or estimate on-the-fly."""
    if "regime" in df.columns:
        val = df.iloc[idx].get("regime", 0)
        try:
            return int(val)
        except Exception:
            return 0
    # Fallback: use 20-day rolling return of BTC as proxy
    if idx < 20:
        return 0
    btc_col = next((c for c in df.columns if "BTC" in c and "Return" in c), None)
    if btc_col is None:
        return 0
    recent = df[btc_col].iloc[max(0, idx - 20): idx + 1].mean()
    if recent > 0.001:
        return 1
    if recent < -0.001:
        return -1
    return 0


def regime_label(code: int) -> str:
    return {1: "BULL", 0: "NEUTRAL", -1: "BEAR"}.get(code, "NEUTRAL")


def compute_sharpe(returns: list, annualise=252) -> float:
    if len(returns) < 5:
        return 0.0
    r = np.array(returns)
    std = r.std()
    if std < 1e-10:
        return 0.0
    return float(r.mean() / std * np.sqrt(annualise))


def compute_max_drawdown(values: list) -> float:
    if len(values) < 2:
        return 0.0
    v = np.array(values)
    peak = np.maximum.accumulate(v)
    dd   = (v - peak) / np.maximum(peak, 1e-9)
    return float(-dd.min())


def compute_volatility(returns: list, annualise=252) -> float:
    if len(returns) < 5:
        return 0.0
    return float(np.std(returns) * np.sqrt(annualise))


def top_signal(weights: np.ndarray) -> str:
    idx = int(np.argmax(weights))
    pct = weights[idx] * 100
    return f"{ASSETS[idx]} strong trend ({pct:.0f}%)"


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_ppo(safe: bool = False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    import gymnasium as gym
    from gymnasium import spaces

    name      = "safe_ppo_portfolio" if safe else "ppo_portfolio"
    zip_path  = os.path.join(MODEL_DIR, f"{name}.zip")
    norm_path = os.path.join(MODEL_DIR, f"{name}_vecnorm.pkl")

    # Dummy env just to satisfy VecNormalize
    obs_dim  = N_ASSETS * 5 + N_ASSETS + 1 + 3   # must match trading_environment
    act_dim  = N_ASSETS

    class _Dummy(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
            self.action_space      = spaces.Box(0.0, 1.0, (act_dim,), np.float32)
        def reset(self, **_):
            return np.zeros(obs_dim, np.float32), {}
        def step(self, _):
            return np.zeros(obs_dim, np.float32), 0.0, True, False, {}

    dummy_vec = DummyVecEnv([lambda: _Dummy()])

    vecnorm = None
    if os.path.exists(norm_path):
        vecnorm = VecNormalize.load(norm_path, dummy_vec)
        vecnorm.training    = False
        vecnorm.norm_reward = False

    model = PPO.load(zip_path, device="cpu")
    print(f"  Loaded PPO model: {zip_path}")
    return model, vecnorm


def load_dqn():
    from stable_baselines3 import DQN

    zip_path    = os.path.join(MODEL_DIR, "dqn_portfolio.zip")
    table_path  = os.path.join(MODEL_DIR, "dqn_action_table.npy")

    model        = DQN.load(zip_path, device="cpu")
    action_table = np.load(table_path)
    print(f"  Loaded DQN model: {zip_path}  ({len(action_table)} actions)")
    return model, action_table


# ── Observation builder ────────────────────────────────────────────────────────

BASE_FEATS  = ["Close", "Return", "RSI"]
EXTRA_FEATS = ["MA20", "Volatility"]

def build_observation(df: pd.DataFrame, idx: int,
                       weights: np.ndarray, port_value: float,
                       initial_value: float) -> np.ndarray:
    """
    Reconstruct the observation vector matching PortfolioTradingEnv._get_obs().
    Layout: [market_features (N*5), weights (N), norm_value (1), regime (3)]
    """
    row = df.iloc[idx]
    feats = []
    for asset in ASSETS:
        for f in BASE_FEATS + EXTRA_FEATS:
            col = f"{asset}_{f}"
            feats.append(_safe_float(row.get(col, 0.0)))

    norm_value = (port_value / initial_value) - 1.0

    regime_val      = _safe_float(row.get("regime", 0))
    trend_strength  = _safe_float(row.get("trend_strength", 0))
    market_vol      = _safe_float(row.get("market_volatility", 0))

    obs = np.array(
        feats + list(weights) + [norm_value, regime_val, trend_strength, market_vol],
        dtype=np.float32
    )
    return obs


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_weights(agent_type: str, obs: np.ndarray,
                    model, aux, prev_weights: np.ndarray) -> np.ndarray:
    """Return normalised portfolio weights from model prediction."""
    if agent_type in ("ppo", "safe_ppo"):
        vecnorm = aux
        obs_in  = obs.reshape(1, -1)
        if vecnorm is not None:
            obs_in = vecnorm.normalize_obs(obs_in)
        action, _ = model.predict(obs_in[0], deterministic=True)
        weights   = np.array(action, dtype=np.float32).flatten()
    else:  # dqn
        action_table = aux
        obs_in       = obs.reshape(1, -1)
        action, _    = model.predict(obs_in[0], deterministic=True)
        weights      = action_table[int(action)].copy()

    # Ensure valid distribution
    weights = np.abs(weights)
    weights = np.clip(weights, 0.0, 0.40)
    total   = weights.sum()
    if total < 1e-8:
        weights = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
    else:
        weights = weights / total
    return weights.astype(np.float32)


# ── Console printer ────────────────────────────────────────────────────────────

def print_step(step: int, date_str: str, regime_code: int,
               port_value: float, daily_return: float,
               weights: np.ndarray, returns_hist: list,
               values_hist: list):
    sharpe  = compute_sharpe(returns_hist)
    max_dd  = compute_max_drawdown(values_hist)
    signal  = top_signal(weights)
    r_label = regime_label(regime_code)

    sep = "─" * 54
    print(f"\n{sep}")
    print(f" STEP {step:>3d}/365   DATE: {date_str}")
    print(f" REGIME: {r_label}")
    print(sep)
    print(f" PORTFOLIO VALUE:  ${port_value:>12,.2f}")
    sign = "+" if daily_return >= 0 else ""
    print(f" DAILY RETURN:     {sign}{daily_return * 100:.2f}%")
    print()
    print(" ALLOCATIONS:")
    for asset, w in zip(ASSETS, weights):
        bar = "█" * int(w * 40)
        print(f"   {asset:<8s}  {bar:<16s} {w*100:5.1f}%")
    print()
    print(f" TOP SIGNAL:       {signal}")
    print(f" SHARPE (rolling): {sharpe:.2f}")
    print(f" MAX DRAWDOWN:     {max_dd * 100:.1f}%")
    print(sep)


# ── State persistence (for dashboard) ─────────────────────────────────────────

def save_state(step: int, date_str: str, regime_code: int,
               port_value: float, daily_return: float,
               weights: np.ndarray, values_hist: list,
               returns_hist: list, agent_name: str):
    state = {
        "step":          step,
        "date":          date_str,
        "regime":        regime_label(regime_code),
        "portfolio_value": round(port_value, 2),
        "daily_return":  round(daily_return * 100, 4),
        "weights":       {a: round(float(w), 4) for a, w in zip(ASSETS, weights)},
        "portfolio_history": [round(v, 2) for v in values_hist],
        "returns_history":   [round(r, 6) for r in returns_hist],
        "sharpe":        round(compute_sharpe(returns_hist), 4),
        "max_drawdown":  round(compute_max_drawdown(values_hist) * 100, 2),
        "volatility":    round(compute_volatility(returns_hist) * 100, 2),
        "agent":         agent_name,
        "total_return":  round((port_value / INITIAL_CAPITAL - 1) * 100, 2),
        "timestamp":     time.time(),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_demo(agent_type: str):
    agent_type = agent_type.lower().strip()
    assert agent_type in ("ppo", "safe_ppo", "dqn"), \
        f"Unknown agent: {agent_type}. Choose ppo | safe_ppo | dqn"

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  AI PORTFOLIO MANAGER — LIVE DEMO")
    print(f"  Agent: {agent_type.upper()}")
    print(f"{'='*54}\n")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Demo data not found: {DATA_PATH}\n"
            f"Please place your 365-row CSV at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col=None)
    df.columns = [c.strip() for c in df.columns]

    # Normalise date column
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col and date_col != "date":
        df.rename(columns={date_col: "date"}, inplace=True)

    # Attach regime features if absent
    if "regime" not in df.columns:
        try:
            sys.path.insert(0, PROJECT_ROOT)
            from data_pipeline.regime_detection import detect_market_regime
            df_indexed = df.set_index("date")
            df_indexed = detect_market_regime(df_indexed)
            df = df_indexed.reset_index()
            print("  Regime features attached via regime_detection module.")
        except Exception as e:
            print(f"  [WARN] Could not attach regime features ({e}). Using neutral.")
            df["regime"]           = 0
            df["trend_strength"]   = 0.0
            df["market_volatility"] = 0.01

    df = df.head(365).reset_index(drop=True)
    total_steps = len(df)
    print(f"  Loaded {total_steps} rows  ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
    print(f"  Estimated runtime: {total_steps * STEP_SLEEP / 60:.1f} minutes\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    print("  Loading model ...")
    if agent_type == "dqn":
        model, aux = load_dqn()
    else:
        model, aux = load_ppo(safe=(agent_type == "safe_ppo"))

    # ── Simulation state ───────────────────────────────────────────────────────
    weights       = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
    port_value    = INITIAL_CAPITAL
    values_hist   = [INITIAL_CAPITAL]
    returns_hist  = []
    return_cols   = [f"{a}_Return" for a in ASSETS]

    print(f"\n  Starting simulation — initial capital: ${INITIAL_CAPITAL:,.0f}\n")

    for step in range(total_steps):
        row       = df.iloc[step]
        date_str  = str(row.get("date", f"Day {step+1}"))[:10]
        regime    = detect_regime(df, step)

        # Build observation
        obs = build_observation(df, step, weights, port_value, INITIAL_CAPITAL)

        # Agent prediction
        new_weights = predict_weights(agent_type, obs, model, aux, weights)

        # Portfolio return
        asset_rets = np.array(
            [_safe_float(row.get(col, 0.0)) for col in return_cols],
            dtype=np.float64
        )
        portfolio_return = float(np.dot(new_weights, asset_rets))
        turnover         = float(np.abs(new_weights - weights).sum())
        txn_cost         = 0.001 * turnover
        net_return       = portfolio_return - txn_cost

        weights      = new_weights
        port_value   = max(port_value * (1.0 + net_return), 1.0)
        values_hist.append(port_value)
        returns_hist.append(net_return)

        # Print step
        print_step(step + 1, date_str, regime,
                   port_value, net_return,
                   weights, returns_hist, values_hist)

        # Save state for dashboard
        save_state(step + 1, date_str, regime,
                   port_value, net_return,
                   weights, values_hist, returns_hist,
                   agent_type.upper())

        if step < total_steps - 1:
            time.sleep(STEP_SLEEP)

    # ── Final summary ──────────────────────────────────────────────────────────
    total_ret = (port_value / INITIAL_CAPITAL - 1) * 100
    sharpe    = compute_sharpe(returns_hist)
    max_dd    = compute_max_drawdown(values_hist) * 100
    vol       = compute_volatility(returns_hist) * 100

    print(f"\n{'='*54}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*54}")
    print(f"  Agent:          {agent_type.upper()}")
    print(f"  Final Value:    ${port_value:>12,.2f}")
    print(f"  Total Return:   {total_ret:+.2f}%")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Max Drawdown:   {max_dd:.1f}%")
    print(f"  Volatility:     {vol:.1f}%")
    print(f"{'='*54}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Portfolio Demo")
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "safe_ppo", "dqn"],
        help="RL agent to use: ppo | safe_ppo | dqn",
    )
    args = parser.parse_args()
    run_demo(args.agent)