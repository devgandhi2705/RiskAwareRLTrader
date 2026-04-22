"""
evaluate_agent.py  (v4 — Risk-Aware Hierarchical Agents)
---------------------------------------------------------
Evaluates the v6 risk-aware agents (Hier-PPO, Safe-Hier-PPO, DQN Risk-Aware)
against classic baselines on 2023–2024 test data.

Agents evaluated
----------------
  hier_ppo_portfolio         : Hierarchical PPO (multi-objective reward)
  safe_hier_ppo_portfolio    : Safe Hierarchical PPO (Lagrangian constraint)
  dqn_risk_aware_portfolio   : Distributional DQN with hierarchical action filter

Metrics computed
----------------
  Total Return, Sharpe Ratio, Sortino Ratio, Max Drawdown,
  CVaR (5%), Annual Volatility, Avg Turnover

Baselines
---------
  Equal Weight, Risk Parity, Momentum (60d)

Plots saved
-----------
  portfolio_value_curves.png
  drawdown_curves.png
  rolling_sharpe.png
  metrics_comparison.png
  allocation_<agent>.png  (per agent)

Logs to: logs/evaluation_log.txt

Run:
    python evaluation/evaluate_agent.py

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

from env.trading_environment import PortfolioTradingEnv, ASSETS, N_ASSETS

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = PROJECT_ROOT / "logs" / "evaluation_log.txt"

def log(msg: str) -> None:
    print(msg)
    os.makedirs(str(LOG_FILE.parent), exist_ok=True)
    with open(str(LOG_FILE), "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DATA_PATH = PROJECT_ROOT / "data"    / "test_dataset.csv"
MODEL_DIR      = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results"

# ── Agent registry ─────────────────────────────────────────────────────────────
# algo_variant: "PPO_HIER" | "SAFE_PPO_HIER" | "DQN_RISK"
AGENT_CONFIGS = {
    "hier_ppo": {
        "label":        "Hier-PPO (Alpha v6)",
        "algo":         "PPO",
        "algo_variant": "PPO_HIER",
        "safe_reward":  False,
        "model_path":   str(MODEL_DIR / "hier_ppo_portfolio"),
        "vecnorm_path": str(MODEL_DIR / "hier_ppo_portfolio_vecnorm.pkl"),
        "color":        "#DD8452",
    },
    "safe_hier_ppo": {
        "label":        "Safe-Hier-PPO (Risk-Aware)",
        "algo":         "PPO",
        "algo_variant": "SAFE_PPO_HIER",
        "safe_reward":  True,
        "model_path":   str(MODEL_DIR / "safe_hier_ppo_portfolio"),
        "vecnorm_path": str(MODEL_DIR / "safe_hier_ppo_portfolio_vecnorm.pkl"),
        "color":        "#55A868",
    },
    "dqn_risk_aware": {
        "label":        "DQN Risk-Aware (Dist. v6)",
        "algo":         "DQN",
        "algo_variant": "DQN_RISK",
        "safe_reward":  False,
        "model_path":   str(MODEL_DIR / "dqn_risk_aware_portfolio"),
        "vecnorm_path": None,
        "color":        "#4C72B0",
    },
}


# ── Metric helpers ─────────────────────────────────────────────────────────────

def total_return(values):
    v = np.array(values)
    return float((v[-1] - v[0]) / v[0]) if len(v) >= 2 and v[0] > 0 else 0.0

def sharpe(returns, ann=252):
    r = np.array(returns)
    s = r.std()
    return float(r.mean() / s * np.sqrt(ann)) if s > 1e-10 else 0.0

def sortino(returns, ann=252):
    r   = np.array(returns)
    neg = r[r < 0]
    ds  = neg.std() if len(neg) > 0 else 1e-10
    return float(r.mean() / ds * np.sqrt(ann)) if ds > 1e-10 else 0.0

def max_drawdown(values):
    v    = np.array(values)
    peak = np.maximum.accumulate(v)
    dd   = (peak - v) / np.where(peak > 0, peak, 1e-9)
    return float(dd.max())

def cvar_metric(returns, alpha=0.05):
    r   = np.array(returns)
    var = np.percentile(r, alpha * 100)
    t   = r[r <= var]
    return float(t.mean()) if len(t) > 0 else float(var)

def ann_vol(returns, ann=252):
    return float(np.std(returns) * np.sqrt(ann))

def drawdown_series(values):
    v    = np.array(values)
    peak = np.maximum.accumulate(v)
    return (peak - v) / np.where(peak > 0, peak, 1e-9)

def rolling_sharpe(returns, window=60, ann=252):
    r  = pd.Series(returns)
    rs = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(ann)
    return rs.fillna(0.0).values


def _build_result(label, color, dates, values, returns, turnovers, weights_h):
    return {
        "label":          label,
        "color":          color,
        "dates":          dates,
        "values":         np.array(values),
        "returns":        np.array(returns),
        "turnovers":      np.array(turnovers),
        "weights_h":      weights_h,
        "total_return":   total_return(values),
        "sharpe":         sharpe(returns),
        "sortino":        sortino(returns),
        "max_drawdown":   max_drawdown(values),
        "cvar":           cvar_metric(returns),
        "ann_vol":        ann_vol(returns),
        "avg_turnover":   float(np.mean(turnovers)) if len(turnovers) > 0 else 0.0,
        "dd_series":      drawdown_series(values),
        "rolling_sharpe": rolling_sharpe(returns),
    }


# ── Baselines ──────────────────────────────────────────────────────────────────

def _ret_cols(df):
    return [f"{a}_return" for a in ASSETS if f"{a}_return" in df.columns]


def equal_weight_strategy(df):
    cols  = _ret_cols(df)
    w     = np.ones(len(cols)) / len(cols)
    daily = df[cols].fillna(0.0).values @ w
    vals  = np.insert(np.cumprod(1 + daily) * 100_000.0, 0, 100_000.0)
    n     = min(len(df), len(vals))
    return _build_result(
        "Equal Weight", "#888888", df.index[:n],
        vals, daily, np.zeros(len(daily)), np.tile(w, (len(daily), 1))
    )


def risk_parity_strategy(df):
    cols       = _ret_cols(df)
    ret_mat    = df[cols].fillna(0.0).values
    n_days, na = ret_mat.shape
    vals       = [100_000.0]; rets = []; turns = []; wh = []
    w_prev     = np.ones(na) / na

    for t in range(n_days):
        if t % 21 == 0 and t >= 60:
            vol = ret_mat[max(0, t - 60):t].std(axis=0)
            vol = np.where(vol < 1e-8, 1e-8, vol)
            w   = 1.0 / vol
            w   = np.minimum(w / w.sum(), 0.40)
            w   = w / w.sum()
        else:
            w = w_prev
        r    = float(ret_mat[t] @ w)
        turn = float(np.abs(w - w_prev).sum())
        nr   = r - 0.001 * turn
        vals.append(vals[-1] * (1 + nr))
        rets.append(nr); turns.append(turn); wh.append(w.tolist())
        w_prev = w.copy()

    n = min(len(df), len(vals))
    return _build_result(
        "Risk Parity", "#9467BD", df.index[:n],
        np.array(vals), np.array(rets), np.array(turns), np.array(wh)
    )


def momentum_strategy(df):
    cols      = _ret_cols(df)
    close_c   = [c.replace("_return", "_Close") for c in cols]
    available = [c for c in close_c if c in df.columns]
    if not available:
        return equal_weight_strategy(df)

    ret_mat   = df[cols].fillna(0.0).values
    close_mat = df[available].ffill().values
    na        = len(available); n_days = len(df)
    vals      = [100_000.0]; rets = []; turns = []; wh = []
    w_prev    = np.ones(na) / na

    for t in range(n_days):
        if t % 21 == 0 and t >= 60:
            mom = (close_mat[t] - close_mat[max(0, t - 60)]) / np.where(
                close_mat[max(0, t - 60)] > 0, close_mat[max(0, t - 60)], 1.0
            )
            mom = np.maximum(mom, 0.0); s = mom.sum()
            w   = mom / s if s > 1e-8 else np.ones(na) / na
            w   = np.minimum(w, 0.40); w = w / w.sum()
        else:
            w = w_prev
        r    = float(ret_mat[t, :na] @ w)
        turn = float(np.abs(w - w_prev).sum())
        nr   = r - 0.001 * turn
        vals.append(vals[-1] * (1 + nr))
        rets.append(nr); turns.append(turn); wh.append(w.tolist())
        w_prev = w.copy()

    n = min(len(df), len(vals))
    return _build_result(
        "Momentum (60d)", "#E377C2", df.index[:n],
        np.array(vals), np.array(rets), np.array(turns), np.array(wh)
    )


# ── Eval env builders ──────────────────────────────────────────────────────────

def _build_eval_env_ppo(df, safe_reward):
    """Wrap base env with HierarchicalWrapper to match training obs space (46 dims)."""
    from train_ppo_risk_aware import HierarchicalWrapper, MultiObjectiveRewardWrapper
    base = PortfolioTradingEnv(df, safe_reward=safe_reward, random_start=False)
    env  = MultiObjectiveRewardWrapper(base)
    env  = HierarchicalWrapper(env)
    return env, base


def _build_eval_env_dqn(df):
    """Wrap base env to match DQN training stack (Hierarchical + Discrete)."""
    from train_ppo_risk_aware import HierarchicalWrapper, MultiObjectiveRewardWrapper
    from train_dqn_risk_aware import DiscretePortfolioWrapper

    at_path = str(MODEL_DIR / "dqn_risk_action_table.npy")
    if not os.path.isfile(at_path):
        raise FileNotFoundError(
            f"DQN action table not found: {at_path}\n"
            "Run agents/train_dqn_risk_aware.py first."
        )
    action_table = np.load(at_path)

    base = PortfolioTradingEnv(df, safe_reward=False, random_start=False)
    env  = MultiObjectiveRewardWrapper(base)
    env  = HierarchicalWrapper(env)
    env  = DiscretePortfolioWrapper(env, action_table)
    return env, base


# ── RL agent rollout ───────────────────────────────────────────────────────────

def run_rl_episode(agent_key, cfg, df):
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    model_zip = cfg["model_path"] + ".zip"
    if not os.path.isfile(model_zip):
        log(f"  ! Model not found: {model_zip} — skipping {agent_key}")
        return None

    variant   = cfg["algo_variant"]
    AlgoClass = DQN if cfg["algo"] == "DQN" else PPO
    model     = AlgoClass.load(cfg["model_path"])
    log(f"  + Loaded {cfg['algo']} ({variant}) from {model_zip}")

    # ── Build eval env matching the training wrapper stack ─────────────────────
    vecnorm = None
    if variant in ("PPO_HIER", "SAFE_PPO_HIER"):
        env, base_env = _build_eval_env_ppo(df, cfg["safe_reward"])
        vn_path = cfg.get("vecnorm_path")
        if vn_path and os.path.isfile(vn_path):
            dummy   = DummyVecEnv([lambda: Monitor(
                _build_eval_env_ppo(df, cfg["safe_reward"])[0]
            )])
            vecnorm = VecNormalize.load(vn_path, dummy)
            vecnorm.training    = False
            vecnorm.norm_reward = False
    elif variant == "DQN_RISK":
        env, base_env = _build_eval_env_dqn(df)
    else:
        raise ValueError(f"Unknown algo_variant: {variant}")

    obs, _   = env.reset()
    done     = False
    vals     = [base_env.initial_value]
    rets     = []; turns = []; weights_h = []

    while not done:
        obs_in = (vecnorm.normalize_obs(obs.reshape(1, -1))[0]
                  if vecnorm else obs)
        action, _ = model.predict(obs_in, deterministic=True)
        obs, _, done, trunc, info = env.step(action)
        done = done or trunc
        vals.append(info.get("portfolio_value", vals[-1]))
        rets.append(info.get("net_return",       0.0))
        turns.append(info.get("turnover",        0.0))
        weights_h.append(info.get("weights",     []))

    n     = min(len(df), len(vals))
    dates = df.index[:n]
    return _build_result(
        cfg["label"], cfg["color"], dates,
        np.array(vals), np.array(rets), np.array(turns), np.array(weights_h)
    )


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_values(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in results:
        n = min(len(r["dates"]), len(r["values"]))
        ax.plot(r["dates"][:n], r["values"][:n],
                label=r["label"], color=r["color"], linewidth=1.8)
    ax.set_title("Portfolio Value — Test Period (2023–2024)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout(); plt.savefig(str(save_path), dpi=150); plt.close()
    log(f"Saved → {save_path}")


def _plot_drawdowns(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for r in results:
        n  = min(len(r["dates"]), len(r["dd_series"]))
        dd = r["dd_series"][:n] * 100
        ax.fill_between(r["dates"][:n], -dd, 0, alpha=0.2, color=r["color"])
        ax.plot(r["dates"][:n], -dd, label=r["label"],
                color=r["color"], linewidth=1.5)
    ax.set_title("Drawdown — Test Period (2023–2024)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(str(save_path), dpi=150); plt.close()
    log(f"Saved → {save_path}")


def _plot_rolling_sharpe(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for r in results:
        rs = r["rolling_sharpe"]; n = min(len(r["dates"]), len(rs))
        ax.plot(r["dates"][:n], rs[:n], label=r["label"],
                color=r["color"], linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling 60-Day Sharpe — Test Period",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(str(save_path), dpi=150); plt.close()
    log(f"Saved → {save_path}")


def _plot_metrics_bar(metrics_df, save_path):
    cols      = ["Total Return (%)", "Sharpe Ratio", "Sortino Ratio",
                 "Max Drawdown (%)", "CVaR (%)", "Ann. Volatility (%)"]
    color_map = {v["label"]: v["color"] for v in AGENT_CONFIGS.values()}
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
    for ax, col in zip(axes, cols):
        labels  = metrics_df["Agent"].values
        vals    = metrics_df[col].values
        bcolors = [color_map.get(l, "#aaaaaa") for l in labels]
        bars    = ax.bar(labels, vals, color=bcolors, edgecolor="white")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7,
            )
    plt.suptitle("Agent Performance Comparison — Test Period",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {save_path}")


def _plot_allocation(result, save_path):
    wh = result["weights_h"]
    if len(wh) == 0:
        return
    wh_arr = np.array(wh)
    if wh_arr.ndim != 2:
        return
    dates = result["dates"][:len(wh_arr)]
    names = ASSETS[:wh_arr.shape[1]]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(dates, wh_arr.T * 100, labels=names, alpha=0.8)
    ax.set_title(f"Allocation Weights — {result['label']}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Allocation (%)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(str(save_path), dpi=150); plt.close()
    log(f"Saved → {save_path}")


# ── Master evaluation entry point ─────────────────────────────────────────────

def evaluate_all(agent_keys=None):
    """
    Evaluate all agents and baselines on the test dataset.

    Parameters
    ----------
    agent_keys : list[str] | None
        Subset of AGENT_CONFIGS keys. None = all.
    """
    with open(str(LOG_FILE), "w", encoding="utf-8") as f:
        f.write("")

    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    log("=" * 65)
    log("  AGENT EVALUATION — TEST PERIOD (2023–2024)")
    log("=" * 65)

    df = pd.read_csv(str(TEST_DATA_PATH), index_col="Date", parse_dates=True)
    log(f"Test data: {df.shape}  "
        f"({df.index.min().date()} → {df.index.max().date()})\n")

    keys_to_eval = agent_keys or list(AGENT_CONFIGS.keys())
    all_results  = []
    rows         = []

    def _add(res):
        all_results.append(res)
        rows.append({
            "Agent":               res["label"],
            "Total Return (%)":    round(res["total_return"]  * 100, 2),
            "Sharpe Ratio":        round(res["sharpe"],              3),
            "Sortino Ratio":       round(res["sortino"],             3),
            "Max Drawdown (%)":    round(res["max_drawdown"]  * 100, 2),
            "CVaR (%)":            round(res["cvar"]          * 100, 2),
            "Ann. Volatility (%)": round(res["ann_vol"]       * 100, 2),
            "Avg Turnover":        round(res["avg_turnover"],        4),
        })

    log("Computing baseline strategies …")
    _add(equal_weight_strategy(df))
    _add(risk_parity_strategy(df))
    _add(momentum_strategy(df))

    log("\nEvaluating RL agents …")
    for key in keys_to_eval:
        if key not in AGENT_CONFIGS:
            continue
        res = run_rl_episode(key, AGENT_CONFIGS[key], df)
        if res:
            _add(res)
            _plot_allocation(res, RESULTS_DIR / f"allocation_{key}.png")

    metrics_df = pd.DataFrame(rows)
    log("\n" + "─" * 80)
    log(metrics_df.to_string(index=False))
    log("─" * 80)

    csv_path = RESULTS_DIR / "performance_metrics.csv"
    metrics_df.to_csv(str(csv_path), index=False)
    log(f"\nMetrics saved → {csv_path}")

    _plot_values(all_results,         RESULTS_DIR / "portfolio_value_curves.png")
    _plot_drawdowns(all_results,      RESULTS_DIR / "drawdown_curves.png")
    _plot_rolling_sharpe(all_results, RESULTS_DIR / "rolling_sharpe.png")
    if len(rows) >= 2:
        _plot_metrics_bar(metrics_df, RESULTS_DIR / "metrics_comparison.png")

    log(f"\nEvaluation complete.  Log saved → {LOG_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RL portfolio agents.")
    parser.add_argument("--agents", nargs="*", default=None,
                        choices=list(AGENT_CONFIGS.keys()),
                        help="Agents to evaluate (default: all).")
    args = parser.parse_args()
    evaluate_all(agent_keys=args.agents)
