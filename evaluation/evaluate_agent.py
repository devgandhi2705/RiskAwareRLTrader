"""
evaluate_agent.py  (v3 — Regime-Aware)
----------------------------------------
Evaluates all trained agents plus classic baselines.

New in v3
---------
  R1 : regime features injected before environment creation
  R8 : per-regime performance breakdown (bull / neutral / bear)
  New plots: regime_performance.png

Retained from v2
----------------
  Equal Weight / Risk Parity / Momentum baselines
  Full metric suite: Sharpe, Sortino, MaxDD, CVaR, Volatility, Turnover
  Portfolio value, drawdown, rolling Sharpe, allocation plots

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.trading_environment import PortfolioTradingEnv, ASSETS, N_ASSETS
from agents.train_dqn import DiscretePortfolioWrapper
from data_pipeline.regime_detection import detect_market_regime, get_regime_stats

TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_dataset.csv")
MODEL_DIR      = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")

AGENT_CONFIGS = {
    "dqn": {
        "label":        "DQN (Baseline)",
        "algo":         "DQN",
        "safe_reward":  False,
        "model_path":   os.path.join(MODEL_DIR, "dqn_portfolio"),
        "vecnorm_path": None,
        "color":        "#4C72B0",
    },
    "ppo": {
        "label":        "PPO (Baseline)",
        "algo":         "PPO",
        "safe_reward":  False,
        "model_path":   os.path.join(MODEL_DIR, "ppo_portfolio"),
        "vecnorm_path": os.path.join(MODEL_DIR, "ppo_portfolio_vecnorm.pkl"),
        "color":        "#DD8452",
    },
    "safe_ppo": {
        "label":        "Safe PPO (Risk-Aware)",
        "algo":         "PPO",
        "safe_reward":  True,
        "model_path":   os.path.join(MODEL_DIR, "safe_ppo_portfolio"),
        "vecnorm_path": os.path.join(MODEL_DIR, "safe_ppo_portfolio_vecnorm.pkl"),
        "color":        "#55A868",
    },
}

REGIME_COLORS = {1: "#f4a261", 0: "#adb5bd", -1: "#457b9d"}
REGIME_NAMES  = {1: "Bull", 0: "Neutral", -1: "Bear"}


# ── Metric helpers ────────────────────────────────────────────────────────────

def total_return(values):
    if len(values) < 2 or values[0] <= 0: return 0.0
    return float((values[-1] - values[0]) / values[0])

def sharpe(returns, rf=0.0, ann=252):
    r = np.array(returns) - rf
    s = r.std()
    return float(r.mean() / s * np.sqrt(ann)) if s > 1e-10 else 0.0

def sortino(returns, rf=0.0, ann=252):
    r   = np.array(returns) - rf
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

def ann_volatility(returns, ann=252):
    return float(np.std(returns) * np.sqrt(ann))

def avg_turnover(t):
    return float(np.mean(t)) if len(t) > 0 else 0.0

def drawdown_series(values):
    v    = np.array(values)
    peak = np.maximum.accumulate(v)
    return (peak - v) / np.where(peak > 0, peak, 1e-9)

def rolling_sharpe(returns, window=60, ann=252):
    r  = pd.Series(returns)
    rs = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(ann)
    return rs.fillna(0.0).values


# ── Result builder ────────────────────────────────────────────────────────────

def _build_result(label, color, dates, values, returns, turnovers,
                  weights_h, regimes=None):
    return {
        "label":          label,
        "color":          color,
        "dates":          dates,
        "values":         np.array(values),
        "returns":        np.array(returns),
        "turnovers":      np.array(turnovers),
        "weights_h":      weights_h,
        "regimes":        np.array(regimes) if regimes is not None else None,
        "total_return":   total_return(values),
        "sharpe":         sharpe(returns),
        "sortino":        sortino(returns),
        "max_drawdown":   max_drawdown(values),
        "cvar":           cvar_metric(returns),
        "ann_vol":        ann_volatility(returns),
        "avg_turnover":   avg_turnover(turnovers),
        "dd_series":      drawdown_series(values),
        "rolling_sharpe": rolling_sharpe(returns),
    }


# ── RL agent rollout ──────────────────────────────────────────────────────────

def run_rl_episode(agent_key, cfg, df):
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    model_zip = cfg["model_path"] + ".zip"
    if not os.path.isfile(model_zip):
        print(f"  ! Model not found: {model_zip} — skipping {agent_key}")
        return None

    AlgoClass = DQN if cfg["algo"] == "DQN" else PPO
    model     = AlgoClass.load(cfg["model_path"])
    print(f"  + Loaded {cfg['algo']} from {model_zip}")

    vecnorm = None
    vn_path = cfg.get("vecnorm_path")
    if vn_path and os.path.isfile(vn_path):
        dummy   = DummyVecEnv([lambda: Monitor(
            PortfolioTradingEnv(df, safe_reward=cfg["safe_reward"], random_start=False)
        )])
        vecnorm = VecNormalize.load(vn_path, dummy)
        vecnorm.training    = False
        vecnorm.norm_reward = False

    base_env = PortfolioTradingEnv(df, safe_reward=cfg["safe_reward"], random_start=False)

    if cfg["algo"] == "DQN":
        at_path = os.path.join(MODEL_DIR, "dqn_action_table.npy")
        if not os.path.isfile(at_path):
            raise FileNotFoundError(f"Action table missing — re-run train_dqn.py")
        env = DiscretePortfolioWrapper(base_env, np.load(at_path))
    else:
        env = base_env

    obs, _  = env.reset()
    done    = False
    values  = [base_env.initial_value]
    returns = []; turnovers = []; weights_h = []; regimes = []

    while not done:
        obs_in = vecnorm.normalize_obs(obs.reshape(1, -1))[0] if vecnorm else obs
        action, _ = model.predict(obs_in, deterministic=True)
        obs, _, done, trunc, info = env.step(action)
        done = done or trunc
        values.append(info["portfolio_value"])
        returns.append(info["net_return"])
        turnovers.append(info["turnover"])
        weights_h.append(info["weights"])
        regimes.append(info.get("regime", 0))

    n     = min(len(df), len(values))
    dates = df.index[:n]
    return _build_result(cfg["label"], cfg["color"], dates,
                         np.array(values), np.array(returns),
                         np.array(turnovers), np.array(weights_h),
                         regimes=regimes[:n-1])


# ── Classic baselines ─────────────────────────────────────────────────────────

def _ret_cols(df):
    return [f"{a}_Return" for a in ASSETS if f"{a}_Return" in df.columns]

def equal_weight_strategy(df):
    cols   = _ret_cols(df)
    w      = np.ones(len(cols)) / len(cols)
    daily  = df[cols].fillna(0.0).values @ w
    values = np.cumprod(1 + daily) * 100_000.0
    values = np.insert(values, 0, 100_000.0)
    turns  = np.zeros(len(daily))
    regs   = df.get("regime", pd.Series(0, index=df.index)).values
    n      = min(len(df), len(values))
    return _build_result("Equal Weight", "#888888", df.index[:n],
                         values, daily, turns,
                         np.tile(w, (len(daily), 1)), regimes=regs)

def risk_parity_strategy(df):
    cols       = _ret_cols(df)
    ret_mat    = df[cols].fillna(0.0).values
    n_days, na = ret_mat.shape
    values     = [100_000.0]; returns = []; turnovers = []; wh = []
    regs       = df.get("regime", pd.Series(0, index=df.index)).values
    w_prev     = np.ones(na) / na

    for t in range(n_days):
        if t % 21 == 0 and t >= 60:
            vol = ret_mat[max(0,t-60):t].std(axis=0)
            vol = np.where(vol < 1e-8, 1e-8, vol)
            w   = (1.0 / vol); w = np.minimum(w / w.sum(), 0.40); w = w / w.sum()
        else:
            w = w_prev
        r    = float(ret_mat[t] @ w)
        turn = float(np.abs(w - w_prev).sum())
        nr   = r - 0.001 * turn
        values.append(values[-1] * (1 + nr))
        returns.append(nr); turnovers.append(turn); wh.append(w.tolist())
        w_prev = w.copy()

    n = min(len(df), len(values))
    return _build_result("Risk Parity", "#9467BD", df.index[:n],
                         np.array(values), np.array(returns),
                         np.array(turnovers), np.array(wh), regimes=regs)

def momentum_strategy(df):
    cols      = _ret_cols(df)
    close_c   = [c.replace("_Return","_Close") for c in cols]
    available = [c for c in close_c if c in df.columns]
    if not available:
        return equal_weight_strategy(df)
    ret_mat   = df[cols].fillna(0.0).values
    close_mat = df[available].ffill().values
    na        = len(available); n_days = len(df)
    values    = [100_000.0]; returns = []; turnovers = []; wh = []
    regs      = df.get("regime", pd.Series(0, index=df.index)).values
    w_prev    = np.ones(na) / na

    for t in range(n_days):
        if t % 21 == 0 and t >= 60:
            mom = (close_mat[t] - close_mat[max(0,t-60)]) / np.where(close_mat[max(0,t-60)] > 0, close_mat[max(0,t-60)], 1.0)
            mom = np.maximum(mom, 0.0); s = mom.sum()
            w   = mom / s if s > 1e-8 else np.ones(na) / na
            w   = np.minimum(w, 0.40); w = w / w.sum()
        else:
            w = w_prev
        r    = float(ret_mat[t, :na] @ w)
        turn = float(np.abs(w - w_prev).sum())
        nr   = r - 0.001 * turn
        values.append(values[-1] * (1 + nr))
        returns.append(nr); turnovers.append(turn); wh.append(w.tolist())
        w_prev = w.copy()

    n = min(len(df), len(values))
    return _build_result("Momentum (60d)", "#E377C2", df.index[:n],
                         np.array(values), np.array(returns),
                         np.array(turnovers), np.array(wh), regimes=regs)


# ── Regime performance breakdown (R8) ────────────────────────────────────────

def compute_regime_performance(result, df):
    """
    R8: Split test period by regime and compute per-regime metrics.

    Returns dict: {regime_int: {return, sharpe, max_drawdown}}
    """
    if result["regimes"] is None or "regime" not in df.columns:
        return {}

    returns = result["returns"]
    regimes = result["regimes"]
    n       = min(len(returns), len(regimes))
    perf    = {}

    for regime_val in [-1, 0, 1]:
        mask = regimes[:n] == regime_val
        if mask.sum() < 5:
            continue
        r_slice = returns[:n][mask]
        # Reconstruct mini value curve for drawdown
        mini_values = np.cumprod(1 + r_slice) * 100_000.0
        mini_values = np.insert(mini_values, 0, 100_000.0)
        perf[regime_val] = {
            "return":       float(r_slice.sum()),
            "sharpe":       sharpe(r_slice),
            "max_drawdown": max_drawdown(mini_values),
            "n_days":       int(mask.sum()),
        }
    return perf


def print_regime_report(all_results, df):
    """R8: Print regime performance table to console."""
    print("\n" + "=" * 65)
    print("  REGIME PERFORMANCE REPORT")
    print("=" * 65)

    for regime_val, regime_name in [(1,"Bull Market"), (0,"Neutral/Sideways"), (-1,"Bear Market")]:
        n_days = int((df.get("regime", pd.Series(0, index=df.index)) == regime_val).sum())
        print(f"\n  {regime_name}  ({n_days} test days)")
        print(f"  {'Agent':<30} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10}")
        print(f"  {'-'*60}")
        for res in all_results:
            perf = res.get("_regime_perf", {})
            if regime_val in perf:
                p = perf[regime_val]
                print(f"  {res['label']:<30} "
                      f"{p['return']*100:>+9.1f}%  "
                      f"{p['sharpe']:>9.2f}  "
                      f"{p['max_drawdown']*100:>9.1f}%")


def plot_regime_performance(all_results, df, save_path):
    """R8: Bar chart of per-regime returns for all agents."""
    regime_vals  = [-1, 0, 1]
    regime_names = ["Bear", "Neutral", "Bull"]
    agents       = [r for r in all_results if r.get("_regime_perf")]
    if not agents:
        return

    n_agents  = len(agents)
    x         = np.arange(len(regime_vals))
    width     = 0.8 / max(n_agents, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    metrics = ["return", "sharpe", "max_drawdown"]
    titles  = ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    scales  = [100, 1, 100]

    for ax, metric, title, scale in zip(axes, metrics, titles, scales):
        for j, res in enumerate(agents):
            perf   = res.get("_regime_perf", {})
            values = [perf.get(rv, {}).get(metric, 0.0) * scale
                      for rv in regime_vals]
            offset = (j - n_agents / 2 + 0.5) * width
            ax.bar(x + offset, values, width * 0.9,
                   label=res["label"], color=res["color"], alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(regime_names)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        if agents:
            ax.legend(fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_agents,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("Performance by Market Regime — Test Period (2023–2024)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


# ── Standard plots ────────────────────────────────────────────────────────────

def plot_values(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in results:
        n = min(len(r["dates"]), len(r["values"]))
        ax.plot(r["dates"][:n], r["values"][:n],
                label=r["label"], color=r["color"], linewidth=1.8)
    ax.set_title("Portfolio Value — Test Period (2023–2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved -> {save_path}")

def plot_drawdowns(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for r in results:
        n  = min(len(r["dates"]), len(r["dd_series"]))
        dd = r["dd_series"][:n] * 100
        ax.fill_between(r["dates"][:n], -dd, 0, alpha=0.2, color=r["color"])
        ax.plot(r["dates"][:n], -dd, label=r["label"], color=r["color"], linewidth=1.5)
    ax.set_title("Drawdown — Test Period (2023–2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved -> {save_path}")

def plot_rolling_sharpe(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for r in results:
        rs = r["rolling_sharpe"]; n = min(len(r["dates"]), len(rs))
        ax.plot(r["dates"][:n], rs[:n], label=r["label"], color=r["color"], linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling 60-Day Sharpe Ratio — Test Period", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved -> {save_path}")

def plot_allocation(result, save_path):
    wh = result["weights_h"]
    if len(wh) == 0: return
    wh_arr = np.array(wh)
    if wh_arr.ndim != 2: return
    dates  = result["dates"][:len(wh_arr)]
    names  = ASSETS[:wh_arr.shape[1]]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(dates, wh_arr.T * 100, labels=names, alpha=0.8)
    ax.set_title(f"Allocation Weights — {result['label']}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Allocation (%)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved -> {save_path}")

def plot_metrics_bar(metrics_df, save_path):
    cols  = ["Total Return (%)", "Sharpe Ratio", "Sortino Ratio",
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
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    plt.suptitle("Agent Performance Comparison — Test Period",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved -> {save_path}")


# ── Master evaluation ─────────────────────────────────────────────────────────

def evaluate_all(agent_keys=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  AGENT EVALUATION — TEST PERIOD (2023–2024)")
    print("=" * 65)

    # R1: attach regime features to test data
    raw_df = pd.read_csv(TEST_DATA_PATH, index_col="Date", parse_dates=True)
    df     = detect_market_regime(raw_df)
    print(f"Test data: {df.shape}  ({df.index.min().date()} -> {df.index.max().date()})")

    stats = get_regime_stats(df)
    print(f"Test regime dist — Bull: {stats.get('bull_pct',0):.1f}%  "
          f"Neutral: {stats.get('neutral_pct',0):.1f}%  "
          f"Bear: {stats.get('bear_pct',0):.1f}%\n")

    keys_to_eval = agent_keys or list(AGENT_CONFIGS.keys())
    all_results  = []
    rows         = []

    def _add(res):
        # Compute regime performance and attach to result
        res["_regime_perf"] = compute_regime_performance(res, df)
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

    print("Computing baseline strategies ...")
    _add(equal_weight_strategy(df))
    _add(risk_parity_strategy(df))
    _add(momentum_strategy(df))

    print("\nEvaluating RL agents ...")
    for key in keys_to_eval:
        if key not in AGENT_CONFIGS:
            continue
        res = run_rl_episode(key, AGENT_CONFIGS[key], df)
        if res:
            _add(res)
            plot_allocation(res, os.path.join(RESULTS_DIR, f"allocation_{key}.png"))

    metrics_df = pd.DataFrame(rows)
    print("\n" + "─" * 80)
    print(metrics_df.to_string(index=False))
    print("─" * 80)

    csv_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved -> {csv_path}")

    # R8: regime report
    print_regime_report(all_results, df)

    # Save regime report CSV
    regime_rows = []
    for res in all_results:
        for rv, rn in REGIME_NAMES.items():
            p = res.get("_regime_perf", {}).get(rv, {})
            if p:
                regime_rows.append({
                    "Agent":   res["label"],
                    "Regime":  rn,
                    "Return (%)":     round(p["return"] * 100, 2),
                    "Sharpe":         round(p["sharpe"],       3),
                    "Max Drawdown (%)": round(p["max_drawdown"] * 100, 2),
                    "N Days":         p["n_days"],
                })
    if regime_rows:
        rdf = pd.DataFrame(regime_rows)
        rdf.to_csv(os.path.join(RESULTS_DIR, "regime_performance.csv"), index=False)
        print(f"Regime CSV saved -> {os.path.join(RESULTS_DIR, 'regime_performance.csv')}")

    # Plots
    plot_values(all_results,    os.path.join(RESULTS_DIR, "portfolio_value_curves.png"))
    plot_drawdowns(all_results, os.path.join(RESULTS_DIR, "drawdown_curves.png"))
    plot_rolling_sharpe(all_results, os.path.join(RESULTS_DIR, "rolling_sharpe.png"))
    if len(rows) >= 2:
        plot_metrics_bar(metrics_df, os.path.join(RESULTS_DIR, "metrics_comparison.png"))
    plot_regime_performance(
        all_results, df,
        os.path.join(RESULTS_DIR, "regime_performance.png")
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="*", default=None,
                        choices=list(AGENT_CONFIGS.keys()))
    args = parser.parse_args()
    evaluate_all(agent_keys=args.agents)