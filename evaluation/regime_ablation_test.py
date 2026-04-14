"""
regime_ablation_test.py
========================
Regime Ablation Study for the Safe RL Portfolio Management project.

PURPOSE
-------
Evaluate whether trained RL agents (DQN, PPO, Safe PPO) actually USE regime
information when making portfolio allocation decisions.

METHOD
------
1. Run each agent on the normal test dataset  (regime features intact).
2. Run each agent again on a zeroed-out copy  (all regime columns set to 0.0).
3. Compare Total Return and Sharpe Ratio between the two runs.
4. A large drop in performance when regime is removed  →  agent uses regimes.
5. No drop  →  agent ignores regime features.

USAGE
-----
    python evaluation/regime_ablation_test.py

OUTPUTS
-------
    results/regime_ablation_results.csv
    results/regime_ablation_comparison.png

Part of: Safe RL for Risk-Constrained Portfolio Management
"""

import os
import sys
import logging
import warnings
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
# Works whether called from repo root or from inside evaluation/
_THIS_DIR    = Path(__file__).resolve().parent          # .../evaluation/
PROJECT_ROOT = _THIS_DIR.parent                         # .../project/

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "env"))
sys.path.insert(0, str(PROJECT_ROOT / "environment"))   # alternate folder name
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Path constants ─────────────────────────────────────────────────────────────
DATA_PATH       = PROJECT_ROOT / "data"  / "test_dataset.csv"
MODEL_DIR       = PROJECT_ROOT / "models"
RESULTS_DIR     = PROJECT_ROOT / "results"
EVAL_AGENT_PATH = _THIS_DIR   / "evaluate_agent.py"

MODEL_FILES = {
    "DQN":      MODEL_DIR / "dqn_portfolio.zip",
    "PPO":      MODEL_DIR / "ppo_portfolio.zip",
    "Safe PPO": MODEL_DIR / "safe_ppo_portfolio.zip",
}
VECNORM_FILES = {
    "DQN":      MODEL_DIR / "dqn_portfolio_vecnorm.pkl",
    "PPO":      MODEL_DIR / "ppo_portfolio_vecnorm.pkl",
    "Safe PPO": MODEL_DIR / "safe_ppo_portfolio_vecnorm.pkl",
}

# ── Regime columns to zero-out in ablation ────────────────────────────────────
# Any column whose name contains one of these sub-strings will be zeroed.
REGIME_KEYWORDS = [
    "regime",
    "trend_strength",
    "volatility",
    "bull_prob",
    "neutral_prob",
    "bear_prob",
]

# Return columns must NOT be zeroed — they are price data, not regime signals
RETURN_KEYWORD_EXCEPTIONS = ["_Return", "_return"]

# Annualised trading days (for Sharpe/Sortino scaling)
TRADING_DAYS = 252

# ── Interpretation thresholds ─────────────────────────────────────────────────
STRONG_DEPENDENCY_THRESHOLD = 0.10   # 10 % absolute return drop
WEAK_DEPENDENCY_THRESHOLD   = 0.02   # 2 %  — below this = "not using regime"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Optional import from evaluate_agent.py
# ══════════════════════════════════════════════════════════════════════════════

def _try_import_evaluate_agent():
    """
    Attempt to import helper functions from evaluate_agent.py.
    Returns the module if successful, None otherwise.
    All downstream code falls back to local implementations if this fails.
    """
    if not EVAL_AGENT_PATH.exists():
        logger.warning(
            "evaluate_agent.py not found at %s — using built-in metrics.",
            EVAL_AGENT_PATH,
        )
        return None
    try:
        spec   = importlib.util.spec_from_file_location("evaluate_agent", EVAL_AGENT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info("Imported evaluate_agent.py successfully.")
        return module
    except Exception as exc:
        logger.warning(
            "Could not import evaluate_agent.py (%s) — using built-in metrics.", exc
        )
        return None


_eval_module = _try_import_evaluate_agent()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Performance metrics (built-in, always available)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics_local(portfolio_values: list, initial_value: float) -> dict:
    """
    Compute standard portfolio performance metrics from a list of daily
    portfolio values.

    Metrics
    -------
    total_return  : (final − initial) / initial
    sharpe        : annualised Sharpe ratio (rf = 0)
    sortino       : annualised Sortino ratio (rf = 0, downside only)
    max_drawdown  : maximum peak-to-trough drawdown fraction
    volatility    : annualised daily-return std dev
    """
    values = np.array(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return {
            "total_return": 0.0,
            "sharpe":       0.0,
            "sortino":      0.0,
            "max_drawdown": 0.0,
            "volatility":   0.0,
        }

    daily_rets = np.diff(values) / np.maximum(values[:-1], 1e-9)

    total_return = (values[-1] - initial_value) / initial_value

    vol = float(np.std(daily_rets)) * np.sqrt(TRADING_DAYS)

    mean_ret = float(np.mean(daily_rets)) * TRADING_DAYS
    sharpe   = (mean_ret / vol) if vol > 1e-10 else 0.0

    downside = daily_rets[daily_rets < 0]
    down_vol = float(np.std(downside)) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else 1e-10
    sortino  = (mean_ret / down_vol) if down_vol > 1e-10 else 0.0

    running_max = np.maximum.accumulate(values)
    drawdowns   = (running_max - values) / np.maximum(running_max, 1e-9)
    max_dd      = float(np.max(drawdowns))

    return {
        "total_return": float(total_return),
        "sharpe":       float(sharpe),
        "sortino":      float(sortino),
        "max_drawdown": float(max_dd),
        "volatility":   float(vol),
    }


def compute_metrics(portfolio_values: list, initial_value: float) -> dict:
    """
    Compute performance metrics.
    Delegates to evaluate_agent.py if available, else uses local implementation.
    """
    if _eval_module is not None:
        for fn_name in ("compute_metrics", "calculate_metrics", "get_metrics"):
            fn = getattr(_eval_module, fn_name, None)
            if callable(fn):
                try:
                    return fn(portfolio_values, initial_value)
                except Exception:
                    pass
    return _compute_metrics_local(portfolio_values, initial_value)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Model loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(agent_name: str):
    """
    Load a trained SB3 model from disk.

    Returns
    -------
    model object  or  None if the file does not exist / loading fails.
    """
    model_path = MODEL_FILES[agent_name]
    if not model_path.exists():
        logger.warning(
            "[%s] Model file not found: %s  →  skipping agent.", agent_name, model_path
        )
        return None

    try:
        if agent_name == "DQN":
            from stable_baselines3 import DQN
            model = DQN.load(str(model_path))
        else:
            from stable_baselines3 import PPO
            model = PPO.load(str(model_path))
        logger.info("[%s] Model loaded from %s", agent_name, model_path)
        return model
    except Exception as exc:
        logger.error("[%s] Failed to load model: %s", agent_name, exc)
        return None


def _load_vecnorm(agent_name: str):
    """
    Load a VecNormalize statistics file if available.

    Returns
    -------
    VecNormalize object  or  None.
    """
    vn_path = VECNORM_FILES.get(agent_name)
    if vn_path is None or not vn_path.exists():
        return None
    try:
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        # Create a dummy env just to load the stats — obs space must match
        # We restore it later after env creation per run
        return str(vn_path)   # return the path; load per-episode inside runner
    except Exception:
        return None


def load_all_models() -> dict:
    """
    Load all three agent models.

    Returns
    -------
    dict  {agent_name: model}  — missing models are omitted.
    """
    models = {}
    print("\n" + "=" * 60)
    print("  LOADING TRAINED MODELS")
    print("=" * 60)
    for name in MODEL_FILES:
        m = _load_model(name)
        if m is not None:
            models[name] = m
            print(f"  ✓ {name:<12}  loaded")
        else:
            print(f"  ✗ {name:<12}  NOT FOUND — will be skipped")
    return models


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def load_test_dataset() -> pd.DataFrame:
    """
    Load the test CSV and print basic info.

    Returns
    -------
    pd.DataFrame with DatetimeIndex.
    """
    print("\n" + "=" * 60)
    print("  LOADING TEST DATASET")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found at: {DATA_PATH}\n"
            f"Run data_pipeline/build_dataset.py first."
        )

    df = pd.read_csv(str(DATA_PATH), index_col="Date", parse_dates=True)

    print(f"  Path       : {DATA_PATH}")
    print(f"  Shape      : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Start date : {df.index.min().date()}")
    print(f"  End date   : {df.index.max().date()}")
    print(f"  Date range : {(df.index.max() - df.index.min()).days} calendar days")
    print(f"  NaN rate   : {df.isna().mean().mean() * 100:.2f}%")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Ablation dataset construction
# ══════════════════════════════════════════════════════════════════════════════

def _is_regime_column(col: str) -> bool:
    """
    Return True if a column name contains a regime-related keyword AND
    is not a price-return column (which must remain intact).
    """
    col_lower = col.lower()

    # Never zero out plain return columns — they are price data
    for exc in RETURN_KEYWORD_EXCEPTIONS:
        if exc.lower() in col_lower:
            return False

    # Check for regime keywords
    for kw in REGIME_KEYWORDS:
        if kw.lower() in col_lower:
            return True

    return False


def create_ablation_dataset(df: pd.DataFrame) -> tuple:
    """
    Create a regime-ablated copy of the DataFrame.

    All columns whose names contain regime-related keywords are set to 0.0.
    Column structure is preserved — no columns are dropped.

    Returns
    -------
    (df_no_regime, ablated_columns)
        df_no_regime   : pd.DataFrame — copy with regime cols zeroed.
        ablated_columns: list[str]    — names of zeroed columns.
    """
    df_no_regime     = df.copy()
    ablated_columns  = []

    for col in df_no_regime.columns:
        if _is_regime_column(col):
            df_no_regime[col] = 0.0
            ablated_columns.append(col)

    print("\n" + "=" * 60)
    print("  REGIME ABLATION DATASET")
    print("=" * 60)
    print(f"  Total columns        : {len(df.columns)}")
    print(f"  Columns zeroed       : {len(ablated_columns)}")
    print(f"  Columns preserved    : {len(df.columns) - len(ablated_columns)}")
    print(f"  Dataset shape        : unchanged — {df_no_regime.shape}")

    if ablated_columns:
        # Group by category for a cleaner print
        per_asset_cols = [c for c in ablated_columns if any(
            c.startswith(a + "_") for a in
            ["BTC", "ETH", "SPY", "GLD", "Silver", "Nifty", "Sensex"]
        )]
        global_cols = [c for c in ablated_columns if c not in per_asset_cols]

        if global_cols:
            print(f"\n  Global regime cols zeroed ({len(global_cols)}):")
            for c in global_cols:
                print(f"    - {c}")
        if per_asset_cols:
            print(f"\n  Per-asset regime cols zeroed ({len(per_asset_cols)}):")
            # Print at most 10 as sample
            for c in per_asset_cols[:10]:
                print(f"    - {c}")
            if len(per_asset_cols) > 10:
                print(f"    ... and {len(per_asset_cols) - 10} more")
    else:
        print("\n  WARNING: No regime columns detected in dataset!")
        print("  The dataset may not have regime features yet.")
        print("  Run data_pipeline/build_dataset.py first.")

    return df_no_regime, ablated_columns


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 & 6 — Evaluation engine
# ══════════════════════════════════════════════════════════════════════════════

def _run_episode(
    model,
    df: pd.DataFrame,
    agent_name: str,
    vecnorm_path: str = None,
) -> dict:
    """
    Run one deterministic episode of agent on the given DataFrame.

    Loads PortfolioTradingEnv, steps through the full dataset from the first
    row to the last, and collects portfolio values at every step.

    Parameters
    ----------
    model        : SB3 model (DQN or PPO).
    df           : DataFrame to use as the environment data.
    agent_name   : Name string for logging.
    vecnorm_path : Path to VecNormalize pkl file (optional).

    Returns
    -------
    dict with performance metrics.
    """
    # Import env — try both common folder names
    PortfolioTradingEnv = None
    for mod_path in [
        "env.trading_environment",
        "environment.trading_environment",
        "trading_environment",
    ]:
        try:
            mod = __import__(mod_path, fromlist=["PortfolioTradingEnv"])
            PortfolioTradingEnv = mod.PortfolioTradingEnv
            break
        except ImportError:
            continue

    if PortfolioTradingEnv is None:
        # Last-resort: direct file import
        env_candidates = [
            PROJECT_ROOT / "env"         / "trading_environment.py",
            PROJECT_ROOT / "environment" / "trading_environment.py",
            PROJECT_ROOT / "trading_environment.py",
        ]
        for path in env_candidates:
            if path.exists():
                spec   = importlib.util.spec_from_file_location(
                    "trading_environment", path
                )
                _mod   = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_mod)
                PortfolioTradingEnv = _mod.PortfolioTradingEnv
                break

    if PortfolioTradingEnv is None:
        raise ImportError(
            "Cannot find PortfolioTradingEnv.  "
            "Ensure trading_environment.py is in env/ or environment/."
        )

    # Build env — safe_reward=False so no extra CVaR penalties during eval
    env = PortfolioTradingEnv(df, safe_reward=False, random_start=False)

    # Optionally load VecNormalize stats for obs normalisation
    vecnorm = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        try:
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            from stable_baselines3.common.monitor import Monitor
            dummy_fn = lambda: Monitor(
                PortfolioTradingEnv(df, safe_reward=False, random_start=False)
            )
            dummy_vec = DummyVecEnv([dummy_fn])
            vecnorm   = VecNormalize.load(vecnorm_path, dummy_vec)
            vecnorm.training    = False
            vecnorm.norm_reward = False
        except Exception as exc:
            logger.debug(
                "[%s] VecNormalize not loaded: %s — using raw obs.", agent_name, exc
            )
            vecnorm = None

    obs, _           = env.reset()
    done             = False
    portfolio_values = [env.portfolio_value]

    while not done:
        # Apply obs normalisation if VecNormalize is available
        if vecnorm is not None:
            obs_in = vecnorm.normalize_obs(obs.reshape(1, -1))[0]
        else:
            obs_in = obs

        action, _ = model.predict(obs_in, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
        done = done or trunc

    metrics = compute_metrics(portfolio_values, env.initial_value)
    return metrics


def evaluate_agents(
    models: dict,
    df: pd.DataFrame,
    label: str = "NORMAL",
) -> dict:
    """
    Evaluate all loaded agents on a given DataFrame.

    Parameters
    ----------
    models  : {agent_name: model}
    df      : DataFrame to evaluate on (normal or ablated).
    label   : String label printed during evaluation.

    Returns
    -------
    dict  {agent_name: metrics_dict}
    """
    results = {}

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION — {label}")
    print(f"{'=' * 60}")

    for agent_name, model in models.items():
        print(f"\n  Running {agent_name} on {label} dataset …", flush=True)
        try:
            vn_path = str(VECNORM_FILES.get(agent_name, ""))
            metrics = _run_episode(model, df, agent_name, vecnorm_path=vn_path)
            results[agent_name] = metrics
            print(f"    Total Return : {metrics['total_return'] * 100:+.2f}%")
            print(f"    Sharpe Ratio : {metrics['sharpe']:.4f}")
            print(f"    Sortino      : {metrics['sortino']:.4f}")
            print(f"    Max Drawdown : {metrics['max_drawdown'] * 100:.2f}%")
            print(f"    Volatility   : {metrics['volatility'] * 100:.2f}%")
        except Exception as exc:
            logger.error("[%s] Evaluation failed: %s", agent_name, exc, exc_info=True)
            results[agent_name] = {
                "total_return": np.nan,
                "sharpe":       np.nan,
                "sortino":      np.nan,
                "max_drawdown": np.nan,
                "volatility":   np.nan,
            }
            print(f"    ERROR: {exc}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(
    results_normal: dict,
    results_no_regime: dict,
) -> pd.DataFrame:
    """
    Build a comparison DataFrame with normal vs. ablated performance and deltas.

    Columns
    -------
    Agent, Return_Normal, Return_NoRegime, Return_Drop,
    Sharpe_Normal, Sharpe_NoRegime, Sharpe_Drop,
    Sortino_Normal, Sortino_NoRegime,
    MaxDD_Normal, MaxDD_NoRegime,
    Vol_Normal, Vol_NoRegime
    """
    rows = []
    all_agents = sorted(
        set(results_normal.keys()) | set(results_no_regime.keys())
    )

    for agent in all_agents:
        norm = results_normal.get(agent, {})
        nreg = results_no_regime.get(agent, {})

        r_norm  = norm.get("total_return", np.nan)
        r_nreg  = nreg.get("total_return", np.nan)
        sh_norm = norm.get("sharpe",       np.nan)
        sh_nreg = nreg.get("sharpe",       np.nan)

        rows.append({
            "Agent":             agent,
            "Return_Normal":     r_norm,
            "Return_NoRegime":   r_nreg,
            "Return_Drop":       r_norm - r_nreg  if not (np.isnan(r_norm) or np.isnan(r_nreg))  else np.nan,
            "Sharpe_Normal":     sh_norm,
            "Sharpe_NoRegime":   sh_nreg,
            "Sharpe_Drop":       sh_norm - sh_nreg if not (np.isnan(sh_norm) or np.isnan(sh_nreg)) else np.nan,
            "Sortino_Normal":    norm.get("sortino",      np.nan),
            "Sortino_NoRegime":  nreg.get("sortino",      np.nan),
            "MaxDD_Normal":      norm.get("max_drawdown", np.nan),
            "MaxDD_NoRegime":    nreg.get("max_drawdown", np.nan),
            "Vol_Normal":        norm.get("volatility",   np.nan),
            "Vol_NoRegime":      nreg.get("volatility",   np.nan),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Interpretation
# ══════════════════════════════════════════════════════════════════════════════

def interpret_results(table: pd.DataFrame) -> None:
    """
    Print a plain-language interpretation of each agent's regime dependency.

    Rules
    -----
    Return_Drop > STRONG_DEPENDENCY_THRESHOLD (10%)
        → "Agent strongly depends on regime signals."
    WEAK_DEPENDENCY_THRESHOLD ≤ Return_Drop ≤ STRONG_DEPENDENCY_THRESHOLD
        → "Agent shows moderate regime dependency."
    Return_Drop < WEAK_DEPENDENCY_THRESHOLD  (2%)
        → "Agent is not using regime information."
    Negative Return_Drop  (ablated > normal)
        → "Ablation improved performance — regime features may be hurting this agent."
    """
    print("\n" + "=" * 60)
    print("  REGIME DEPENDENCY INTERPRETATION")
    print("=" * 60)
    print(
        f"  Thresholds:  Strong >{STRONG_DEPENDENCY_THRESHOLD*100:.0f}%  |  "
        f"Weak <{WEAK_DEPENDENCY_THRESHOLD*100:.0f}%  (absolute return drop)"
    )

    for _, row in table.iterrows():
        agent = row["Agent"]
        drop  = row["Return_Drop"]

        print(f"\n  [{agent}]")
        print(f"    Return (normal)    : {row['Return_Normal'] * 100:+.2f}%")
        print(f"    Return (no regime) : {row['Return_NoRegime'] * 100:+.2f}%")
        print(f"    Return drop        : {drop * 100:+.2f}%")
        print(f"    Sharpe drop        : {row['Sharpe_Drop']:+.4f}")

        if np.isnan(drop):
            verdict = "⚠  Could not evaluate (NaN — check model/data)."
        elif drop < 0:
            verdict = (
                "⚠  Ablation IMPROVED performance. "
                "Regime features may be hurting this agent — review reward shaping."
            )
        elif drop < WEAK_DEPENDENCY_THRESHOLD:
            verdict = (
                "✗  Agent is NOT using regime information. "
                "Consider increasing regime signal weight in observation or reward."
            )
        elif drop < STRONG_DEPENDENCY_THRESHOLD:
            verdict = (
                "~  Agent shows MODERATE regime dependency. "
                "Some regime signals are being used but not fully exploited."
            )
        else:
            verdict = (
                "✓  Agent STRONGLY depends on regime signals. "
                "Regime features are meaningfully driving allocation decisions."
            )

        print(f"    Verdict: {verdict}")

    # Overall summary
    valid_drops = table["Return_Drop"].dropna()
    if len(valid_drops) > 0:
        avg_drop = valid_drops.mean()
        print(f"\n  Average return drop across all agents: {avg_drop * 100:+.2f}%")
        if avg_drop > STRONG_DEPENDENCY_THRESHOLD:
            print("  OVERALL: The regime-aware system is meaningfully better — regime features are working.")
        elif avg_drop < WEAK_DEPENDENCY_THRESHOLD:
            print("  OVERALL: Agents are largely ignoring regime features — investigate observation or training.")
        else:
            print("  OVERALL: Partial regime usage detected — the system can be improved further.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Save results
# ══════════════════════════════════════════════════════════════════════════════

def save_results(table: pd.DataFrame) -> None:
    """Save comparison table to CSV and print it to terminal."""
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    csv_path = RESULTS_DIR / "regime_ablation_results.csv"
    table.to_csv(str(csv_path), index=False, float_format="%.6f")
    print(f"\n  Results saved → {csv_path}")

    # Pretty terminal print
    print("\n" + "=" * 60)
    print("  REGIME ABLATION COMPARISON TABLE")
    print("=" * 60)

    # Format percentages for readability
    display = table.copy()
    pct_cols = [
        "Return_Normal", "Return_NoRegime", "Return_Drop",
        "MaxDD_Normal",  "MaxDD_NoRegime",
        "Vol_Normal",    "Vol_NoRegime",
    ]
    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x*100:+.2f}%" if not np.isnan(x) else "N/A"
            )
    float_cols = [
        "Sharpe_Normal", "Sharpe_NoRegime", "Sharpe_Drop",
        "Sortino_Normal", "Sortino_NoRegime",
    ]
    for col in float_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
            )

    print(display.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(table: pd.DataFrame) -> None:
    """
    Save a grouped bar chart comparing Return_Normal vs Return_NoRegime
    for each agent, plus a secondary panel for Sharpe drop.
    """
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    save_path = RESULTS_DIR / "regime_ablation_comparison.png"

    agents         = table["Agent"].tolist()
    return_normal  = table["Return_Normal"].fillna(0).tolist()
    return_no_reg  = table["Return_NoRegime"].fillna(0).tolist()
    return_drop    = table["Return_Drop"].fillna(0).tolist()
    sharpe_drop    = table["Sharpe_Drop"].fillna(0).tolist()

    n      = len(agents)
    x      = np.arange(n)
    width  = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Regime Ablation Study — RL Portfolio Agents\n"
        "Comparing performance with and without regime information",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # ── Panel 1: Total Return comparison ──────────────────────────────────────
    ax1 = axes[0]
    bars_n = ax1.bar(
        x - width / 2, [r * 100 for r in return_normal],
        width, label="With Regime", color="#2196F3", alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )
    bars_a = ax1.bar(
        x + width / 2, [r * 100 for r in return_no_reg],
        width, label="No Regime (Ablated)", color="#FF5722", alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )

    ax1.set_title("Total Return (%)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, fontsize=10)
    ax1.set_ylabel("Return (%)")
    ax1.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars_n:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )
    for bar in bars_a:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    # ── Panel 2: Return Drop ──────────────────────────────────────────────────
    ax2 = axes[1]
    colors_drop = [
        "#4CAF50" if d > STRONG_DEPENDENCY_THRESHOLD
        else "#FFC107" if d > WEAK_DEPENDENCY_THRESHOLD
        else "#9E9E9E"
        for d in return_drop
    ]
    bars_d = ax2.bar(
        x, [d * 100 for d in return_drop],
        width * 1.5, color=colors_drop, alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )
    ax2.set_title("Return Drop When Regime Removed\n(Normal − Ablated)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents, fontsize=10)
    ax2.set_ylabel("Return Drop (%)")
    ax2.axhline(0,                                     color="black",  linewidth=0.7, linestyle="--", alpha=0.4)
    ax2.axhline(STRONG_DEPENDENCY_THRESHOLD * 100,     color="#4CAF50", linewidth=1.2, linestyle=":",  alpha=0.8, label=f"Strong threshold ({STRONG_DEPENDENCY_THRESHOLD*100:.0f}%)")
    ax2.axhline(WEAK_DEPENDENCY_THRESHOLD   * 100,     color="#9E9E9E", linewidth=1.2, linestyle=":",  alpha=0.8, label=f"Weak threshold ({WEAK_DEPENDENCY_THRESHOLD*100:.0f}%)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars_d:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # ── Panel 3: Sharpe Ratio comparison ──────────────────────────────────────
    ax3 = axes[2]
    sharpe_normal  = table["Sharpe_Normal"].fillna(0).tolist()
    sharpe_no_reg  = table["Sharpe_NoRegime"].fillna(0).tolist()
    ax3.bar(
        x - width / 2, sharpe_normal,
        width, label="With Regime", color="#2196F3", alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )
    ax3.bar(
        x + width / 2, sharpe_no_reg,
        width, label="No Regime (Ablated)", color="#FF5722", alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )
    ax3.set_title("Sharpe Ratio Comparison", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(agents, fontsize=10)
    ax3.set_ylabel("Sharpe Ratio")
    ax3.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    # ── Colour legend note ─────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#4CAF50", label=f"Strong regime dependency (drop >{STRONG_DEPENDENCY_THRESHOLD*100:.0f}%)"),
        Patch(color="#FFC107", label="Moderate regime dependency"),
        Patch(color="#9E9E9E", label="Weak / no regime dependency"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved   → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — orchestrate the full ablation test
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation_test() -> pd.DataFrame:
    """
    Full regime ablation pipeline.

    Steps
    -----
    1. Load test dataset.
    2. Load trained models (DQN, PPO, Safe PPO).
    3. Evaluate on normal dataset   → results_normal.
    4. Build ablated dataset        → df_no_regime.
    5. Evaluate on ablated dataset  → results_no_regime.
    6. Build comparison table.
    7. Print interpretation.
    8. Save CSV.
    9. Save chart.

    Returns
    -------
    comparison DataFrame.
    """
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    REGIME ABLATION TEST — Safe RL Portfolio Management   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    df = load_test_dataset()

    # ── Step 2: Load models ────────────────────────────────────────────────────
    models = load_all_models()
    if not models:
        print("\n  ERROR: No models could be loaded.")
        print("  Train agents first using:")
        print("    python agents/train_ppo.py")
        print("    python agents/train_safe_ppo.py")
        print("    python agents/train_dqn.py")
        return pd.DataFrame()

    # ── Step 3: Normal evaluation ──────────────────────────────────────────────
    results_normal = evaluate_agents(models, df, label="NORMAL (Regime Intact)")

    # ── Step 4: Build ablated dataset ─────────────────────────────────────────
    df_no_regime, ablated_cols = create_ablation_dataset(df)

    if not ablated_cols:
        print(
            "\n  WARNING: No regime columns were found to ablate.\n"
            "  The ablation test cannot distinguish regime vs. no-regime.\n"
            "  Rebuild the dataset with regime features via build_dataset.py.\n"
        )

    # ── Step 5: Ablated evaluation ─────────────────────────────────────────────
    results_no_regime = evaluate_agents(
        models, df_no_regime, label="ABLATED (Regime Zeroed)"
    )

    # ── Step 6: Comparison table ───────────────────────────────────────────────
    table = build_comparison_table(results_normal, results_no_regime)

    # ── Step 7: Interpretation ─────────────────────────────────────────────────
    interpret_results(table)

    # ── Step 8: Save CSV ───────────────────────────────────────────────────────
    save_results(table)

    # ── Step 9: Save chart ─────────────────────────────────────────────────────
    try:
        plot_comparison(table)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

    print("\n" + "=" * 60)
    print("  ABLATION TEST COMPLETE")
    print(f"  CSV    : {RESULTS_DIR / 'regime_ablation_results.csv'}")
    print(f"  Chart  : {RESULTS_DIR / 'regime_ablation_comparison.png'}")
    print("=" * 60 + "\n")

    return table


if __name__ == "__main__":
    run_ablation_test()