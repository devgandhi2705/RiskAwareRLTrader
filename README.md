# RiskAwareRLTrader

**Safe Reinforcement Learning for Risk-Constrained Multi-Asset Portfolio Management**

A research-grade RL system that trains three agents — DQN, PPO, and a regime-aware Safe PPO — to manage a 7-asset global portfolio while respecting explicit risk constraints (drawdown, CVaR, volatility). The key contribution is a **market-regime-conditional penalty framework** that dynamically tightens risk controls during bear markets and relaxes them during bull markets.

---

## Results (Out-of-Sample, 2023–2025)

| Agent | Final Value | Total Return | Sharpe | Sortino | Max Drawdown |
|---|---|---|---|---|---|
| DQN (Baseline) | $129,890 | +29.89% | 1.024 | 1.664 | 16.73% |
| PPO (Baseline) | $149,547 | +49.55% | 1.721 | 2.800 | 11.59% |
| Safe PPO (Risk-Aware) | $130,332 | +30.33% | 1.440 | 2.426 | 12.04% |

Initial capital: $100,000. Safe PPO trades ~4% of absolute return for a ~4.7pp reduction in maximum drawdown vs. DQN.

---

## Domain Context

### The Portfolio Management Problem

At each trading day, an agent observes market conditions and decides how to allocate capital across assets. The goal is to maximize risk-adjusted returns while avoiding catastrophic losses. Classical strategies (equal-weight, risk parity, momentum) are rule-based and regime-agnostic. RL agents can learn dynamic, non-linear allocation policies directly from data.

### Why Safe RL?

Standard RL maximizes cumulative reward without safety guarantees. In finance this leads to agents that take excessive concentration risk during training and blow up on drawdowns in production. Safe RL adds explicit penalty terms for undesirable risk behaviors — here: current drawdown exceeding 20%, CVaR of recent returns below −5%, and portfolio volatility spikes. The penalties are scaled by a **regime multiplier** so the agent is aggressively penalized in bear markets but is allowed to run more exposure in bull markets.

### Market Regime Detection

Regime is derived from BTC price (used as a global risk-on/off proxy):
- **Bull (1):** price > MA200 and trend strength > +2%
- **Bear (−1):** price < MA200 and trend strength < −2%
- **Neutral (0):** everything else

Trend strength = (MA50 − MA200) / MA200. Market volatility = 30-day rolling std of BTC returns. Both are added to the observation vector so agents can condition their behavior on market state.

---

## Asset Universe

| Asset | Class | Ticker Source |
|---|---|---|
| BTC | Crypto | Bitcoin/USD |
| ETH | Crypto | Ethereum/USD |
| SPY | Equity | S&P 500 ETF |
| GLD | Commodity | Gold ETF |
| Silver | Commodity | Silver spot |
| Nifty50 | Equity | Indian large-cap index |
| Sensex | Equity | Indian BSE index |

The mix of crypto, US equities, gold, and Indian equities creates a cross-asset, cross-geography portfolio with varied risk profiles and low/negative correlations.

---

## Technical Architecture

### Environment: `PortfolioTradingEnv`

A custom [Gymnasium](https://gymnasium.farama.org/) environment wrapping the daily portfolio MDP.

**Observation space** — `Box(−∞, +∞, shape=(73,))`:

```
[ BTC_Close, BTC_Return, BTC_RSI, BTC_MA20, BTC_Volatility,  ← 5 features × 7 assets = 35
  ETH_Close, ETH_Return, ...                                   ↑
  ...                                                          |
  portfolio_weights (7)                                        |
  norm_portfolio_value (1)                                     = 46 market + portfolio
  market_regime (1)                                            |
  trend_strength (1)                                           + 3 regime features
  market_volatility (1) ]                                      = 49 → padded to 73
```

**Action space**:
- PPO / Safe PPO: `Box(0, 1, shape=(7,))` — continuous weights, clipped + renormalized + max-position-capped at 40%
- DQN: `Discrete(240)` — index into a pre-built action table of Dirichlet-sampled portfolio vectors

**Key environment mechanics**:
- Transaction cost: 0.1% per unit turnover
- Rebalancing gate: weights only updated every 5 steps
- Action smoothing: 30% new action, 70% carry-forward (reduces churn)
- Equal-weight initialization at episode start
- Randomized episode start during training (prevents overfitting to a fixed start date)
- Minimum episode length: 252 steps (≈ 1 trading year)

### Reward Functions

**Base reward (DQN and PPO)**:
```
r = net_return − λ_turn × turnover
```

**Safe reward (Safe PPO)**:
```
r = net_return
    + exposure_bonus          # +5% × |weights|  during bull regime
    − λ_turn × turnover
    − scale × (dd_penalty + cvar_penalty + vol_penalty)
    − 0.05 × tail_cvar
```

Where:
- `dd_penalty = λ_dd × max(0, drawdown − 0.20)` — kicks in only above 20% drawdown
- `cvar_penalty = λ_cvar × max(0, −(CVaR + 0.05))` — kicks in only when tail losses exceed −5%
- `vol_penalty = λ_vol × rolling_portfolio_std`
- `scale = min(1, step / 200k) × risk_penalty_scale × regime_multiplier` — warm-up over 200k steps
- `regime_multiplier`: Bull=0.1, Neutral=0.5, Bear=1.0

All rewards are multiplied by 100 to improve gradient magnitude.

### Agents

**DQN (Baseline)**
- Network: `[256, 256, 128]` Q-network
- Buffer: 50,000 transitions, epsilon-greedy (50% → 2%)
- Discrete action space via `DiscretePortfolioWrapper`
- 400,000 training timesteps

**PPO (Baseline)**
- Network: `pi=[256, 256, 128]`, `vf=[256, 256, 128]`
- `lr=1e-4`, `n_steps=2048`, `batch=64`, `n_epochs=10`, `ent_coef=0.02`, `clip=0.1`
- VecNormalize: observation normalization ON, reward normalization OFF
- 500,000 total timesteps across walk-forward windows + full fine-tune

**Safe PPO (Risk-Aware)**
- Same architecture as PPO
- Regime-conditional safe reward with 200k-step warm-up
- Separate VecNormalize state (`safe_ppo_portfolio_vecnorm.pkl`)

### Training Pipeline

```
Walk-forward validation (5 windows):
  Window 1: train 2016–2017 → validate 2018
  Window 2: train 2017–2018 → validate 2019
  Window 3: train 2018–2019 → validate 2020
  Window 4: train 2019–2020 → validate 2021
  Window 5: train 2020–2021 → validate 2022
  → select best validation checkpoint
  → fine-tune on full 2016–2022 dataset

Test: 2023–2025 (out-of-sample, never seen during training)
```

Walk-forward prevents look-ahead bias and gives a more realistic estimate of out-of-sample performance than a single train/test split.

### Data Pipeline

```
data_pipeline/
  download_data.py      — fetch OHLCV from sources
  feature_engineering.py — compute RSI, MA20, rolling volatility
  build_dataset.py      — merge assets, align dates, produce train/test CSVs
  regime_detection.py   — attach regime, trend_strength, market_volatility columns
```

Regime features are always injected **after** loading and **before** environment creation, so every training window and the test set all receive consistent regime labels.

---

## Project Structure

```
project/
├── agents/
│   ├── train_dqn.py          # DQN training + DiscretePortfolioWrapper
│   ├── train_ppo.py          # PPO / Safe PPO training with walk-forward
│   └── train_safe_ppo.py     # Thin wrapper: train_ppo(safe_reward=True)
├── data_pipeline/
│   ├── build_dataset.py
│   ├── download_data.py
│   ├── feature_engineering.py
│   └── regime_detection.py
├── env/
│   └── trading_environment.py  # PortfolioTradingEnv (Gymnasium)
├── evaluation/
│   └── evaluate_agent.py       # Full evaluation suite + baselines + plots
├── models/                     # Saved model artifacts (gitignored)
│   ├── dqn_portfolio.zip
│   ├── dqn_action_table.npy
│   ├── ppo_portfolio.zip
│   ├── ppo_portfolio_vecnorm.pkl
│   ├── safe_ppo_portfolio.zip
│   └── safe_ppo_portfolio_vecnorm.pkl
├── data/                       # CSV datasets (gitignored)
│   ├── train_dataset.csv       # 2016–2022
│   └── test_dataset.csv        # 2023–2025
├── results/                    # Generated plots (gitignored)
├── run_models.py               # Quick inference: load all models, print metrics
├── train_all.py                # Master training script
├── validate_dataset.py
├── verify_gpu.py
└── .gitignore
```

---

## Installation

```bash
# Clone and create virtual environment
git clone <repo-url>
cd project
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install stable-baselines3[extra] gymnasium torch pandas numpy matplotlib
```

### Dependencies

| Package | Purpose |
|---|---|
| `stable-baselines3` | DQN, PPO implementations |
| `gymnasium` | Environment interface |
| `torch` | Neural network backend |
| `pandas` | Data manipulation |
| `numpy` | Numerical ops |
| `matplotlib` | Plotting |

---

## Usage

### Quick inference (pre-trained models)

```bash
python run_models.py
```

Loads all three models from `models/`, runs a full deterministic episode on `data/test_dataset.csv`, and prints a metric table.

### Full evaluation with plots

```bash
python evaluation/evaluate_agent.py
```

Produces portfolio value curves, drawdown charts, rolling Sharpe plots, allocation heatmaps, and regime performance breakdowns in `results/`.

### Retrain all agents

```bash
# Build datasets first (if not already present)
python data_pipeline/build_dataset.py

# Train all three agents
python train_all.py

# Train and immediately evaluate
python train_all.py --evaluate

# Skip agents whose model files already exist
python train_all.py --skip-existing

# Train a specific agent only
python train_all.py --agents ppo
```

### Train individual agents

```bash
python agents/train_dqn.py
python agents/train_ppo.py          # PPO baseline
python agents/train_safe_ppo.py     # Safe PPO
```

---

## Key Design Decisions

**BTC as regime proxy** — Bitcoin is highly correlated with global risk appetite and reacts faster than equity indices. Using BTC price vs. its MA200 gives a clean, forward-looking risk-on/off signal that benefits all assets in the portfolio.

**No reward normalization in VecNormalize** — `norm_reward=False` keeps raw reward signal for the agent. Normalizing rewards can destroy the scale information the safe penalty terms depend on.

**Warm-up for safe penalties** — Applying full penalties from step 0 destabilizes early training because the agent has not yet learned a baseline policy. The 200k-step linear ramp lets the agent first learn to trade profitably, then refine toward risk control.

**Rebalance gate** — Updating weights every 5 steps instead of every step reduces the number of transactions and prevents the agent from optimizing for short-term noise rather than multi-day trends.

**Max position cap (40%)** — Hard constraint applied in `_process_weights` before the action is executed, regardless of what the network outputs. This prevents single-asset concentration that would violate basic portfolio diversification.
