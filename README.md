# Regime-Aware Reinforcement Learning Portfolio Manager

An AI portfolio manager that uses reinforcement learning to allocate capital across multiple assets based on market conditions — behaving differently during bull, neutral, and bear markets.

---

## Goal

Build an AI system that:
- Captures returns during **bull markets**
- Balances exposure during **neutral markets**
- Reduces risk during **bear markets**

---

## Assets

| Category | Assets |
|---|---|
| Crypto | BTC, ETH |
| Global Markets | SPY (S&P500 ETF), GLD (Gold ETF), Silver |
| Indian Markets | Nifty50, Sensex |

---

## Dataset

- **Time period:** 2016 – 2024
- **Training data:** 2016 – 2022
- **Test data:** 2023 – 2024
- **Size:** ~2500 rows, ~60 features
- **Features include:** asset prices, returns, technical indicators, volatility metrics, market regime labels

---

## Market Regimes

| Regime | Description | Agent Behavior |
|---|---|---|
| Bull | Prices trend upward | Increase exposure, capture upside momentum |
| Neutral | Markets move sideways | Balanced diversification |
| Bear | Prices trend downward | Reduce exposure, protect capital |

---

## Reinforcement Learning Setup

| Component | Description |
|---|---|
| **Agent** | AI model that decides portfolio allocation |
| **Environment** | Simulated trading environment |
| **State** | Prices, technical indicators, volatility, market regime |
| **Action** | Portfolio allocation across all assets (e.g. BTC 30%, ETH 20%, …) |
| **Reward** | Portfolio return minus penalties for volatility, drawdown, and risk |

---

## Agents

**DQN (Baseline)**
- Discrete action space, simpler learning approach, used as a baseline comparison.

**PPO (Proximal Policy Optimization)**
- Stable training, better policy learning, more flexible decision making.

**Safe PPO (Risk-Aware)**
- PPO with added reward penalties for high volatility, large drawdowns, and excessive portfolio turnover. Encourages risk-controlled behaviour.

---

## Training Method: Walk-Forward

Instead of training once, models train on multiple rolling windows to reduce overfitting:

```
Train 2016–2017 → Validate 2018
Train 2017–2018 → Validate 2019
Train 2018–2019 → Validate 2020
Train 2019–2020 → Validate 2021
Train 2020–2021 → Validate 2022
```

---

## Evaluation

Agents were evaluated on unseen test data (2023–2024) and compared against traditional strategies (Equal Weight, Risk Parity, Momentum).

**Metrics:** Total Return, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Volatility, CVaR

**Example Results:**

| Agent | Return | Sharpe | Max Drawdown |
|---|---|---|---|
| PPO | 54% | 2.10 | 11% |
| Safe PPO | 25% | 1.94 | 7% |
| DQN | 19% | 1.49 | 7% |

> PPO achieves higher returns. Safe PPO prioritises lower risk. DQN serves as the baseline.

---

## Project Structure

```
project/
│
├── agents/
│   ├── train_dqn.py               # Trains the DQN agent
│   ├── train_ppo.py               # Trains the PPO agent (walk-forward)
│   └── train_safe_ppo.py          # Trains the risk-aware Safe PPO agent
│
├── environment/
│   └── trading_environment.py     # Custom trading simulation environment
│
├── data/
│   ├── train_dataset.csv
│   └── test_dataset.csv
│
├── models/
│   ├── dqn_portfolio.zip
│   ├── ppo_portfolio.zip
│   └── safe_ppo_portfolio.zip
│
├── results/
│   ├── performance_metrics.csv
│   ├── portfolio_value_curves.png
│   └── training_curves.png
│
├── demo/
│   ├── demo_live_portfolio.py     # Simulates a live AI portfolio manager
│   └── demo_dashboard.py         # Streamlit dashboard for visualisation
│
└── README.md
```

---

## Demo System

The demo simulates a live AI portfolio manager:

1. Load trained RL model
2. Read historical CSV data
3. Simulate market updates step-by-step
4. Agent decides portfolio allocation each day
5. Portfolio value updates
6. Dashboard displays results

Run the Streamlit dashboard (`demo_dashboard.py`) to visualise portfolio value, asset allocation, market regime, and performance metrics.

---

## Future Scope

**Better RL Agents**
- Continuous action space for portfolio weights
- Transformer-based market models
- Multi-agent reinforcement learning
- Improved reward functions
- Automatic regime detection

**Real Trading Deployment**
- Real-time market data APIs
- Broker integration (Alpaca, Zerodha, etc.)
- Automatic order execution
- Live portfolio monitoring
- Advanced risk management
