# Regime-Aware Reinforcement Learning Portfolio Manager

An AI portfolio manager that uses **reinforcement learning** to dynamically allocate capital across multiple assets based on market conditions.

The system learns to behave differently during **bull, neutral, and bear markets**, balancing **profit generation and risk control**.

---

# Goal

Build an AI portfolio management system that:

- Captures returns during **bull markets**
- Maintains balanced allocation during **sideways markets**
- Reduces exposure during **bear markets**
- Allocates capital **dynamically across multiple assets**
- Maintains **risk-aware portfolio behavior**

---

# Assets Used

| Category | Assets |
|---|---|
| Crypto | BTC, ETH |
| Global Markets | SPY (S&P500 ETF), GLD (Gold ETF), Silver |
| Indian Markets | Nifty50, Sensex |

This allows the agent to learn **cross-asset portfolio allocation strategies**.

---

# Dataset

| Property | Value |
|---|---|
| Time Period | 2016 – 2024 |
| Training Data | 2016 – 2022 |
| Test Data | 2023 – 2024 |
| Dataset Size | ~2500 rows |
| Features | ~60+ |

Features include:

- Asset prices
- Daily returns
- Technical indicators
- Rolling volatility
- Market regime labels
- Trend strength signals

---

# Market Regime Detection

Market regimes are estimated using **trend signals derived from moving averages**.

Trend strength is calculated as:

```
trend_strength = (MA50 − MA200) / MA200
```

Regime classification:

| Regime | Description |
|---|---|
| Bull | Strong upward trend |
| Neutral | Sideways or weak trend |
| Bear | Strong downward trend |

Each asset is classified independently, allowing the model to detect situations like:

```
BTC → Bull
ETH → Bull
SPY → Neutral
GLD → Bull
Nifty → Bear
Sensex → Bear
```

This allows the RL agent to allocate capital according to **individual asset conditions**.

---

# Reinforcement Learning Setup

The portfolio problem is modeled as a **Markov Decision Process (MDP)**.

| Component | Description |
|---|---|
| State | Market features (returns, trend strength, volatility, regime) |
| Action | Portfolio allocation across all assets |
| Environment | Simulated trading system |
| Reward | Portfolio return adjusted by risk penalties |

The agent observes market signals and decides **how to distribute capital across assets**.

---

# Reward Function

The reward function encourages **profit generation while controlling risk**.

Reward components:

### Portfolio Return
Primary objective is to maximize portfolio returns.

### Risk Penalties

The agent is penalized for excessive risk.

| Penalty | Purpose |
|---|---|
| Volatility Penalty | discourages unstable portfolios |
| Drawdown Penalty | discourages large losses |
| Turnover Penalty | discourages excessive trading |

These penalties encourage **stable portfolio growth instead of aggressive speculation**.

---

# Agents Implemented

### DQN (Baseline)

- Uses discrete actions
- Simpler RL approach
- Serves as a baseline model

---

### PPO (Proximal Policy Optimization)

- Stable policy learning
- Handles complex environments better
- Produces stronger portfolio strategies

---

### Safe PPO (Risk-Aware PPO)

Safe PPO extends PPO by including additional **risk penalties** in the reward function:

- Volatility penalties
- Drawdown penalties
- Turnover penalties

This encourages the agent to maintain **controlled risk exposure**.

---

# Training Method — Walk-Forward Validation

To avoid overfitting, the system uses **rolling training windows**.

```
Train 2016-2017 → Validate 2018
Train 2017-2018 → Validate 2019
Train 2018-2019 → Validate 2020
Train 2019-2020 → Validate 2021
Train 2020-2021 → Validate 2022
```

This ensures the model learns **robust strategies across different market conditions**.

---

# Evaluation

Agents are evaluated on **unseen test data (2023–2024)**.

Baseline strategies used for comparison:

- Equal Weight Portfolio
- Risk Parity Portfolio
- Momentum Strategy

Evaluation metrics include:

- Total Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Volatility
- CVaR (tail risk)

Example results:

| Agent | Return | Sharpe | Max Drawdown |
|---|---|---|---|
| PPO | ~54% | ~2.1 | ~11% |
| Safe PPO | ~25% | ~1.9 | ~7% |
| DQN | ~19% | ~1.5 | ~7% |

PPO achieves higher returns while Safe PPO maintains **lower risk and smaller drawdowns**.

---

# Project Structure

```
project/

agents/
    train_dqn.py
    train_ppo.py
    train_safe_ppo.py

environment/
    trading_environment.py

features/
    regime_detection.py

data/
    train_dataset.csv
    test_dataset.csv

models/
    dqn_portfolio.zip
    ppo_portfolio.zip
    safe_ppo_portfolio.zip

results/
    performance_metrics.csv
    portfolio_value_curves.png
    drawdown_curves.png
    training_curves.png

demo/
    demo_live_portfolio.py
    demo_dashboard.py

README.md
```

---

# Demo System

The project includes a simple **AI portfolio manager simulation**.

Workflow:

1. Load trained RL model
2. Read historical market data
3. Simulate daily market updates
4. Agent decides portfolio allocation
5. Portfolio value updates
6. Dashboard visualizes results

The **Streamlit dashboard** displays:

- Portfolio value
- Asset allocation
- Market regime
- Performance metrics

---

# Future Scope

## Improving the RL Model

Potential improvements:

- Continuous portfolio allocation models
- Transformer-based market prediction
- Multi-agent reinforcement learning
- Improved reward design
- Advanced regime detection models

---

## Real Trading Deployment

Future production deployment could include:

- Real-time market data APIs
- Broker API integration (Alpaca, Zerodha, etc.)
- Automated order execution
- Live portfolio monitoring
- Advanced risk management tools

---

# Project Outcome

This project demonstrates how **reinforcement learning can be applied to multi-asset portfolio management**, enabling an AI system to dynamically adapt its allocation strategy based on changing market conditions while maintaining risk awareness.
