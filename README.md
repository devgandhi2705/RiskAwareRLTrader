# Reinforcement Learning Portfolio Manager

A multi-asset portfolio management system that uses **Reinforcement Learning (RL)** to dynamically allocate capital across global markets.

The system trains multiple RL agents to learn **portfolio allocation strategies** that aim to outperform traditional investment strategies such as **Equal Weight, Risk Parity, and Momentum**.

---

# Project Goal

The objective of this project is to build an RL-based portfolio manager that can:

- Generate **alpha** by outperforming traditional strategies  
- Adapt to different market conditions  
- Manage risk during market downturns  
- Allocate capital dynamically across multiple assets  

The project compares three agents:

| Agent | Role |
|------|------|
| **DQN** | Baseline RL strategy |
| **PPO** | Alpha generation agent |
| **Safe PPO** | Risk-aware portfolio manager |

---

# Assets Used

The portfolio is built across **7 diversified assets**:

| Category | Assets |
|--------|--------|
| Crypto | BTC, ETH |
| Global Markets | SPY (S&P500 ETF) |
| Commodities | Gold (GLD), Silver |
| Indian Markets | Nifty50, Sensex |

This diversification allows the RL agents to learn **cross-asset allocation strategies**.

---

# Dataset

The dataset contains **daily market data with technical indicators**.

### Training Data
2016-01-01 → 2022-12-31

### Test Data
2023-01-01 → 2024-12-31

Each asset includes only the **most important indicators** to keep the observation space manageable.

### Features per Asset

- Return  
- MA20  
- MA50  
- RSI  
- Rolling Volatility  

Total features:
