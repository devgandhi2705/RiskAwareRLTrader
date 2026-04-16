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

```

7 assets × 5 indicators = 35 features

* 7 portfolio weights
* 1 portfolio value

---

43 total observation features

```

---

# Reinforcement Learning Setup

### Observation (State)

The RL agent observes:

- Asset indicators  
- Portfolio weights  
- Portfolio value  

### Action Space

The agent outputs **continuous portfolio weights**:

```

[BTC, ETH, SPY, GLD, Silver, Nifty50, Sensex]

```

Weights are normalized so that:

```

sum(weights) = 1

```

Maximum allocation per asset:

```

max_weight = 0.90

```

---

# Reward Functions

### PPO (Alpha Generator)

PPO is trained to **beat the Momentum strategy**.

```

reward =
portfolio_return

* 0.5 * (portfolio_return - momentum_return)

- 0.0025 * turnover

```

This encourages the agent to:

- maximize portfolio returns  
- outperform momentum  
- avoid excessive trading  

---

### Safe PPO (Risk Protection)

Safe PPO behaves like PPO normally, but **switches to risk control during market downturns**.

During bearish conditions:

```

reward =
0.5 * portfolio_return

* 2 * volatility
* 3 * drawdown

```

This encourages:

- lower volatility  
- lower drawdowns  
- capital protection  

---

# Training Method

Instead of walk-forward retraining, the system uses **randomized training episodes**.

```

episode_length = 255 trading days

```

Episodes start at random points in the dataset, allowing the agent to experience:

- bull markets  
- bear markets  
- sideways markets  

Training timesteps:

```

1,000,000 steps

```

---

# Baseline Strategies

The RL agents are compared against three traditional portfolio strategies:

| Strategy | Description |
|-------|-------------|
| Equal Weight | Equal allocation across all assets |
| Risk Parity | Risk-balanced portfolio |
| Momentum (60d) | Invest in strongest trending assets |

Momentum serves as the **primary benchmark for alpha generation**.

---

# Evaluation Metrics

Agents are evaluated on **unseen test data (2023-2024)** using:

- Total Return  
- Sharpe Ratio  
- Sortino Ratio  
- Maximum Drawdown  
- CVaR  
- Annual Volatility  
- Turnover  

---

# Project Structure

```

project/
│
├── data/
│   ├── raw_market_data.csv
│   ├── train_dataset.csv
│   └── test_dataset.csv
│
├── data_pipeline/
│   ├── download_data.py
│   ├── feature_engineering.py
│   ├── build_dataset.py
│   └── validate_dataset.py
│
├── environment/
│   └── trading_environment.py
│
├── agents/
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── train_safe_ppo.py
│
├── evaluation/
│   ├── evaluate_agent.py
│   └── evaluate_and_ablate.py
│
├── results/
│
├── train_all.py
├── requirements.txt
└── README.md

```

---

# Running the Project

### 1. Install dependencies

```

pip install -r requirements.txt

```

---

### 2. Build the dataset

```

python data_pipeline/build_dataset.py

```

---

### 3. Train all agents

```

python train_all.py

```

This will:

- build dataset  
- validate dataset  
- train DQN  
- train PPO  
- train Safe PPO  
- run evaluation  

---

### 4. View results

Results will be saved in:

```

results/

```

Including:

- portfolio_value_curves.png  
- drawdown_curves.png  
- rolling_sharpe.png  
- metrics_comparison.png  

---

# Key Design Decisions

The final system was simplified to improve RL training stability:

- Removed regime detection features  
- Reduced observation feature size  
- Simplified reward functions  
- Allowed aggressive allocations (max_weight = 0.90)  
- Used randomized training episodes instead of walk-forward retraining  

These changes help the RL agents learn **more stable and interpretable strategies**.

---

# Future Improvements

Potential extensions include:

- multi-agent reinforcement learning  
- transformer-based market representations  
- macroeconomic data integration  
- dynamic risk budgeting  
- live trading deployment  

---

# Disclaimer

This project is for **research and educational purposes only**.  
It is **not financial advice** and should not be used directly for live trading without extensive testing.
