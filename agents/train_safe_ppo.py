"""
train_safe_ppo.py  (v2)
------------------------
Trains Safe PPO with risk-penalised reward.
Delegates entirely to train_ppo(safe_reward=True).

Reward includes:
  - Turnover penalty     (S3)
  - Drawdown penalty     (S7)
  - CVaR penalty         (S7)
  - Volatility penalty   (S8)
"""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from agents.train_ppo import train_ppo

def train_safe_ppo():
    train_ppo(safe_reward=True)

if __name__ == "__main__":
    train_safe_ppo()