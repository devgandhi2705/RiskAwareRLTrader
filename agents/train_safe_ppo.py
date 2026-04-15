"""
train_safe_ppo.py  (v3)
------------------------
Trains Safe PPO with risk-switched reward.
Delegates entirely to train_ppo(safe_reward=True).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.train_ppo import train_ppo, log


def train_safe_ppo():
    log("\n[train_safe_ppo] Delegating to train_ppo(safe_reward=True)")
    train_ppo(safe_reward=True)


if __name__ == "__main__":
    train_safe_ppo()