"""
Alpha PPO V1 Training Script
Paper-ready / Competition-ready

Alpha = entry decision maker
Guardian = external safety layer (NOT in reward)

Author: Competition System
"""

import os
import argparse
import json
import numpy as np
import random
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# =========================
# 🔧 CONFIG
# =========================

DEFAULT_TIMESTEPS = 5_000_000
DEFAULT_SEED = 42

LOG_ROOT = "runs/alpha_v1"
MODEL_ROOT = "models"

os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

# =========================
# 🧠 ENV IMPORT
# =========================

from src.envs.market_env_v3 import MarketEnvV3

# =========================
# 🎯 REWARD PROFILE
# =========================

def competition_reward(info: dict) -> float:
    """
    Reward shaping for competition:
    - maximize pnl
    - minimize DD
    - avoid overtrading
    
    Paper Statement:
        "We design a multi-objective reward function that balances
         profit maximization with risk constraints."
    """
    reward = 0.0

    # Main PnL component
    reward += info.get("pnl", 0.0)
    
    # DD penalty (risk-aware)
    reward -= 0.5 * info.get("dd_inc", 0.0)
    
    # Overtrading penalty
    reward -= 0.1 * info.get("overtrade", 0.0)
    
    # Trend alignment bonus
    reward += 0.2 * info.get("trend_align", 0.0)

    return float(reward)


# =========================
# 🧪 ENV FACTORY
# =========================

def make_env(seed: int):
    def _init():
        env = MarketEnvV3(
            mode="train",
            reward_fn=competition_reward,
            enable_guardian=False,   # ❌ Guardian NOT used in training
            shadow_mode=False
        )
        env = Monitor(env)
        return env
    return _init


# =========================
# 🚀 TRAIN
# =========================

def train(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"alpha_ppo_v1_{run_id}"

    log_dir = os.path.join(LOG_ROOT, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Reproducibility ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)

    # ---- Env ----
    env = DummyVecEnv([make_env(args.seed)])

    # ---- Model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log=None,  # Disabled for Windows compatibility
        seed=args.seed,
        device=device   # Auto GPU if available
    )

    # ---- Checkpoints ----
    checkpoint_cb = CheckpointCallback(
        save_freq=200_000,
        save_path=MODEL_ROOT,
        name_prefix=run_name
    )

    # ---- Train ----
    print("\n" + "=" * 60)
    print("🧠 Alpha PPO V1 Training Started")
    print("=" * 60)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seed: {args.seed}")
    print(f"Log dir: {log_dir}")
    print("=" * 60 + "\n")
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        progress_bar=True
    )

    # ---- Save final ----
    final_path = os.path.join(
        MODEL_ROOT,
        f"{run_name}_FINAL.zip"
    )
    model.save(final_path)

    # ---- Save metadata (paper-friendly) ----
    meta = {
        "run_name": run_name,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "reward": "competition",
        "policy": "PPO-Mlp",
        "guardian_in_training": False,
        "date": run_id
    }

    with open(os.path.join(log_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Alpha PPO V1 Training Complete")
    print(f"📦 Model: {final_path}")
    print(f"📊 Logs : {log_dir}")
    print("=" * 60 + "\n")


# =========================
# 🏁 MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alpha PPO V1")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                       help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="Random seed for reproducibility")

    args = parser.parse_args()
    train(args)
