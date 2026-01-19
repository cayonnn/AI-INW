# src/rl/guardian_ppo.py
"""
Guardian PPO Training
======================

Train Guardian agent using Proximal Policy Optimization.
Guardian learns to maximize survival + capital efficiency.

Objective:
- NOT maximize profit
- Maximize survival + capital efficiency
- Penalize late response (asymmetric reward)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_PPO")


# =============================================================================
# Guardian Actions
# =============================================================================

GUARDIAN_ACTIONS = {
    0: "ALLOW",
    1: "REDUCE_RISK", 
    2: "FORCE_HOLD",
    3: "EMERGENCY_FREEZE"
}


# =============================================================================
# Reward Shaping (Reviewer-level)
# =============================================================================

def guardian_reward(state: Dict, action: int) -> float:
    """
    Asymmetric reward function for Guardian.
    
    Heavily penalizes:
    - Late response to high DD
    - Margin starvation without action
    
    Rewards:
    - Correct defensive reactions
    - Stability in safe zones
    
    Penalizes:
    - Overreaction (freeze when not needed)
    """
    dd = state.get("daily_dd", 0.0)
    margin = state.get("margin_ratio", 1.0)
    error_detected = state.get("error_detected", False)
    
    reward = 0.0
    
    # 1️⃣ Capital protection dominates
    if dd > 0.10:
        reward -= 10.0  # Severe penalty for high DD
        if action >= 2:  # FORCE_HOLD or EMERGENCY_FREEZE
            reward += 4.0  # Reward defensive action
    
    # 2️⃣ Margin starvation defense
    if margin < 0.25:
        reward -= 5.0  # Penalty for low margin
        if action >= 1:  # Any defensive action
            reward += 3.0  # Reward for taking action
    
    # 3️⃣ Overreaction penalty
    if action == 3 and dd < 0.05:  # EMERGENCY_FREEZE when DD is low
        reward -= 3.0  # Penalize panic
    
    # 4️⃣ Stability bonus (safe zone)
    if 0.3 < margin < 0.7 and dd < 0.05:
        reward += 0.2  # Small bonus for maintaining stability
    
    # 5️⃣ Chaos/Error handling
    if error_detected and action >= 2:
        reward += 2.0  # Reward correct defensive reaction to errors
    
    return reward


# =============================================================================
# Gymnasium Environment
# =============================================================================

class GuardianGymEnv(gym.Env):
    """
    Gymnasium-compatible environment for Guardian PPO training.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, max_steps: int = 1000):
        super().__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Observation space: 8 continuous features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1, 0, 0], dtype=np.float32),
            high=np.array([2, 1, 10, 10, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # State tracking
        self.state = None
        self._init_state()
    
    def _init_state(self):
        """Initialize state with random values."""
        self.state = {
            "margin_ratio": np.random.uniform(0.5, 1.0),
            "daily_dd": 0.0,
            "open_positions": np.random.randint(0, 5),
            "margin_block_count": 0,
            "equity_volatility": np.random.uniform(0, 0.1),
            "alpha_signal": np.random.choice([-1, 0, 1]),
            "time_in_trade": np.random.uniform(0, 3600),
            "error_detected": False,
        }
        self.current_step = 0
    
    def _get_obs(self) -> np.ndarray:
        """Convert state to observation array."""
        return np.array([
            self.state["margin_ratio"],
            self.state["daily_dd"],
            self.state["open_positions"] / 10.0,
            self.state["margin_block_count"] / 5.0,
            self.state["equity_volatility"],
            (self.state["alpha_signal"] + 1) / 2.0,
            min(self.state["time_in_trade"] / 3600, 1.0),
            float(self.state["error_detected"]),
        ], dtype=np.float32)
    
    def _simulate_market(self, action: int):
        """Simulate market dynamics."""
        # Random market movement
        dd_change = np.random.uniform(-0.005, 0.015)  # Slight upward bias for DD
        margin_change = np.random.uniform(-0.03, 0.02)  # Slight margin pressure
        
        # Action effects
        if action == 2:  # FORCE_HOLD
            dd_change *= 0.5  # Reduce DD growth
            margin_change *= 0.3  # Reduce margin loss
        elif action == 1:  # REDUCE_RISK
            dd_change *= 0.7
            margin_change *= 0.5
        
        # Apply changes
        self.state["daily_dd"] = np.clip(
            self.state["daily_dd"] + dd_change, 0, 0.3
        )
        self.state["margin_ratio"] = np.clip(
            self.state["margin_ratio"] + margin_change, 0.05, 1.5
        )
        
        # Random position changes
        if action < 2:
            self.state["open_positions"] = min(10, self.state["open_positions"] + np.random.choice([0, 0, 1]))
        else:
            self.state["open_positions"] = max(0, self.state["open_positions"] - 1)
        
        # Random errors (chaos simulation)
        self.state["error_detected"] = np.random.random() < 0.02
        
        # Update block count based on margin
        if self.state["margin_ratio"] < 0.3:
            self.state["margin_block_count"] += 1
        else:
            self.state["margin_block_count"] = max(0, self.state["margin_block_count"] - 1)
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action."""
        self.current_step += 1
        
        # Calculate reward
        reward = guardian_reward(self.state, action)
        
        # Simulate market
        self._simulate_market(action)
        
        # Check termination
        terminated = False
        truncated = False
        
        if action == 3:  # EMERGENCY_FREEZE
            terminated = True
            
        if self.state["daily_dd"] > 0.15:
            terminated = True
            reward -= 20.0  # Severe penalty for catastrophic loss
        
        if self.current_step >= self.max_steps:
            truncated = True
        
        obs = self._get_obs()
        info = {
            "action_name": GUARDIAN_ACTIONS[action],
            "daily_dd": self.state["daily_dd"],
            "margin_ratio": self.state["margin_ratio"],
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render current state."""
        print(
            f"Step {self.current_step} | "
            f"DD={self.state['daily_dd']*100:.1f}% | "
            f"Margin={self.state['margin_ratio']*100:.0f}% | "
            f"Positions={self.state['open_positions']}"
        )


# =============================================================================
# Training Functions
# =============================================================================

def train_guardian_ppo(
    total_timesteps: int = 100_000,
    save_path: str = "models/guardian_ppo",
    verbose: int = 1
) -> "PPO":
    """
    Train Guardian PPO model.
    
    Args:
        total_timesteps: Training steps
        save_path: Path to save model
        verbose: Logging verbosity
        
    Returns:
        Trained PPO model
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        logger.error("stable_baselines3 not installed. Run: pip install stable-baselines3")
        return None
    
    logger.info("Creating Guardian PPO environment...")
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: GuardianGymEnv()])
    
    # Create PPO model (CPU for MLP policy)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=verbose,
        device="cpu",  # MLP policy runs better on CPU
    )
    
    logger.info(f"Training Guardian PPO for {total_timesteps:,} timesteps...")
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    logger.info(f"✅ Guardian PPO saved to {save_path}")
    
    return model


def load_guardian_ppo(model_path: str = "models/guardian_ppo"):
    """Load trained Guardian PPO model."""
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        logger.info(f"✅ Guardian PPO loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Guardian PPO: {e}")
        return None


class GuardianPPOAgent:
    """
    Guardian agent that uses trained PPO model.
    
    Drop-in replacement for rule-based GuardianAgent.
    """
    
    def __init__(self, model_path: str = "models/guardian_ppo"):
        self.model = load_guardian_ppo(model_path)
        self.mode = "ppo"
    
    def decide(self, state: Dict) -> str:
        """
        Make decision using PPO model.
        
        Args:
            state: Guardian state dictionary
            
        Returns:
            Action name string
        """
        if self.model is None:
            return "ALLOW"  # Fallback
        
        # Convert state to observation
        obs = np.array([
            state.get("margin_ratio", 1.0),
            state.get("daily_dd", 0.0),
            state.get("open_positions", 0) / 10.0,
            state.get("margin_block_count", 0) / 5.0,
            state.get("equity_volatility", 0.0),
            (state.get("alpha_signal", 0) + 1) / 2.0,
            min(state.get("time_in_trade", 0) / 3600, 1.0),
            float(state.get("error_detected", False)),
        ], dtype=np.float32)
        
        # Predict action
        action, _ = self.model.predict(obs, deterministic=True)
        
        return GUARDIAN_ACTIONS[int(action)]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Guardian PPO Training")
    parser.add_argument("--train", action="store_true", help="Train new model")
    parser.add_argument("--steps", type=int, default=100_000, help="Training steps")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    
    args = parser.parse_args()
    
    if args.train:
        print("\n" + "=" * 60)
        print("🧠 GUARDIAN PPO TRAINING")
        print("=" * 60)
        model = train_guardian_ppo(total_timesteps=args.steps)
        print("=" * 60 + "\n")
    
    if args.test:
        print("\n" + "=" * 60)
        print("🧪 GUARDIAN PPO TEST")
        print("=" * 60)
        
        agent = GuardianPPOAgent()
        env = GuardianGymEnv()
        
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(20):
            state = {
                "margin_ratio": obs[0],
                "daily_dd": obs[1],
                "open_positions": int(obs[2] * 10),
                "margin_block_count": int(obs[3] * 5),
                "equity_volatility": obs[4],
                "alpha_signal": int(obs[5] * 2 - 1),
                "time_in_trade": obs[6] * 3600,
                "error_detected": bool(obs[7]),
            }
            
            action_name = agent.decide(state)
            action = list(GUARDIAN_ACTIONS.keys())[list(GUARDIAN_ACTIONS.values()).index(action_name)]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {step:2d} | DD={info['daily_dd']*100:5.1f}% | Margin={info['margin_ratio']*100:5.0f}% | Action={action_name} | Reward={reward:+.1f}")
            
            if terminated or truncated:
                break
        
        print("=" * 60)
        print(f"📊 Total Reward: {total_reward:.2f}")
        print("=" * 60 + "\n")
