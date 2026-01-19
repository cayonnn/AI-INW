# src/rl/guardian_ppo_v2.py
"""
Guardian PPO V2 - Competition-Grade Training
==============================================

Production-grade PPO with proper reward shaping.
Guardian learns to minimize tail-risk and failure cascades.

State Space (8-dim):
- daily_dd
- margin_ratio
- free_margin_ratio
- open_positions / max_pos
- recent_error_rate_5m
- recent_latency_avg
- chaos_flag
- alpha_intent

Action Space:
- 0: ALLOW
- 1: REDUCE_RISK
- 2: FORCE_HOLD
- 3: KILL_SWITCH
"""

import numpy as np
from pathlib import Path
import csv
from datetime import datetime

# Gymnasium (modern) vs gym (legacy)
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# Action Definitions
# =============================================================================

ACTIONS = {
    0: "ALLOW",
    1: "REDUCE_RISK",
    2: "FORCE_HOLD",
    3: "KILL_SWITCH"
}


# =============================================================================
# Guardian Environment V2
# =============================================================================

class GuardianEnvV2(gym.Env):
    """
    Production-grade Guardian RL Environment.
    
    Key improvements over V1:
    - Better reward shaping
    - Chaos simulation
    - Over-conservative penalty
    - Survival bonus
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, max_steps: int = 200):
        super().__init__()
        
        self.max_steps = max_steps
        self.actions = ACTIONS
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 8 continuous features [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Initialize or reset state variables."""
        self.daily_dd = 0.0
        self.margin_ratio = 0.9
        self.free_margin_ratio = np.random.uniform(0.4, 0.8)
        self.open_pos_ratio = np.random.uniform(0.0, 0.3)
        self.error_rate = 0.0
        self.latency = 0.1
        self.chaos = 0
        self.alpha_intent = 0.5  # 0=SELL, 0.5=HOLD, 1=BUY
        
        self.step_count = 0
        self.total_reward = 0.0
        self.chaos_events = 0
        self.blocks = 0
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            np.clip(self.daily_dd, 0, 1),
            np.clip(self.margin_ratio, 0, 1),
            np.clip(self.free_margin_ratio, 0, 1),
            np.clip(self.open_pos_ratio, 0, 1),
            np.clip(self.error_rate, 0, 1),
            np.clip(self.latency / 5.0, 0, 1),  # Normalize latency
            float(self.chaos),
            self.alpha_intent,
        ], dtype=np.float32)
    
    def _simulate_market(self):
        """Simulate market dynamics and chaos events."""
        # Random chaos events (10% chance)
        if np.random.rand() < 0.10:
            self.chaos = 1
            self.chaos_events += 1
            self.error_rate = min(1.0, self.error_rate + 0.2)
            self.latency += np.random.uniform(0.5, 2.0)
        else:
            self.chaos = 0
            self.error_rate = max(0, self.error_rate - 0.05)
            self.latency = max(0.1, self.latency - 0.1)
        
        # Gradual risk drift
        self.daily_dd += np.random.uniform(-0.002, 0.015)  # Bias toward increase
        self.daily_dd = np.clip(self.daily_dd, 0, 0.3)
        
        self.free_margin_ratio += np.random.uniform(-0.03, 0.01)  # Bias toward decrease
        self.free_margin_ratio = np.clip(self.free_margin_ratio, 0.05, 1.0)
        
        # Alpha sends random intents
        self.alpha_intent = np.random.choice([0, 0.5, 1], p=[0.3, 0.3, 0.4])
    
    def _calculate_reward(self, action: int) -> float:
        """
        Production-grade reward function.
        
        Key principles:
        1. Safety first - reward defensive actions during danger
        2. Penalize over-protection - don't HOLD when safe
        3. Kill switch is last resort - high reward only when needed
        """
        reward = 0.0
        
        # === Chaos Handling ===
        if self.chaos == 1:
            if action in [1, 2]:  # REDUCE_RISK or FORCE_HOLD
                reward += 1.5
            elif action == 0:  # ALLOW during chaos
                reward -= 3.0
        
        # === Drawdown Control ===
        if self.daily_dd > 0.05:
            if action in [1, 2]:
                reward += 1.0
            elif action == 0:
                reward -= 2.0
        
        # === Margin Starvation ===
        if self.free_margin_ratio < 0.20:
            if action == 2:  # FORCE_HOLD
                reward += 2.0
            else:
                reward -= 3.0
        
        # === Over-Conservative Penalty ===
        no_risk = (self.daily_dd < 0.03 and 
                   self.free_margin_ratio > 0.5 and 
                   self.chaos == 0)
        if no_risk and action == 2:  # FORCE_HOLD when safe
            reward -= 0.2
        
        # === Kill Switch (Very Rare) ===
        if action == 3:
            imminent_failure = (self.daily_dd > 0.10 or 
                               self.free_margin_ratio < 0.10)
            reward += 5.0 if imminent_failure else -5.0
        
        # === Survival Bonus ===
        reward += 0.1
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed) if hasattr(super(), 'reset') else None
        self._reset_state()
        return self._get_obs(), {}
    
    def step(self, action: int):
        """Execute action."""
        self.step_count += 1
        
        # Simulate market
        self._simulate_market()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.total_reward += reward
        
        # Track blocks
        if action >= 1:
            self.blocks += 1
        
        # Apply action effects
        if action == 2:  # FORCE_HOLD reduces risk
            self.daily_dd = max(0, self.daily_dd - 0.005)
            self.free_margin_ratio = min(1, self.free_margin_ratio + 0.02)
        elif action == 1:  # REDUCE_RISK slight protection
            self.daily_dd = max(0, self.daily_dd - 0.002)
        
        # Check termination
        terminated = False
        truncated = False
        
        if action == 3:  # KILL_SWITCH ends episode
            terminated = True
        
        if self.daily_dd > 0.15:  # Bankruptcy
            terminated = True
            reward -= 10.0
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        obs = self._get_obs()
        info = {
            "action": self.actions[action],
            "daily_dd": self.daily_dd,
            "free_margin": self.free_margin_ratio,
            "chaos_events": self.chaos_events,
            "blocks": self.blocks,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render current state."""
        print(
            f"Step {self.step_count:3d} | "
            f"DD={self.daily_dd*100:5.1f}% | "
            f"Margin={self.free_margin_ratio*100:4.0f}% | "
            f"Chaos={self.chaos} | "
            f"Reward={self.total_reward:.1f}"
        )


# =============================================================================
# Training Functions
# =============================================================================

def train_v2(total_timesteps: int = 200_000, save_path: str = "models/guardian_ppo_v2"):
    """Train Guardian PPO V2 with improved reward."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("❌ stable_baselines3 not installed")
        return None
    
    print("\n" + "=" * 60)
    print("🧠 GUARDIAN PPO V2 TRAINING")
    print("=" * 60)
    
    env = DummyVecEnv([lambda: GuardianEnvV2()])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        device="cpu",
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    
    print("=" * 60)
    print(f"✅ Model saved to {save_path}")
    print("=" * 60 + "\n")
    
    return model


def test_v2(model_path: str = "models/guardian_ppo_v2"):
    """Test trained model."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("❌ stable_baselines3 not installed")
        return
    
    print("\n" + "=" * 60)
    print("🧪 GUARDIAN PPO V2 TEST")
    print("=" * 60)
    
    env = GuardianEnvV2()
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    
    action_counts = {a: 0 for a in ACTIONS.values()}
    
    for step in range(30):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        action_name = ACTIONS[int(action)]
        action_counts[action_name] += 1
        
        chaos_indicator = "💥" if env.chaos else "  "
        print(
            f"  Step {step:2d} {chaos_indicator} | "
            f"DD={info['daily_dd']*100:5.1f}% | "
            f"Margin={info['free_margin']*100:4.0f}% | "
            f"Action={action_name:12} | "
            f"Reward={reward:+.1f}"
        )
        
        if terminated or truncated:
            break
    
    print("=" * 60)
    print(f"📊 Total Reward: {env.total_reward:.2f}")
    print(f"📊 Chaos Events: {env.chaos_events}")
    print(f"📊 Blocks: {env.blocks}")
    print(f"📊 Action Distribution: {action_counts}")
    print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Guardian PPO V2")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--steps", type=int, default=200_000)
    
    args = parser.parse_args()
    
    if args.train:
        train_v2(total_timesteps=args.steps)
    
    if args.test:
        test_v2()
    
    if not args.train and not args.test:
        print("Usage: python -m src.rl.guardian_ppo_v2 --train --steps 200000")
        print("       python -m src.rl.guardian_ppo_v2 --test")
