# src/rl/guardian_ppo_v3.py
"""
Guardian PPO V3 - Winner-Grade Training
========================================

Improvements over V2:
1. Dense reward shaping (no sparse rewards)
2. Curriculum learning (3 stages)
3. Proper DD/margin pressure modeling
4. Ready for competition

Curriculum:
- Stage 1: Low chaos (2%), max DD 5%
- Stage 2: Medium chaos (10%), max DD 10%
- Stage 3: High chaos (20%), bankruptcy enabled
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import random

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

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_V3")


# =============================================================================
# Action Space
# =============================================================================

ACTIONS = {
    0: "ALLOW",
    1: "REDUCE_RISK",
    2: "FORCE_HOLD",
    3: "KILL_SWITCH"
}


# =============================================================================
# Guardian Environment V3
# =============================================================================

class GuardianEnvV3(gym.Env):
    """
    Winner-grade Guardian RL Environment.
    
    Key improvements:
    - Dense reward (no sparse rewards)
    - Configurable chaos probability
    - Bankruptcy simulation
    - Curriculum-ready
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        chaos_prob: float = 0.05,
        max_dd: float = 0.10,
        bankruptcy: bool = False,
        max_steps: int = 500
    ):
        super().__init__()
        
        self.chaos_prob = chaos_prob
        self.max_dd = max_dd
        self.bankruptcy = bankruptcy
        self.max_steps = max_steps
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 4 normalized features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Initialize state variables."""
        self.daily_dd = 0.0
        self.free_margin_ratio = np.random.uniform(0.7, 1.0)
        self.chaos = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        self.chaos_events = 0
        self.actions_taken = {a: 0 for a in ACTIONS.values()}
    
    def _get_obs(self) -> np.ndarray:
        """Get normalized observation."""
        return np.array([
            min(self.daily_dd / self.max_dd, 1.0),
            self.free_margin_ratio,
            self.chaos,
            self.step_count / self.max_steps
        ], dtype=np.float32)
    
    def _simulate_market(self):
        """Simulate market dynamics and chaos."""
        # Random chaos events
        if random.random() < self.chaos_prob:
            self.chaos = 1.0
            self.chaos_events += 1
            self.daily_dd += random.uniform(0.002, 0.01)
            self.free_margin_ratio -= random.uniform(0.02, 0.05)
        else:
            self.chaos = 0.0
            self.daily_dd += random.uniform(-0.001, 0.003)
            self.free_margin_ratio -= random.uniform(0, 0.01)
        
        # Clamp values
        self.daily_dd = max(0.0, self.daily_dd)
        self.free_margin_ratio = max(0.0, min(1.0, self.free_margin_ratio))
    
    def _calculate_reward(self, action: int) -> float:
        """
        Dense reward function.
        
        No sparse rewards - every action gets meaningful feedback.
        """
        # Pressure metrics (0 to 1)
        dd_pressure = min(self.daily_dd / self.max_dd, 1.0)
        margin_pressure = max(0.0, 1.0 - self.free_margin_ratio / 0.3)
        
        reward = 0.05  # Survival bonus (every step)
        
        # Penalty for risk (always applied)
        reward -= dd_pressure * 2.0
        reward -= margin_pressure * 2.5
        
        # Action rewards (context-dependent)
        if action == 2:  # FORCE_HOLD
            reward += 1.5 * (dd_pressure + margin_pressure)
        elif action == 1:  # REDUCE_RISK
            reward += 1.0 * dd_pressure
        elif action == 0:  # ALLOW
            reward += 0.5 * (1.0 - dd_pressure)
        elif action == 3:  # KILL_SWITCH
            reward += 5.0 if self.daily_dd > self.max_dd * 1.2 else -3.0
        
        # Chaos handling bonus
        if self.chaos > 0 and action in [1, 2]:
            reward += 0.5
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed) if hasattr(super(), 'reset') else None
        self._reset_state()
        return self._get_obs(), {}
    
    def step(self, action: int):
        """Execute action."""
        self.step_count += 1
        self.actions_taken[ACTIONS[int(action)]] += 1
        
        # Simulate market
        self._simulate_market()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.total_reward += reward
        
        # Apply action effects
        if action == 2:  # FORCE_HOLD stabilizes
            self.daily_dd = max(0, self.daily_dd - 0.002)
            self.free_margin_ratio = min(1, self.free_margin_ratio + 0.01)
        elif action == 1:  # REDUCE_RISK helps slightly
            self.daily_dd = max(0, self.daily_dd - 0.001)
        
        # Check termination
        terminated = False
        truncated = False
        
        if action == 3:  # KILL_SWITCH
            terminated = True
        
        if self.daily_dd >= self.max_dd:
            terminated = True
            if self.bankruptcy:
                reward -= 10.0
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        obs = self._get_obs()
        info = {
            "action": ACTIONS[int(action)],
            "daily_dd": self.daily_dd,
            "free_margin": self.free_margin_ratio,
            "chaos_events": self.chaos_events,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render state."""
        print(
            f"Step {self.step_count:3d} | "
            f"DD={self.daily_dd*100:5.2f}% | "
            f"Margin={self.free_margin_ratio*100:4.0f}% | "
            f"Chaos={self.chaos_events}"
        )


# =============================================================================
# Curriculum Training
# =============================================================================

CURRICULUM = [
    {
        "name": "Stage 1: Low Chaos",
        "chaos_prob": 0.02,
        "max_dd": 0.05,
        "bankruptcy": False,
        "steps": 50_000
    },
    {
        "name": "Stage 2: Medium Chaos",
        "chaos_prob": 0.10,
        "max_dd": 0.10,
        "bankruptcy": False,
        "steps": 75_000
    },
    {
        "name": "Stage 3: High Chaos + Bankruptcy",
        "chaos_prob": 0.20,
        "max_dd": 0.10,
        "bankruptcy": True,
        "steps": 75_000
    },
]


def train_v3(save_path: str = "models/guardian_ppo_v3"):
    """
    Train Guardian PPO V3 with curriculum learning.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("❌ stable_baselines3 not installed")
        return None
    
    print("\n" + "=" * 60)
    print("🧠 GUARDIAN PPO V3 TRAINING (Curriculum)")
    print("=" * 60)
    
    model = None
    
    for stage_idx, stage in enumerate(CURRICULUM):
        print(f"\n🎓 {stage['name']}")
        print(f"   chaos_prob={stage['chaos_prob']}, max_dd={stage['max_dd']}, bankruptcy={stage['bankruptcy']}")
        print(f"   steps={stage['steps']:,}")
        
        # Create environment for this stage
        def make_env():
            return GuardianEnvV3(
                chaos_prob=stage["chaos_prob"],
                max_dd=stage["max_dd"],
                bankruptcy=stage["bankruptcy"]
            )
        
        env = DummyVecEnv([make_env])
        
        if model is None:
            # First stage: create model
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=2.5e-4,
                n_steps=2048,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.02,  # More exploration
                vf_coef=0.4,    # Balanced value learning
                verbose=1,
                device="cpu",
            )
        else:
            # Subsequent stages: reuse model, new env
            model.set_env(env)
        
        # Train this stage
        model.learn(total_timesteps=stage["steps"])
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = f"{save_path}_{timestamp}"
    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)
    
    print("\n" + "=" * 60)
    print(f"✅ Guardian PPO V3 saved to: {final_path}")
    print("=" * 60 + "\n")
    
    return model


def test_v3(model_path: str = None):
    """Test trained V3 model."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("❌ stable_baselines3 not installed")
        return
    
    # Find latest model if not specified
    if model_path is None:
        models_dir = Path("models")
        v3_models = list(models_dir.glob("guardian_ppo_v3_*.zip"))
        if not v3_models:
            print("❌ No V3 model found. Train first with --train")
            return
        model_path = str(sorted(v3_models)[-1])
    
    print("\n" + "=" * 60)
    print("🧪 GUARDIAN PPO V3 TEST")
    print(f"   Model: {model_path}")
    print("=" * 60)
    
    # Test with high chaos to see robustness
    env = GuardianEnvV3(chaos_prob=0.15, max_dd=0.10, bankruptcy=True)
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    action_counts = {a: 0 for a in ACTIONS.values()}
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        action_name = ACTIONS[int(action)]
        action_counts[action_name] += 1
        
        chaos_indicator = "💥" if env.chaos else "  "
        print(
            f"  Step {step:2d} {chaos_indicator} | "
            f"DD={info['daily_dd']*100:5.2f}% | "
            f"Margin={info['free_margin']*100:4.0f}% | "
            f"Action={action_name:12} | "
            f"Reward={reward:+.2f}"
        )
        
        if terminated or truncated:
            if terminated and env.daily_dd >= env.max_dd:
                print(f"\n  💀 BANKRUPTCY at step {step}")
            elif action == 3:
                print(f"\n  ☠️ KILL SWITCH at step {step}")
            break
    
    print("\n" + "=" * 60)
    print(f"📊 Results:")
    print(f"   Total Reward: {env.total_reward:.2f}")
    print(f"   Chaos Events: {env.chaos_events}")
    print(f"   Action Distribution: {action_counts}")
    print(f"   Survived: {'✅ YES' if env.daily_dd < env.max_dd else '❌ NO'}")
    print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Guardian PPO V3 (Winner-Grade)")
    parser.add_argument("--train", action="store_true", help="Train with curriculum")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--model", type=str, default=None, help="Model path for testing")
    
    args = parser.parse_args()
    
    if args.train:
        train_v3()
    
    if args.test:
        test_v3(args.model)
    
    if not args.train and not args.test:
        print("Usage:")
        print("  python -m src.rl.guardian_ppo_v3 --train")
        print("  python -m src.rl.guardian_ppo_v3 --test")
