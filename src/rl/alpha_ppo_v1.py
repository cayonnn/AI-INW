# src/rl/alpha_ppo_v1.py
"""
Alpha PPO V1 Training & Inference
=================================

Train and deploy Alpha PPO for shadow/live trading.

Usage:
    # Training
    python src/rl/alpha_ppo_v1.py --mode train --episodes 100000
    
    # Evaluation
    python src/rl/alpha_ppo_v1.py --mode eval --model models/alpha_ppo_v1.zip

Architecture:
    AlphaTradingEnv (alpha_env.py) 
        → PPO (Stable-Baselines3)
        → AlphaPPOInference (this file)
        → Shadow Mode in live_loop
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rl.alpha_env import AlphaTradingEnv, AlphaEnvConfig, AlphaStateNormalizer, AlphaAction
from src.utils.logger import get_logger

logger = get_logger("ALPHA_PPO")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AlphaPPOConfig:
    """Training & inference configuration."""
    # Model paths
    model_path: str = "models/alpha_ppo_v1.zip"
    tensorboard_log: str = "logs/alpha_ppo_tb"
    
    # Training hyperparameters
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    
    # Environment config
    max_positions: int = 5
    dd_penalty: float = 2.0
    guardian_penalty: float = 3.0
    
    # Inference config
    confidence_threshold: float = 0.60
    

# =============================================================================
# Training
# =============================================================================

def train_alpha_ppo(
    config: Optional[AlphaPPOConfig] = None,
    data_path: str = "data/imitation_full_dataset.csv"
) -> str:
    """
    Train Alpha PPO V1 using historical data.
    
    Args:
        config: Training configuration
        data_path: Path to training dataset
        
    Returns:
        Path to saved model
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    except ImportError:
        logger.error("stable_baselines3 not installed. Run: pip install stable-baselines3")
        raise
    
    config = config or AlphaPPOConfig()
    
    logger.info("=" * 60)
    logger.info("🚀 Alpha PPO V1 Training Started")
    logger.info("=" * 60)
    
    # Load data for environment
    if Path(data_path).exists():
        df = pd.read_csv(data_path)
        logger.info(f"📊 Loaded training data: {len(df)} rows")
    else:
        logger.warning(f"No data found at {data_path}, using random environment")
        df = None
    
    # Create environment
    env_config = AlphaEnvConfig(
        max_positions=config.max_positions,
        dd_penalty=config.dd_penalty,
        guardian_penalty=config.guardian_penalty,
        max_steps=1000
    )
    
    def make_env():
        return AlphaTradingEnv(config=env_config)
    
    vec_env = DummyVecEnv([make_env])
    
    # Create model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        tensorboard_log=config.tensorboard_log,
        verbose=1
    )
    
    # Callbacks
    os.makedirs("models/checkpoints/alpha", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/checkpoints/alpha",
        name_prefix="alpha_ppo"
    )
    
    # Train
    logger.info(f"🎯 Training for {config.total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    model.save(config.model_path)
    logger.info(f"✅ Model saved to: {config.model_path}")
    
    return config.model_path


# =============================================================================
# Inference
# =============================================================================

class AlphaPPOInference:
    """
    Alpha PPO V1 Inference for Shadow Mode.
    
    Usage:
        inference = AlphaPPOInference()
        if inference.enabled:
            action, confidence = inference.predict(state_dict)
    """
    
    def __init__(
        self,
        model_path: str = "models/alpha_ppo_v1.zip",
        enabled: bool = True,
        confidence_threshold: float = 0.60
    ):
        self.model_path = model_path
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.normalizer = AlphaStateNormalizer()
        
        # Stats
        self.decisions = 0
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model."""
        if not self.enabled:
            return
            
        if not Path(self.model_path).exists():
            logger.warning(f"Alpha PPO model not found: {self.model_path}")
            self.enabled = False
            return
        
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(self.model_path)
            logger.info(f"✅ Alpha PPO loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load Alpha PPO: {e}")
            self.enabled = False
    
    def predict(
        self,
        state: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Predict action from state.
        
        Args:
            state: Dictionary with market/account features
                
        Returns:
            (action_name, confidence) tuple
            action_name: "HOLD", "BUY", or "SELL"
            confidence: Probability of chosen action [0, 1]
        """
        if not self.enabled or self.model is None:
            return "HOLD", 0.0
        
        try:
            # Build 12-feature observation (matching MarketEnvV3)
            obs = np.array([
                state.get("ret_1", state.get("ema_fast_diff", 0.0)),          # 0: return 1
                state.get("ret_5", 0.0),                                        # 1: return 5
                state.get("ema_diff", state.get("ema_fast_diff", 0.0)),        # 2: ema diff
                state.get("rsi", 50.0) / 100.0 - 0.5,                          # 3: rsi normalized
                state.get("atr", 10.0) / 100.0,                                # 4: atr normalized
                state.get("volatility", 0.02),                                 # 5: volatility
                np.tanh(state.get("open_positions", 0) / 5),                   # 6: position
                state.get("margin_used", 0.1),                                 # 7: margin
                state.get("time_of_day", 0.5) * 2 - 1,                         # 8: hour normalized
                state.get("spread", 0.1) / 10.0,                               # 9: spread
                state.get("floating_dd", 0.0) * 4,                             # 10: unrealized pnl
                state.get("guardian_state", 0) / 2.0                           # 11: time in position
            ], dtype=np.float32)
            
            # Get action and probabilities
            action, _states = self.model.predict(obs, deterministic=False)
            
            # Get action probabilities for confidence
            obs_tensor = self.model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().detach().numpy()[0]
            
            action = int(action)
            confidence = float(probs[action])
            
            # Track stats
            self.decisions += 1
            self.action_counts[action] += 1
            
            action_name = AlphaAction(action).name
            return action_name, confidence
            
        except Exception as e:
            logger.error(f"Alpha PPO prediction error: {e}")
            return "HOLD", 0.0
    
    def predict_from_df(
        self,
        row,
        open_positions: int,
        floating_dd: float,
        guardian_state: int
    ) -> Tuple[str, float]:
        """
        Predict from DataFrame row (convenience method).
        
        Args:
            row: DataFrame row with market data
            open_positions: Current position count
            floating_dd: Current drawdown
            guardian_state: 0=OK, 1=WARNING, 2=BLOCK
            
        Returns:
            (action_name, confidence)
        """
        state = self.normalizer.normalize(
            row=row,
            open_positions=open_positions,
            floating_dd=floating_dd,
            guardian_state=guardian_state
        )
        return self.predict(state)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        total = max(self.decisions, 1)
        return {
            "total_decisions": self.decisions,
            "hold_rate": self.action_counts[0] / total,
            "buy_rate": self.action_counts[1] / total,
            "sell_rate": self.action_counts[2] / total,
            "enabled": self.enabled
        }


# =============================================================================
# Singleton Access
# =============================================================================

_alpha_ppo: Optional[AlphaPPOInference] = None


def get_alpha_ppo(
    enabled: bool = True,
    model_path: str = "models/alpha_ppo_v1.zip"
) -> AlphaPPOInference:
    """Get singleton Alpha PPO inference instance."""
    global _alpha_ppo
    if _alpha_ppo is None:
        _alpha_ppo = AlphaPPOInference(
            model_path=model_path,
            enabled=enabled
        )
    return _alpha_ppo


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpha PPO V1 Training & Inference")
    parser.add_argument("--mode", choices=["train", "eval", "test"], default="test")
    parser.add_argument("--model", default="models/alpha_ppo_v1.zip")
    parser.add_argument("--episodes", type=int, default=500000)
    parser.add_argument("--data", default="data/imitation_full_dataset.csv")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        config = AlphaPPOConfig(
            total_timesteps=args.episodes,
            model_path=args.model
        )
        train_alpha_ppo(config, args.data)
        
    elif args.mode == "eval":
        inference = AlphaPPOInference(model_path=args.model)
        print(f"Model loaded: {inference.enabled}")
        
        # Test prediction
        test_state = {
            "ema_fast_diff": 0.1,
            "rsi": 0.2,
            "atr": 0.3,
            "spread": 0.1,
            "time_of_day": 0.5,
            "open_positions": 1,
            "floating_dd": 0.02,
            "guardian_state": 0
        }
        
        for i in range(10):
            action, conf = inference.predict(test_state)
            print(f"Prediction {i+1}: {action} (conf={conf:.2f})")
        
        print(f"\nStats: {inference.get_stats()}")
        
    else:  # test mode
        print("=" * 60)
        print("Alpha PPO V1 Module Test")
        print("=" * 60)
        
        # Test environment
        from src.rl.alpha_env import AlphaTradingEnv, AlphaEnvConfig
        
        config = AlphaEnvConfig(max_steps=10)
        env = AlphaTradingEnv(config=config)
        
        obs, info = env.reset()
        print(f"Env observation shape: {obs.shape}")
        print(f"Env action space: {env.action_space}")
        
        # Test inference (may fail if no model)
        inference = get_alpha_ppo(enabled=True)
        print(f"Inference enabled: {inference.enabled}")
        
        print("\n✅ Module test passed")
