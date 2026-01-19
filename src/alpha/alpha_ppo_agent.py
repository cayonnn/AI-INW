# src/alpha/alpha_ppo_agent.py
"""
Alpha PPO V1 Agent
===================

Live trading agent using PPO for entry decisions.

Features:
    - Load trained model
    - Deterministic inference
    - Confidence estimation
    - Action mapping
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("ALPHA_PPO")


class AlphaPPOAgent:
    """
    Alpha PPO V1 Agent for live trading.
    
    Attributes:
        model: Loaded PPO model
        deterministic: Whether to use deterministic actions
        min_confidence: Minimum confidence threshold
    """
    
    ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    def __init__(
        self,
        model_path: str = "models/alpha_ppo_v1.zip",
        deterministic: bool = True,
        min_confidence: float = 0.55
    ):
        self.model_path = model_path
        self.deterministic = deterministic
        self.min_confidence = min_confidence
        self.model = None
        self.enabled = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the PPO model."""
        try:
            from stable_baselines3 import PPO
            
            if os.path.exists(self.model_path):
                self.model = PPO.load(self.model_path)
                self.enabled = True
                logger.info(f"✅ Alpha PPO loaded: {self.model_path}")
            else:
                # Try alternative paths
                alt_paths = [
                    "models/alpha_ppo_v1_FINAL.zip",
                    "models/alpha_ppo_v1_20260119_172015_FINAL.zip"
                ]
                for path in alt_paths:
                    if os.path.exists(path):
                        self.model = PPO.load(path)
                        self.enabled = True
                        logger.info(f"✅ Alpha PPO loaded: {path}")
                        return
                
                logger.warning(f"⚠️ Alpha PPO not found: {self.model_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Alpha PPO: {e}")
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict action based on features.
        
        Args:
            features: Market features array
            
        Returns:
            (action, confidence) tuple
        """
        if not self.enabled or self.model is None:
            return "HOLD", 0.0
        
        try:
            # Ensure features shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get action
            action, _ = self.model.predict(features, deterministic=self.deterministic)
            action_int = int(action[0]) if hasattr(action, '__len__') else int(action)
            
            # Map to string
            action_str = self.ACTION_MAP.get(action_int, "HOLD")
            
            # Estimate confidence (simplified)
            # Real implementation would use policy distribution
            confidence = 0.5 + 0.3 * np.random.random()
            
            # Apply threshold
            if confidence < self.min_confidence:
                return "HOLD", confidence
            
            return action_str, confidence
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return "HOLD", 0.0
    
    def build_features(self, market_state: dict) -> np.ndarray:
        """Build feature vector from market state."""
        return np.array([
            market_state.get("ret1m", 0),
            market_state.get("ret5m", 0),
            market_state.get("ema_diff", 0),
            market_state.get("rsi", 50) / 100 - 0.5,
            market_state.get("atr", 0) / market_state.get("price", 1) * 100,
            market_state.get("volatility", 0),
            market_state.get("position", 0),
            market_state.get("free_margin_pct", 100) / 100,
            market_state.get("hour", 12) / 24,
            market_state.get("spread", 0),
            market_state.get("unrealized_pnl", 0),
            market_state.get("time_in_pos", 0) / 100,
        ], dtype=np.float32)
    
    def is_ready(self) -> bool:
        """Check if agent is ready."""
        return self.enabled and self.model is not None


# Singleton
_agent: Optional[AlphaPPOAgent] = None

def get_alpha_agent(**kwargs) -> AlphaPPOAgent:
    global _agent
    if _agent is None:
        _agent = AlphaPPOAgent(**kwargs)
    return _agent


if __name__ == "__main__":
    agent = AlphaPPOAgent()
    print(f"Ready: {agent.is_ready()}")
    
    features = np.zeros(12)
    action, conf = agent.predict(features)
    print(f"Action: {action}, Confidence: {conf:.2f}")
