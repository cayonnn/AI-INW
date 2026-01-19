# src/rl/guardian_ppo_infer.py
"""
Guardian PPO Inference Module
==============================

Loads trained PPO V3 model and provides inference API
for hybrid integration with rule-based Guardian.

Usage:
    advisor = GuardianPPOAdvisor("models/guardian_ppo_v3_*.zip")
    action, confidence = advisor.decide(state)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import glob

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_PPO_INFER")


# Action mapping
ACTIONS = {
    0: "ALLOW",
    1: "REDUCE_RISK",
    2: "FORCE_HOLD",
    3: "EMERGENCY_FREEZE"
}


class GuardianPPOAdvisor:
    """
    PPO-based Guardian Advisor.
    
    Acts as a soft advisor layer on top of rule-based Guardian.
    Does NOT override safety rules - only provides recommendations.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.65,
        enabled: bool = True
    ):
        """
        Initialize PPO advisor.
        
        Args:
            model_path: Path to trained PPO model (.zip)
            confidence_threshold: Minimum confidence to act on advice
            enabled: Whether advisor is active
        """
        self.model = None
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Stats
        self.decisions = 0
        self.blocks = 0
        
        if enabled:
            self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load PPO model from path or find latest."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.warning("stable_baselines3 not installed - PPO disabled")
            self.enabled = False
            return
        
        # Find model
        if model_path is None:
            # Find latest V3 model
            models_dir = Path("models")
            v3_models = list(models_dir.glob("guardian_ppo_v3_*.zip"))
            if not v3_models:
                logger.warning("No PPO V3 model found - advisor disabled")
                self.enabled = False
                return
            model_path = str(sorted(v3_models)[-1])
        
        try:
            self.model = PPO.load(model_path, device='cuda')
            self.model_path = model_path
            logger.info(f"✅ Guardian PPO Advisor loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            self.enabled = False
    
    def _state_to_obs(self, state: Dict) -> np.ndarray:
        """Convert state dict to observation array."""
        # Get values with defaults
        dd = state.get("daily_dd", 0.0)
        margin = state.get("margin_ratio", 1.0)
        free_margin = state.get("free_margin_ratio", margin)
        
        # Normalize to match training environment
        max_dd = 0.10
        
        return np.array([
            min(dd / max_dd, 1.0),           # DD pressure
            free_margin,                      # Free margin ratio
            float(state.get("chaos", 0)),     # Chaos flag
            state.get("step", 0) / 500        # Time progress
        ], dtype=np.float32)
    
    def decide(self, state: Dict) -> Tuple[str, float]:
        """
        Get PPO recommendation.
        
        Args:
            state: Current trading state
            
        Returns:
            Tuple of (action_name, confidence)
        """
        self.decisions += 1
        
        if not self.enabled or self.model is None:
            return "ALLOW", 0.0
        
        try:
            obs = self._state_to_obs(state)
            
            # Get action probabilities
            obs_tensor = self.model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            
            # Get predicted action
            action = int(np.argmax(probs))
            confidence = float(probs[action])
            
            action_name = ACTIONS.get(action, "ALLOW")
            
            if action_name != "ALLOW" and confidence >= self.confidence_threshold:
                self.blocks += 1
            
            return action_name, confidence
            
        except Exception as e:
            logger.debug(f"PPO inference error: {e}")
            return "ALLOW", 0.0
    
    def get_stats(self) -> Dict:
        """Get advisor statistics."""
        return {
            "enabled": self.enabled,
            "model_path": self.model_path,
            "decisions": self.decisions,
            "blocks": self.blocks,
            "block_rate": self.blocks / max(self.decisions, 1),
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.decisions = 0
        self.blocks = 0


# Singleton
_advisor: Optional[GuardianPPOAdvisor] = None


def get_ppo_advisor(enabled: bool = True) -> GuardianPPOAdvisor:
    """Get singleton PPO advisor."""
    global _advisor
    if _advisor is None:
        _advisor = GuardianPPOAdvisor(enabled=enabled)
    return _advisor


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("\n=== Guardian PPO Advisor Test ===\n")
    
    advisor = GuardianPPOAdvisor()
    
    # Test states
    test_states = [
        {"daily_dd": 0.02, "margin_ratio": 0.8, "chaos": 0},
        {"daily_dd": 0.05, "margin_ratio": 0.5, "chaos": 0},
        {"daily_dd": 0.08, "margin_ratio": 0.3, "chaos": 1},
        {"daily_dd": 0.12, "margin_ratio": 0.2, "chaos": 1},
    ]
    
    for i, state in enumerate(test_states):
        action, conf = advisor.decide(state)
        print(f"  State {i+1}: DD={state['daily_dd']*100:.0f}%, Margin={state['margin_ratio']*100:.0f}%")
        print(f"    → PPO: {action} (conf={conf:.2f})\n")
    
    print(f"Stats: {advisor.get_stats()}")
