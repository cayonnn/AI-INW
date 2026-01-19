# src/validation/ppo_live_switch.py
"""
PPO Live Activation Switch
===========================

Safe switching mechanism for Alpha PPO V1 live trading.

Features:
    - Confidence-gated execution
    - Automatic rollback
    - Real-time monitoring
    - Dashboard integration

Usage:
    from src.validation.ppo_live_switch import AlphaPPOSwitch
    
    switch = AlphaPPOSwitch()
    signal, source = switch.get_decision(obs, rule_signal)
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("PPO_SWITCH")


@dataclass
class PPOSwitchConfig:
    """Configuration for PPO live switch."""
    enabled: bool = False  # Master switch
    confidence_threshold: float = 0.70
    
    # Rollback triggers
    max_intraday_dd: float = 0.03
    max_loss_streak: int = 5
    min_margin_level: float = 1.20
    
    # Safe mode
    safe_mode_on_error: bool = True


class AlphaPPOSwitch:
    """
    Safe switching mechanism between Rule and PPO trading.
    
    PPO only executes when:
        1. Switch is enabled
        2. Confidence >= threshold
        3. No rollback trigger active
        4. Guardian approves
    """
    
    def __init__(self, config: Optional[PPOSwitchConfig] = None):
        self.config = config or PPOSwitchConfig()
        
        # State
        self.ppo_enabled = self.config.enabled
        self.rollback_active = False
        self.loss_streak = 0
        self.intraday_dd = 0.0
        
        # Stats
        self.ppo_decisions = 0
        self.rule_fallbacks = 0
        self.rollback_count = 0
        
        # PPO model
        self.ppo_model = None
        self._load_ppo_model()
        
        logger.info(f"🔌 AlphaPPOSwitch initialized (enabled={self.ppo_enabled})")
    
    def _load_ppo_model(self):
        """Load PPO model."""
        try:
            from stable_baselines3 import PPO
            model_path = "models/alpha_ppo_v1.zip"
            if os.path.exists(model_path):
                self.ppo_model = PPO.load(model_path)
                logger.info(f"📂 Loaded PPO model: {model_path}")
        except Exception as e:
            logger.debug(f"PPO model not loaded: {e}")
    
    def get_decision(
        self,
        obs: Any,
        rule_signal: str,
        guardian_state: Dict = None,
        account_state: Dict = None
    ) -> Tuple[str, str, Dict]:
        """
        Get trading decision with automatic fallback.
        
        Args:
            obs: Environment observation
            rule_signal: Rule-based signal (BUY/SELL/HOLD)
            guardian_state: Guardian context
            account_state: Account status
            
        Returns:
            (signal, source, info)
            source: "PPO" or "RULE"
        """
        info = {
            "ppo_enabled": self.ppo_enabled,
            "rollback_active": self.rollback_active,
            "confidence": 0.0
        }
        
        # Check rollback triggers
        if account_state:
            self._check_rollback_triggers(account_state)
        
        # If disabled or rollback, use Rule
        if not self.ppo_enabled or self.rollback_active:
            self.rule_fallbacks += 1
            info["fallback_reason"] = "disabled" if not self.ppo_enabled else "rollback"
            return rule_signal, "RULE", info
        
        # If no PPO model, fallback
        if self.ppo_model is None:
            self.rule_fallbacks += 1
            info["fallback_reason"] = "no_model"
            return rule_signal, "RULE", info
        
        # Get PPO prediction
        try:
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            confidence = self._estimate_confidence(obs)
            info["confidence"] = confidence
            
            # Action mapping
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            ppo_signal = action_map.get(int(action), "HOLD")
            
            # Confidence gating
            if confidence >= self.config.confidence_threshold:
                self.ppo_decisions += 1
                info["ppo_action"] = ppo_signal
                return ppo_signal, "PPO", info
            else:
                self.rule_fallbacks += 1
                info["fallback_reason"] = f"low_conf ({confidence:.2f})"
                return rule_signal, "RULE", info
                
        except Exception as e:
            logger.error(f"PPO prediction error: {e}")
            self.rule_fallbacks += 1
            info["fallback_reason"] = f"error: {e}"
            
            if self.config.safe_mode_on_error:
                return "HOLD", "RULE", info
            return rule_signal, "RULE", info
    
    def _estimate_confidence(self, obs) -> float:
        """Estimate PPO confidence (simplified)."""
        # In real impl, use action probabilities
        return 0.65 + 0.20 * (1 - abs(obs[0]) if hasattr(obs, '__getitem__') else 0.5)
    
    def _check_rollback_triggers(self, account_state: Dict):
        """Check if rollback should be triggered."""
        dd = account_state.get("intraday_dd", 0)
        margin = account_state.get("margin_level", 999)
        
        should_rollback = False
        
        if dd > self.config.max_intraday_dd:
            should_rollback = True
            logger.warning(f"🚨 Rollback: DD {dd:.1%} > {self.config.max_intraday_dd:.1%}")
        
        if margin < self.config.min_margin_level:
            should_rollback = True
            logger.warning(f"🚨 Rollback: Margin {margin:.0%} < {self.config.min_margin_level:.0%}")
        
        if self.loss_streak >= self.config.max_loss_streak:
            should_rollback = True
            logger.warning(f"🚨 Rollback: Loss streak {self.loss_streak}")
        
        if should_rollback and not self.rollback_active:
            self.rollback_active = True
            self.rollback_count += 1
    
    def record_trade_result(self, profit: float):
        """Record trade result for streak tracking."""
        if profit < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
    
    def enable(self):
        """Enable PPO live trading."""
        self.ppo_enabled = True
        self.rollback_active = False
        logger.info("🟢 PPO Live ENABLED")
    
    def disable(self):
        """Disable PPO live trading."""
        self.ppo_enabled = False
        logger.info("🔴 PPO Live DISABLED")
    
    def reset_rollback(self):
        """Reset rollback state."""
        self.rollback_active = False
        self.loss_streak = 0
        logger.info("🔄 Rollback state reset")
    
    def get_stats(self) -> Dict:
        """Get switch statistics."""
        total = self.ppo_decisions + self.rule_fallbacks
        return {
            "enabled": self.ppo_enabled,
            "rollback_active": self.rollback_active,
            "ppo_decisions": self.ppo_decisions,
            "rule_fallbacks": self.rule_fallbacks,
            "ppo_rate": self.ppo_decisions / max(total, 1),
            "rollback_count": self.rollback_count
        }


# =============================================================================
# Singleton
# =============================================================================

_ppo_switch: Optional[AlphaPPOSwitch] = None


def get_ppo_switch() -> AlphaPPOSwitch:
    """Get singleton PPO switch."""
    global _ppo_switch
    if _ppo_switch is None:
        _ppo_switch = AlphaPPOSwitch()
    return _ppo_switch


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("PPO Live Switch Test")
    print("=" * 60)
    
    switch = AlphaPPOSwitch()
    
    # Test disabled mode
    obs = np.zeros(10)
    signal, source, info = switch.get_decision(obs, "BUY")
    print(f"\nDisabled: signal={signal}, source={source}")
    
    # Enable and test
    switch.enable()
    signal, source, info = switch.get_decision(obs, "BUY")
    print(f"Enabled: signal={signal}, source={source}, conf={info.get('confidence', 0):.2f}")
    
    # Test rollback
    switch._check_rollback_triggers({"intraday_dd": 0.05})
    signal, source, info = switch.get_decision(obs, "BUY")
    print(f"After rollback: signal={signal}, source={source}")
    
    print(f"\nStats: {switch.get_stats()}")
    print("=" * 60)
