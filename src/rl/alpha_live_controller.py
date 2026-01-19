# src/rl/alpha_live_controller.py
"""
Alpha PPO Live Controller
==========================

Controls Alpha PPO V1 in live trading mode.

Features:
    - Live order execution (via Guardian)
    - Confidence gating
    - Alpha-specific kill switch
    - Phase-based configuration

Flow:
    Market → Alpha PPO → Guardian → Execute/Block
"""

import os
import sys
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("ALPHA_LIVE")


@dataclass
class AlphaDecisionLog:
    """Structured decision log for Alpha PPO."""
    timestamp: str
    alpha_action: str
    alpha_confidence: float
    guardian_decision: str
    final_execution: str
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "alpha_action": self.alpha_action,
            "alpha_confidence": f"{self.alpha_confidence:.2f}",
            "guardian_decision": self.guardian_decision,
            "final_execution": self.final_execution,
            "reason": self.reason
        }


class AlphaKillSwitch:
    """Alpha-specific kill switch."""
    
    def __init__(
        self,
        dd_limit: float = 0.03,
        error_limit: int = 3,
        chaos_force_hold: bool = True
    ):
        self.dd_limit = dd_limit
        self.error_limit = error_limit
        self.chaos_force_hold = chaos_force_hold
        
        # State
        self.alpha_dd = 0.0
        self.error_count = 0
        self.triggered = False
        self.trigger_reason = ""
    
    def check(
        self,
        alpha_dd: float = 0.0,
        is_error: bool = False,
        is_chaos: bool = False,
        alpha_action: str = "HOLD"
    ) -> Tuple[bool, str]:
        """Check if kill switch should trigger."""
        if self.triggered:
            return True, self.trigger_reason
        
        # DD limit
        self.alpha_dd = alpha_dd
        if alpha_dd > self.dd_limit:
            self.triggered = True
            self.trigger_reason = f"Alpha DD {alpha_dd:.1%} > {self.dd_limit:.1%}"
            return True, self.trigger_reason
        
        # Error count
        if is_error:
            self.error_count += 1
        if self.error_count >= self.error_limit:
            self.triggered = True
            self.trigger_reason = f"Error count {self.error_count} >= {self.error_limit}"
            return True, self.trigger_reason
        
        # Chaos + BUY
        if self.chaos_force_hold and is_chaos and alpha_action in ["BUY", "SELL"]:
            return True, "Chaos event - force HOLD"
        
        return False, ""
    
    def reset(self):
        """Reset kill switch state."""
        self.triggered = False
        self.trigger_reason = ""
        self.error_count = 0
        self.alpha_dd = 0.0


class AlphaLiveController:
    """
    Controls Alpha PPO V1 in live trading mode.
    
    Responsibilities:
        - Load trained Alpha PPO model
        - Make trading decisions
        - Apply confidence gating
        - Respect Guardian override
        - Log all decisions
    """
    
    def __init__(
        self,
        model_path: str = "models/alpha_ppo_v1_FINAL.zip",
        confidence_threshold: float = 0.55,
        enabled: bool = True
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled
        
        self.model = None
        self.kill_switch = AlphaKillSwitch()
        self.decision_logs: list = []
        
        # Stats
        self.total_decisions = 0
        self.alpha_executed = 0
        self.guardian_blocked = 0
        self.low_confidence_skips = 0
        
        self._load_model()
        logger.info(f"🧠 AlphaLiveController initialized (enabled={enabled})")
    
    def _load_model(self):
        """Load trained Alpha PPO model."""
        try:
            from stable_baselines3 import PPO
            
            # Find latest model
            if os.path.exists(self.model_path):
                self.model = PPO.load(self.model_path)
                logger.info(f"📂 Loaded: {self.model_path}")
            else:
                # Try to find any alpha model
                import glob
                models = glob.glob("models/alpha_ppo_v1*.zip")
                if models:
                    latest = max(models, key=os.path.getctime)
                    self.model = PPO.load(latest)
                    logger.info(f"📂 Loaded: {latest}")
                else:
                    logger.warning("⚠️ No Alpha PPO model found")
        except Exception as e:
            logger.error(f"Failed to load Alpha model: {e}")
    
    def decide(
        self,
        obs: np.ndarray,
        guardian_state: Dict,
        account_state: Dict,
        is_chaos: bool = False
    ) -> Tuple[str, float, str, Dict]:
        """
        Make Alpha trading decision.
        
        Args:
            obs: Environment observation
            guardian_state: Guardian context
            account_state: Account status
            is_chaos: Whether chaos event is active
            
        Returns:
            (action, confidence, source, info)
        """
        self.total_decisions += 1
        timestamp = datetime.now().isoformat()
        
        # Check kill switch
        kill_triggered, kill_reason = self.kill_switch.check(
            alpha_dd=account_state.get("alpha_dd", 0),
            is_chaos=is_chaos,
            alpha_action="HOLD"  # Pre-check
        )
        
        if kill_triggered:
            logger.warning(f"🚨 Alpha Kill Switch: {kill_reason}")
            return "HOLD", 0.0, "KILL_SWITCH", {"reason": kill_reason}
        
        # If disabled or no model, fallback
        if not self.enabled or self.model is None:
            return "HOLD", 0.0, "DISABLED", {"reason": "Alpha disabled"}
        
        try:
            # Get Alpha decision
            action, _ = self.model.predict(obs, deterministic=True)
            action_int = int(action)
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            alpha_action = action_map.get(action_int, "HOLD")
            
            # Estimate confidence (simplified)
            confidence = 0.5 + 0.3 * np.random.random()  # Placeholder
            
            # Re-check kill switch with actual action
            kill_triggered, kill_reason = self.kill_switch.check(
                alpha_dd=account_state.get("alpha_dd", 0),
                is_chaos=is_chaos,
                alpha_action=alpha_action
            )
            
            if kill_triggered:
                return "HOLD", confidence, "KILL_SWITCH", {"reason": kill_reason}
            
            # Confidence gating
            if confidence < self.confidence_threshold:
                self.low_confidence_skips += 1
                return "HOLD", confidence, "LOW_CONFIDENCE", {"threshold": self.confidence_threshold}
            
            # Guardian check (simulated - real impl uses Guardian module)
            guardian_blocked = guardian_state.get("blocked", False)
            if guardian_blocked:
                self.guardian_blocked += 1
                return "HOLD", confidence, "GUARDIAN_BLOCKED", {"guardian": guardian_state}
            
            # Execute Alpha decision
            self.alpha_executed += 1
            
            # Log decision
            log = AlphaDecisionLog(
                timestamp=timestamp,
                alpha_action=alpha_action,
                alpha_confidence=confidence,
                guardian_decision="ALLOW",
                final_execution=alpha_action,
                reason=f"Confidence {confidence:.0%} >= {self.confidence_threshold:.0%}"
            )
            self.decision_logs.append(log)
            
            logger.info(f"🧠 Alpha: {alpha_action} (conf={confidence:.0%})")
            
            return alpha_action, confidence, "ALPHA_PPO", {"log": log.to_dict()}
            
        except Exception as e:
            self.kill_switch.check(is_error=True)
            logger.error(f"Alpha decision error: {e}")
            return "HOLD", 0.0, "ERROR", {"error": str(e)}
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return {
            "enabled": self.enabled,
            "total_decisions": self.total_decisions,
            "alpha_executed": self.alpha_executed,
            "guardian_blocked": self.guardian_blocked,
            "low_confidence_skips": self.low_confidence_skips,
            "kill_switch_triggered": self.kill_switch.triggered,
            "alpha_execution_rate": self.alpha_executed / max(self.total_decisions, 1)
        }
    
    def enable(self):
        """Enable Alpha live trading."""
        self.enabled = True
        self.kill_switch.reset()
        logger.info("🟢 Alpha Live ENABLED")
    
    def disable(self):
        """Disable Alpha live trading."""
        self.enabled = False
        logger.info("🔴 Alpha Live DISABLED")


# =============================================================================
# Singleton
# =============================================================================

_alpha_controller: Optional[AlphaLiveController] = None


def get_alpha_live_controller(**kwargs) -> AlphaLiveController:
    """Get singleton Alpha controller."""
    global _alpha_controller
    if _alpha_controller is None:
        _alpha_controller = AlphaLiveController(**kwargs)
    return _alpha_controller


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha Live Controller Test")
    print("=" * 60)
    
    controller = AlphaLiveController(enabled=True)
    
    # Test decision
    obs = np.zeros(12)
    action, conf, source, info = controller.decide(
        obs=obs,
        guardian_state={"blocked": False},
        account_state={"alpha_dd": 0.01}
    )
    
    print(f"\nDecision: {action}")
    print(f"Confidence: {conf:.2f}")
    print(f"Source: {source}")
    print(f"\nStats: {controller.get_stats()}")
    
    print("=" * 60)
