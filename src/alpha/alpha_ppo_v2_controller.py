# src/alpha/alpha_ppo_v2_controller.py
"""
🧠 Alpha PPO V2 Controller — Live Trading Decision Engine
=========================================================
Features:
- Hybrid/Full mode switching
- Automatic v1 fallback on error
- Confidence-based decision
- Emergency disable capability
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger("AlphaPPOv2")

@dataclass
class AlphaV2Decision:
    """Structured decision output from Alpha PPO v2"""
    action: str  # BUY, SELL, HOLD
    confidence: float
    source: str  # v2, v1_fallback, rule
    reason: str

class AlphaPPOv2Controller:
    """
    Alpha PPO v2 Controller with safe fallback mechanism.
    
    Hierarchy:
    1. Try Alpha v2 prediction
    2. If error or low confidence → fallback to v1
    3. If both fail → fallback to rule signal
    4. Guardian PPO v3 always has final authority
    """
    
    ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    def __init__(
        self,
        model_path: str,
        fallback_model_path: Optional[str] = None,
        mode: str = "hybrid",
        min_trade_conf: float = 0.55,
        override_rule_conf: float = 0.65,
        auto_disable_dd: float = 8.0
    ):
        """
        Initialize Alpha PPO v2 Controller.
        
        Args:
            model_path: Path to v2 model
            fallback_model_path: Path to v1 model (fallback)
            mode: 'shadow' | 'hybrid' | 'full'
            min_trade_conf: Minimum confidence to trade
            override_rule_conf: Confidence to override rule signal
            auto_disable_dd: DD threshold for auto-disable (%)
        """
        self.mode = mode
        self.min_trade_conf = min_trade_conf
        self.override_rule_conf = override_rule_conf
        self.auto_disable_dd = auto_disable_dd
        
        self.model = None
        self.fallback = None
        self.enabled = True
        self.disabled_reason = None
        
        # Stats
        self.stats = {
            "total_predictions": 0,
            "v2_used": 0,
            "v1_fallback_used": 0,
            "rule_fallback_used": 0,
            "errors": 0
        }
        
        # Load models
        self._load_models(model_path, fallback_model_path)
        
    def _load_models(self, model_path: str, fallback_path: Optional[str]):
        """Load v2 and optional v1 fallback models."""
        try:
            from stable_baselines3 import PPO
            
            if Path(model_path).exists():
                self.model = PPO.load(model_path)
                log.info(f"✅ Alpha PPO v2 loaded: {model_path}")
            else:
                log.warning(f"⚠️ Alpha PPO v2 not found: {model_path}")
                self.enabled = False
                self.disabled_reason = "Model not found"
            
            if fallback_path and Path(fallback_path).exists():
                self.fallback = PPO.load(fallback_path)
                log.info(f"✅ Alpha PPO v1 fallback loaded: {fallback_path}")
                
        except Exception as e:
            log.error(f"❌ Failed to load Alpha models: {e}")
            self.enabled = False
            self.disabled_reason = str(e)
    
    def set_mode(self, mode: str):
        """Switch operating mode."""
        if mode not in ["shadow", "hybrid", "full"]:
            log.error(f"Invalid mode: {mode}")
            return
        
        old_mode = self.mode
        self.mode = mode
        log.warning(f"🔄 Alpha PPO v2 mode: {old_mode} → {mode}")
    
    def enable(self):
        """Re-enable after being disabled."""
        self.enabled = True
        self.disabled_reason = None
        log.info("✅ Alpha PPO v2 RE-ENABLED")
    
    def disable(self, reason: str = "Manual disable"):
        """Disable Alpha v2 (will use fallback)."""
        self.enabled = False
        self.disabled_reason = reason
        log.warning(f"🚨 Alpha PPO v2 DISABLED: {reason}")
    
    def check_emergency_disable(self, daily_dd: float) -> bool:
        """Check if DD exceeds threshold and auto-disable."""
        if daily_dd >= self.auto_disable_dd:
            self.disable(f"DD {daily_dd:.1f}% >= {self.auto_disable_dd}%")
            return True
        return False
    
    def predict(self, obs: np.ndarray) -> Tuple[str, float]:
        """
        Get Alpha v2 prediction with confidence.
        
        Args:
            obs: Observation array from environment
            
        Returns:
            (action, confidence) tuple
        """
        self.stats["total_predictions"] += 1
        
        # Check if disabled
        if not self.enabled or self.model is None:
            return self._fallback_predict(obs, "v2_disabled")
        
        try:
            # V2 prediction
            action, _states = self.model.predict(obs, deterministic=True)
            conf = self._estimate_confidence(obs, action)
            
            self.stats["v2_used"] += 1
            return self._decode_action(action), conf
            
        except Exception as e:
            log.error(f"Alpha v2 prediction error: {e}")
            self.stats["errors"] += 1
            return self._fallback_predict(obs, f"error: {e}")
    
    def _fallback_predict(self, obs: np.ndarray, reason: str) -> Tuple[str, float]:
        """Use v1 fallback or return HOLD."""
        if self.fallback is not None:
            try:
                action, _ = self.fallback.predict(obs, deterministic=True)
                self.stats["v1_fallback_used"] += 1
                log.debug(f"🔁 Using Alpha v1 fallback ({reason})")
                return self._decode_action(action), 0.45  # Lower confidence for fallback
            except Exception as e:
                log.error(f"Alpha v1 fallback error: {e}")
        
        self.stats["rule_fallback_used"] += 1
        return "HOLD", 0.0  # Safe default
    
    def _decode_action(self, action) -> str:
        """Convert action index to string."""
        return self.ACTION_MAP.get(int(action), "HOLD")
    
    def _estimate_confidence(self, obs: np.ndarray, action: int) -> float:
        """
        Estimate prediction confidence.
        
        Uses action probability if available, otherwise uses heuristics.
        """
        try:
            # Try to get action probabilities from policy
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'get_distribution'):
                obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
                return float(probs[action])
        except:
            pass
        
        # Fallback: use observation volatility as proxy
        obs_std = np.std(obs) if len(obs) > 0 else 0.5
        return float(np.clip(0.5 + obs_std * 0.3, 0.4, 0.95))
    
    def decide(
        self,
        rule_signal: str,
        obs: np.ndarray,
        daily_dd: float = 0.0
    ) -> AlphaV2Decision:
        """
        Main decision function with mode-aware logic.
        
        Args:
            rule_signal: Signal from rule-based engine
            obs: Observation for PPO
            daily_dd: Current daily drawdown %
            
        Returns:
            AlphaV2Decision with action, confidence, source, reason
        """
        # Emergency check
        if self.check_emergency_disable(daily_dd):
            return AlphaV2Decision(
                action=rule_signal,
                confidence=0.0,
                source="rule",
                reason=f"Alpha v2 auto-disabled (DD={daily_dd:.1f}%)"
            )
        
        # Get Alpha prediction
        alpha_signal, alpha_conf = self.predict(obs)
        
        # Mode-based decision
        if self.mode == "shadow":
            # Shadow mode: always use rule, just log Alpha
            return AlphaV2Decision(
                action=rule_signal,
                confidence=alpha_conf,
                source="rule",
                reason=f"Shadow mode (Alpha would: {alpha_signal} @ {alpha_conf:.0%})"
            )
        
        elif self.mode == "hybrid":
            # Hybrid: use Alpha only if high confidence
            if alpha_conf >= self.override_rule_conf:
                return AlphaV2Decision(
                    action=alpha_signal,
                    confidence=alpha_conf,
                    source="v2",
                    reason=f"High confidence override ({alpha_conf:.0%} >= {self.override_rule_conf:.0%})"
                )
            else:
                return AlphaV2Decision(
                    action=rule_signal,
                    confidence=alpha_conf,
                    source="rule",
                    reason=f"Low confidence ({alpha_conf:.0%} < {self.override_rule_conf:.0%})"
                )
        
        else:  # full
            # Full mode: always use Alpha if above min threshold
            if alpha_conf >= self.min_trade_conf:
                return AlphaV2Decision(
                    action=alpha_signal,
                    confidence=alpha_conf,
                    source="v2",
                    reason=f"Full mode ({alpha_conf:.0%})"
                )
            else:
                return AlphaV2Decision(
                    action="HOLD",
                    confidence=alpha_conf,
                    source="v2",
                    reason=f"Below min confidence ({alpha_conf:.0%} < {self.min_trade_conf:.0%})"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total = max(self.stats["total_predictions"], 1)
        return {
            **self.stats,
            "v2_usage_pct": self.stats["v2_used"] / total * 100,
            "fallback_pct": (self.stats["v1_fallback_used"] + self.stats["rule_fallback_used"]) / total * 100,
            "mode": self.mode,
            "enabled": self.enabled
        }
    
    def summary(self) -> str:
        """Get formatted summary string."""
        stats = self.get_stats()
        return (
            f"🧠 Alpha v2 | Mode={self.mode} | Enabled={self.enabled} | "
            f"v2={stats['v2_usage_pct']:.0f}% | fallback={stats['fallback_pct']:.0f}% | "
            f"errors={stats['errors']}"
        )
