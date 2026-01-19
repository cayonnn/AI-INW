# src/rl/alpha_hybrid_controller.py
"""
Alpha Hybrid Controller
=======================

Controls Alpha PPO operating modes and hybrid decision logic.

Modes:
    SHADOW: PPO thinks, Rule executes (current default)
    HYBRID: PPO executes if confident, else Rule fallback
    FULL_PPO: PPO executes, Rule disabled

Safety Features:
    - Confidence threshold enforcement
    - Guardian compatibility pre-check
    - Mode switching via command line
    - Emergency fallback to Rule

Command Line:
    python live_loop_v3.py --alpha-mode shadow
    python live_loop_v3.py --alpha-mode hybrid
    python live_loop_v3.py --alpha-mode full
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rl.alpha_decision import (
    AlphaDecision, AlphaAction, AlphaMode, MarketRegime,
    AlphaTradeResult, create_alpha_decision
)
from src.utils.logger import get_logger

logger = get_logger("ALPHA_HYBRID")


@dataclass
class AlphaHybridConfig:
    """Configuration for Alpha Hybrid Controller."""
    # Mode (can be overridden by command line)
    default_mode: AlphaMode = AlphaMode.SHADOW
    
    # Confidence thresholds
    min_confidence_for_trade: float = 0.60
    min_confidence_for_override: float = 0.75  # To override Rule disagreement
    
    # Risk thresholds
    max_risk_score: float = 0.70
    
    # Safety constraints (stricter than Rule)
    alpha_max_dd_soft: float = 0.08   # 8% - Alpha stops trading
    alpha_max_dd_hard: float = 0.10   # 10% - Alpha forced HOLD
    alpha_margin_buffer: float = 1.20  # 20% extra margin required
    
    # Fallback settings
    fallback_to_rule_on_error: bool = True
    log_all_decisions: bool = True


class AlphaHybridController:
    """
    Hybrid decision engine for Alpha PPO.
    
    Manages the transition from Rule-based to PPO-based trading
    with full safety controls.
    
    Decision Flow:
        1. Get PPO decision (AlphaDecision)
        2. Get Rule decision (for comparison/fallback)
        3. Apply hybrid logic based on mode
        4. Check Guardian compatibility
        5. Return final decision
    """
    
    def __init__(self, config: Optional[AlphaHybridConfig] = None):
        self.config = config or AlphaHybridConfig()
        self.mode = self.config.default_mode
        
        # Stats
        self.total_decisions = 0
        self.ppo_used = 0
        self.rule_fallback = 0
        self.guardian_overrides = 0
        
        # Safety state
        self.emergency_mode = False
        self.last_error = None
        
        logger.info(f"🧠 AlphaHybridController initialized (mode={self.mode.value})")
    
    def set_mode(self, mode: str):
        """
        Set operating mode.
        
        Args:
            mode: "shadow", "hybrid", or "full"
        """
        try:
            self.mode = AlphaMode(mode.lower())
            logger.info(f"🔄 Alpha mode changed to: {self.mode.value}")
        except ValueError:
            logger.error(f"Invalid alpha mode: {mode}")
            self.mode = AlphaMode.SHADOW
    
    def decide(
        self,
        ppo_decision: AlphaDecision,
        rule_signal: str,
        guardian_state: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Make hybrid decision.
        
        Args:
            ppo_decision: Decision from Alpha PPO
            rule_signal: Signal from Rule engine (BUY/SELL/HOLD)
            guardian_state: Current Guardian state
            market_state: Current market conditions
            
        Returns:
            (final_action, source, info_dict)
            - final_action: "BUY", "SELL", or "HOLD"
            - source: "PPO", "RULE", or "GUARDIAN_OVERRIDE"
            - info_dict: Detailed decision info
        """
        self.total_decisions += 1
        
        # Build info dict
        info = {
            "mode": self.mode.value,
            "ppo_action": ppo_decision.action_name,
            "ppo_confidence": ppo_decision.confidence,
            "ppo_risk": ppo_decision.risk_score,
            "rule_signal": rule_signal,
            "agreed": ppo_decision.action_name == rule_signal,
            "timestamp": datetime.now().isoformat()
        }
        
        # Emergency mode check
        if self.emergency_mode:
            logger.warning("🚨 EMERGENCY MODE: Forcing Rule fallback")
            self.rule_fallback += 1
            return rule_signal, "RULE_EMERGENCY", info
        
        # Mode-specific logic
        if self.mode == AlphaMode.SHADOW:
            # Shadow mode: Always use Rule
            self.rule_fallback += 1
            return rule_signal, "RULE", info
        
        elif self.mode == AlphaMode.FULL_PPO:
            # Full PPO mode: Always use PPO (but Guardian can still block)
            final_action = ppo_decision.action_name
            
            # Check extra safety constraints for Alpha
            blocked, reason = self._check_alpha_safety(
                ppo_decision, guardian_state, market_state
            )
            
            if blocked:
                self.guardian_overrides += 1
                info["blocked_reason"] = reason
                return "HOLD", "GUARDIAN_OVERRIDE", info
            
            self.ppo_used += 1
            return final_action, "PPO", info
        
        else:  # HYBRID mode
            return self._hybrid_decision(
                ppo_decision, rule_signal, guardian_state, market_state, info
            )
    
    def _hybrid_decision(
        self,
        ppo_decision: AlphaDecision,
        rule_signal: str,
        guardian_state: Dict[str, Any],
        market_state: Dict[str, Any],
        info: Dict
    ) -> Tuple[str, str, Dict]:
        """
        Hybrid decision logic.
        
        Rules:
            1. If PPO confidence < threshold → use Rule
            2. If PPO agrees with Rule → use PPO
            3. If PPO disagrees but very confident → use PPO
            4. Otherwise → use Rule
        """
        # Check confidence threshold
        if ppo_decision.confidence < self.config.min_confidence_for_trade:
            self.rule_fallback += 1
            info["fallback_reason"] = "LOW_CONFIDENCE"
            logger.info(
                f"📊 Hybrid: PPO conf={ppo_decision.confidence:.0%} < threshold, "
                f"using Rule={rule_signal}"
            )
            return rule_signal, "RULE", info
        
        # Check risk threshold
        if ppo_decision.risk_score > self.config.max_risk_score:
            self.rule_fallback += 1
            info["fallback_reason"] = "HIGH_RISK"
            logger.info(
                f"📊 Hybrid: PPO risk={ppo_decision.risk_score:.0%} > threshold, "
                f"using Rule={rule_signal}"
            )
            return rule_signal, "RULE", info
        
        # Check Alpha-specific safety
        blocked, reason = self._check_alpha_safety(
            ppo_decision, guardian_state, market_state
        )
        
        if blocked:
            self.guardian_overrides += 1
            info["blocked_reason"] = reason
            return "HOLD", "GUARDIAN_OVERRIDE", info
        
        # Agreement check
        ppo_action = ppo_decision.action_name
        
        if ppo_action == rule_signal:
            # Agreement → use PPO (for consistent logging)
            self.ppo_used += 1
            info["agreed"] = True
            return ppo_action, "PPO", info
        
        # Disagreement → need higher confidence to override Rule
        if ppo_decision.confidence >= self.config.min_confidence_for_override:
            self.ppo_used += 1
            info["override"] = True
            logger.info(
                f"🧠 Hybrid: PPO overrides Rule ({ppo_action} vs {rule_signal}) "
                f"with conf={ppo_decision.confidence:.0%}"
            )
            return ppo_action, "PPO", info
        
        # Default to Rule
        self.rule_fallback += 1
        info["fallback_reason"] = "DISAGREEMENT_LOW_CONF"
        return rule_signal, "RULE", info
    
    def _check_alpha_safety(
        self,
        ppo_decision: AlphaDecision,
        guardian_state: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Extra safety checks specific to Alpha PPO.
        
        Alpha has STRICTER constraints than Rule.
        
        Returns:
            (blocked, reason) - True if Alpha should be blocked
        """
        # Check DD soft limit
        current_dd = guardian_state.get("current_dd", 0)
        if current_dd > self.config.alpha_max_dd_soft:
            if ppo_decision.is_trade:
                return True, f"ALPHA_DD_SOFT ({current_dd:.1%} > {self.config.alpha_max_dd_soft:.1%})"
        
        # Check DD hard limit
        if current_dd > self.config.alpha_max_dd_hard:
            return True, f"ALPHA_DD_HARD ({current_dd:.1%})"
        
        # Check margin (Alpha needs more buffer)
        margin_ratio = market_state.get("margin_ratio", 1.0)
        if margin_ratio < self.config.alpha_margin_buffer - 1:
            if ppo_decision.is_trade:
                return True, f"ALPHA_MARGIN ({margin_ratio:.0%} margin)"
        
        # Check volatility spike
        if market_state.get("regime") == "VOLATILE" and ppo_decision.risk_score > 0.5:
            if ppo_decision.is_trade:
                return True, "ALPHA_VOLATILE_HIGH_RISK"
        
        # Check Guardian latch
        if guardian_state.get("is_latched", False):
            return True, "GUARDIAN_LATCHED"
        
        return False, None
    
    def enter_emergency_mode(self, reason: str):
        """Enter emergency mode - forces Rule fallback."""
        self.emergency_mode = True
        self.last_error = reason
        logger.critical(f"🚨 ALPHA EMERGENCY MODE: {reason}")
    
    def exit_emergency_mode(self):
        """Exit emergency mode."""
        if self.emergency_mode:
            self.emergency_mode = False
            logger.info("✅ Alpha emergency mode cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        total = max(self.total_decisions, 1)
        return {
            "mode": self.mode.value,
            "total_decisions": self.total_decisions,
            "ppo_used_rate": self.ppo_used / total,
            "rule_fallback_rate": self.rule_fallback / total,
            "guardian_override_rate": self.guardian_overrides / total,
            "emergency_mode": self.emergency_mode
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        stats = self.get_stats()
        return (
            f"🧠 Alpha Hybrid: mode={stats['mode']} | "
            f"PPO={stats['ppo_used_rate']:.0%} | "
            f"Rule={stats['rule_fallback_rate']:.0%} | "
            f"Override={stats['guardian_override_rate']:.0%}"
        )


# =============================================================================
# Singleton
# =============================================================================

_controller: Optional[AlphaHybridController] = None


def get_alpha_hybrid_controller(
    mode: str = "shadow"
) -> AlphaHybridController:
    """Get singleton Alpha Hybrid Controller."""
    global _controller
    if _controller is None:
        _controller = AlphaHybridController()
        _controller.set_mode(mode)
    return _controller


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpha Hybrid Controller")
    parser.add_argument("--mode", choices=["shadow", "hybrid", "full"], default="hybrid")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Alpha Hybrid Controller Test")
    print("=" * 60)
    
    controller = AlphaHybridController()
    controller.set_mode(args.mode)
    
    # Test decisions
    test_cases = [
        # (ppo_action, ppo_conf, ppo_risk, rule_signal, current_dd)
        (1, 0.85, 0.30, "BUY", 0.03),   # Agreement, high conf
        (1, 0.45, 0.30, "BUY", 0.03),   # Agreement, low conf
        (2, 0.82, 0.40, "BUY", 0.03),   # Disagreement, high conf
        (2, 0.65, 0.40, "BUY", 0.03),   # Disagreement, medium conf
        (1, 0.90, 0.30, "SELL", 0.09),  # High DD
    ]
    
    for i, (action, conf, risk, rule, dd) in enumerate(test_cases):
        ppo_decision = create_alpha_decision(
            action=action,
            confidence=conf,
            risk_score=risk,
            regime="TREND"
        )
        
        final, source, info = controller.decide(
            ppo_decision=ppo_decision,
            rule_signal=rule,
            guardian_state={"current_dd": dd, "is_latched": False},
            market_state={"margin_ratio": 0.5, "regime": "TREND"}
        )
        
        print(f"\nTest {i+1}: PPO={ppo_decision.action_name}({conf:.0%}) Rule={rule} DD={dd:.0%}")
        print(f"  → Final={final} (source={source})")
    
    print(f"\n{controller.summary()}")
    print("=" * 60)
