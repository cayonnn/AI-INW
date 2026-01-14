# src/core/mode_controller.py
"""
Mode Controller - Dynamic Mode Decision Engine
===============================================

Decides trading mode in real-time based on:
- LiveScore: Competition performance
- Drawdown: Current loss from peak
- Volatility: Market conditions

Mode transitions are smooth with hysteresis to prevent flip-flopping.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta

from src.core.trading_mode import TradingMode, MODE_PROFILES, get_mode_profile
from src.utils.logger import get_logger

logger = get_logger("MODE_CONTROLLER")


@dataclass
class ModeDecision:
    """Mode decision result."""
    mode: TradingMode
    reason: str
    score: float
    drawdown: float
    volatility: float
    profile: dict


class ModeController:
    """
    Dynamic Mode Decision Engine.
    
    Decision Rules:
    ┌─────────────┬─────────────────────────────────────┐
    │ Mode        │ Conditions                          │
    ├─────────────┼─────────────────────────────────────┤
    │ ALPHA       │ Score > 70 AND DD < 5%              │
    │ DEFENSIVE   │ Score < 40 OR DD > 8%               │
    │ NEUTRAL     │ Otherwise                           │
    └─────────────┴─────────────────────────────────────┘
    
    Hysteresis:
    - Mode must persist for 5 cycles before change
    - Prevents rapid mode switching
    """
    
    # Mode thresholds
    ALPHA_SCORE_THRESHOLD = 70
    ALPHA_DD_THRESHOLD = 5.0
    
    DEFENSIVE_SCORE_THRESHOLD = 40
    DEFENSIVE_DD_THRESHOLD = 8.0
    
    # Hysteresis settings
    MIN_CYCLES_BEFORE_CHANGE = 3
    
    def __init__(self):
        """Initialize Mode Controller."""
        self.current_mode = TradingMode.NEUTRAL
        self.mode_since = datetime.now()
        self.cycles_in_mode = 0
        self.pending_mode: Optional[TradingMode] = None
        self.pending_cycles = 0
        
        logger.info(f"ModeController initialized: starting in {self.current_mode.value}")
    
    def decide(
        self,
        score: float,
        drawdown: float,
        volatility: float = 0.5
    ) -> ModeDecision:
        """
        Decide trading mode based on current conditions.
        
        Args:
            score: Current LiveScore (0-100)
            drawdown: Current drawdown percentage
            volatility: Market volatility (0-1)
            
        Returns:
            ModeDecision with mode and profile
        """
        # Determine target mode
        target_mode, reason = self._evaluate_conditions(score, drawdown, volatility)
        
        # Apply hysteresis
        final_mode = self._apply_hysteresis(target_mode)
        
        # Get profile for mode
        profile = get_mode_profile(final_mode)
        
        # Update cycles
        self.cycles_in_mode += 1
        
        return ModeDecision(
            mode=final_mode,
            reason=reason,
            score=score,
            drawdown=drawdown,
            volatility=volatility,
            profile=profile
        )
    
    def _evaluate_conditions(
        self,
        score: float,
        drawdown: float,
        volatility: float
    ) -> tuple[TradingMode, str]:
        """Evaluate conditions and determine target mode."""
        
        # DEFENSIVE conditions (checked first)
        if score < self.DEFENSIVE_SCORE_THRESHOLD:
            return TradingMode.DEFENSIVE, f"Low score ({score:.1f} < {self.DEFENSIVE_SCORE_THRESHOLD})"
        
        if drawdown > self.DEFENSIVE_DD_THRESHOLD:
            return TradingMode.DEFENSIVE, f"High DD ({drawdown:.1f}% > {self.DEFENSIVE_DD_THRESHOLD}%)"
        
        # ALPHA conditions
        if score > self.ALPHA_SCORE_THRESHOLD and drawdown < self.ALPHA_DD_THRESHOLD:
            return TradingMode.ALPHA, f"High score ({score:.1f}) + Low DD ({drawdown:.1f}%)"
        
        # NEUTRAL (default)
        return TradingMode.NEUTRAL, "Normal conditions"
    
    def _apply_hysteresis(self, target_mode: TradingMode) -> TradingMode:
        """Apply hysteresis to prevent rapid mode switching."""
        
        if target_mode == self.current_mode:
            # Same mode, reset pending
            self.pending_mode = None
            self.pending_cycles = 0
            return self.current_mode
        
        # Different mode requested
        if self.pending_mode == target_mode:
            # Same pending mode, increment counter
            self.pending_cycles += 1
            
            if self.pending_cycles >= self.MIN_CYCLES_BEFORE_CHANGE:
                # Switch mode
                old_mode = self.current_mode
                self.current_mode = target_mode
                self.mode_since = datetime.now()
                self.cycles_in_mode = 0
                self.pending_mode = None
                self.pending_cycles = 0
                
                logger.info(f"Mode changed: {old_mode.value} -> {target_mode.value}")
                return self.current_mode
        else:
            # New pending mode
            self.pending_mode = target_mode
            self.pending_cycles = 1
        
        # Return current mode (not changed yet)
        return self.current_mode
    
    def force_mode(self, mode: TradingMode, reason: str = "manual") -> None:
        """Force switch to a specific mode immediately."""
        old_mode = self.current_mode
        self.current_mode = mode
        self.mode_since = datetime.now()
        self.cycles_in_mode = 0
        self.pending_mode = None
        self.pending_cycles = 0
        
        logger.warning(f"Mode FORCED: {old_mode.value} -> {mode.value} ({reason})")
    
    def get_status(self) -> dict:
        """Get current mode status."""
        return {
            "current_mode": self.current_mode.value,
            "mode_since": self.mode_since.isoformat(),
            "cycles_in_mode": self.cycles_in_mode,
            "pending_mode": self.pending_mode.value if self.pending_mode else None,
            "pending_cycles": self.pending_cycles,
            "profile": get_mode_profile(self.current_mode)
        }


# Singleton instance
_controller: Optional[ModeController] = None


def get_mode_controller() -> ModeController:
    """Get or create singleton ModeController."""
    global _controller
    if _controller is None:
        _controller = ModeController()
    return _controller


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ModeController Test")
    print("=" * 60)
    
    controller = ModeController()
    
    # Test scenarios
    scenarios = [
        {"score": 75, "dd": 3, "vol": 0.5},   # Should be ALPHA
        {"score": 75, "dd": 3, "vol": 0.5},   # Still ALPHA
        {"score": 75, "dd": 3, "vol": 0.5},   # Still ALPHA (confirm)
        {"score": 55, "dd": 6, "vol": 0.5},   # Should switch to NEUTRAL
        {"score": 55, "dd": 6, "vol": 0.5},   # Still pending
        {"score": 55, "dd": 6, "vol": 0.5},   # Now NEUTRAL
        {"score": 30, "dd": 10, "vol": 0.8},  # High DD -> DEFENSIVE pending
        {"score": 30, "dd": 10, "vol": 0.8},  # Still pending
        {"score": 30, "dd": 10, "vol": 0.8},  # Now DEFENSIVE
    ]
    
    print("\n--- Mode Transitions ---")
    for i, s in enumerate(scenarios, 1):
        decision = controller.decide(s["score"], s["dd"], s["vol"])
        print(
            f"Cycle {i}: Score={s['score']}, DD={s['dd']}% -> "
            f"Mode={decision.mode.value}, Risk Mult={decision.profile['risk_mult']}"
        )
