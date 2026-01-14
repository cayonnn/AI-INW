# src/safety/progressive_guard.py
"""
Progressive Safety Guard
=========================

Multi-level response to system stress:
- Level 1 (75%): Reduce pyramid
- Level 2 (80%): Freeze new entries  
- Level 3 (85%): Kill switch

Can monitor memory, DD, or any metric.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from src.utils.logger import get_logger

logger = get_logger("PROGRESSIVE_GUARD")


class AlertLevel(Enum):
    """Progressive alert levels."""
    OK = 0
    LEVEL_1 = 1      # Yellow - reduce risk
    LEVEL_2 = 2      # Orange - freeze new entries
    LEVEL_3 = 3      # Red - kill switch


@dataclass
class AlertThresholds:
    """Configurable thresholds."""
    level_1: float
    level_2: float
    level_3: float


class ProgressiveGuard:
    """
    Progressive Safety Response with LATCHED KILL.
    
    Memory Thresholds:
    - 75%: Reduce pyramid, lower risk
    - 80%: Freeze new entries
    - 85%: Kill switch (LATCHED - cannot reset!)
    
    DD Thresholds:
    - 2%: Reduce risk
    - 3%: Freeze entries
    - 5%: Kill switch (LATCHED)
    
    IMPORTANT: Once LEVEL_3 is triggered, state is LATCHED.
    Cannot reset without explicit unlock (requires manual intervention).
    """
    
    # Default memory thresholds
    MEMORY_THRESHOLDS = AlertThresholds(75, 80, 85)
    
    # Default DD thresholds
    DD_THRESHOLDS = AlertThresholds(2.0, 3.0, 5.0)
    
    def __init__(self):
        """Initialize progressive guard."""
        self.current_level = AlertLevel.OK
        self.memory_thresholds = self.MEMORY_THRESHOLDS
        self.dd_thresholds = self.DD_THRESHOLDS
        
        # LATCHED STATE - once True, cannot be reset without unlock
        self.kill_latched = False
        self.latch_time: Optional[datetime] = None
        self.latch_reason: Optional[str] = None
        
        # Action callbacks
        self._level_actions: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.LEVEL_1: [],
            AlertLevel.LEVEL_2: [],
            AlertLevel.LEVEL_3: [],
        }
        
        logger.info("ProgressiveGuard initialized (with latched kill)")
    
    def check_memory(self) -> AlertLevel:
        """Check memory and return alert level."""
        # If latched, always return LEVEL_3
        if self.kill_latched:
            return AlertLevel.LEVEL_3
        
        if not HAS_PSUTIL:
            return AlertLevel.OK
        
        try:
            mem = psutil.virtual_memory()
            percent = mem.percent
            
            level = self._get_level(percent, self.memory_thresholds)
            
            if level != self.current_level:
                self._on_level_change("memory", level, percent)
                self.current_level = level
            
            return level
            
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return AlertLevel.OK
    
    def check_dd(self, dd_percent: float) -> AlertLevel:
        """Check drawdown and return alert level."""
        # If latched, always return LEVEL_3
        if self.kill_latched:
            return AlertLevel.LEVEL_3
        
        level = self._get_level(dd_percent, self.dd_thresholds)
        
        if level != self.current_level:
            self._on_level_change("dd", level, dd_percent)
            self.current_level = level
        
        return level
    
    def _get_level(
        self,
        value: float,
        thresholds: AlertThresholds
    ) -> AlertLevel:
        """Determine level based on value."""
        if value >= thresholds.level_3:
            return AlertLevel.LEVEL_3
        elif value >= thresholds.level_2:
            return AlertLevel.LEVEL_2
        elif value >= thresholds.level_1:
            return AlertLevel.LEVEL_1
        return AlertLevel.OK
    
    def _on_level_change(
        self,
        source: str,
        new_level: AlertLevel,
        value: float
    ) -> None:
        """Handle level change."""
        if new_level == AlertLevel.LEVEL_1:
            logger.warning(f"⚠️ LEVEL 1 ({source}={value:.1f}%): Reducing risk")
            self._execute_level_1()
        
        elif new_level == AlertLevel.LEVEL_2:
            logger.warning(f"🟠 LEVEL 2 ({source}={value:.1f}%): Freezing entries")
            self._execute_level_2()
        
        elif new_level == AlertLevel.LEVEL_3:
            logger.critical(f"🔴 LEVEL 3 ({source}={value:.1f}%): Kill switch!")
            self._execute_level_3(source, value)
        
        elif new_level == AlertLevel.OK:
            logger.info(f"✅ Level OK ({source}={value:.1f}%)")
        
        # Execute registered callbacks
        for callback in self._level_actions.get(new_level, []):
            try:
                callback(source, value)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _execute_level_1(self) -> None:
        """Level 1: Reduce pyramid and risk."""
        os.environ["PYRAMID_DISABLED"] = "1"
        os.environ["RISK_REDUCED"] = "1"
        logger.info("Level 1 actions: Pyramid disabled, risk reduced")
    
    def _execute_level_2(self) -> None:
        """Level 2: Freeze new entries."""
        os.environ["ENTRIES_FROZEN"] = "1"
        logger.info("Level 2 actions: New entries frozen")
    
    def _execute_level_3(self, source: str = "unknown", value: float = 0) -> None:
        """Level 3: Kill switch - LATCHED."""
        # LATCH THE STATE - cannot be undone without unlock
        self.kill_latched = True
        self.latch_time = datetime.now()
        self.latch_reason = f"{source}={value:.1f}%"
        self.current_level = AlertLevel.LEVEL_3
        
        # Set all restrictions
        os.environ["PYRAMID_DISABLED"] = "1"
        os.environ["RISK_REDUCED"] = "1"
        os.environ["ENTRIES_FROZEN"] = "1"
        os.environ["KILL_LATCHED"] = "1"
        
        from src.safety.kill_switch import get_kill_switch, KillReason
        
        kill = get_kill_switch()
        kill._trigger(KillReason.SYSTEM_ERROR, "Progressive guard level 3")
        
        logger.critical(f"🔒 KILL LATCHED - Manual unlock required")
    
    def on_level(self, level: AlertLevel, callback: Callable) -> None:
        """Register callback for specific level."""
        if level in self._level_actions:
            self._level_actions[level].append(callback)
    
    def reset(self) -> None:
        """
        Reset to OK level.
        
        WARNING: Does NOT work if kill is latched!
        Use unlock() for latched state.
        """
        if self.kill_latched:
            logger.error("❌ Cannot reset - KILL IS LATCHED. Use unlock() instead.")
            return
        
        self.current_level = AlertLevel.OK
        os.environ.pop("PYRAMID_DISABLED", None)
        os.environ.pop("RISK_REDUCED", None)
        os.environ.pop("ENTRIES_FROZEN", None)
        logger.info("Progressive guard reset to OK")
    
    def unlock(self, confirmation: str = "") -> bool:
        """
        Unlock latched kill state.
        
        Requires confirmation string: "CONFIRM_UNLOCK"
        This is a dangerous operation - use with caution.
        """
        if confirmation != "CONFIRM_UNLOCK":
            logger.error("❌ Unlock failed - invalid confirmation")
            return False
        
        if not self.kill_latched:
            logger.info("No latch to unlock")
            return True
        
        self.kill_latched = False
        self.latch_time = None
        self.latch_reason = None
        self.current_level = AlertLevel.OK
        
        os.environ.pop("PYRAMID_DISABLED", None)
        os.environ.pop("RISK_REDUCED", None)
        os.environ.pop("ENTRIES_FROZEN", None)
        os.environ.pop("KILL_LATCHED", None)
        
        logger.warning("🔓 Kill latch UNLOCKED (manual intervention)")
        return True
    
    def get_status(self) -> Dict:
        """Get current status."""
        mem_percent = 0
        if HAS_PSUTIL:
            try:
                mem_percent = psutil.virtual_memory().percent
            except:
                pass
        
        return {
            "level": self.current_level.name,
            "level_value": self.current_level.value,
            "kill_latched": self.kill_latched,
            "latch_time": self.latch_time.isoformat() if self.latch_time else None,
            "latch_reason": self.latch_reason,
            "memory_percent": mem_percent,
            "pyramid_disabled": os.environ.get("PYRAMID_DISABLED") == "1",
            "entries_frozen": os.environ.get("ENTRIES_FROZEN") == "1",
            "risk_reduced": os.environ.get("RISK_REDUCED") == "1",
        }


# Convenience functions
def is_pyramid_disabled() -> bool:
    """Check if pyramid is disabled by guard."""
    return os.environ.get("PYRAMID_DISABLED") == "1"


def is_entries_frozen() -> bool:
    """Check if new entries are frozen."""
    return os.environ.get("ENTRIES_FROZEN") == "1"


def is_risk_reduced() -> bool:
    """Check if risk is reduced."""
    return os.environ.get("RISK_REDUCED") == "1"


# Singleton
_guard: Optional[ProgressiveGuard] = None


def get_progressive_guard() -> ProgressiveGuard:
    """Get singleton ProgressiveGuard."""
    global _guard
    if _guard is None:
        _guard = ProgressiveGuard()
    return _guard


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Progressive Guard Test")
    print("=" * 50)
    
    guard = ProgressiveGuard()
    
    # Test memory
    level = guard.check_memory()
    print(f"\nMemory Level: {level.name}")
    
    # Test DD levels
    for dd in [1.0, 2.5, 3.5, 6.0]:
        level = guard.check_dd(dd)
        print(f"DD {dd}%: Level {level.name}")
        guard.reset()
    
    print(f"\nStatus: {guard.get_status()}")
