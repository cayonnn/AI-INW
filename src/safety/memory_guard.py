# src/safety/memory_guard.py
"""
Memory Guard - Memory Pressure Monitor
=======================================

Monitors system memory and triggers actions:
- WARN (85%): Reduce batch size, pause shadow agents
- CRITICAL (92%): Trigger kill switch

Prevents silent failures from memory exhaustion.
"""

import os
from typing import Tuple, Optional
from enum import Enum

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("MEMORY_GUARD")


class MemoryLevel(Enum):
    """Memory pressure levels."""
    OK = "OK"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class MemoryGuard:
    """
    Memory Pressure Monitor.
    
    Thresholds:
    - OK: < 85%
    - WARN: 85-92%
    - CRITICAL: > 92%
    
    Actions:
    - WARN: Reduce batch, pause shadow
    - CRITICAL: Kill switch
    """
    
    WARN_THRESHOLD = 85
    CRITICAL_THRESHOLD = 92
    
    def __init__(
        self,
        warn_threshold: int = WARN_THRESHOLD,
        critical_threshold: int = CRITICAL_THRESHOLD
    ):
        """
        Initialize memory guard.
        
        Args:
            warn_threshold: Percentage for WARN
            critical_threshold: Percentage for CRITICAL
        """
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold
        self.last_level = MemoryLevel.OK
        
        if not HAS_PSUTIL:
            logger.warning("psutil not installed, memory monitoring disabled")
        
        logger.info(
            f"MemoryGuard initialized: WARN={warn_threshold}%, "
            f"CRITICAL={critical_threshold}%"
        )
    
    def check(self) -> MemoryLevel:
        """
        Check current memory pressure.
        
        Returns:
            MemoryLevel (OK, WARN, CRITICAL)
        """
        if not HAS_PSUTIL:
            return MemoryLevel.OK
        
        try:
            mem = psutil.virtual_memory()
            percent = mem.percent
            
            if percent > self.critical_threshold:
                level = MemoryLevel.CRITICAL
            elif percent > self.warn_threshold:
                level = MemoryLevel.WARN
            else:
                level = MemoryLevel.OK
            
            # Log level changes
            if level != self.last_level:
                if level == MemoryLevel.CRITICAL:
                    logger.critical(f"🚨 MEMORY CRITICAL: {percent:.1f}%")
                elif level == MemoryLevel.WARN:
                    logger.warning(f"⚠️ MEMORY WARN: {percent:.1f}%")
                else:
                    logger.info(f"Memory OK: {percent:.1f}%")
                
                self.last_level = level
            
            return level
            
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return MemoryLevel.OK
    
    def get_usage(self) -> Tuple[float, float, float]:
        """
        Get memory usage details.
        
        Returns:
            (used_gb, available_gb, percent)
        """
        if not HAS_PSUTIL:
            return (0, 0, 0)
        
        try:
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            avail_gb = mem.available / (1024 ** 3)
            return (used_gb, avail_gb, mem.percent)
        except:
            return (0, 0, 0)


def handle_memory_pressure(level: MemoryLevel) -> None:
    """
    Handle memory pressure based on level.
    
    Args:
        level: Current memory level
    """
    if level == MemoryLevel.WARN:
        logger.warning("Memory WARN: Reducing load...")
        reduce_batch_size()
        pause_shadow_agents()
    
    elif level == MemoryLevel.CRITICAL:
        logger.critical("Memory CRITICAL: Triggering kill switch...")
        from src.safety.kill_switch import get_kill_switch, KillReason
        kill = get_kill_switch()
        kill._trigger(KillReason.SYSTEM_ERROR, "Memory pressure critical")


def reduce_batch_size() -> None:
    """Reduce batch sizes to free memory."""
    os.environ["BATCH_SIZE_REDUCED"] = "1"
    logger.info("Batch size reduced")


def pause_shadow_agents() -> None:
    """Pause shadow agents to reduce load."""
    os.environ["SHADOW_PAUSED"] = "1"
    logger.info("Shadow agents paused")


def resume_shadow_agents() -> None:
    """Resume shadow agents."""
    os.environ.pop("SHADOW_PAUSED", None)
    logger.info("Shadow agents resumed")


def is_shadow_paused() -> bool:
    """Check if shadow agents are paused."""
    return os.environ.get("SHADOW_PAUSED") == "1"


# Singleton
_guard: Optional[MemoryGuard] = None


def get_memory_guard() -> MemoryGuard:
    """Get singleton MemoryGuard."""
    global _guard
    if _guard is None:
        _guard = MemoryGuard()
    return _guard


def check_memory() -> MemoryLevel:
    """Convenience function to check memory."""
    guard = get_memory_guard()
    level = guard.check()
    
    if level != MemoryLevel.OK:
        handle_memory_pressure(level)
    
    return level


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Memory Guard Test")
    print("=" * 50)
    
    guard = MemoryGuard()
    level = guard.check()
    used, avail, percent = guard.get_usage()
    
    print(f"\nLevel: {level.value}")
    print(f"Used: {used:.1f} GB")
    print(f"Available: {avail:.1f} GB")
    print(f"Percent: {percent:.1f}%")
    
    print(f"\nThresholds:")
    print(f"  WARN: {guard.warn_threshold}%")
    print(f"  CRITICAL: {guard.critical_threshold}%")
