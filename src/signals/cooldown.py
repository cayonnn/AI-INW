"""
cooldown.py
===========
Signal Cooldown Manager (Fund Discipline Layer)

Prevents:
- Duplicate consecutive signals
- Too frequent trading
- Same context signals (avoid redundancy)
"""

import time
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class CooldownStatus:
    """Cooldown check result."""
    allowed: bool
    reason: str
    time_remaining: float  # seconds until allowed


class SignalCooldown:
    """
    Signal cooldown manager.
    
    Fund-Grade discipline:
    - Minimum interval between signals
    - No duplicate direction in a row
    - Context change required
    """
    
    def __init__(
        self,
        min_interval: int = 60,
        prevent_duplicate_direction: bool = True,
        require_context_change: bool = True
    ):
        """
        Args:
            min_interval: Minimum seconds between signals
            prevent_duplicate_direction: Block same direction consecutively
            require_context_change: Block if context (indicators) unchanged
        """
        self.min_interval = min_interval
        self.prevent_duplicate_direction = prevent_duplicate_direction
        self.require_context_change = require_context_change
        
        # State tracking per symbol
        self.last_time: Dict[str, float] = {}
        self.last_direction: Dict[str, str] = {}
        self.last_context_hash: Dict[str, str] = {}
    
    def _hash_context(self, context: dict) -> str:
        """Create hash of context for comparison."""
        # Round values to avoid floating point noise
        rounded = {}
        for k, v in context.items():
            if isinstance(v, float):
                rounded[k] = round(v, 2)
            else:
                rounded[k] = v
        
        raw = str(sorted(rounded.items()))
        return hashlib.md5(raw.encode()).hexdigest()[:8]
    
    def check(self, symbol: str, direction: str, context: dict = None) -> CooldownStatus:
        """
        Check if signal is allowed.
        
        Args:
            symbol: Trading symbol
            direction: Signal direction
            context: Signal context (indicators, etc.)
            
        Returns:
            CooldownStatus with allowed flag and reason
        """
        now = time.time()
        context = context or {}
        
        # Check time interval
        last_time = self.last_time.get(symbol, 0)
        elapsed = now - last_time
        
        if elapsed < self.min_interval:
            remaining = self.min_interval - elapsed
            return CooldownStatus(
                allowed=False,
                reason=f"Cooldown: {remaining:.0f}s remaining",
                time_remaining=remaining
            )
        
        # Check duplicate direction
        if self.prevent_duplicate_direction:
            last_dir = self.last_direction.get(symbol)
            if last_dir == direction:
                return CooldownStatus(
                    allowed=False,
                    reason=f"Duplicate direction blocked (last={last_dir})",
                    time_remaining=0
                )
        
        # Check context change
        if self.require_context_change and context:
            ctx_hash = self._hash_context(context)
            last_hash = self.last_context_hash.get(symbol)
            
            if ctx_hash == last_hash:
                return CooldownStatus(
                    allowed=False,
                    reason="Context unchanged - signal redundant",
                    time_remaining=0
                )
        
        return CooldownStatus(
            allowed=True,
            reason="OK",
            time_remaining=0
        )
    
    def record(self, symbol: str, direction: str, context: dict = None):
        """Record a signal that was executed."""
        self.last_time[symbol] = time.time()
        self.last_direction[symbol] = direction
        
        if context:
            self.last_context_hash[symbol] = self._hash_context(context)
    
    def reset(self, symbol: str = None):
        """Reset cooldown state."""
        if symbol:
            self.last_time.pop(symbol, None)
            self.last_direction.pop(symbol, None)
            self.last_context_hash.pop(symbol, None)
        else:
            self.last_time.clear()
            self.last_direction.clear()
            self.last_context_hash.clear()
