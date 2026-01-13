"""
cooldown.py
===========
Signal Cooldown Manager (Fund Discipline Layer)

Prevents:
- Duplicate consecutive signals (when position still open)
- Too frequent trading
- Same context signals (avoid redundancy)

🔧 FIX: Now checks actual MT5 positions, not just signal memory!
"""

import time
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False


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
    - No duplicate direction IF position still open
    - Context change required
    
    🔧 FIX: Duplicate check now uses MT5 positions!
    - If no open position → allow re-entry same direction
    - If position open → block duplicate
    """
    
    def __init__(
        self,
        min_interval: int = 60,
        prevent_duplicate_direction: bool = True,
        require_context_change: bool = True,
        check_mt5_positions: bool = True  # NEW: Check actual positions
    ):
        """
        Args:
            min_interval: Minimum seconds between signals
            prevent_duplicate_direction: Block same direction IF position open
            require_context_change: Block if context (indicators) unchanged
            check_mt5_positions: Use MT5 to verify open positions (recommended)
        """
        self.min_interval = min_interval
        self.prevent_duplicate_direction = prevent_duplicate_direction
        self.require_context_change = require_context_change
        self.check_mt5_positions = check_mt5_positions
        
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
    
    def _has_open_position(self, symbol: str, direction: str) -> bool:
        """
        Check if there's an open position for symbol in given direction.
        
        Returns:
            True if open position exists, False otherwise
        """
        if not self.check_mt5_positions or not HAS_MT5:
            return False  # Can't check, assume no position
        
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                return False
            
            # Check if any position matches direction
            for pos in positions:
                pos_dir = "BUY" if pos.type == 0 else "SELL"
                if pos_dir == direction:
                    return True
            
            return False
        except Exception:
            return False  # Error checking, assume no position
    
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
        
        # 🔧 FIX: Check duplicate direction - NOW WITH MT5 POSITION CHECK!
        if self.prevent_duplicate_direction:
            last_dir = self.last_direction.get(symbol)
            
            if last_dir == direction:
                # Key fix: Only block if position is actually open
                if self._has_open_position(symbol, direction):
                    return CooldownStatus(
                        allowed=False,
                        reason=f"Duplicate direction blocked - {direction} position still open",
                        time_remaining=0
                    )
                else:
                    # Position closed (TP/SL hit) - allow re-entry!
                    # Reset the last_direction to allow this trade
                    pass  # Don't block, fall through to allow
        
        # Check context change
        if self.require_context_change and context:
            ctx_hash = self._hash_context(context)
            last_hash = self.last_context_hash.get(symbol)
            
            if ctx_hash == last_hash:
                # Also check if position is closed
                if self._has_open_position(symbol, direction):
                    return CooldownStatus(
                        allowed=False,
                        reason="Context unchanged and position still open",
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
    
    def on_trade_closed(self, symbol: str):
        """
        Called when a trade closes (TP/SL hit).
        Resets duplicate lock for this symbol to allow re-entry.
        """
        self.last_direction.pop(symbol, None)
        self.last_context_hash.pop(symbol, None)
        # Keep last_time to maintain minimum interval

