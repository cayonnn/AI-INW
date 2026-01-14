# src/execution/pyramid_manager.py
"""
Multi-Entry Pyramid Manager - Competition Grade
================================================

Smart pyramid scaling for winning trades only:
- Entry 1: Base signal + AI pass → 1.0R
- Entry 2: Price ≥ +1R → 0.7R  
- Entry 3: Price ≥ +2R → 0.4R

Safety guards:
- No pyramid if drawdown > threshold
- No pyramid if regime changes
- Max 3 entries per trade
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger("PYRAMID_MANAGER")


@dataclass
class PyramidEntry:
    """Single pyramid entry record."""
    ticket: int
    entry_price: float
    risk_multiplier: float
    entry_time: datetime
    r_level: int  # 0=initial, 1=+1R, 2=+2R


@dataclass 
class PyramidState:
    """Current pyramid state for a position."""
    position_id: int
    symbol: str
    direction: str  # BUY or SELL
    initial_price: float
    sl_distance: float
    entries: List[PyramidEntry]
    max_entries: int = 3


class PyramidManager:
    """
    Multi-Entry Pyramid Manager.
    
    Risk Multipliers:
    ┌─────────┬───────────┬────────────┐
    │ Entry # │ Condition │ Risk Mult  │
    ├─────────┼───────────┼────────────┤
    │ 1       │ Signal    │ 1.0        │
    │ 2       │ +1R       │ 0.7        │
    │ 3       │ +2R       │ 0.4        │
    └─────────┴───────────┴────────────┘
    """
    
    # Risk multipliers for each entry level
    RISK_MULTIPLIERS = [1.0, 0.7, 0.4]
    
    def __init__(
        self,
        max_entries: int = 3,
        min_r_for_pyramid: float = 1.0,  # Minimum +1R to add
        dd_threshold: float = 0.03,      # No pyramid if DD > 3%
        mode: str = "smart"              # smart, safe, aggressive
    ):
        """
        Initialize Pyramid Manager.
        
        Args:
            max_entries: Maximum entries per position (default 3)
            min_r_for_pyramid: Minimum R profit to add entry
            dd_threshold: Max drawdown % to allow pyramid
            mode: "smart" (default), "safe", "aggressive"
        """
        self.max_entries = max_entries
        self.min_r_for_pyramid = min_r_for_pyramid
        self.dd_threshold = dd_threshold
        self.mode = mode
        
        # Active pyramids by position_id
        self.active_pyramids: dict[int, PyramidState] = {}
        
        # Mode-specific settings
        if mode == "safe":
            self.RISK_MULTIPLIERS = [1.0, 0.5, 0.3]
            self.max_entries = 2
        elif mode == "aggressive":
            self.RISK_MULTIPLIERS = [1.0, 0.8, 0.6]
            self.max_entries = 4
        
        logger.info(
            f"PyramidManager initialized: mode={mode}, "
            f"max_entries={self.max_entries}, dd_threshold={dd_threshold:.1%}"
        )
    
    def start_pyramid(
        self,
        position_id: int,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_distance: float,
        ticket: int
    ) -> None:
        """
        Start tracking a new position for pyramid.
        
        Args:
            position_id: MT5 position ID
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Initial entry price
            sl_distance: SL distance in points
            ticket: Trade ticket
        """
        initial_entry = PyramidEntry(
            ticket=ticket,
            entry_price=entry_price,
            risk_multiplier=self.RISK_MULTIPLIERS[0],
            entry_time=datetime.now(),
            r_level=0
        )
        
        self.active_pyramids[position_id] = PyramidState(
            position_id=position_id,
            symbol=symbol,
            direction=direction,
            initial_price=entry_price,
            sl_distance=sl_distance,
            entries=[initial_entry],
            max_entries=self.max_entries
        )
        
        logger.info(f"Pyramid started: {symbol} {direction} @ {entry_price}")
    
    def can_add_entry(
        self,
        position_id: int,
        current_price: float,
        current_dd: float = 0.0,
        regime_stable: bool = True
    ) -> tuple[bool, str]:
        """
        Check if we can add another pyramid entry.
        
        Args:
            position_id: Position to check
            current_price: Current market price
            current_dd: Current drawdown %
            regime_stable: Is market regime stable?
            
        Returns:
            (can_add, reason)
        """
        if position_id not in self.active_pyramids:
            return False, "Position not tracked"
        
        state = self.active_pyramids[position_id]
        
        # Check max entries
        if len(state.entries) >= state.max_entries:
            return False, f"Max entries reached ({state.max_entries})"
        
        # Check drawdown threshold
        if current_dd > self.dd_threshold:
            return False, f"DD too high ({current_dd:.1%} > {self.dd_threshold:.1%})"
        
        # Check regime stability
        if not regime_stable:
            return False, "Regime unstable"
        
        # Calculate current R-multiple
        current_r = self._calc_r_multiple(state, current_price)
        required_r = len(state.entries)  # Entry 2 needs +1R, Entry 3 needs +2R
        
        if current_r < required_r:
            return False, f"R too low ({current_r:.2f}R < {required_r}R)"
        
        return True, f"OK - Position at +{current_r:.2f}R"
    
    def _calc_r_multiple(self, state: PyramidState, current_price: float) -> float:
        """Calculate current R-multiple for position."""
        if state.sl_distance == 0:
            return 0
            
        if state.direction == "BUY":
            profit_points = current_price - state.initial_price
        else:  # SELL
            profit_points = state.initial_price - current_price
        
        return profit_points / state.sl_distance
    
    def get_next_risk_multiplier(self, position_id: int) -> float:
        """
        Get risk multiplier for next entry.
        
        Returns:
            Risk multiplier (1.0, 0.7, 0.4, etc.)
        """
        if position_id not in self.active_pyramids:
            return self.RISK_MULTIPLIERS[0]
        
        state = self.active_pyramids[position_id]
        entry_num = min(len(state.entries), len(self.RISK_MULTIPLIERS) - 1)
        
        return self.RISK_MULTIPLIERS[entry_num]
    
    def register_entry(
        self,
        position_id: int,
        ticket: int,
        entry_price: float
    ) -> None:
        """Register a new pyramid entry."""
        if position_id not in self.active_pyramids:
            logger.warning(f"Position {position_id} not found for pyramid")
            return
        
        state = self.active_pyramids[position_id]
        r_level = len(state.entries)
        
        entry = PyramidEntry(
            ticket=ticket,
            entry_price=entry_price,
            risk_multiplier=self.get_next_risk_multiplier(position_id),
            entry_time=datetime.now(),
            r_level=r_level
        )
        
        state.entries.append(entry)
        
        logger.info(
            f"Pyramid entry #{r_level + 1}: {state.symbol} @ {entry_price} "
            f"(risk mult: {entry.risk_multiplier})"
        )
    
    def close_pyramid(self, position_id: int) -> Optional[PyramidState]:
        """Close and return pyramid state."""
        if position_id in self.active_pyramids:
            state = self.active_pyramids.pop(position_id)
            logger.info(
                f"Pyramid closed: {state.symbol} with {len(state.entries)} entries"
            )
            return state
        return None
    
    def get_state(self, position_id: int) -> Optional[PyramidState]:
        """Get current pyramid state."""
        return self.active_pyramids.get(position_id)
    
    def get_stats(self) -> dict:
        """Get pyramid manager statistics."""
        total_entries = sum(len(s.entries) for s in self.active_pyramids.values())
        
        return {
            "active_pyramids": len(self.active_pyramids),
            "total_entries": total_entries,
            "mode": self.mode,
            "max_entries": self.max_entries,
        }


# Singleton instance
_pyramid_manager: Optional[PyramidManager] = None


def get_pyramid_manager(mode: str = "smart") -> PyramidManager:
    """Get or create singleton PyramidManager."""
    global _pyramid_manager
    if _pyramid_manager is None:
        _pyramid_manager = PyramidManager(mode=mode)
    return _pyramid_manager


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PyramidManager Test")
    print("=" * 60)
    
    pm = PyramidManager(mode="smart")
    
    # Simulate a winning trade
    pm.start_pyramid(
        position_id=1,
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.00,
        sl_distance=10.0,
        ticket=12345
    )
    
    print("\n--- Checking pyramid entries ---")
    
    # Check at various price levels
    test_prices = [2655, 2660, 2665, 2670, 2680]
    
    for price in test_prices:
        can_add, reason = pm.can_add_entry(1, price)
        r_mult = pm.get_next_risk_multiplier(1)
        print(f"Price {price}: can_add={can_add}, mult={r_mult}, reason={reason}")
        
        if can_add:
            pm.register_entry(1, 12346 + len(pm.active_pyramids[1].entries), price)
    
    print("\n--- Final State ---")
    state = pm.get_state(1)
    if state:
        print(f"Entries: {len(state.entries)}")
        for e in state.entries:
            print(f"  #{e.r_level + 1}: price={e.entry_price}, mult={e.risk_multiplier}")
