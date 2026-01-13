"""
strategy_pool.py
=================
Self-Pruning Strategy Pool

ระบบ "ตัดกิ่ง" อัตโนมัติ

Strategy Lifecycle:
Candidate → Active → At-Risk → Frozen → Retired

Fund Rule: Strategy ที่ "ไม่ชัด" อันตรายกว่า strategy ที่แพ้
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from src.utils.logger import get_logger

logger = get_logger("STRATEGY_POOL")


class StrategyState(str, Enum):
    """Strategy lifecycle states."""
    CANDIDATE = "CANDIDATE"     # New, being evaluated
    ACTIVE = "ACTIVE"           # Full capital allocation
    REDUCED = "REDUCED"         # Reduced capital (at-risk)
    FROZEN = "FROZEN"           # No new trades
    RETIRED = "RETIRED"         # Permanently disabled


@dataclass
class StrategyEntry:
    """Strategy entry in the pool."""
    name: str
    state: StrategyState
    capital_multiplier: float    # 0.0 to 1.0
    decay_score: float = 0.0
    alpha_confidence: float = 0.5
    consecutive_losses: int = 0
    last_trade: Optional[datetime] = None
    state_entered_at: datetime = field(default_factory=datetime.now)
    freeze_reason: str = ""
    total_trades: int = 0
    win_rate: float = 0.5


class SelfPruningStrategyPool:
    """
    Self-Pruning Strategy Pool.
    
    Automatically manages strategy lifecycle based on performance.
    """

    def __init__(self):
        self.strategies: Dict[str, StrategyEntry] = {}
        
        # Thresholds
        self.decay_threshold = 0.6
        self.alpha_confidence_threshold = 0.3
        self.max_consecutive_losses = 5
        self.max_freeze_days = 14
        self.candidate_min_trades = 30

    # -------------------------------------------------
    # Strategy management
    # -------------------------------------------------
    def add_strategy(self, name: str, as_candidate: bool = True) -> StrategyEntry:
        """Add new strategy to pool."""
        state = StrategyState.CANDIDATE if as_candidate else StrategyState.ACTIVE
        capital = 0.2 if as_candidate else 1.0
        
        entry = StrategyEntry(
            name=name,
            state=state,
            capital_multiplier=capital,
        )
        self.strategies[name] = entry
        
        logger.info(f"Strategy added: {name} as {state.value}")
        return entry

    def get_strategy(self, name: str) -> Optional[StrategyEntry]:
        """Get strategy entry."""
        return self.strategies.get(name)

    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy names."""
        return [
            name for name, entry in self.strategies.items()
            if entry.state in [StrategyState.ACTIVE, StrategyState.REDUCED]
        ]

    # -------------------------------------------------
    # Lifecycle transitions
    # -------------------------------------------------
    def evaluate(self, name: str, decay_score: float, alpha_confidence: float):
        """
        Evaluate strategy and handle transitions.
        
        Args:
            name: Strategy name
            decay_score: From decay detector (0-1)
            alpha_confidence: From alpha attribution (0-1)
        """
        if name not in self.strategies:
            return
        
        entry = self.strategies[name]
        entry.decay_score = decay_score
        entry.alpha_confidence = alpha_confidence
        
        # State transitions
        if entry.state == StrategyState.CANDIDATE:
            self._evaluate_candidate(entry)
        elif entry.state == StrategyState.ACTIVE:
            self._evaluate_active(entry)
        elif entry.state == StrategyState.REDUCED:
            self._evaluate_reduced(entry)
        elif entry.state == StrategyState.FROZEN:
            self._evaluate_frozen(entry)

    def _evaluate_candidate(self, entry: StrategyEntry):
        """Evaluate candidate strategy."""
        if entry.total_trades >= self.candidate_min_trades:
            if entry.alpha_confidence >= 0.5 and entry.decay_score < self.decay_threshold:
                self._transition(entry, StrategyState.ACTIVE, 1.0)
                logger.info(f"Strategy {entry.name} promoted to ACTIVE")
            else:
                self._transition(entry, StrategyState.RETIRED, 0.0, "Failed evaluation")

    def _evaluate_active(self, entry: StrategyEntry):
        """Evaluate active strategy."""
        # Check for decay
        if entry.decay_score > self.decay_threshold:
            self._transition(entry, StrategyState.REDUCED, 0.5)
            logger.warning(f"Strategy {entry.name} moved to REDUCED (decay)")
        
        # Check consecutive losses
        elif entry.consecutive_losses >= self.max_consecutive_losses:
            self._transition(entry, StrategyState.REDUCED, 0.5)
            logger.warning(f"Strategy {entry.name} moved to REDUCED (losses)")

    def _evaluate_reduced(self, entry: StrategyEntry):
        """Evaluate reduced strategy."""
        # Confirm decay and freeze
        if (entry.decay_score > self.decay_threshold and 
            entry.alpha_confidence < self.alpha_confidence_threshold):
            self._transition(entry, StrategyState.FROZEN, 0.0, 
                           f"Decay confirmed: {entry.decay_score:.2f}")
            logger.warning(f"Strategy {entry.name} FROZEN")
        
        # Recovery possible
        elif entry.decay_score < 0.4 and entry.alpha_confidence > 0.5:
            self._transition(entry, StrategyState.ACTIVE, 1.0)
            logger.info(f"Strategy {entry.name} recovered to ACTIVE")

    def _evaluate_frozen(self, entry: StrategyEntry):
        """Evaluate frozen strategy."""
        days_frozen = (datetime.now() - entry.state_entered_at).days
        
        # Check for retirement
        if days_frozen >= self.max_freeze_days:
            self._transition(entry, StrategyState.RETIRED, 0.0,
                           f"Frozen for {days_frozen} days")
            logger.warning(f"Strategy {entry.name} RETIRED")
        
        # Allow reactivation if metrics improve significantly
        elif entry.alpha_confidence > 0.7 and entry.decay_score < 0.3:
            self._transition(entry, StrategyState.REDUCED, 0.3)
            logger.info(f"Strategy {entry.name} unfrozen to REDUCED")

    def _transition(self, entry: StrategyEntry, new_state: StrategyState,
                   capital: float, reason: str = ""):
        """Transition strategy to new state."""
        old_state = entry.state
        entry.state = new_state
        entry.capital_multiplier = capital
        entry.state_entered_at = datetime.now()
        if reason:
            entry.freeze_reason = reason
        
        logger.info(f"Strategy {entry.name}: {old_state.value} → {new_state.value}")

    # -------------------------------------------------
    # Trade updates
    # -------------------------------------------------
    def record_trade(self, name: str, is_win: bool):
        """Record trade result."""
        if name not in self.strategies:
            return
        
        entry = self.strategies[name]
        entry.total_trades += 1
        entry.last_trade = datetime.now()
        
        if is_win:
            entry.consecutive_losses = 0
        else:
            entry.consecutive_losses += 1
        
        # Update win rate
        # Simple exponential moving average
        alpha = 0.1
        win_value = 1.0 if is_win else 0.0
        entry.win_rate = entry.win_rate * (1 - alpha) + win_value * alpha

    # -------------------------------------------------
    # Capital allocation
    # -------------------------------------------------
    def get_capital_multiplier(self, name: str) -> float:
        """Get capital multiplier for strategy."""
        entry = self.strategies.get(name)
        if not entry:
            return 0.0
        return entry.capital_multiplier

    def get_allocatable_strategies(self) -> Dict[str, float]:
        """Get strategies that can receive capital."""
        return {
            name: entry.capital_multiplier
            for name, entry in self.strategies.items()
            if entry.state in [StrategyState.ACTIVE, StrategyState.REDUCED, StrategyState.CANDIDATE]
            and entry.capital_multiplier > 0
        }

    # -------------------------------------------------
    # Manual controls
    # -------------------------------------------------
    def freeze_strategy(self, name: str, reason: str = "Manual freeze"):
        """Manually freeze a strategy."""
        if name in self.strategies:
            self._transition(self.strategies[name], StrategyState.FROZEN, 0.0, reason)

    def unfreeze_strategy(self, name: str):
        """Manually unfreeze a strategy."""
        if name in self.strategies:
            self._transition(self.strategies[name], StrategyState.REDUCED, 0.3)

    def retire_strategy(self, name: str, reason: str = "Manual retirement"):
        """Manually retire a strategy."""
        if name in self.strategies:
            self._transition(self.strategies[name], StrategyState.RETIRED, 0.0, reason)

    # -------------------------------------------------
    # Status
    # -------------------------------------------------
    def get_status(self) -> Dict:
        """Get pool status."""
        by_state = {}
        for entry in self.strategies.values():
            state = entry.state.value
            if state not in by_state:
                by_state[state] = []
            by_state[state].append(entry.name)
        
        return {
            "total": len(self.strategies),
            "by_state": by_state,
            "active_count": len([e for e in self.strategies.values() 
                               if e.state in [StrategyState.ACTIVE, StrategyState.REDUCED]]),
        }
