# src/registry/promotion_engine.py
"""
Auto-Promotion Governance Engine
=================================

State machine for profile promotion with:
- Strict promotion rules
- Auto-rollback on DD spike
- No human error in promotion
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("PROMOTION_ENGINE")


class ProfileState(Enum):
    """Profile states."""
    LIVE = "LIVE"
    SHADOW = "SHADOW"
    CANDIDATE = "CANDIDATE"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class ProfileMetrics:
    """Profile performance metrics."""
    profile_id: str
    score: float
    max_dd: float
    win_days: int
    total_days: int
    dd_today: float = 0.0


class PromotionEngine:
    """
    Auto-Promotion Governance.
    
    States:
    - LIVE: Currently trading
    - SHADOW: Being tested
    - CANDIDATE: Ready for promotion review
    - PROMOTED: Just promoted to live
    - REJECTED: Failed promotion criteria
    - ROLLED_BACK: Reverted due to DD
    
    Promotion Rule:
    - Shadow score > Live score
    - Win days >= 3 consecutive
    - Shadow DD <= Live DD
    
    Auto-Rollback:
    - If DD today > 3%, rollback immediately
    """
    
    PROMOTION_MIN_WIN_DAYS = 3
    ROLLBACK_DD_THRESHOLD = 0.03  # 3% DD triggers rollback
    
    def __init__(self, state_file: str = "profiles/promotion_state.json"):
        """Initialize promotion engine."""
        self.state_file = state_file
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        
        self.current_live: Optional[str] = None
        self.previous_live: Optional[str] = None  # For rollback
        self.state_history: list = []
        
        self._load_state()
        logger.info("PromotionEngine initialized")
    
    def evaluate(
        self,
        shadow: ProfileMetrics,
        live: ProfileMetrics
    ) -> str:
        """
        Evaluate if shadow should be promoted.
        
        Returns:
            "PROMOTE", "HOLD", or "REJECT"
        """
        # Check win days
        if shadow.win_days < self.PROMOTION_MIN_WIN_DAYS:
            logger.info(f"HOLD: win_days={shadow.win_days} < {self.PROMOTION_MIN_WIN_DAYS}")
            return "HOLD"
        
        # Check score
        if shadow.score <= live.score:
            logger.info(f"HOLD: score={shadow.score:.2f} <= live={live.score:.2f}")
            return "HOLD"
        
        # Check DD
        if shadow.max_dd > live.max_dd:
            logger.info(f"REJECT: DD={shadow.max_dd:.1%} > live={live.max_dd:.1%}")
            return "REJECT"
        
        logger.info(f"PROMOTE: {shadow.profile_id}")
        return "PROMOTE"
    
    def promote(self, shadow_id: str) -> bool:
        """Promote shadow to live."""
        self.previous_live = self.current_live
        self.current_live = shadow_id
        
        self._record_state_change(shadow_id, ProfileState.PROMOTED)
        self._save_state()
        
        logger.info(f"Promoted {shadow_id} to LIVE")
        return True
    
    def monitor(self, live: ProfileMetrics) -> Optional[str]:
        """
        Monitor live profile for rollback triggers.
        
        Returns:
            "ROLLBACK" if threshold exceeded, else None
        """
        if live.dd_today > self.ROLLBACK_DD_THRESHOLD:
            logger.warning(
                f"DD spike detected: {live.dd_today:.1%} > "
                f"{self.ROLLBACK_DD_THRESHOLD:.0%}"
            )
            return "ROLLBACK"
        return None
    
    def rollback(self) -> bool:
        """Rollback to previous live profile."""
        if not self.previous_live:
            logger.error("No previous profile to rollback to")
            return False
        
        rolled_back = self.current_live
        self.current_live = self.previous_live
        self.previous_live = None
        
        self._record_state_change(rolled_back, ProfileState.ROLLED_BACK)
        self._save_state()
        
        logger.warning(f"Rolled back from {rolled_back} to {self.current_live}")
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get current promotion state."""
        return {
            "current_live": self.current_live,
            "previous_live": self.previous_live,
            "history_count": len(self.state_history),
        }
    
    def _record_state_change(
        self,
        profile_id: str,
        new_state: ProfileState
    ) -> None:
        """Record state change."""
        self.state_history.append({
            "profile": profile_id,
            "state": new_state.value,
            "timestamp": datetime.now().isoformat(),
        })
    
    def _save_state(self) -> None:
        """Save state to file."""
        state = {
            "current_live": self.current_live,
            "previous_live": self.previous_live,
            "history": self.state_history[-20:],  # Keep last 20
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self) -> None:
        """Load state from file."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.current_live = state.get("current_live")
            self.previous_live = state.get("previous_live")
            self.state_history = state.get("history", [])
        except:
            pass


# Singleton
_engine: Optional[PromotionEngine] = None


def get_promotion_engine() -> PromotionEngine:
    """Get singleton PromotionEngine."""
    global _engine
    if _engine is None:
        _engine = PromotionEngine()
    return _engine
