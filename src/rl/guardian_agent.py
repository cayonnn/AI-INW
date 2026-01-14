# src/rl/guardian_agent.py
"""
Guardian Agent - Risk Intelligence Advisor
============================================

Guardian Agent is an ADVISOR, not an executor.
- Observes system state
- Suggests risk adjustments
- CANNOT override ProgressiveGuard

Architecture:
    Alpha Agent → GuardianAgent → ProgressiveGuard → Live Loop
    (propose)      (advise)        (enforce)         (execute)

Guardian suggestions are SOFT - can be ignored
ProgressiveGuard is HARD - cannot be bypassed
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_AGENT")


@dataclass
class GuardianSuggestion:
    """Guardian's risk adjustment suggestion."""
    risk_multiplier: float = 1.0      # 0.0-1.5, applied to base risk
    freeze_entries: bool = False       # Suggest no new entries
    disable_pyramid: bool = False      # Suggest no pyramid adds
    confidence_penalty: float = 0.0    # Reduce confidence score
    reason: str = ""                   # Why this suggestion
    urgency: int = 0                   # 0=normal, 1=warning, 2=urgent


class GuardianAction(Enum):
    """Actions Guardian can suggest."""
    HOLD = 0              # No change
    REDUCE_RISK = 1       # Lower risk multiplier
    FREEZE_ENTRIES = 2    # Stop new trades
    DISABLE_PYRAMID = 3   # No pyramid adds
    FULL_DEFENSIVE = 4    # All restrictions
    RESUME_NORMAL = 5     # Clear restrictions


@dataclass
class GuardianState:
    """Observable state for Guardian."""
    dd_today: float           # Daily drawdown %
    dd_total: float           # Total drawdown %
    memory_percent: float     # System memory %
    win_rate_24h: float       # Recent win rate
    volatility: float         # Market volatility
    guard_level: str          # Current ProgressiveGuard level
    open_positions: int       # Number of open positions
    hours_since_trade: float  # Time since last trade
    
    def to_vector(self) -> np.ndarray:
        """Convert to neural network input."""
        guard_map = {"OK": 0, "LEVEL_1": 1, "LEVEL_2": 2, "LEVEL_3": 3}
        return np.array([
            self.dd_today / 10,
            self.dd_total / 20,
            self.memory_percent / 100,
            self.win_rate_24h,
            min(self.volatility, 3) / 3,
            guard_map.get(self.guard_level, 0) / 3,
            self.open_positions / 5,
            min(self.hours_since_trade, 24) / 24,
        ], dtype=np.float32)


class GuardianAgent:
    """
    Guardian Agent - Risk Intelligence Advisor.
    
    Responsibilities:
    - Observe system state (DD, memory, volatility)
    - Suggest risk adjustments to Alpha agent
    - Learn optimal defensive strategies via RL
    
    Constraints:
    - Cannot override ProgressiveGuard
    - Cannot reset kill latch
    - Suggestions are advisory only
    """
    
    STATE_DIM = 8
    ACTION_DIM = 6
    
    def __init__(
        self,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        model_path: str = "models/guardian_agent.npz"
    ):
        """Initialize Guardian agent."""
        self.lr = learning_rate
        self.gamma = gamma
        self.model_path = model_path
        
        # Simple linear policy (can upgrade to neural network)
        self.policy_weights = np.random.randn(self.STATE_DIM, self.ACTION_DIM) * 0.1
        self.value_weights = np.random.randn(self.STATE_DIM) * 0.1
        
        self.last_state: Optional[GuardianState] = None
        self.last_action: Optional[GuardianAction] = None
        self.trajectory: List[Dict] = []
        
        self._load_model()
        logger.info("GuardianAgent initialized (advisor mode)")
    
    def observe(self, state: GuardianState) -> None:
        """Observe current system state."""
        self.last_state = state
    
    def act(self) -> GuardianSuggestion:
        """
        Generate risk adjustment suggestion.
        
        Uses:
        1. Rule-based fallback for critical situations
        2. RL policy for nuanced decisions
        """
        if self.last_state is None:
            return GuardianSuggestion()
        
        s = self.last_state
        
        # === RULE-BASED OVERRIDES (deterministic) ===
        
        # Level 2 = aggressive defense
        if s.guard_level == "LEVEL_2":
            return GuardianSuggestion(
                risk_multiplier=0.4,
                freeze_entries=True,
                disable_pyramid=True,
                confidence_penalty=0.3,
                reason="Guard LEVEL_2 triggered",
                urgency=2
            )
        
        # Level 1 = moderate caution
        if s.guard_level == "LEVEL_1":
            return GuardianSuggestion(
                risk_multiplier=0.6,
                disable_pyramid=True,
                reason="Guard LEVEL_1 triggered",
                urgency=1
            )
        
        # High DD = reduce risk
        if s.dd_today > 2.5:
            return GuardianSuggestion(
                risk_multiplier=0.5,
                freeze_entries=True,
                disable_pyramid=True,
                reason=f"High DD: {s.dd_today:.1f}%",
                urgency=2
            )
        
        if s.dd_today > 1.5:
            return GuardianSuggestion(
                risk_multiplier=0.7,
                disable_pyramid=True,
                reason=f"Elevated DD: {s.dd_today:.1f}%",
                urgency=1
            )
        
        # High memory = reduce load
        if s.memory_percent > 85:
            return GuardianSuggestion(
                risk_multiplier=0.5,
                freeze_entries=True,
                reason=f"High memory: {s.memory_percent:.0f}%",
                urgency=2
            )
        
        if s.memory_percent > 75:
            return GuardianSuggestion(
                risk_multiplier=0.8,
                reason=f"Elevated memory: {s.memory_percent:.0f}%",
                urgency=1
            )
        
        # Low win rate = reduce aggression
        if s.win_rate_24h < 0.35:
            return GuardianSuggestion(
                risk_multiplier=0.7,
                disable_pyramid=True,
                reason=f"Low win rate: {s.win_rate_24h:.0%}",
                urgency=1
            )
        
        # High volatility = caution
        if s.volatility > 2.0:
            return GuardianSuggestion(
                risk_multiplier=0.8,
                reason=f"High volatility: {s.volatility:.1f}",
                urgency=1
            )
        
        # === RL POLICY (for normal conditions) ===
        state_vec = s.to_vector()
        logits = state_vec @ self.policy_weights
        probs = self._softmax(logits)
        
        action_idx = np.argmax(probs)  # Greedy for production
        action = GuardianAction(action_idx)
        self.last_action = action
        
        return self._action_to_suggestion(action)
    
    def _action_to_suggestion(self, action: GuardianAction) -> GuardianSuggestion:
        """Convert action to suggestion."""
        if action == GuardianAction.REDUCE_RISK:
            return GuardianSuggestion(
                risk_multiplier=0.7,
                reason="RL suggests caution"
            )
        
        if action == GuardianAction.FREEZE_ENTRIES:
            return GuardianSuggestion(
                freeze_entries=True,
                reason="RL suggests no new entries"
            )
        
        if action == GuardianAction.DISABLE_PYRAMID:
            return GuardianSuggestion(
                disable_pyramid=True,
                reason="RL suggests no pyramid"
            )
        
        if action == GuardianAction.FULL_DEFENSIVE:
            return GuardianSuggestion(
                risk_multiplier=0.5,
                freeze_entries=True,
                disable_pyramid=True,
                reason="RL suggests full defensive",
                urgency=1
            )
        
        if action == GuardianAction.RESUME_NORMAL:
            return GuardianSuggestion(
                risk_multiplier=1.0,
                reason="RL suggests normal operations"
            )
        
        return GuardianSuggestion()  # HOLD
    
    def update(
        self,
        reward: float,
        new_state: GuardianState
    ) -> None:
        """Store experience for training."""
        if self.last_state is not None and self.last_action is not None:
            self.trajectory.append({
                "state": self.last_state,
                "action": self.last_action,
                "reward": reward,
                "next_state": new_state,
            })
        
        self.last_state = new_state
    
    def train(self) -> Dict[str, float]:
        """Train policy on collected trajectory."""
        if len(self.trajectory) < 10:
            return {"loss": 0, "samples": 0}
        
        # Calculate returns
        returns = []
        G = 0
        for exp in reversed(self.trajectory):
            G = exp["reward"] + self.gamma * G
            returns.insert(0, G)
        
        # Policy gradient update
        total_loss = 0
        for i, exp in enumerate(self.trajectory):
            state_vec = exp["state"].to_vector()
            advantage = returns[i] - state_vec @ self.value_weights
            
            action_idx = exp["action"].value
            one_hot = np.zeros(self.ACTION_DIM)
            one_hot[action_idx] = 1
            grad = np.outer(state_vec, one_hot)
            
            self.policy_weights += self.lr * advantage * grad
            self.value_weights += self.lr * advantage * state_vec
            
            total_loss += advantage ** 2
        
        avg_loss = total_loss / len(self.trajectory)
        n_samples = len(self.trajectory)
        
        self.trajectory = []
        self._save_model()
        
        logger.info(f"Guardian trained: loss={avg_loss:.4f}, samples={n_samples}")
        
        return {"loss": avg_loss, "samples": n_samples}
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _save_model(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        np.savez(
            self.model_path,
            policy=self.policy_weights,
            value=self.value_weights,
        )
    
    def _load_model(self) -> None:
        if os.path.exists(self.model_path + ".npz"):
            try:
                data = np.load(self.model_path + ".npz")
                self.policy_weights = data["policy"]
                self.value_weights = data["value"]
                logger.info("Guardian model loaded")
            except:
                pass


# Singleton
_guardian: Optional[GuardianAgent] = None


def get_guardian_agent() -> GuardianAgent:
    """Get singleton Guardian agent."""
    global _guardian
    if _guardian is None:
        _guardian = GuardianAgent()
    return _guardian
