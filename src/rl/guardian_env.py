# src/rl/guardian_env.py
"""
Guardian RL Environment (Gym-style)
====================================

Guardian learns WHEN TO BRAKE, not how to trade.
This is a meta-controller for risk management.

State Space (Continuous):
- margin_ratio: free_margin / equity
- daily_dd: (start_balance - equity) / start_balance
- open_positions: int
- margin_block_count: int
- equity_volatility: std(equity_last_n)
- alpha_signal: -1 sell | 0 hold | 1 buy
- time_in_trade: avg holding time

Action Space (Discrete):
- 0: ALLOW - no interference
- 1: REDUCE_RISK - clamp lot / RR
- 2: FORCE_HOLD - block new trades
- 3: EMERGENCY_FREEZE - kill switch
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_ENV")


class GuardianAction(IntEnum):
    """Discrete action space for Guardian."""
    ALLOW = 0
    REDUCE_RISK = 1
    FORCE_HOLD = 2
    EMERGENCY_FREEZE = 3


@dataclass
class GuardianState:
    """Observable state for Guardian RL."""
    margin_ratio: float = 1.0          # free_margin / equity
    daily_dd: float = 0.0              # (start_balance - equity) / start_balance
    open_positions: int = 0
    margin_block_count: int = 0
    equity_volatility: float = 0.0     # std(equity_last_n)
    alpha_signal: int = 0              # -1 sell | 0 hold | 1 buy
    time_in_trade: float = 0.0         # avg holding time in seconds
    # V4 Context Features
    market_regime: float = 0.5         # 0=Range, 1=Trend
    session_time: float = 0.5          # 0.0-1.0 (Day progress)
    recent_win_rate: float = 0.5       # Last 20 trades WR
    
    def to_array(self) -> np.ndarray:
        """Convert to observation array (10 dims)."""
        return np.array([
            self.margin_ratio,
            self.daily_dd,
            self.open_positions / 10.0,
            self.margin_block_count / 5.0,
            self.equity_volatility,
            (self.alpha_signal + 1) / 2.0,
            min(self.time_in_trade / 3600, 1.0),
            # New Features
            self.market_regime,
            self.session_time,
            self.recent_win_rate
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "margin_ratio": self.margin_ratio,
            "daily_dd": self.daily_dd,
            "open_positions": self.open_positions,
            "margin_block_count": self.margin_block_count,
            "equity_volatility": self.equity_volatility,
            "alpha_signal": self.alpha_signal,
            "time_in_trade": self.time_in_trade,
            "market_regime": self.market_regime,
            "session_time": self.session_time,
            "recent_win_rate": self.recent_win_rate
        }


class GuardianEnv:
    """
    Gym-style RL Environment for Guardian Agent.
    
    Guardian learns to maximize capital preservation while
    allowing profitable trades to execute.
    
    Reward Shaping:
    - Survival > Profit
    - Penalize both over-protection and under-protection
    """
    
    # State dimension (Updated for V4 Context)
    STATE_DIM = 10
    ACTION_DIM = 4
    
    def __init__(self):
        """Initialize environment."""
        self.state: Optional[GuardianState] = None
        self.done: bool = False
        self.step_count: int = 0
        self.total_reward: float = 0.0
        
        # Episode history for analysis
        self.history: List[Dict] = []
    
    def reset(self, initial_state: Optional[GuardianState] = None) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Args:
            initial_state: Starting state (if None, use defaults)
            
        Returns:
            Initial observation array
        """
        self.state = initial_state or GuardianState()
        self.done = False
        self.step_count = 0
        self.total_reward = 0.0
        self.history = []
        
        return self.state.to_array()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (obs, reward, done, info).
        
        Args:
            action: Guardian action (0-3)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.state is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        action = GuardianAction(action)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Apply state transition
        self._transition(action)
        
        # Update counters
        self.step_count += 1
        self.total_reward += reward
        
        # Log history
        self.history.append({
            "step": self.step_count,
            "action": action.name,
            "reward": reward,
            "state": self.state.to_dict(),
            "done": self.done,
        })
        
        info = {
            "action_name": action.name,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
        }
        
        return self.state.to_array(), reward, self.done, info
    
    def _calculate_reward(self, action: GuardianAction) -> float:
        """
        Calculate reward based on action and state.
        
        Reward Shaping:
        - Survival > Profit
        - Penalize overreaction (freezing when not needed)
        - Penalize underreaction (not protecting when needed)
        """
        dd = self.state.daily_dd
        margin = self.state.margin_ratio
        
        # CRITICAL: Failed to protect at high DD
        if dd > 0.10 and action.value < GuardianAction.FORCE_HOLD:
            return -5.0  # Severe penalty
        
        # OVERREACTION: Emergency freeze when not needed
        if action == GuardianAction.EMERGENCY_FREEZE and dd < 0.08:
            return -2.0  # Moderate penalty
        
        # GOOD: Strong protection at low margin
        if margin < 0.3 and action.value >= GuardianAction.FORCE_HOLD:
            return +2.0  # Good protection
        
        # GOOD: Reduce risk at warning level
        if dd > 0.05 and action == GuardianAction.REDUCE_RISK:
            return +1.0  # Proactive caution
        
        # NEUTRAL: Normal operation
        if action == GuardianAction.ALLOW and dd < 0.03 and margin > 0.5:
            return +0.2  # Reward allowing profitable trades
        
        # DEFAULT: Small survival bonus
        return +0.1
    
    def _transition(self, action: GuardianAction) -> None:
        """
        Apply state transition based on action.
        
        Emergency freeze ends the episode.
        """
        if action == GuardianAction.EMERGENCY_FREEZE:
            self.done = True
            logger.info("Guardian triggered EMERGENCY_FREEZE - Episode ended")
        
        elif action == GuardianAction.FORCE_HOLD:
            # Reduce block count slowly (recovery)
            self.state.margin_block_count = max(0, self.state.margin_block_count - 1)
        
        elif action == GuardianAction.REDUCE_RISK:
            # Slight recovery
            self.state.margin_block_count = max(0, self.state.margin_block_count - 0.5)
    
    def render(self) -> str:
        """Return string representation of current state."""
        if self.state is None:
            return "Environment not initialized"
        
        return (
            f"Step {self.step_count} | "
            f"DD={self.state.daily_dd*100:.1f}% | "
            f"Margin={self.state.margin_ratio*100:.0f}% | "
            f"Blocks={self.state.margin_block_count} | "
            f"Reward={self.total_reward:.2f}"
        )


# =============================================================================
# Stress Test Infrastructure
# =============================================================================

class FakeAccount:
    """Simulated account for stress testing."""
    
    def __init__(self, balance: float = 1000.0):
        self.initial_balance = balance
        self.balance = balance
        self.equity = balance
        self.margin_used = 0.0
    
    def apply_stress(self, step: int) -> None:
        """Apply incremental stress to account."""
        # Simulate drawdown
        self.equity -= step * 15
        self.equity = max(0, self.equity)
        
        # Simulate margin pressure
        self.margin_used += step * 30
        self.margin_used = min(self.margin_used, self.balance * 0.95)
    
    @property
    def margin_ratio(self) -> float:
        """Free margin ratio."""
        free = self.balance - self.margin_used
        return max(0.01, free / max(self.balance, 1))
    
    @property
    def daily_dd(self) -> float:
        """Daily drawdown percentage."""
        return max(0, (self.initial_balance - self.equity) / self.initial_balance)
    
    def __repr__(self) -> str:
        return f"FakeAccount(balance={self.balance:.0f}, equity={self.equity:.0f}, dd={self.daily_dd*100:.1f}%)"


def run_guardian_stress_test(agent, max_steps: int = 20) -> List[Dict]:
    """
    Run stress test to validate Guardian behavior.
    
    Args:
        agent: Guardian agent with decide(state) method
        max_steps: Maximum stress steps
        
    Returns:
        List of log entries showing escalation behavior
    """
    acc = FakeAccount()
    logs = []
    
    print("\n" + "=" * 60)
    print("🧪 GUARDIAN STRESS TEST")
    print("=" * 60)
    
    for t in range(max_steps):
        acc.apply_stress(t)
        
        state = {
            "margin_ratio": acc.margin_ratio,
            "daily_dd": acc.daily_dd,
            "open_positions": 5,
            "margin_block_count": t // 3,
            "equity_volatility": 0.04,
            "alpha_signal": 1,
            "time_in_trade": 120,
        }
        
        action = agent.decide(state)
        action_name = action.name if hasattr(action, 'name') else str(action)
        
        log_entry = {
            "step": t,
            "margin_ratio": acc.margin_ratio,
            "daily_dd": acc.daily_dd,
            "action": action_name,
            "equity": acc.equity,
        }
        logs.append(log_entry)
        
        print(f"  Step {t:2d} | DD={acc.daily_dd*100:5.1f}% | Margin={acc.margin_ratio*100:5.1f}% | Action={action_name}")
        
        if action_name == "EMERGENCY_FREEZE":
            print("  🛑 EMERGENCY_FREEZE triggered - Test complete")
            break
    
    print("=" * 60)
    print(f"📊 Total steps: {len(logs)} | Final DD: {acc.daily_dd*100:.1f}%")
    print("=" * 60 + "\n")
    
    return logs


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    # Test environment
    env = GuardianEnv()
    
    print("\n=== Guardian Environment Test ===\n")
    
    # Reset with initial state
    initial = GuardianState(
        margin_ratio=0.8,
        daily_dd=0.02,
        open_positions=2,
    )
    obs = env.reset(initial)
    print(f"Initial obs: {obs}")
    print(f"State: {env.render()}")
    
    # Simulate some steps
    actions = [0, 0, 1, 1, 2, 2, 3]  # Escalating actions
    
    for action in actions:
        obs, reward, done, info = env.step(action)
        print(f"Action: {info['action_name']:15} | Reward: {reward:+.1f} | {env.render()}")
        if done:
            break
    
    print(f"\n=== Episode Complete ===")
    print(f"Total Reward: {env.total_reward:.2f}")
    print(f"Steps: {env.step_count}")
