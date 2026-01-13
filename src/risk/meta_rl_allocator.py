"""
meta_rl_allocator.py
=====================
Self-Learning Capital Allocation (Meta-Reinforcement Learning)

AI เรียนรู้ว่า ควรให้เงินกับกลยุทธ์ไหน เมื่อไหร่

กองทุนที่รอด ไม่ใช่กองทุนที่หากลยุทธ์เก่ง
แต่คือกองทุนที่ จัดเงินเก่ง

State: Portfolio state (alpha, correlation, volatility, drawdown, liquidity)
Action: Adjust weights, freeze/revive, shift to cash
Reward: alpha - drawdown_penalty - tail_risk_penalty
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import random
import math
from src.utils.logger import get_logger

logger = get_logger("META_RL_ALLOCATOR")


@dataclass
class PortfolioState:
    """State representation for Meta-RL agent."""
    # Strategy states
    strategy_alphas: Dict[str, float] = field(default_factory=dict)
    strategy_volatilities: Dict[str, float] = field(default_factory=dict)
    strategy_drawdowns: Dict[str, float] = field(default_factory=dict)
    
    # Portfolio metrics
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    portfolio_volatility: float = 0.0
    portfolio_drawdown: float = 0.0
    drawdown_slope: float = 0.0    # Positive = recovering, Negative = worsening
    
    # Market conditions
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME
    liquidity_score: float = 1.0       # 0-1
    
    # Current allocation
    current_weights: Dict[str, float] = field(default_factory=dict)
    cash_weight: float = 0.0

    def to_vector(self) -> List[float]:
        """Convert state to numerical vector for RL."""
        vector = []
        
        # Strategy metrics
        for name in sorted(self.strategy_alphas.keys()):
            vector.extend([
                self.strategy_alphas.get(name, 0),
                self.strategy_volatilities.get(name, 0),
                self.strategy_drawdowns.get(name, 0),
                self.current_weights.get(name, 0),
            ])
        
        # Portfolio metrics
        vector.extend([
            self.portfolio_volatility,
            self.portfolio_drawdown,
            self.drawdown_slope,
            self.liquidity_score,
            self.cash_weight,
        ])
        
        # Regime encoding
        regime_map = {"LOW": 0.25, "NORMAL": 0.5, "HIGH": 0.75, "EXTREME": 1.0}
        vector.append(regime_map.get(self.volatility_regime, 0.5))
        
        return vector


@dataclass
class AllocationAction:
    """Action taken by Meta-RL agent."""
    weight_changes: Dict[str, float]   # Strategy -> delta weight
    freeze_strategies: List[str]
    revive_strategies: List[str]
    target_cash_weight: float


@dataclass
class RLExperience:
    """Single experience for replay buffer."""
    state: PortfolioState
    action: AllocationAction
    reward: float
    next_state: PortfolioState
    done: bool


class MetaRLAllocator:
    """
    Meta-Reinforcement Learning Capital Allocator.
    
    Learns optimal capital allocation across strategies dynamically.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
        # Policy parameters (simplified - would be neural net in production)
        self.policy_weights: Dict[str, float] = {}
        self.value_estimates: Dict[str, float] = {}
        
        # Experience replay
        self.replay_buffer: List[RLExperience] = []
        self.max_buffer_size = 10000
        
        # Allocation state
        self.current_weights: Dict[str, float] = {}
        self.frozen_strategies: set = set()
        
        # Constraints
        self.min_weight = 0.0
        self.max_weight = 0.4
        self.max_cash = 0.5
        self.min_strategies = 2
        
        # Reward parameters
        self.drawdown_penalty_mult = 2.0
        self.tail_risk_penalty_mult = 3.0
        self.turnover_penalty_mult = 0.1
        
        # Training state
        self.episode = 0
        self.total_reward = 0.0

    # -------------------------------------------------
    # Main decision loop
    # -------------------------------------------------
    def decide_allocation(self, state: PortfolioState) -> AllocationAction:
        """
        Decide new allocation based on current state.
        
        Returns optimal allocation action.
        """
        # Get action from policy
        action = self._get_action(state)
        
        # Apply constraints
        action = self._apply_constraints(action, state)
        
        logger.debug(f"RL Action: {len(action.weight_changes)} changes, "
                    f"cash={action.target_cash_weight:.1%}")
        
        return action

    def _get_action(self, state: PortfolioState) -> AllocationAction:
        """Get action from policy (epsilon-greedy + learned policy)."""
        epsilon = max(0.1, 1.0 - self.episode * 0.01)
        
        if random.random() < epsilon:
            # Exploration: random action
            return self._random_action(state)
        else:
            # Exploitation: use learned policy
            return self._policy_action(state)

    def _random_action(self, state: PortfolioState) -> AllocationAction:
        """Generate random action for exploration."""
        weight_changes = {}
        
        for strategy in state.strategy_alphas.keys():
            if strategy not in self.frozen_strategies:
                weight_changes[strategy] = random.uniform(-0.1, 0.1)
        
        return AllocationAction(
            weight_changes=weight_changes,
            freeze_strategies=[],
            revive_strategies=[],
            target_cash_weight=random.uniform(0.05, 0.2),
        )

    def _policy_action(self, state: PortfolioState) -> AllocationAction:
        """Generate action from learned policy."""
        weight_changes = {}
        freeze = []
        revive = []
        
        for strategy, alpha in state.strategy_alphas.items():
            current_weight = state.current_weights.get(strategy, 0)
            vol = state.strategy_volatilities.get(strategy, 0.1)
            dd = state.strategy_drawdowns.get(strategy, 0)
            
            # Simple policy: reward high alpha, penalize high vol/dd
            score = alpha - vol * 0.5 - abs(dd) * 1.0
            
            # Adjust weight based on score
            if score > 0.1:
                weight_changes[strategy] = min(0.05, self.max_weight - current_weight)
            elif score < -0.1:
                weight_changes[strategy] = max(-0.05, -current_weight)
            else:
                weight_changes[strategy] = 0
            
            # Freeze decision
            if dd < -0.1 and vol > 0.2:
                freeze.append(strategy)
            
            # Revive decision
            if strategy in self.frozen_strategies and alpha > 0.05 and vol < 0.15:
                revive.append(strategy)
        
        # Cash decision based on regime
        regime_cash = {
            "LOW": 0.05,
            "NORMAL": 0.1,
            "HIGH": 0.2,
            "EXTREME": 0.4,
        }
        target_cash = regime_cash.get(state.volatility_regime, 0.1)
        
        return AllocationAction(
            weight_changes=weight_changes,
            freeze_strategies=freeze,
            revive_strategies=revive,
            target_cash_weight=target_cash,
        )

    def _apply_constraints(self, action: AllocationAction, 
                          state: PortfolioState) -> AllocationAction:
        """Apply allocation constraints."""
        # Freeze strategies
        for s in action.freeze_strategies:
            self.frozen_strategies.add(s)
            action.weight_changes[s] = -state.current_weights.get(s, 0)
        
        # Revive strategies
        for s in action.revive_strategies:
            self.frozen_strategies.discard(s)
        
        # Calculate new weights
        new_weights = {}
        for s, current in state.current_weights.items():
            delta = action.weight_changes.get(s, 0)
            new = max(self.min_weight, min(self.max_weight, current + delta))
            new_weights[s] = new
        
        # Ensure min strategies
        active = [s for s, w in new_weights.items() if w > 0.05]
        if len(active) < self.min_strategies:
            # Don't reduce below minimum
            action.weight_changes = {s: 0 for s in action.weight_changes}
        
        # Cap cash
        action.target_cash_weight = min(action.target_cash_weight, self.max_cash)
        
        return action

    # -------------------------------------------------
    # Learning
    # -------------------------------------------------
    def observe_outcome(self, state: PortfolioState, action: AllocationAction,
                       realized_pnl: float, new_state: PortfolioState):
        """
        Observe outcome and learn.
        
        reward = alpha - drawdown_penalty - tail_risk_penalty
        """
        # Calculate reward
        alpha = realized_pnl
        dd_penalty = abs(new_state.portfolio_drawdown) * self.drawdown_penalty_mult
        tail_penalty = (1 - new_state.liquidity_score) * self.tail_risk_penalty_mult
        
        # Turnover penalty
        turnover = sum(abs(d) for d in action.weight_changes.values())
        turnover_penalty = turnover * self.turnover_penalty_mult
        
        reward = alpha - dd_penalty - tail_penalty - turnover_penalty
        
        # Store experience
        experience = RLExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=new_state,
            done=False,
        )
        self._store_experience(experience)
        
        # Update policy
        self._update_policy(experience)
        
        self.total_reward += reward
        
        logger.debug(f"RL Reward: {reward:.4f} (alpha={alpha:.4f}, dd_pen={dd_penalty:.4f})")

    def _store_experience(self, exp: RLExperience):
        """Store experience in replay buffer."""
        self.replay_buffer.append(exp)
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

    def _update_policy(self, exp: RLExperience):
        """Update policy based on experience (simplified TD learning)."""
        # Simplified update - would use neural net in production
        for strategy, weight_change in exp.action.weight_changes.items():
            current_value = self.policy_weights.get(strategy, 0)
            
            # TD error approximation
            td_error = exp.reward - current_value
            
            # Update
            self.policy_weights[strategy] = current_value + self.learning_rate * td_error

    def end_episode(self):
        """End current episode."""
        self.episode += 1
        logger.info(f"Episode {self.episode} complete. Total reward: {self.total_reward:.4f}")
        self.total_reward = 0.0

    # -------------------------------------------------
    # Getters
    # -------------------------------------------------
    def get_target_weights(self) -> Dict[str, float]:
        """Get target weights from last action."""
        return self.current_weights.copy()

    def is_frozen(self, strategy: str) -> bool:
        """Check if strategy is frozen."""
        return strategy in self.frozen_strategies

    def get_status(self) -> Dict:
        """Get allocator status."""
        return {
            "episode": self.episode,
            "buffer_size": len(self.replay_buffer),
            "frozen_count": len(self.frozen_strategies),
            "policy_strategies": len(self.policy_weights),
        }
