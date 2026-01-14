# src/rl/ppo_agent.py
"""
PPO Agent for Risk Control
===========================

RL agent that controls risk/pyramid/mode decisions.
Does NOT decide trades - only meta-parameters.

Action Space:
- INC_RISK, DEC_RISK
- ENABLE_PYRAMID, DISABLE_PYRAMID
- SWITCH_ALPHA, SWITCH_DEFENSIVE
- HOLD

Reward = Leaderboard Score Δ
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("PPO_AGENT")


class Action(Enum):
    """Risk control actions."""
    INC_RISK = 0
    DEC_RISK = 1
    ENABLE_PYRAMID = 2
    DISABLE_PYRAMID = 3
    SWITCH_ALPHA = 4
    SWITCH_DEFENSIVE = 5
    HOLD = 6


@dataclass
class State:
    """Agent state vector."""
    score_24h: float
    win_rate_24h: float
    max_dd_24h: float
    regime_id: int  # 0=TREND, 1=CHOP, 2=HIGH_VOL
    volatility: float
    open_positions: int
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.score_24h / 10,  # Normalize
            self.win_rate_24h,
            self.max_dd_24h * 10,
            self.regime_id / 2,
            self.volatility,
            self.open_positions / 3,
        ], dtype=np.float32)


@dataclass
class Experience:
    """Single experience for training."""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


class RiskPPOAgent:
    """
    PPO Agent for Risk Meta-control.
    
    Does NOT decide entry/exit.
    Only decides: risk level, pyramid, mode.
    
    Optimizes for: Leaderboard Score
    """
    
    STATE_DIM = 6
    ACTION_DIM = 7
    
    def __init__(
        self,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        epsilon_clip: float = 0.2,
        model_path: str = "models/ppo_risk.pth"
    ):
        """
        Initialize PPO agent.
        
        Args:
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_clip: PPO clip parameter
            model_path: Path to save/load model
        """
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.model_path = model_path
        
        # Simple policy (can be replaced with neural network)
        self.policy_weights = np.zeros((self.STATE_DIM, self.ACTION_DIM))
        self.value_weights = np.zeros(self.STATE_DIM)
        
        self.trajectory: List[Experience] = []
        self.training_enabled = False
        
        self._load_model()
        logger.info("RiskPPOAgent initialized")
    
    def act(self, state: State) -> Action:
        """
        Select action based on current state.
        
        Uses policy to sample action with exploration.
        """
        state_vec = state.to_vector()
        logits = state_vec @ self.policy_weights
        
        # Softmax probabilities
        probs = self._softmax(logits)
        
        # Sample action
        action_idx = np.random.choice(self.ACTION_DIM, p=probs)
        action = Action(action_idx)
        
        logger.debug(f"Action: {action.name} (p={probs[action_idx]:.2f})")
        
        return action
    
    def get_action_probs(self, state: State) -> Dict[str, float]:
        """Get action probabilities for visualization."""
        state_vec = state.to_vector()
        logits = state_vec @ self.policy_weights
        probs = self._softmax(logits)
        
        return {Action(i).name: float(probs[i]) for i in range(self.ACTION_DIM)}
    
    def calculate_reward(
        self,
        score_prev: float,
        score_now: float,
        dd_today: float
    ) -> float:
        """
        Calculate reward.
        
        Reward = Score Δ - DD penalty
        """
        score_delta = score_now - score_prev
        dd_penalty = max(0, dd_today - 2.5) * 2
        
        reward = score_delta - dd_penalty
        
        logger.debug(f"Reward: {reward:.3f} (Δscore={score_delta:.2f}, dd_pen={dd_penalty:.2f})")
        
        return reward
    
    def store_experience(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool = False
    ) -> None:
        """Store experience for training."""
        exp = Experience(state, action, reward, next_state, done)
        self.trajectory.append(exp)
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns training metrics.
        """
        if len(self.trajectory) < 10:
            return {"loss": 0, "samples": 0}
        
        # Calculate returns
        returns = self._compute_returns()
        
        # Simplified PPO update
        total_loss = 0
        for i, exp in enumerate(self.trajectory):
            state_vec = exp.state.to_vector()
            advantage = returns[i] - state_vec @ self.value_weights
            
            # Policy gradient
            action_idx = exp.action.value
            grad = np.outer(state_vec, np.eye(self.ACTION_DIM)[action_idx])
            self.policy_weights += self.lr * advantage * grad
            
            # Value update
            self.value_weights += self.lr * advantage * state_vec
            
            total_loss += advantage ** 2
        
        avg_loss = total_loss / len(self.trajectory)
        
        # Clear trajectory
        n_samples = len(self.trajectory)
        self.trajectory = []
        
        self._save_model()
        
        logger.info(f"PPO update: loss={avg_loss:.4f}, samples={n_samples}")
        
        return {"loss": avg_loss, "samples": n_samples}
    
    def _compute_returns(self) -> List[float]:
        """Compute discounted returns."""
        returns = []
        G = 0
        
        for exp in reversed(self.trajectory):
            G = exp.reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _save_model(self) -> None:
        """Save model weights."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        np.savez(
            self.model_path,
            policy=self.policy_weights,
            value=self.value_weights,
        )
    
    def _load_model(self) -> None:
        """Load model weights if exists."""
        if os.path.exists(self.model_path + ".npz"):
            data = np.load(self.model_path + ".npz")
            self.policy_weights = data["policy"]
            self.value_weights = data["value"]
            logger.info(f"Loaded model from {self.model_path}")


def apply_action(action: Action, current_config: Dict) -> Dict:
    """Apply action to current config."""
    config = current_config.copy()
    
    if action == Action.INC_RISK:
        config["risk_mult"] = min(config.get("risk_mult", 1.0) + 0.2, 1.5)
    
    elif action == Action.DEC_RISK:
        config["risk_mult"] = max(config.get("risk_mult", 1.0) - 0.2, 0.5)
    
    elif action == Action.ENABLE_PYRAMID:
        config["pyramid_enabled"] = True
    
    elif action == Action.DISABLE_PYRAMID:
        config["pyramid_enabled"] = False
    
    elif action == Action.SWITCH_ALPHA:
        config["mode"] = "ALPHA"
    
    elif action == Action.SWITCH_DEFENSIVE:
        config["mode"] = "DEFENSIVE"
    
    # HOLD = no change
    
    return config


# Singleton
_agent: Optional[RiskPPOAgent] = None


def get_ppo_agent() -> RiskPPOAgent:
    """Get singleton PPO agent."""
    global _agent
    if _agent is None:
        _agent = RiskPPOAgent()
    return _agent
