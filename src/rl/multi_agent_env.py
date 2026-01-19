# src/rl/multi_agent_env.py
"""
Multi-Agent Co-Learning Environment
====================================

Bi-level RL system where Alpha and Guardian learn together.

Architecture:
    Alpha PPO (Trader) ←→ Guardian PPO (Risk)
           ↓                    ↓
         Market Environment
           ↓
      Separate Rewards

Key Innovation:
    "We train risk management as an intelligent agent,
     not a static constraint."

Paper Statement:
    "Our Guardian learns to protect capital without killing opportunity,
     creating an adversarial yet cooperative dynamic that produces
     robust trading policies."
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from enum import IntEnum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("MULTI_AGENT_ENV")


# =============================================================================
# Action Definitions
# =============================================================================

class AlphaAction(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


class GuardianAction(IntEnum):
    ALLOW = 0
    SCALE_DOWN = 1  # Reduce position size by 50%
    BLOCK = 2       # Block this trade
    FREEZE = 3      # Freeze all trading


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent environment."""
    # Environment
    max_steps: int = 1000
    max_positions: int = 5
    
    # Alpha rewards
    alpha_profit_scale: float = 0.1
    alpha_guardian_allow: float = 0.2
    alpha_guardian_block: float = -0.3
    alpha_trigger_freeze: float = -1.0
    alpha_hold_in_risk: float = 0.1
    
    # Guardian rewards
    guardian_dd_avoided: float = 0.5
    guardian_profit_preserved: float = 0.3
    guardian_over_block: float = -0.2
    guardian_late_block: float = -0.4
    guardian_unnecessary_freeze: float = -1.0
    
    # Stability
    guardian_update_interval: int = 5  # Update every N episodes
    kl_regularization: float = 0.01


# =============================================================================
# Multi-Agent Environment
# =============================================================================

class MultiAgentTradingEnv(gym.Env):
    """
    Multi-Agent Environment for Alpha ↔ Guardian Co-Learning.
    
    Observation Spaces:
        Alpha:  [price_features, account_state, guardian_risk_level, 
                 guardian_block_rate, alpha_confidence_history]
        Guardian: [alpha_action, alpha_confidence, account_state,
                   volatility_state, drawdown_slope]
    
    Action Spaces:
        Alpha: Discrete(3) - HOLD, BUY, SELL
        Guardian: Discrete(4) - ALLOW, SCALE_DOWN, BLOCK, FREEZE
    
    Rewards:
        Alpha: Rewarded for profit, penalized for Guardian blocks
        Guardian: Rewarded for DD avoided, penalized for over-blocking
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: Optional[MultiAgentConfig] = None):
        super().__init__()
        
        self.config = config or MultiAgentConfig()
        
        # --- Alpha Space ---
        self.alpha_obs_dim = 10
        self.alpha_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.alpha_obs_dim,), dtype=np.float32
        )
        self.alpha_action_space = spaces.Discrete(3)
        
        # --- Guardian Space ---
        self.guardian_obs_dim = 8
        self.guardian_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.guardian_obs_dim,), dtype=np.float32
        )
        self.guardian_action_space = spaces.Discrete(4)
        
        # Internal state
        self._reset_state()
    
    def _reset_state(self):
        """Reset all internal state."""
        self.step_count = 0
        self.equity = 1000.0
        self.start_equity = 1000.0
        self.positions = 0
        self.floating_pnl = 0.0
        self.max_equity = 1000.0
        self.current_dd = 0.0
        self.dd_slope = 0.0
        
        # Tracking
        self.alpha_actions_history = []
        self.guardian_actions_history = []
        self.blocks_count = 0
        self.freezes_count = 0
        self.profit_total = 0.0
        self.dd_avoided_count = 0
        
        # Guardian state
        self.is_frozen = False
        self.freeze_until = 0
        
        # Market state (simulated)
        self._ema_diff = 0.0
        self._rsi = 0.0
        self._atr = 0.0
        self._volatility = 0.0
    
    # =========================================================================
    # Gym Interface
    # =========================================================================
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self._reset_state()
        
        return self._get_observations(), {}
    
    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict]:
        """
        Execute one step with both agents.
        
        Args:
            actions: {"alpha": action, "guardian": action}
            
        Returns:
            observations, rewards, terminated, truncated, info
        """
        self.step_count += 1
        
        alpha_action = AlphaAction(actions.get("alpha", 0))
        guardian_action = GuardianAction(actions.get("guardian", 0))
        
        # Record actions
        self.alpha_actions_history.append(alpha_action)
        self.guardian_actions_history.append(guardian_action)
        
        # Apply Guardian decision
        final_action, lot_multiplier = self._apply_guardian(
            alpha_action, guardian_action
        )
        
        # Execute trade (simulated)
        profit, dd_change = self._simulate_trade(final_action, lot_multiplier)
        
        # Update state
        self._update_state(profit, dd_change)
        
        # Calculate rewards
        alpha_reward = self._calculate_alpha_reward(
            alpha_action, guardian_action, profit
        )
        guardian_reward = self._calculate_guardian_reward(
            alpha_action, guardian_action, dd_change, profit
        )
        
        # Check termination
        terminated = self.current_dd >= 0.25 or self.equity <= 0
        truncated = self.step_count >= self.config.max_steps
        
        rewards = {"alpha": alpha_reward, "guardian": guardian_reward}
        info = self._get_info()
        
        return self._get_observations(), rewards, terminated, truncated, info
    
    # =========================================================================
    # Observations
    # =========================================================================
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for both agents."""
        return {
            "alpha": self._get_alpha_obs(),
            "guardian": self._get_guardian_obs()
        }
    
    def _get_alpha_obs(self) -> np.ndarray:
        """
        Alpha observation:
            - price_features (3): EMA diff, RSI, ATR
            - account_state (3): equity %, positions, floating PnL
            - guardian_context (2): risk level, block rate
            - confidence_history (2): recent confidence stats
        """
        # Guardian risk level (0-1 based on recent blocks)
        recent_blocks = sum(
            1 for a in self.guardian_actions_history[-20:] 
            if a >= GuardianAction.BLOCK
        )
        guardian_risk = min(recent_blocks / 20.0, 1.0)
        
        # Block rate
        total_guardian_actions = len(self.guardian_actions_history)
        block_rate = self.blocks_count / max(total_guardian_actions, 1)
        
        return np.array([
            # Price features
            self._ema_diff,
            self._rsi,
            self._atr,
            # Account state
            (self.equity / self.start_equity) - 1,  # Equity change %
            self.positions / self.config.max_positions,
            self.floating_pnl / 100.0,
            # Guardian context
            guardian_risk,
            block_rate,
            # Confidence proxy (action consistency)
            self._alpha_action_consistency(),
            self.current_dd * 4  # Scaled DD
        ], dtype=np.float32)
    
    def _get_guardian_obs(self) -> np.ndarray:
        """
        Guardian observation:
            - alpha_intent (2): proposed action, confidence
            - account_state (3): equity %, DD, margin
            - risk_context (3): volatility, DD slope, freeze status
        """
        # Latest alpha action (one-hot encoded as single value)
        last_alpha = self.alpha_actions_history[-1] if self.alpha_actions_history else 0
        alpha_intent = last_alpha / 2.0  # Normalize 0-2 to 0-1
        
        return np.array([
            # Alpha intent
            alpha_intent,
            self._alpha_action_consistency(),
            # Account state
            (self.equity / self.start_equity) - 1,
            self.current_dd * 4,
            self.positions / self.config.max_positions,
            # Risk context
            self._volatility,
            self.dd_slope,
            float(self.is_frozen)
        ], dtype=np.float32)
    
    def _alpha_action_consistency(self) -> float:
        """Calculate how consistent Alpha's recent actions are."""
        if len(self.alpha_actions_history) < 5:
            return 0.5
        recent = self.alpha_actions_history[-10:]
        mode_action = max(set(recent), key=recent.count)
        consistency = sum(1 for a in recent if a == mode_action) / len(recent)
        return consistency
    
    # =========================================================================
    # Guardian Logic
    # =========================================================================
    
    def _apply_guardian(
        self,
        alpha_action: AlphaAction,
        guardian_action: GuardianAction
    ) -> Tuple[AlphaAction, float]:
        """
        Apply Guardian decision to Alpha's action.
        
        Returns:
            (final_action, lot_multiplier)
        """
        # Check freeze status
        if self.is_frozen:
            if self.step_count >= self.freeze_until:
                self.is_frozen = False
            else:
                return AlphaAction.HOLD, 0.0
        
        # Apply Guardian action
        if guardian_action == GuardianAction.ALLOW:
            return alpha_action, 1.0
        
        elif guardian_action == GuardianAction.SCALE_DOWN:
            return alpha_action, 0.5  # 50% lot size
        
        elif guardian_action == GuardianAction.BLOCK:
            self.blocks_count += 1
            return AlphaAction.HOLD, 0.0
        
        else:  # FREEZE
            self.freezes_count += 1
            self.is_frozen = True
            self.freeze_until = self.step_count + 10
            return AlphaAction.HOLD, 0.0
    
    # =========================================================================
    # Trade Simulation
    # =========================================================================
    
    def _simulate_trade(
        self,
        action: AlphaAction,
        lot_multiplier: float
    ) -> Tuple[float, float]:
        """
        Simulate trade outcome.
        
        Returns:
            (profit, dd_change)
        """
        profit = 0.0
        dd_change = 0.0
        
        if action == AlphaAction.HOLD:
            # Small cost for holding
            profit = -0.01
        else:
            # Simplified P&L simulation
            direction = 1 if action == AlphaAction.BUY else -1
            market_move = np.random.normal(0, 0.5)  # Random market
            
            # Add trend component based on EMA
            if (self._ema_diff > 0 and direction == 1) or \
               (self._ema_diff < 0 and direction == -1):
                market_move += 0.2  # Trend bonus
            
            profit = market_move * lot_multiplier * 10.0  # Scaled profit
            
            if profit < 0:
                dd_change = abs(profit) / self.equity
        
        return profit, dd_change
    
    def _update_state(self, profit: float, dd_change: float):
        """Update internal state after trade."""
        self.equity += profit
        self.profit_total += profit
        
        old_dd = self.current_dd
        
        if self.equity > self.max_equity:
            self.max_equity = self.equity
            self.current_dd = 0.0
        else:
            self.current_dd = (self.max_equity - self.equity) / self.max_equity
        
        # DD slope (acceleration of drawdown)
        self.dd_slope = self.current_dd - old_dd
        
        # Update simulated market
        self._ema_diff = np.random.uniform(-0.5, 0.5)
        self._rsi = np.random.uniform(-0.5, 0.5)
        self._atr = np.random.uniform(0, 0.5)
        self._volatility = abs(np.random.normal(0, 0.3))
    
    # =========================================================================
    # Reward Functions
    # =========================================================================
    
    def _calculate_alpha_reward(
        self,
        alpha_action: AlphaAction,
        guardian_action: GuardianAction,
        profit: float
    ) -> float:
        """
        Calculate Alpha's reward.
        
        Alpha learns to:
            - Make profitable trades
            - Get Guardian approval
            - Avoid triggering blocks/freezes
        """
        cfg = self.config
        reward = 0.0
        
        # Profit component
        if profit > 0:
            reward += profit * cfg.alpha_profit_scale
        else:
            reward += profit * cfg.alpha_profit_scale * 1.5  # Penalize losses more
        
        # Guardian interaction
        if guardian_action == GuardianAction.ALLOW:
            reward += cfg.alpha_guardian_allow
        elif guardian_action == GuardianAction.BLOCK:
            reward += cfg.alpha_guardian_block
        elif guardian_action == GuardianAction.FREEZE:
            reward += cfg.alpha_trigger_freeze
        
        # HOLD during high DD is good
        if alpha_action == AlphaAction.HOLD and self.current_dd > 0.05:
            reward += cfg.alpha_hold_in_risk
        
        return reward
    
    def _calculate_guardian_reward(
        self,
        alpha_action: AlphaAction,
        guardian_action: GuardianAction,
        dd_change: float,
        profit: float
    ) -> float:
        """
        Calculate Guardian's reward.
        
        Guardian learns to:
            - Avoid DD spikes
            - Preserve profits
            - Not over-block (kill opportunity)
        """
        cfg = self.config
        reward = 0.0
        
        # DD avoided (blocked a trade that would have caused DD)
        if guardian_action >= GuardianAction.BLOCK and dd_change > 0:
            reward += cfg.guardian_dd_avoided
            self.dd_avoided_count += 1
        
        # Profit preserved (allowed trade that was profitable)
        if guardian_action == GuardianAction.ALLOW and profit > 0:
            reward += cfg.guardian_profit_preserved
        
        # Over-blocking penalty
        if guardian_action >= GuardianAction.BLOCK and dd_change <= 0:
            reward += cfg.guardian_over_block  # Blocked unnecessarily
        
        # Late block (DD already happened)
        if guardian_action >= GuardianAction.BLOCK and self.dd_slope > 0.05:
            reward += cfg.guardian_late_block
        
        # Unnecessary freeze
        if guardian_action == GuardianAction.FREEZE and self.current_dd < 0.03:
            reward += cfg.guardian_unnecessary_freeze
        
        return reward
    
    # =========================================================================
    # Info
    # =========================================================================
    
    def _get_info(self) -> Dict[str, Any]:
        """Get step info."""
        return {
            "step": self.step_count,
            "equity": self.equity,
            "current_dd": self.current_dd,
            "positions": self.positions,
            "blocks": self.blocks_count,
            "freezes": self.freezes_count,
            "dd_avoided": self.dd_avoided_count,
            "profit_total": self.profit_total,
            "is_frozen": self.is_frozen
        }
    
    def render(self):
        """Render current state."""
        print(
            f"[Step {self.step_count}] Equity=${self.equity:.2f} | "
            f"DD={self.current_dd:.1%} | Blocks={self.blocks_count} | "
            f"Freezes={self.freezes_count} | Profit=${self.profit_total:.2f}"
        )


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent Trading Environment Test")
    print("=" * 60)
    
    env = MultiAgentTradingEnv()
    obs, _ = env.reset()
    
    print(f"\nAlpha obs shape: {obs['alpha'].shape}")
    print(f"Guardian obs shape: {obs['guardian'].shape}")
    
    print("\n--- Running 20 random steps ---\n")
    
    total_alpha_reward = 0
    total_guardian_reward = 0
    
    for i in range(20):
        actions = {
            "alpha": env.alpha_action_space.sample(),
            "guardian": env.guardian_action_space.sample()
        }
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        total_alpha_reward += rewards["alpha"]
        total_guardian_reward += rewards["guardian"]
        
        alpha_name = AlphaAction(actions["alpha"]).name
        guardian_name = GuardianAction(actions["guardian"]).name
        
        print(
            f"Step {i+1}: Alpha={alpha_name:5s} Guardian={guardian_name:10s} | "
            f"R_alpha={rewards['alpha']:+.2f} R_guard={rewards['guardian']:+.2f}"
        )
        
        if terminated or truncated:
            print("\n--- Episode ended ---")
            break
    
    print(f"\nTotal Alpha Reward: {total_alpha_reward:.2f}")
    print(f"Total Guardian Reward: {total_guardian_reward:.2f}")
    print(f"Final info: {info}")
    print("=" * 60)
