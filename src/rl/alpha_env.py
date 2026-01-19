# src/rl/alpha_env.py
"""
Alpha PPO V1 Environment
========================

Competition-Ready Gymnasium environment for training Alpha PPO.

Features:
- Shadow-mode safe (no real order execution)
- Guardian-aware (penalizes actions during warnings/blocks)
- Explainable state representation
- Ready for ablation & chaos testing

Architecture Position:
    Alpha PPO (this env) → Guardian Layer → MT5 Execution
    
Note: Alpha PPO CANNOT bypass Guardian. This is by design.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum


class AlphaAction(IntEnum):
    """Alpha Agent action space."""
    HOLD = 0
    BUY = 1
    SELL = 2


class GuardianState(IntEnum):
    """Guardian state levels."""
    OK = 0
    WARNING = 1
    BLOCK = 2


@dataclass
class AlphaEnvConfig:
    """Configuration for Alpha Trading Environment."""
    max_positions: int = 5
    dd_penalty: float = 2.0
    guardian_penalty: float = 3.0
    trade_cost: float = 0.1
    hold_bonus: float = 0.01
    max_steps: int = 1000
    dd_terminate_threshold: float = 0.25
    
    # Reward shaping
    profit_scale: float = 0.1
    loss_scale: float = 0.15  # Penalize losses more
    overtrade_penalty: float = 1.0
    consistency_bonus: float = 0.2


class AlphaTradingEnv(gym.Env):
    """
    Alpha PPO V1 Environment.
    
    Shadow-mode safe, Guardian-aware, Competition-ready.
    
    Observation Space (8-dim, normalized to [-1, 1]):
        0: EMA20 - EMA50 (normalized trend)
        1: RSI (scaled from 0-100 to -1..1)
        2: ATR (normalized volatility)
        3: Spread (normalized cost)
        4: Time of Day (0-1, session encoding)
        5: Open Positions (normalized by max)
        6: Floating DD (scaled drawdown)
        7: Guardian State (0=OK, 0.5=WARNING, 1=BLOCK)
    
    Action Space (Discrete 3):
        0: HOLD
        1: BUY
        2: SELL
    
    Reward Function:
        + profit (scaled)
        - loss (scaled more heavily)
        - trade cost
        - overtrade penalty
        - guardian violation penalty
        - drawdown penalty
        + hold bonus (small)
        + consistency bonus
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        config: Optional[AlphaEnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.config = config or AlphaEnvConfig()
        self.render_mode = render_mode
        
        # ---- Action Space ----
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        
        # ---- Observation Space ----
        # All features normalized to [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )
        
        # ---- Internal State ----
        self._reset_internal_state()
        
    def _reset_internal_state(self):
        """Reset all internal tracking variables."""
        self.step_count = 0
        self.position_count = 0
        self.floating_dd = 0.0
        self.guardian_state = GuardianState.OK
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Market state (injected externally or simulated)
        self._ema_fast_diff = 0.0
        self._rsi = 0.0
        self._atr = 0.0
        self._spread = 0.0
        self._time_of_day = 0.0
        self._last_profit = 0.0
        
    # =========================================================================
    # Core Gymnasium Interface
    # =========================================================================
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._reset_internal_state()
        
        return self._get_obs(), self._get_info()
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        This is SHADOW evaluation only - no real orders are placed.
        The reward simulates what would happen if this action were taken.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # ---- Action Processing ----
        action_enum = AlphaAction(action)
        
        if action_enum == AlphaAction.HOLD:
            # Small bonus for holding (patience)
            reward += self.config.hold_bonus
            
        elif action_enum in (AlphaAction.BUY, AlphaAction.SELL):
            # Trade cost
            reward -= self.config.trade_cost
            self.trade_count += 1
            self.position_count += 1
            
            # Overtrade penalty
            if self.position_count > self.config.max_positions:
                reward -= self.config.overtrade_penalty
                
            # Simulate profit/loss (in real mode, this comes from market)
            if self._last_profit > 0:
                reward += self._last_profit * self.config.profit_scale
                self.win_count += 1
            elif self._last_profit < 0:
                reward += self._last_profit * self.config.loss_scale
                self.loss_count += 1
        
        # ---- Guardian Awareness ----
        # Penalize trading during Guardian warnings/blocks
        if self.guardian_state == GuardianState.BLOCK:
            if action != AlphaAction.HOLD:
                reward -= self.config.guardian_penalty
                
        elif self.guardian_state == GuardianState.WARNING:
            if action != AlphaAction.HOLD:
                reward -= self.config.guardian_penalty * 0.5
        
        # ---- Drawdown Penalty ----
        # Continuous penalty proportional to DD
        reward -= self.floating_dd * self.config.dd_penalty
        
        # ---- Consistency Bonus ----
        if self.trade_count > 0:
            win_rate = self.win_count / self.trade_count
            if win_rate > 0.55:
                reward += self.config.consistency_bonus
        
        # ---- Update Tracking ----
        self.last_reward = reward
        self.cumulative_reward += reward
        
        # ---- Termination Conditions ----
        if self.floating_dd >= self.config.dd_terminate_threshold:
            terminated = True
            reward -= 5.0  # Heavy penalty for blowing up
            
        if self.step_count >= self.config.max_steps:
            truncated = True
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    # =========================================================================
    # External State Injection (for Live/Shadow Mode)
    # =========================================================================
    
    def update_market_state(
        self,
        ema_fast_diff: float,
        rsi: float,
        atr: float,
        spread: float,
        time_of_day: float,
        open_positions: int,
        floating_dd: float,
        guardian_state: int,
        last_profit: float = 0.0
    ):
        """
        Inject current market state from live loop or backtester.
        
        All values should already be normalized to appropriate ranges:
        - ema_fast_diff: Centered around 0, scaled by typical range
        - rsi: Scaled from [0,100] to [-1,1]
        - atr: Scaled by typical ATR value
        - spread: Scaled by typical spread
        - time_of_day: [0,1] representing trading session
        - guardian_state: 0=OK, 1=WARNING, 2=BLOCK
        
        Args:
            ema_fast_diff: EMA20 - EMA50 (normalized)
            rsi: RSI value (normalized to [-1,1])
            atr: ATR value (normalized)
            spread: Current spread (normalized)
            time_of_day: Time encoding [0,1]
            open_positions: Current open position count
            floating_dd: Current floating drawdown [0,1]
            guardian_state: Guardian state enum value
            last_profit: Profit/loss from last closed trade
        """
        self._ema_fast_diff = np.clip(ema_fast_diff, -1, 1)
        self._rsi = np.clip(rsi, -1, 1)
        self._atr = np.clip(atr, -1, 1)
        self._spread = np.clip(spread, -1, 1)
        self._time_of_day = np.clip(time_of_day, 0, 1)
        self.position_count = open_positions
        self.floating_dd = np.clip(floating_dd, 0, 1)
        self.guardian_state = GuardianState(guardian_state)
        self._last_profit = last_profit
    
    # =========================================================================
    # Observation & Info Helpers
    # =========================================================================
    
    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        return np.array([
            self._ema_fast_diff,
            self._rsi,
            self._atr,
            self._spread,
            self._time_of_day * 2 - 1,  # Scale to [-1, 1]
            np.tanh(self.position_count / self.config.max_positions),
            np.clip(self.floating_dd * 4, -1, 1),  # Scale DD for sensitivity
            self.guardian_state / 2.0  # 0=0, 1=0.5, 2=1.0
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary."""
        return {
            "step": self.step_count,
            "positions": self.position_count,
            "floating_dd": self.floating_dd,
            "guardian_state": self.guardian_state.name,
            "last_reward": self.last_reward,
            "cumulative_reward": self.cumulative_reward,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_count / max(self.trade_count, 1)
        }
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    def render(self):
        """Render current state."""
        if self.render_mode == "human":
            print(self._render_string())
        elif self.render_mode == "ansi":
            return self._render_string()
    
    def _render_string(self) -> str:
        """Generate render string."""
        return (
            f"[AlphaEnv] step={self.step_count:4d} | "
            f"pos={self.position_count} | "
            f"dd={self.floating_dd:.2%} | "
            f"guardian={self.guardian_state.name:7s} | "
            f"reward={self.last_reward:+.3f} | "
            f"cumulative={self.cumulative_reward:+.2f}"
        )


# =============================================================================
# Utility: State Normalizer
# =============================================================================

class AlphaStateNormalizer:
    """
    Normalize raw market data to [-1, 1] for the environment.
    
    Usage in live_loop:
        normalizer = AlphaStateNormalizer()
        state = normalizer.normalize(df.iloc[-1], account_info, guardian_state)
        env.update_market_state(**state)
    """
    
    def __init__(
        self,
        ema_diff_scale: float = 10.0,  # Typical EMA diff range
        atr_scale: float = 5.0,        # Typical ATR value
        spread_scale: float = 0.5      # Typical spread in points
    ):
        self.ema_diff_scale = ema_diff_scale
        self.atr_scale = atr_scale
        self.spread_scale = spread_scale
    
    def normalize(
        self,
        row,  # DataFrame row with market data
        open_positions: int,
        floating_dd: float,
        guardian_state: int,
        last_profit: float = 0.0
    ) -> Dict[str, float]:
        """
        Normalize raw market data for environment.
        
        Args:
            row: DataFrame row with columns: ema20, ema50, rsi14, atr14, spread, etc.
            open_positions: Current open position count
            floating_dd: Current floating DD percentage
            guardian_state: 0=OK, 1=WARNING, 2=BLOCK
            last_profit: Last trade profit/loss
            
        Returns:
            Dictionary ready for env.update_market_state()
        """
        from datetime import datetime
        
        # EMA difference normalized
        ema20 = row.get('ema20', row.get('EMA20', 0))
        ema50 = row.get('ema50', row.get('EMA50', 0))
        ema_diff = (ema20 - ema50) / self.ema_diff_scale
        
        # RSI: 0-100 → -1..1
        rsi = row.get('rsi14', row.get('RSI14', row.get('rsi', 50)))
        rsi_norm = (rsi - 50) / 50
        
        # ATR normalized
        atr = row.get('atr14', row.get('ATR14', row.get('atr', 2.5)))
        atr_norm = atr / self.atr_scale
        
        # Spread normalized
        spread = row.get('spread', 0.1)
        spread_norm = spread / self.spread_scale
        
        # Time of day: Hour as fraction
        hour = row.get('hour', datetime.now().hour)
        time_of_day = hour / 24.0
        
        return {
            "ema_fast_diff": float(np.clip(ema_diff, -1, 1)),
            "rsi": float(np.clip(rsi_norm, -1, 1)),
            "atr": float(np.clip(atr_norm, -1, 1)),
            "spread": float(np.clip(spread_norm, -1, 1)),
            "time_of_day": float(time_of_day),
            "open_positions": int(open_positions),
            "floating_dd": float(np.clip(floating_dd, 0, 1)),
            "guardian_state": int(guardian_state),
            "last_profit": float(last_profit)
        }


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha PPO V1 Environment Test")
    print("=" * 60)
    
    # Create environment
    config = AlphaEnvConfig(max_steps=20)
    env = AlphaTradingEnv(config=config, render_mode="human")
    
    # Random agent test
    obs, info = env.reset()
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    print("\n--- Running 20 random steps ---\n")
    
    for i in range(20):
        # Simulate market state injection
        env.update_market_state(
            ema_fast_diff=np.random.uniform(-0.5, 0.5),
            rsi=np.random.uniform(-0.5, 0.5),
            atr=np.random.uniform(0, 0.5),
            spread=np.random.uniform(0, 0.3),
            time_of_day=i / 24.0,
            open_positions=np.random.randint(0, 3),
            floating_dd=np.random.uniform(0, 0.1),
            guardian_state=np.random.choice([0, 0, 0, 1]),  # Mostly OK
            last_profit=np.random.uniform(-1, 1)
        )
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_name = AlphaAction(action).name
        print(f"Step {i+1:2d}: action={action_name:4s} → {env._render_string()}")
        
        if terminated or truncated:
            print("\n--- Episode ended ---")
            break
    
    print(f"\nFinal stats: {info}")
    print("=" * 60)
