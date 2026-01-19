# src/envs/market_env_v3.py
"""
Market Environment V3
======================

Gymnasium environment for Alpha PPO training.

Features:
    - Multi-timeframe observation
    - Competition reward function
    - Shadow mode support
    - Guardian-free training

Paper Statement:
    "We design a market environment that separates trading
     decisions from risk management, enabling clean policy learning."
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Callable, Dict, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("MARKET_ENV_V3")


class MarketEnvV3(gym.Env):
    """
    Trading environment for Alpha PPO V1.
    
    Observation Space (12 features):
        - price_return_1m, price_return_5m
        - ema_diff (EMA20 - EMA50)
        - rsi, atr, volatility
        - current_position
        - free_margin_pct
        - hour_of_day
        - spread
        - unrealized_pnl
        - time_in_position
    
    Action Space:
        0 = HOLD
        1 = BUY (Long)
        2 = SELL (Short)
    
    Reward:
        Customizable via reward_fn
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        mode: str = "train",
        reward_fn: Optional[Callable] = None,
        enable_guardian: bool = False,
        shadow_mode: bool = False,
        episode_length: int = 1000,
        initial_balance: float = 1000.0
    ):
        super().__init__()
        
        self.mode = mode
        self.reward_fn = reward_fn
        self.enable_guardian = enable_guardian
        self.shadow_mode = shadow_mode
        self.episode_length = episode_length
        self.initial_balance = initial_balance
        
        # State
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.step_idx = 0
        self.entry_price = 0.0
        self.time_in_position = 0
        self.max_equity = initial_balance
        
        # Market simulation
        self.prices = None
        self._generate_market_data()
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
    
    def _generate_market_data(self):
        """Generate synthetic market data."""
        n = self.episode_length + 100
        
        # Random walk with trend
        returns = np.random.randn(n) * 0.001
        trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.0005
        self.returns = returns + trend
        
        # Price series
        self.prices = 2000 * np.exp(np.cumsum(self.returns))
        
        # Indicators
        self.ema20 = self._ema(self.prices, 20)
        self.ema50 = self._ema(self.prices, 50)
        self.rsi = self._rsi(self.prices, 14)
        self.atr = self._atr(self.prices, 14)
        self.volatility = np.std(self.returns.reshape(-1, 10), axis=1).repeat(10)[:n]
    
    def _ema(self, prices, period):
        """Calculate EMA."""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        alpha = 2 / (period + 1)
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _rsi(self, prices, period):
        """Calculate RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            avg_gain[i] = np.mean(gains[i-period:i])
            avg_loss[i] = np.mean(losses[i-period:i])
        
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - 100 / (1 + rs)
    
    def _atr(self, prices, period):
        """Calculate ATR (simplified)."""
        atr = np.zeros_like(prices)
        for i in range(period, len(prices)):
            atr[i] = np.std(prices[i-period:i])
        return atr
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self._generate_market_data()
        
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.step_idx = 50  # Start after warmup
        self.entry_price = 0.0
        self.time_in_position = 0
        self.max_equity = self.initial_balance
        
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_equity = self.equity
        
        price = self.prices[self.step_idx]
        pnl = 0.0
        dd_inc = 0.0
        overtrade = 0.0
        trend_align = 0.0
        
        # Calculate trend
        trend = 1 if self.ema20[self.step_idx] > self.ema50[self.step_idx] else -1
        
        # Execute action
        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.time_in_position = 0
                self.total_trades += 1
                trend_align = 1.0 if trend == 1 else -0.5
            elif self.position == 1:
                overtrade = 1.0
        
        elif action == 2:  # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.time_in_position = 0
                self.total_trades += 1
                trend_align = 1.0 if trend == -1 else -0.5
            elif self.position == -1:
                overtrade = 1.0
        
        else:  # HOLD
            if self.position != 0:
                self.time_in_position += 1
        
        # Close position if opposite signal
        if self.position == 1 and action == 2:
            pnl = (price - self.entry_price) / self.entry_price * 100
            if pnl > 0:
                self.winning_trades += 1
            self.total_pnl += pnl
            self.position = 0
        elif self.position == -1 and action == 1:
            pnl = (self.entry_price - price) / self.entry_price * 100
            if pnl > 0:
                self.winning_trades += 1
            self.total_pnl += pnl
            self.position = 0
        
        # Update equity
        if self.position == 1:
            unrealized = (price - self.entry_price) / self.entry_price * 100
            self.equity = self.balance * (1 + unrealized / 100)
        elif self.position == -1:
            unrealized = (self.entry_price - price) / self.entry_price * 100
            self.equity = self.balance * (1 + unrealized / 100)
        else:
            self.balance += pnl * self.balance / 100
            self.equity = self.balance
        
        # DD tracking
        self.max_equity = max(self.max_equity, self.equity)
        current_dd = (self.max_equity - self.equity) / self.max_equity * 100
        dd_inc = max(0, current_dd - 0)  # Simplified
        
        # Build info
        info = {
            "pnl": pnl,
            "dd_inc": dd_inc,
            "overtrade": overtrade,
            "trend_align": trend_align,
            "equity": self.equity,
            "position": self.position,
            "price": price
        }
        
        # Calculate reward
        if self.reward_fn:
            reward = self.reward_fn(info)
        else:
            reward = pnl - 0.5 * dd_inc
        
        # Step forward
        self.step_idx += 1
        terminated = self.step_idx >= len(self.prices) - 1
        truncated = self.step_idx >= self.episode_length
        
        # Episode end stats
        if terminated or truncated:
            info["total_trades"] = self.total_trades
            info["win_rate"] = self.winning_trades / max(self.total_trades, 1)
            info["total_pnl"] = self.total_pnl
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        i = self.step_idx
        price = self.prices[i]
        
        return np.array([
            self.returns[i] if i < len(self.returns) else 0,  # ret1m
            np.mean(self.returns[max(0,i-5):i]) if i > 0 else 0,  # ret5m
            (self.ema20[i] - self.ema50[i]) / price * 100,  # ema_diff
            (self.rsi[i] - 50) / 50,  # rsi normalized
            self.atr[i] / price * 100,  # atr normalized
            self.volatility[i] * 100 if i < len(self.volatility) else 0,  # vol
            float(self.position),  # position
            min(1.0, self.equity / self.initial_balance),  # margin proxy
            (self.step_idx % 24) / 24,  # hour of day
            0.0001,  # spread (fixed)
            (self.equity - self.balance) / self.balance if self.balance > 0 else 0,  # unrealized
            self.time_in_position / 100,  # time in position
        ], dtype=np.float32)
    
    def render(self):
        print(f"Step {self.step_idx}: Price={self.prices[self.step_idx]:.2f}, "
              f"Position={self.position}, Equity={self.equity:.2f}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MarketEnvV3 Test")
    print("=" * 60)
    
    env = MarketEnvV3()
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if term or trunc:
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("=" * 60)
