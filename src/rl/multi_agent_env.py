# src/rl/multi_agent_env.py
"""
Multi-Agent Trading Environment
================================

Competition-grade environment for Alpha vs Guardian training.

Architecture:
    Alpha Agent  →  proposes trades/pyramid
    Guardian Agent → suggests risk adjustments
    ProgressiveGuard → HARD enforcement (cannot bypass)
    
Reward Structure:
    Alpha: Maximize score/profit
    Guardian: Minimize DD/prevent kill
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.rl.rewards import alpha_reward, guardian_reward

logger = get_logger("MULTI_AGENT_ENV")


@dataclass
class EnvState:
    """Environment state."""
    equity: float
    dd_today: float
    dd_total: float
    positions: int
    win_rate: float
    volatility: float
    memory_percent: float
    guard_level: str
    score: float


class TradingEnvMultiAgent:
    """
    Multi-Agent Trading Environment.
    
    Two agents:
    - Alpha: Entry, pyramid, aggression
    - Guardian: Risk scaling, freeze suggestions
    
    ProgressiveGuard is always enforced first.
    """
    
    def __init__(
        self,
        initial_equity: float = 10000,
        max_dd_limit: float = 10.0,
        base_risk: float = 2.0
    ):
        """Initialize environment."""
        self.initial_equity = initial_equity
        self.max_dd_limit = max_dd_limit
        self.base_risk = base_risk
        
        # State
        self.equity = initial_equity
        self.dd_today = 0.0
        self.dd_total = 0.0
        self.positions = 0
        self.win_rate = 0.5
        self.volatility = 1.0
        self.memory_percent = 50.0
        self.score = 0.0
        
        # Guard state
        self.guard_level = "OK"
        self.kill_latched = False
        self.entries_frozen = False
        self.pyramid_active = True
        
        # History
        self.equity_history: List[float] = [initial_equity]
        self.step_count = 0
        
        logger.info("MultiAgent Environment initialized")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment."""
        self.equity = self.initial_equity
        self.dd_today = 0.0
        self.dd_total = 0.0
        self.positions = 0
        self.win_rate = 0.5
        self.volatility = 1.0
        self.memory_percent = 50.0
        self.score = 0.0
        
        self.guard_level = "OK"
        self.kill_latched = False
        self.entries_frozen = False
        self.pyramid_active = True
        
        self.equity_history = [self.initial_equity]
        self.step_count = 0
        
        return {
            "alpha": self._get_alpha_obs(),
            "guardian": self._get_guardian_obs(),
        }
    
    def step(
        self,
        actions: Dict[str, Dict]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        Execute one step.
        
        Args:
            actions: {
                "alpha": {"entry": 0/1, "pyramid": 0/1, "aggression": float},
                "guardian": {"risk_mult": float, "freeze": bool}
            }
        
        Returns:
            obs, rewards, done, info
        """
        self.step_count += 1
        
        # === 1. Progressive Guard ALWAYS FIRST ===
        self._update_guard()
        
        if self.kill_latched:
            # Episode ends on kill
            return self._terminal_state()
        
        # === 2. Apply Guardian suggestions ===
        guardian_action = actions.get("guardian", {})
        risk_mult = guardian_action.get("risk_mult", 1.0)
        
        if guardian_action.get("freeze", False):
            self.entries_frozen = True
        
        if risk_mult < 0.5:
            self.pyramid_active = False
        
        # === 3. Alpha executes under constraints ===
        alpha_action = actions.get("alpha", {})
        
        if not self.entries_frozen and alpha_action.get("entry", 0):
            self._execute_entry(risk_mult, alpha_action.get("aggression", 1.0))
        
        if self.pyramid_active and alpha_action.get("pyramid", 0):
            self._execute_pyramid(risk_mult)
        
        # === 4. Simulate market movement ===
        self._simulate_market()
        
        # === 5. Calculate rewards ===
        alpha_r = self._calculate_alpha_reward()
        guardian_r = self._calculate_guardian_reward()
        
        # === 6. Check done ===
        done = self.kill_latched or self.dd_total > self.max_dd_limit
        
        obs = {
            "alpha": self._get_alpha_obs(),
            "guardian": self._get_guardian_obs(),
        }
        
        rewards = {
            "alpha": alpha_r,
            "guardian": guardian_r,
        }
        
        info = {
            "equity": self.equity,
            "dd_today": self.dd_today,
            "guard_level": self.guard_level,
            "score": self.score,
        }
        
        return obs, rewards, done, info
    
    def _update_guard(self) -> None:
        """Update Progressive Guard state."""
        # DD-based levels
        if self.dd_today > 5.0:
            self.guard_level = "LEVEL_3"
            self.kill_latched = True
        elif self.dd_today > 3.0:
            self.guard_level = "LEVEL_2"
            self.entries_frozen = True
        elif self.dd_today > 2.0:
            self.guard_level = "LEVEL_1"
            self.pyramid_active = False
        else:
            self.guard_level = "OK"
    
    def _execute_entry(self, risk_mult: float, aggression: float) -> None:
        """Execute entry trade."""
        effective_risk = self.base_risk * risk_mult * aggression
        # Simplified PnL simulation
        win = np.random.random() < self.win_rate
        if win:
            pnl = effective_risk * 1.5  # RR 1:1.5
            self.score += 1
        else:
            pnl = -effective_risk
            self.score -= 0.5
        
        self.equity *= (1 + pnl / 100)
        self.positions += 1
        self._update_dd()
    
    def _execute_pyramid(self, risk_mult: float) -> None:
        """Execute pyramid add."""
        effective_risk = self.base_risk * risk_mult * 0.5
        win = np.random.random() < self.win_rate
        if win:
            pnl = effective_risk
            self.score += 0.5
        else:
            pnl = -effective_risk
        
        self.equity *= (1 + pnl / 100)
        self._update_dd()
    
    def _simulate_market(self) -> None:
        """Simulate market noise."""
        noise = np.random.normal(0, 0.001)
        self.equity *= (1 + noise)
        self.volatility = max(0.5, min(3.0, self.volatility + np.random.normal(0, 0.1)))
        self._update_dd()
    
    def _update_dd(self) -> None:
        """Update drawdown."""
        peak = max(self.equity_history)
        self.dd_total = (peak - self.equity) / peak * 100
        
        # Simple daily DD (reset every 100 steps)
        if self.step_count % 100 == 0:
            self.dd_today = 0
        else:
            day_start = self.equity_history[-min(100, len(self.equity_history))]
            self.dd_today = max(0, (day_start - self.equity) / day_start * 100)
        
        self.equity_history.append(self.equity)
    
    def _calculate_alpha_reward(self) -> float:
        """Calculate Alpha reward."""
        if len(self.equity_history) < 2:
            return 0
        
        pnl = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
        win = pnl > 0
        
        return alpha_reward({
            "pnl": pnl * 100,
            "score_delta": 0.1 if win else -0.05,
            "win": win,
            "dd_delta": max(0, self.dd_today - 1),
        })
    
    def _calculate_guardian_reward(self) -> float:
        """Calculate Guardian reward."""
        return guardian_reward({
            "daily_dd": self.dd_today,
            "equity_slope": (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2] if len(self.equity_history) >= 2 else 0,
            "kill_triggered": self.kill_latched,
            "entries_frozen": self.entries_frozen,
            "pyramid_active": self.pyramid_active,
            "guard_level": self.guard_level,
        })
    
    def _get_alpha_obs(self) -> np.ndarray:
        """Get Alpha observation."""
        return np.array([
            self.equity / self.initial_equity,
            self.positions / 5,
            self.volatility / 3,
            self.win_rate,
            self.score / 100,
            float(not self.entries_frozen),
            float(self.pyramid_active),
        ], dtype=np.float32)
    
    def _get_guardian_obs(self) -> np.ndarray:
        """Get Guardian observation."""
        guard_map = {"OK": 0, "LEVEL_1": 1, "LEVEL_2": 2, "LEVEL_3": 3}
        return np.array([
            self.dd_today / 10,
            self.dd_total / 20,
            self.memory_percent / 100,
            self.win_rate,
            self.volatility / 3,
            guard_map.get(self.guard_level, 0) / 3,
            self.positions / 5,
            float(self.entries_frozen),
        ], dtype=np.float32)
    
    def _terminal_state(self) -> Tuple[Dict, Dict, bool, Dict]:
        """Return terminal state on kill."""
        obs = {
            "alpha": self._get_alpha_obs(),
            "guardian": self._get_guardian_obs(),
        }
        rewards = {
            "alpha": -50,
            "guardian": -100,
        }
        return obs, rewards, True, {"kill": True}


# =============================================================================
# Training Helper
# =============================================================================

def train_multi_agent(
    env: TradingEnvMultiAgent,
    alpha_agent,
    guardian_agent,
    episodes: int = 1000,
    max_steps: int = 500
) -> Dict[str, List[float]]:
    """
    Train multi-agent system.
    
    Args:
        env: Trading environment
        alpha_agent: Alpha RL agent
        guardian_agent: Guardian RL agent
        episodes: Number of episodes
        max_steps: Max steps per episode
    
    Returns:
        Training history
    """
    history = {
        "alpha_rewards": [],
        "guardian_rewards": [],
        "episode_lengths": [],
        "kill_count": 0,
    }
    
    for ep in range(episodes):
        obs = env.reset()
        episode_alpha_reward = 0
        episode_guardian_reward = 0
        
        for step in range(max_steps):
            # Get actions from both agents
            alpha_action = {
                "entry": 1 if np.random.random() < 0.3 else 0,
                "pyramid": 1 if np.random.random() < 0.2 else 0,
                "aggression": np.random.uniform(0.5, 1.5),
            }
            
            guardian_action = {
                "risk_mult": np.random.uniform(0.5, 1.0),
                "freeze": np.random.random() < 0.1,
            }
            
            actions = {
                "alpha": alpha_action,
                "guardian": guardian_action,
            }
            
            obs, rewards, done, info = env.step(actions)
            
            episode_alpha_reward += rewards["alpha"]
            episode_guardian_reward += rewards["guardian"]
            
            if done:
                if info.get("kill"):
                    history["kill_count"] += 1
                break
        
        history["alpha_rewards"].append(episode_alpha_reward)
        history["guardian_rewards"].append(episode_guardian_reward)
        history["episode_lengths"].append(step + 1)
        
        if (ep + 1) % 100 == 0:
            avg_alpha = np.mean(history["alpha_rewards"][-100:])
            avg_guardian = np.mean(history["guardian_rewards"][-100:])
            logger.info(
                f"Episode {ep+1}: Alpha={avg_alpha:.1f}, "
                f"Guardian={avg_guardian:.1f}, "
                f"Kills={history['kill_count']}"
            )
    
    return history
