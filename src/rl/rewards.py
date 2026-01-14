# src/rl/rewards.py
"""
Reward Shaping for Competition-Grade RL
=========================================

Two reward functions:
1. Alpha Reward: Maximize score/profit
2. Guardian Reward: Minimize DD, prevent kill

Guardian learns to "break early" rather than "crash hard".
"""

from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("REWARDS")


def alpha_reward(metrics: Dict[str, Any]) -> float:
    """
    Alpha Agent Reward - Score Hunter.
    
    Maximize:
    - Profit
    - Score
    - Win rate
    
    Penalize:
    - Drawdown
    - Losses
    
    Args:
        metrics: {
            "pnl": float,           # Trade P&L
            "score_delta": float,   # Score change
            "win": bool,            # Trade won?
            "dd_delta": float,      # DD change
        }
    
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Profit reward (primary)
    pnl = metrics.get("pnl", 0)
    reward += pnl * 10  # Scale profit
    
    # Score improvement
    score_delta = metrics.get("score_delta", 0)
    reward += score_delta * 5
    
    # Win bonus
    if metrics.get("win", False):
        reward += 2.0
    else:
        reward -= 1.0
    
    # DD penalty (mild for Alpha)
    dd_delta = metrics.get("dd_delta", 0)
    if dd_delta > 0:
        reward -= dd_delta * 3
    
    return reward


def guardian_reward(metrics: Dict[str, Any]) -> float:
    """
    Guardian Agent Reward - Survival Optimizer.
    
    Primary goal: Prevent kill switch trigger
    Secondary: Minimize drawdown
    
    Args:
        metrics: {
            "daily_dd": float,        # Daily DD %
            "equity_slope": float,    # Equity trend
            "kill_triggered": bool,   # Kill switch fired?
            "entries_frozen": bool,   # Entries blocked?
            "pyramid_active": bool,   # Pyramid enabled?
            "guard_level": str,       # Current guard level
        }
    
    Returns:
        Reward value
    """
    reward = 0.0
    
    # ❌ Kill = SEVERE penalty (primary objective)
    if metrics.get("kill_triggered", False):
        return -100.0
    
    # 🔴 Level 3 reached = very bad
    if metrics.get("guard_level") == "LEVEL_3":
        return -50.0
    
    # 🟡 Level 2 = bad
    if metrics.get("guard_level") == "LEVEL_2":
        reward -= 20.0
    
    # 🟠 Level 1 = warning
    if metrics.get("guard_level") == "LEVEL_1":
        reward -= 5.0
    
    # 🔴 Drawdown penalty (non-linear - punishes high DD more)
    dd = metrics.get("daily_dd", 0)
    reward -= (dd ** 2) * 5.0
    
    # 🟢 Early braking bonus (Guardian learned to be cautious)
    if dd > 2.0 and metrics.get("entries_frozen", False):
        reward += 5.0
    
    if dd > 2.5 and not metrics.get("pyramid_active", True):
        reward += 4.0
    
    if dd > 1.5 and metrics.get("entries_frozen", False):
        reward += 2.0  # Extra credit for early caution
    
    # 🟢 Stability reward
    equity_slope = metrics.get("equity_slope", 0)
    reward += equity_slope * 2.0
    
    # 🟢 Survival time bonus
    hours_alive = metrics.get("hours_since_last_kill", 0)
    reward += min(hours_alive / 24, 5)  # Cap at 5 points
    
    return reward


def combined_reward(
    alpha_metrics: Dict[str, Any],
    guardian_metrics: Dict[str, Any],
    alpha_weight: float = 0.6,
    guardian_weight: float = 0.4
) -> Dict[str, float]:
    """
    Calculate combined rewards for multi-agent system.
    
    Guardian weight increases when DD is high.
    
    Args:
        alpha_metrics: Metrics for Alpha agent
        guardian_metrics: Metrics for Guardian agent
        alpha_weight: Base weight for Alpha
        guardian_weight: Base weight for Guardian
    
    Returns:
        {"alpha": float, "guardian": float, "combined": float}
    """
    alpha_r = alpha_reward(alpha_metrics)
    guardian_r = guardian_reward(guardian_metrics)
    
    # Dynamic weight adjustment based on DD
    dd = guardian_metrics.get("daily_dd", 0)
    if dd > 2.0:
        # Increase Guardian weight when DD is high
        adjusted_guardian_weight = min(guardian_weight + (dd - 2.0) * 0.2, 0.8)
        adjusted_alpha_weight = 1.0 - adjusted_guardian_weight
    else:
        adjusted_alpha_weight = alpha_weight
        adjusted_guardian_weight = guardian_weight
    
    combined = (
        alpha_r * adjusted_alpha_weight +
        guardian_r * adjusted_guardian_weight
    )
    
    return {
        "alpha": alpha_r,
        "guardian": guardian_r,
        "combined": combined,
        "alpha_weight": adjusted_alpha_weight,
        "guardian_weight": adjusted_guardian_weight,
    }


# =============================================================================
# Reward Metrics Helper
# =============================================================================

def calculate_guardian_metrics(
    dd_today: float,
    dd_total: float,
    guard_level: str,
    kill_triggered: bool,
    entries_frozen: bool,
    pyramid_active: bool,
    equity_history: list = None
) -> Dict[str, Any]:
    """Helper to build guardian metrics dict."""
    
    # Calculate equity slope
    equity_slope = 0.0
    if equity_history and len(equity_history) >= 2:
        equity_slope = (equity_history[-1] - equity_history[-2]) / max(equity_history[-2], 1)
    
    return {
        "daily_dd": dd_today,
        "total_dd": dd_total,
        "guard_level": guard_level,
        "kill_triggered": kill_triggered,
        "entries_frozen": entries_frozen,
        "pyramid_active": pyramid_active,
        "equity_slope": equity_slope,
    }
