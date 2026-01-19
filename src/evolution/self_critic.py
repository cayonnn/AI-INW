# src/evolution/self_critic.py
"""
Self-Critic & Self-Repair AI
=============================

AI that monitors itself for degradation and auto-repairs.

Self-Questions:
    - "Am I overfitting?"
    - "Is my edge still valid?"
    - "Should I retrain?"

Actions:
    - Reduce position (if uncertain)
    - Retrain (if decaying)
    - Kill strategy (if dead)
    - Spawn new strategy (if opportunity)

Paper Statement:
    "Our system includes a self-critic module that detects performance
     degradation and autonomously initiates corrective actions."
"""

import os
import sys
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("SELF_CRITIC")


class SelfCriticAction(Enum):
    """Self-repair actions."""
    ALL_GOOD = "all_good"
    REDUCE_RISK = "reduce_risk"
    RETRAIN = "retrain"
    KILL_STRATEGY = "kill_strategy"
    SPAWN_STRATEGY = "spawn_strategy"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class HealthCheck:
    """System health check result."""
    overall_health: float  # 0-1
    issues: List[str]
    action: SelfCriticAction
    confidence: float
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            "health": round(self.overall_health, 3),
            "action": self.action.value,
            "issues": self.issues,
            "reason": self.reason
        }


class SelfCriticAI:
    """
    Self-monitoring and self-repair AI.
    
    Continuously evaluates system health and
    takes corrective actions when needed.
    
    Features:
        - Performance trend analysis
        - Overfit detection
        - Edge validity checks
        - Automatic remediation
    """
    
    def __init__(self):
        # Performance tracking
        self.recent_trades: deque = deque(maxlen=100)
        self.equity_history: deque = deque(maxlen=200)
        self.prediction_accuracy: deque = deque(maxlen=50)
        
        # Health metrics
        self.last_health_check: Optional[HealthCheck] = None
        self.health_history: List[HealthCheck] = []
        
        # Thresholds
        self.overfit_threshold = 0.30  # >30% gap = overfit
        self.edge_decay_threshold = 0.90  # <90% of original = decaying
        self.min_win_rate = 0.45  # Below = problem
        self.max_dd = 0.15  # Above = critical
        
        logger.info("🪞 SelfCriticAI initialized")
    
    def record_trade(self, profit: float, is_win: bool, predicted_correct: bool):
        """Record trade result."""
        self.recent_trades.append({
            "profit": profit,
            "win": is_win,
            "predicted_correct": predicted_correct,
            "timestamp": datetime.now()
        })
        self.prediction_accuracy.append(1 if predicted_correct else 0)
    
    def record_equity(self, equity: float):
        """Record equity point."""
        self.equity_history.append((datetime.now(), equity))
    
    def check_health(self) -> HealthCheck:
        """
        Comprehensive health check.
        
        Analyzes:
            - Recent performance
            - Prediction accuracy
            - Equity trend
            - Overfit signals
        """
        issues = []
        health_scores = []
        
        # Check 1: Win rate
        win_rate_health, win_issues = self._check_win_rate()
        health_scores.append(win_rate_health)
        issues.extend(win_issues)
        
        # Check 2: Prediction accuracy
        pred_health, pred_issues = self._check_prediction_accuracy()
        health_scores.append(pred_health)
        issues.extend(pred_issues)
        
        # Check 3: Equity trend
        equity_health, equity_issues = self._check_equity_trend()
        health_scores.append(equity_health)
        issues.extend(equity_issues)
        
        # Check 4: Overfit detection
        overfit_health, overfit_issues = self._detect_overfit()
        health_scores.append(overfit_health)
        issues.extend(overfit_issues)
        
        # Overall health
        overall_health = np.mean(health_scores) if health_scores else 1.0
        
        # Determine action
        action, reason = self._determine_action(overall_health, issues)
        
        health_check = HealthCheck(
            overall_health=overall_health,
            issues=issues,
            action=action,
            confidence=0.8,
            reason=reason
        )
        
        self.last_health_check = health_check
        self.health_history.append(health_check)
        
        return health_check
    
    def _check_win_rate(self) -> tuple:
        """Check recent win rate."""
        if len(self.recent_trades) < 10:
            return 1.0, []
        
        wins = sum(1 for t in self.recent_trades if t["win"])
        win_rate = wins / len(self.recent_trades)
        
        if win_rate < self.min_win_rate:
            health = win_rate / self.min_win_rate
            return health, [f"LOW_WIN_RATE ({win_rate:.0%})"]
        
        return 1.0, []
    
    def _check_prediction_accuracy(self) -> tuple:
        """Check prediction accuracy."""
        if len(self.prediction_accuracy) < 10:
            return 1.0, []
        
        accuracy = np.mean(list(self.prediction_accuracy))
        
        if accuracy < 0.50:
            return accuracy, [f"LOW_PREDICTION_ACCURACY ({accuracy:.0%})"]
        
        return 1.0, []
    
    def _check_equity_trend(self) -> tuple:
        """Check equity curve trend."""
        if len(self.equity_history) < 20:
            return 1.0, []
        
        equities = [e[1] for e in list(self.equity_history)[-30:]]
        
        # Check if declining
        if len(equities) >= 10:
            recent = np.mean(equities[-10:])
            older = np.mean(equities[:10])
            
            if recent < older * 0.95:  # 5% decline
                health = recent / older
                return health, [f"EQUITY_DECLINING ({(1-health)*100:.1f}% drop)"]
        
        return 1.0, []
    
    def _detect_overfit(self) -> tuple:
        """Detect overfitting signals."""
        # Simple heuristic: check if recent performance diverges from historical
        if len(self.recent_trades) < 20:
            return 1.0, []
        
        recent = list(self.recent_trades)[-10:]
        older = list(self.recent_trades)[-30:-10]
        
        if len(older) < 10:
            return 1.0, []
        
        recent_wr = sum(1 for t in recent if t["win"]) / len(recent)
        older_wr = sum(1 for t in older if t["win"]) / len(older)
        
        gap = abs(recent_wr - older_wr)
        
        if gap > self.overfit_threshold:
            return 1 - gap, [f"POSSIBLE_OVERFIT (gap={gap:.0%})"]
        
        return 1.0, []
    
    def _determine_action(self, health: float, issues: List[str]) -> tuple:
        """Determine corrective action."""
        if health < 0.3:
            return SelfCriticAction.EMERGENCY_STOP, "Critical health - emergency stop"
        
        if health < 0.5:
            if "LOW_WIN_RATE" in str(issues):
                return SelfCriticAction.RETRAIN, "Poor performance - retrain needed"
            return SelfCriticAction.KILL_STRATEGY, "Strategy failing - consider replacement"
        
        if health < 0.7:
            return SelfCriticAction.REDUCE_RISK, "Health declining - reduce exposure"
        
        if health < 0.85:
            if "POSSIBLE_OVERFIT" in str(issues):
                return SelfCriticAction.RETRAIN, "Overfit detected - retrain suggested"
            return SelfCriticAction.REDUCE_RISK, "Minor issues - reduce risk"
        
        return SelfCriticAction.ALL_GOOD, "System healthy"
    
    def should_retrain(self) -> bool:
        """Check if retraining is recommended."""
        if self.last_health_check:
            return self.last_health_check.action == SelfCriticAction.RETRAIN
        return False
    
    def should_stop(self) -> bool:
        """Check if emergency stop is recommended."""
        if self.last_health_check:
            return self.last_health_check.action == SelfCriticAction.EMERGENCY_STOP
        return False
    
    def summary(self) -> str:
        """Generate summary."""
        if self.last_health_check:
            hc = self.last_health_check
            return (
                f"🪞 SelfCritic | Health={hc.overall_health:.0%} | "
                f"Action={hc.action.value} | {hc.reason}"
            )
        return "🪞 SelfCritic | No health check yet"


# =============================================================================
# Singleton
# =============================================================================

_self_critic: Optional[SelfCriticAI] = None


def get_self_critic() -> SelfCriticAI:
    """Get singleton SelfCriticAI."""
    global _self_critic
    if _self_critic is None:
        _self_critic = SelfCriticAI()
    return _self_critic


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Self-Critic AI Test")
    print("=" * 60)
    
    critic = SelfCriticAI()
    
    # Simulate trading history
    for i in range(50):
        profit = np.random.uniform(-10, 15)
        is_win = profit > 0
        predicted_correct = random.random() > 0.4
        
        critic.record_trade(profit, is_win, predicted_correct)
        critic.record_equity(1000 + i * 2 + np.random.uniform(-5, 10))
    
    # Health check
    health = critic.check_health()
    
    print(f"\n{critic.summary()}")
    print(f"\nHealth Check Details:")
    print(f"  Overall Health: {health.overall_health:.0%}")
    print(f"  Issues: {health.issues}")
    print(f"  Action: {health.action.value}")
    print(f"  Reason: {health.reason}")
    
    print(f"\n  Should Retrain: {critic.should_retrain()}")
    print(f"  Should Stop: {critic.should_stop()}")
    
    print("=" * 60)
