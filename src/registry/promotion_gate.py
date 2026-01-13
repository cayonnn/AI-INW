# src/registry/promotion_gate.py
"""
Promotion Gate - Safe Model Deployment
=======================================

❌ ห้าม replace live model ตรง ๆ
✅ ต้องผ่าน Promotion Gate

Rules:
- accuracy >= previous
- drawdown <= previous
- winrate >= previous (optional)
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("PROMOTION_GATE")


@dataclass
class PromotionResult:
    """Promotion gate result."""
    allowed: bool
    reason: str
    checks: Dict[str, bool]


class PromotionGate:
    """
    Safe model promotion with validation.
    
    A new model can only replace the live model if:
    - Accuracy >= previous
    - Drawdown <= previous
    - (Optional) Sharpe >= threshold
    """

    # Default thresholds
    THRESHOLDS = {
        "accuracy_min_delta": 0.0,       # Must be >= previous
        "drawdown_max_delta": 0.0,       # Must be <= previous
        "sharpe_min": 1.0,               # Absolute minimum
        "winrate_min_delta": -0.02       # Allow 2% drop max
    }

    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = {**self.THRESHOLDS, **(thresholds or {})}

    def check(
        self,
        new_metrics: Dict,
        live_metrics: Optional[Dict] = None
    ) -> PromotionResult:
        """
        Check if new model can be promoted.
        
        Args:
            new_metrics: Metrics from new model
            live_metrics: Metrics from current live model
            
        Returns:
            PromotionResult with allowed status
        """
        checks = {}
        failures = []
        
        # If no live metrics, allow promotion (first model)
        if live_metrics is None:
            logger.info("No live model - promotion allowed")
            return PromotionResult(
                allowed=True,
                reason="First model - auto-approved",
                checks={"first_model": True}
            )
        
        # Check accuracy
        new_acc = new_metrics.get("accuracy", 0)
        live_acc = live_metrics.get("accuracy", 0)
        acc_ok = new_acc >= live_acc + self.thresholds["accuracy_min_delta"]
        checks["accuracy"] = acc_ok
        if not acc_ok:
            failures.append(f"accuracy {new_acc:.2%} < {live_acc:.2%}")
        
        # Check drawdown
        new_dd = new_metrics.get("drawdown", 0)
        live_dd = live_metrics.get("drawdown", 1.0)
        dd_ok = new_dd <= live_dd + self.thresholds["drawdown_max_delta"]
        checks["drawdown"] = dd_ok
        if not dd_ok:
            failures.append(f"drawdown {new_dd:.2%} > {live_dd:.2%}")
        
        # Check winrate (optional - allow slight drop)
        new_wr = new_metrics.get("winrate", 0)
        live_wr = live_metrics.get("winrate", 0)
        wr_ok = new_wr >= live_wr + self.thresholds["winrate_min_delta"]
        checks["winrate"] = wr_ok
        if not wr_ok:
            failures.append(f"winrate {new_wr:.2%} < {live_wr:.2%} - {abs(self.thresholds['winrate_min_delta']):.2%}")
        
        # Check Sharpe (absolute threshold)
        new_sharpe = new_metrics.get("sharpe", 0)
        sharpe_ok = new_sharpe >= self.thresholds["sharpe_min"]
        checks["sharpe"] = sharpe_ok
        if not sharpe_ok:
            failures.append(f"sharpe {new_sharpe:.2f} < {self.thresholds['sharpe_min']}")
        
        # Overall result
        allowed = len(failures) == 0
        reason = "All checks passed" if allowed else f"Failed: {', '.join(failures)}"
        
        if allowed:
            logger.info(f"✅ Promotion ALLOWED: {reason}")
        else:
            logger.warning(f"❌ Promotion BLOCKED: {reason}")
        
        return PromotionResult(allowed=allowed, reason=reason, checks=checks)


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def allow_promotion(new_metrics: Dict, live_metrics: Optional[Dict]) -> bool:
    """
    Simple check if promotion is allowed.
    
    Usage:
        if allow_promotion(new_metrics, live_metrics):
            registry.promote("xgb_signal", "v002")
    """
    gate = PromotionGate()
    result = gate.check(new_metrics, live_metrics)
    return result.allowed
