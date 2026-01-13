# src/core/parity_checker.py
"""
Parity Checker - Fund-Grade Consistency Validator
==================================================

"Signal เดียวกัน → Decision เดียวกัน → Risk เดียวกัน"

ตรวจสอบว่า Live / Sandbox / Backtest ให้ผลลัพธ์เหมือนกัน
ถ้าต่าง = PARITY VIOLATION → block trade / alert

นี่คือ regulator-grade
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("PARITY_CHECKER")


@dataclass
class ParityResult:
    """Parity check result."""
    is_consistent: bool
    violations: Dict
    severity: str  # INFO, WARNING, CRITICAL


def compare_decisions(
    live: Dict,
    sandbox: Dict,
    backtest: Dict,
    tolerance: float = 0.001
) -> ParityResult:
    """
    Compare decisions across modes.
    
    Args:
        live: Decision from live mode
        sandbox: Decision from sandbox mode
        backtest: Decision from backtest mode
        tolerance: Numeric tolerance for float comparison
        
    Returns:
        ParityResult with consistency status and violations
    """
    keys = ["action", "direction", "sl", "tp", "lot"]
    violations = {}
    
    for key in keys:
        live_val = live.get(key)
        sandbox_val = sandbox.get(key)
        backtest_val = backtest.get(key)
        
        # Check for differences
        is_different = False
        
        if isinstance(live_val, (int, float)) and isinstance(sandbox_val, (int, float)):
            # Numeric comparison with tolerance
            if abs(live_val - sandbox_val) > tolerance:
                is_different = True
            elif backtest_val is not None and abs(live_val - backtest_val) > tolerance:
                is_different = True
        else:
            # String/other comparison
            if live_val != sandbox_val or (backtest_val is not None and live_val != backtest_val):
                is_different = True
        
        if is_different:
            violations[key] = {
                "live": live_val,
                "sandbox": sandbox_val,
                "backtest": backtest_val
            }
    
    # Determine severity
    if not violations:
        severity = "INFO"
    elif "action" in violations or "direction" in violations:
        severity = "CRITICAL"  # Action mismatch is critical
    elif "lot" in violations:
        severity = "WARNING"   # Lot mismatch is warning
    else:
        severity = "INFO"      # SL/TP minor variance
    
    return ParityResult(
        is_consistent=len(violations) == 0,
        violations=violations,
        severity=severity
    )


def check_parity(
    live: Dict,
    sandbox: Dict,
    backtest: Optional[Dict] = None,
    block_on_violation: bool = True
) -> bool:
    """
    Check parity and take action if needed.
    
    Args:
        live: Live mode decision
        sandbox: Sandbox mode decision
        backtest: Backtest mode decision (optional)
        block_on_violation: Whether to block trade on violation
        
    Returns:
        True if consistent, False if violation found
    """
    result = compare_decisions(
        live=live,
        sandbox=sandbox,
        backtest=backtest or {}
    )
    
    if result.is_consistent:
        logger.debug("✅ Parity check passed")
        return True
    
    # Log violation
    if result.severity == "CRITICAL":
        logger.error(f"🚨 PARITY VIOLATION (CRITICAL): {result.violations}")
    elif result.severity == "WARNING":
        logger.warning(f"⚠️ PARITY VIOLATION (WARNING): {result.violations}")
    else:
        logger.info(f"📊 PARITY VARIANCE: {result.violations}")
    
    if block_on_violation and result.severity in ["CRITICAL", "WARNING"]:
        logger.error("❌ Trade BLOCKED due to parity violation")
        return False
    
    return True


class ParityMonitor:
    """
    Continuous parity monitoring.
    
    Tracks violations over time and can trigger alerts.
    """
    
    def __init__(self, max_violations: int = 3):
        self.max_violations = max_violations
        self.violation_count = 0
        self.violation_history: List[Dict] = []
        self.is_disabled = False
    
    def check(self, live: Dict, sandbox: Dict, backtest: Optional[Dict] = None) -> bool:
        """
        Check parity with monitoring.
        
        Returns:
            True if trade should proceed
        """
        if self.is_disabled:
            logger.warning("🔴 Parity monitor is DISABLED due to repeated violations")
            return False
        
        result = compare_decisions(live, sandbox, backtest or {})
        
        if not result.is_consistent:
            self.violation_count += 1
            self.violation_history.append({
                "violations": result.violations,
                "severity": result.severity
            })
            
            if self.violation_count >= self.max_violations:
                self.is_disabled = True
                logger.critical(
                    f"🛑 PARITY MONITOR DISABLED: {self.violation_count} violations exceeded limit"
                )
                return False
        else:
            # Reset on success
            self.violation_count = max(0, self.violation_count - 1)
        
        return result.is_consistent or result.severity == "INFO"
    
    def reset(self):
        """Reset monitor state."""
        self.violation_count = 0
        self.violation_history = []
        self.is_disabled = False
        logger.info("✅ Parity monitor reset")
