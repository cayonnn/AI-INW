# src/core/safety_checklist.py
"""
Safety Checklist - MANDATORY BEFORE LIVE
==========================================

🚨 Hard Rules (Fail = Stop Bot)

Execution:
- SL/TP บังคับทุก order
- RR ≥ 1.2
- Lot คำนวณจาก CAE เท่านั้น
- No default lot

Risk:
- Daily max loss lock
- Symbol exposure cap
- Kill-switch manual

Model:
- Model version logged
- Confidence threshold enforced
- Retrain rollback enabled

System:
- Sandbox / Live parity check
- Clock sync (UTC)
- MT error handling + retry limit
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("SAFETY")


@dataclass
class CheckResult:
    """Individual check result."""
    name: str
    passed: bool
    message: str
    category: str


class SafetyChecklist:
    """
    MANDATORY Safety Checklist before going live.
    
    ALL checks must pass before live trading is enabled.
    """

    def __init__(self):
        self.checks: List[CheckResult] = []
        self.last_check: str = ""

    # =========================================================
    # EXECUTION CHECKS
    # =========================================================
    
    def check_execution(self, config: Dict) -> List[CheckResult]:
        """Check execution safety rules."""
        results = []
        
        # SL/TP required
        results.append(CheckResult(
            name="SL/TP Required",
            passed=config.get("sl_tp_required", True),
            message="SL/TP is mandatory for all orders" if config.get("sl_tp_required", True) else "❌ SL/TP not enforced!",
            category="EXECUTION"
        ))
        
        # RR minimum
        min_rr = config.get("min_rr", 1.2)
        results.append(CheckResult(
            name="Min RR Check",
            passed=min_rr >= 1.2,
            message=f"RR minimum = {min_rr}" if min_rr >= 1.2 else f"❌ RR {min_rr} < 1.2!",
            category="EXECUTION"
        ))
        
        # CAE lot calculation
        results.append(CheckResult(
            name="CAE Lot Calculation",
            passed=config.get("use_cae", True),
            message="Lot calculated by CAE" if config.get("use_cae", True) else "❌ Default lot in use!",
            category="EXECUTION"
        ))
        
        # No default lot
        default_lot = config.get("default_lot", None)
        results.append(CheckResult(
            name="No Default Lot",
            passed=default_lot is None,
            message="No default lot" if default_lot is None else f"❌ Default lot = {default_lot}!",
            category="EXECUTION"
        ))
        
        return results

    # =========================================================
    # RISK CHECKS
    # =========================================================
    
    def check_risk(self, config: Dict) -> List[CheckResult]:
        """Check risk safety rules."""
        results = []
        
        # Daily max loss
        max_daily = config.get("max_daily_loss", 0.04)
        results.append(CheckResult(
            name="Daily Max Loss",
            passed=max_daily <= 0.05,
            message=f"Max daily loss = {max_daily:.1%}" if max_daily <= 0.05 else f"❌ Daily loss {max_daily:.1%} > 5%!",
            category="RISK"
        ))
        
        # Symbol exposure cap
        max_exposure = config.get("max_symbol_exposure", 0.02)
        results.append(CheckResult(
            name="Symbol Exposure Cap",
            passed=max_exposure <= 0.05,
            message=f"Max exposure = {max_exposure:.1%}" if max_exposure <= 0.05 else f"❌ Exposure {max_exposure:.1%} > 5%!",
            category="RISK"
        ))
        
        # Kill switch
        results.append(CheckResult(
            name="Kill Switch",
            passed=config.get("kill_switch_enabled", True),
            message="Kill switch enabled" if config.get("kill_switch_enabled", True) else "❌ No kill switch!",
            category="RISK"
        ))
        
        return results

    # =========================================================
    # MODEL CHECKS
    # =========================================================
    
    def check_model(self, config: Dict) -> List[CheckResult]:
        """Check model safety rules."""
        results = []
        
        # Model version logging
        results.append(CheckResult(
            name="Model Version Logging",
            passed=config.get("log_model_version", True),
            message="Model version logged" if config.get("log_model_version", True) else "❌ Version not logged!",
            category="MODEL"
        ))
        
        # Confidence threshold
        min_conf = config.get("min_confidence", 0.55)
        results.append(CheckResult(
            name="Confidence Threshold",
            passed=min_conf >= 0.5,
            message=f"Min confidence = {min_conf:.1%}" if min_conf >= 0.5 else f"❌ Confidence {min_conf:.1%} < 50%!",
            category="MODEL"
        ))
        
        # Rollback enabled
        results.append(CheckResult(
            name="Rollback Enabled",
            passed=config.get("rollback_enabled", True),
            message="Rollback enabled" if config.get("rollback_enabled", True) else "❌ No rollback!",
            category="MODEL"
        ))
        
        return results

    # =========================================================
    # SYSTEM CHECKS
    # =========================================================
    
    def check_system(self, config: Dict) -> List[CheckResult]:
        """Check system safety rules."""
        results = []
        
        # Parity check
        results.append(CheckResult(
            name="Sandbox/Live Parity",
            passed=config.get("parity_check_enabled", True),
            message="Parity check enabled" if config.get("parity_check_enabled", True) else "❌ No parity check!",
            category="SYSTEM"
        ))
        
        # MT error handling
        results.append(CheckResult(
            name="MT Error Handling",
            passed=config.get("mt_error_handling", True),
            message="MT errors handled" if config.get("mt_error_handling", True) else "❌ No error handling!",
            category="SYSTEM"
        ))
        
        # Retry limit
        max_retries = config.get("max_retries", 3)
        results.append(CheckResult(
            name="Retry Limit",
            passed=0 < max_retries <= 5,
            message=f"Max retries = {max_retries}" if 0 < max_retries <= 5 else f"❌ Retries = {max_retries}!",
            category="SYSTEM"
        ))
        
        return results

    # =========================================================
    # FULL CHECKLIST
    # =========================================================
    
    def run_all_checks(self, config: Dict = None) -> Tuple[bool, List[CheckResult]]:
        """
        Run all safety checks.
        
        Returns:
            (all_passed, list of results)
        """
        config = config or self._get_default_config()
        
        self.checks = []
        self.checks.extend(self.check_execution(config))
        self.checks.extend(self.check_risk(config))
        self.checks.extend(self.check_model(config))
        self.checks.extend(self.check_system(config))
        
        self.last_check = datetime.now().isoformat()
        
        all_passed = all(c.passed for c in self.checks)
        
        # Log results
        passed_count = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        
        if all_passed:
            logger.info(f"✅ SAFETY CHECK PASSED: {passed_count}/{total}")
        else:
            logger.error(f"❌ SAFETY CHECK FAILED: {passed_count}/{total}")
            for c in self.checks:
                if not c.passed:
                    logger.error(f"   ❌ {c.name}: {c.message}")
        
        return all_passed, self.checks

    def _get_default_config(self) -> Dict:
        """Get default safe configuration."""
        return {
            "sl_tp_required": True,
            "min_rr": 1.2,
            "use_cae": True,
            "default_lot": None,
            "max_daily_loss": 0.04,
            "max_symbol_exposure": 0.02,
            "kill_switch_enabled": True,
            "log_model_version": True,
            "min_confidence": 0.55,
            "rollback_enabled": True,
            "parity_check_enabled": True,
            "mt_error_handling": True,
            "max_retries": 3
        }

    def get_report(self) -> str:
        """Get formatted report."""
        if not self.checks:
            return "No checks run yet"
        
        lines = [
            "=" * 50,
            "🚨 SAFETY CHECKLIST REPORT",
            f"   Time: {self.last_check}",
            "=" * 50
        ]
        
        by_category = {}
        for c in self.checks:
            if c.category not in by_category:
                by_category[c.category] = []
            by_category[c.category].append(c)
        
        for cat, checks in by_category.items():
            lines.append(f"\n[{cat}]")
            for c in checks:
                status = "✅" if c.passed else "❌"
                lines.append(f"  {status} {c.name}: {c.message}")
        
        passed = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        lines.append(f"\n{'=' * 50}")
        lines.append(f"RESULT: {passed}/{total} PASSED")
        lines.append("=" * 50)
        
        return "\n".join(lines)
