"""
auto_disable.py
================
Multi-Layer Kill-Switch (Hedge Fund Grade)

ปิดระบบอัตโนมัติเมื่อความเสี่ยงหลุดกรอบ โดยไม่ต้องรอมนุษย์

5 Layers:
1. Capital Protection - Daily/Max Drawdown
2. Execution Integrity - Slippage/Rejects
3. Strategy Degradation - Win Rate/Expectancy
4. Market Regime Shock - Volatility/Liquidity
5. Model Confidence Collapse - Signal quality

Severity Levels:
- LOW: Reduce position size
- MEDIUM: Disable new entries
- HIGH: Close all positions
- CRITICAL: Kill system + Lock
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger("AUTO_DISABLE")


class DisableSeverity(str, Enum):
    """Kill-switch severity levels."""
    LOW = "LOW"           # Reduce position size
    MEDIUM = "MEDIUM"     # Disable new entries
    HIGH = "HIGH"         # Close all positions
    CRITICAL = "CRITICAL" # Kill system + Lock


class KillReason(str, Enum):
    """Trigger reasons for kill-switch."""
    # Capital Protection Layer
    DAILY_DRAWDOWN_LIMIT = "DAILY_DRAWDOWN_LIMIT"
    MAX_DRAWDOWN_LIMIT = "MAX_DRAWDOWN_LIMIT"
    EQUITY_CRASH = "EQUITY_CRASH"
    
    # Execution Integrity Layer
    SLIPPAGE_ANOMALY = "SLIPPAGE_ANOMALY"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    LATENCY_EXPLOSION = "LATENCY_EXPLOSION"
    
    # Strategy Degradation Layer
    NEGATIVE_EXPECTANCY = "NEGATIVE_EXPECTANCY"
    LOSS_STREAK = "LOSS_STREAK"
    WIN_RATE_COLLAPSE = "WIN_RATE_COLLAPSE"
    
    # Market Regime Shock Layer
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"
    CORRELATION_BREAKDOWN = "CORRELATION_BREAKDOWN"
    
    # Model Confidence Layer
    CONFIDENCE_COLLAPSE = "CONFIDENCE_COLLAPSE"
    MODEL_ERROR = "MODEL_ERROR"
    
    # Manual
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


class SystemState(str, Enum):
    """System state machine."""
    ACTIVE = "ACTIVE"
    WARNING = "WARNING"
    DISABLED = "DISABLED"
    COOLDOWN = "COOLDOWN"


@dataclass
class KillEvent:
    """Record of a kill-switch trigger."""
    timestamp: datetime
    reason: KillReason
    severity: DisableSeverity
    details: str
    metrics_snapshot: Dict = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Metrics for kill-switch evaluation."""
    # Capital
    daily_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    equity_change_1h: float = 0.0
    
    # Execution
    avg_slippage_pips: float = 0.0
    expected_slippage_pips: float = 1.0
    order_reject_rate: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Strategy
    rolling_expectancy: float = 0.0
    consecutive_losses: int = 0
    rolling_win_rate: float = 0.5
    
    # Market
    volatility_zscore: float = 0.0
    liquidity_score: float = 1.0
    correlation_score: float = 1.0
    
    # Model
    avg_signal_confidence: float = 0.5
    model_error_count: int = 0
    
    def is_stable(self) -> bool:
        """Check if metrics indicate stable conditions."""
        return (
            self.daily_drawdown_pct > -0.02 and
            self.volatility_zscore < 2.0 and
            self.rolling_expectancy >= 0 and
            self.avg_signal_confidence >= 0.4
        )


class AutoDisableGuard:
    """
    Multi-Layer Kill-Switch for Hedge Fund Grade Protection.
    
    State Machine:
    ACTIVE → WARNING → DISABLED → COOLDOWN → ACTIVE
    """

    def __init__(self):
        self.state = SystemState.ACTIVE
        self.triggered_reason: Optional[KillReason] = None
        self.current_severity: Optional[DisableSeverity] = None
        self.cooldown_until: Optional[datetime] = None
        self.events: List[KillEvent] = []
        self.warning_count = 0
        
        # Thresholds (can be config-driven)
        self.thresholds = {
            # Capital Protection
            "daily_dd_limit": -0.03,      # -3%
            "max_dd_limit": -0.10,        # -10%
            "equity_crash_1h": -0.02,     # -2% in 1 hour
            
            # Execution Integrity
            "slippage_multiplier": 3.0,
            "reject_rate_limit": 0.15,    # 15%
            "latency_limit_ms": 500,
            
            # Strategy Degradation
            "dynamic_loss_limit": 5,
            "min_win_rate": 0.35,
            
            # Market Regime
            "volatility_zscore_limit": 4.0,
            "min_liquidity": 0.3,
            
            # Model Confidence
            "min_confidence": 0.25,
            "max_model_errors": 5,
        }
        
        # Cooldown durations by severity
        self.cooldown_hours = {
            DisableSeverity.LOW: 1,
            DisableSeverity.MEDIUM: 4,
            DisableSeverity.HIGH: 12,
            DisableSeverity.CRITICAL: 24,
        }

    # -------------------------------------------------
    # Main evaluation
    # -------------------------------------------------
    def evaluate(self, metrics: SystemMetrics):
        """
        Evaluate all layers and trigger kill if needed.
        """
        if self.state == SystemState.DISABLED:
            self._check_reactivation(metrics)
            return
        
        if self.state == SystemState.COOLDOWN:
            if datetime.now() >= self.cooldown_until:
                self._attempt_reactivation(metrics)
            return
        
        # Check all layers
        triggered = self._check_capital_layer(metrics)
        if not triggered:
            triggered = self._check_execution_layer(metrics)
        if not triggered:
            triggered = self._check_strategy_layer(metrics)
        if not triggered:
            triggered = self._check_market_layer(metrics)
        if not triggered:
            triggered = self._check_model_layer(metrics)

    def _check_capital_layer(self, m: SystemMetrics) -> bool:
        """Layer 1: Capital Protection."""
        if m.daily_drawdown_pct <= self.thresholds["daily_dd_limit"]:
            self._trigger(KillReason.DAILY_DRAWDOWN_LIMIT, DisableSeverity.HIGH,
                         f"Daily DD {m.daily_drawdown_pct:.2%}")
            return True
        
        if m.max_drawdown_pct <= self.thresholds["max_dd_limit"]:
            self._trigger(KillReason.MAX_DRAWDOWN_LIMIT, DisableSeverity.CRITICAL,
                         f"Max DD {m.max_drawdown_pct:.2%}")
            return True
        
        if m.equity_change_1h <= self.thresholds["equity_crash_1h"]:
            self._trigger(KillReason.EQUITY_CRASH, DisableSeverity.HIGH,
                         f"Equity crash {m.equity_change_1h:.2%} in 1h")
            return True
        
        return False

    def _check_execution_layer(self, m: SystemMetrics) -> bool:
        """Layer 2: Execution Integrity."""
        if m.avg_slippage_pips > m.expected_slippage_pips * self.thresholds["slippage_multiplier"]:
            self._trigger(KillReason.SLIPPAGE_ANOMALY, DisableSeverity.MEDIUM,
                         f"Slippage {m.avg_slippage_pips:.1f} pips")
            return True
        
        if m.order_reject_rate > self.thresholds["reject_rate_limit"]:
            self._trigger(KillReason.EXECUTION_FAILURE, DisableSeverity.HIGH,
                         f"Reject rate {m.order_reject_rate:.2%}")
            return True
        
        if m.avg_latency_ms > self.thresholds["latency_limit_ms"]:
            self._trigger(KillReason.LATENCY_EXPLOSION, DisableSeverity.MEDIUM,
                         f"Latency {m.avg_latency_ms:.0f}ms")
            return True
        
        return False

    def _check_strategy_layer(self, m: SystemMetrics) -> bool:
        """Layer 3: Strategy Degradation."""
        if m.rolling_expectancy < 0:
            self._trigger(KillReason.NEGATIVE_EXPECTANCY, DisableSeverity.HIGH,
                         f"Expectancy {m.rolling_expectancy:.2f}")
            return True
        
        if m.consecutive_losses >= self.thresholds["dynamic_loss_limit"]:
            self._trigger(KillReason.LOSS_STREAK, DisableSeverity.MEDIUM,
                         f"Consecutive losses: {m.consecutive_losses}")
            return True
        
        if m.rolling_win_rate < self.thresholds["min_win_rate"]:
            self._trigger(KillReason.WIN_RATE_COLLAPSE, DisableSeverity.MEDIUM,
                         f"Win rate {m.rolling_win_rate:.2%}")
            return True
        
        return False

    def _check_market_layer(self, m: SystemMetrics) -> bool:
        """Layer 4: Market Regime Shock."""
        if m.volatility_zscore > self.thresholds["volatility_zscore_limit"]:
            self._trigger(KillReason.VOLATILITY_SHOCK, DisableSeverity.HIGH,
                         f"Volatility z-score {m.volatility_zscore:.1f}")
            return True
        
        if m.liquidity_score < self.thresholds["min_liquidity"]:
            self._trigger(KillReason.LIQUIDITY_CRISIS, DisableSeverity.HIGH,
                         f"Liquidity score {m.liquidity_score:.2f}")
            return True
        
        return False

    def _check_model_layer(self, m: SystemMetrics) -> bool:
        """Layer 5: Model Confidence Collapse."""
        if m.avg_signal_confidence < self.thresholds["min_confidence"]:
            self._trigger(KillReason.CONFIDENCE_COLLAPSE, DisableSeverity.MEDIUM,
                         f"Confidence {m.avg_signal_confidence:.2%}")
            return True
        
        if m.model_error_count >= self.thresholds["max_model_errors"]:
            self._trigger(KillReason.MODEL_ERROR, DisableSeverity.HIGH,
                         f"Model errors: {m.model_error_count}")
            return True
        
        return False

    # -------------------------------------------------
    # Kill-switch execution
    # -------------------------------------------------
    def _trigger(self, reason: KillReason, severity: DisableSeverity, details: str):
        """Execute kill-switch."""
        self.state = SystemState.DISABLED
        self.triggered_reason = reason
        self.current_severity = severity
        self.cooldown_until = datetime.now() + timedelta(hours=self.cooldown_hours[severity])
        
        event = KillEvent(
            timestamp=datetime.now(),
            reason=reason,
            severity=severity,
            details=details,
        )
        self.events.append(event)
        
        logger.critical(f"🛑 KILL SWITCH TRIGGERED: {reason.value}")
        logger.critical(f"   Severity: {severity.value}")
        logger.critical(f"   Details: {details}")
        logger.critical(f"   Cooldown until: {self.cooldown_until}")
        
        # Execute severity-based actions
        if severity == DisableSeverity.CRITICAL:
            logger.critical("   Action: FULL SYSTEM LOCK")
        elif severity == DisableSeverity.HIGH:
            logger.critical("   Action: CLOSE ALL POSITIONS")
        elif severity == DisableSeverity.MEDIUM:
            logger.critical("   Action: DISABLE NEW ENTRIES")
        else:
            logger.warning("   Action: REDUCE POSITION SIZE")

    # -------------------------------------------------
    # Reactivation
    # -------------------------------------------------
    def _check_reactivation(self, metrics: SystemMetrics):
        """Check if cooldown period has passed."""
        if self.cooldown_until and datetime.now() >= self.cooldown_until:
            self.state = SystemState.COOLDOWN
            self._attempt_reactivation(metrics)

    def _attempt_reactivation(self, metrics: SystemMetrics) -> bool:
        """Attempt to reactivate system after cooldown."""
        if not metrics.is_stable():
            logger.info("Reactivation blocked: conditions not stable")
            return False
        
        self.state = SystemState.ACTIVE
        self.triggered_reason = None
        self.current_severity = None
        self.warning_count = 0
        
        logger.info("✅ SYSTEM REACTIVATED")
        return True

    def force_reactivate(self) -> bool:
        """Manual force reactivation (requires acknowledgment)."""
        logger.warning("⚠️ FORCE REACTIVATION - Manual override")
        self.state = SystemState.ACTIVE
        self.triggered_reason = None
        self.current_severity = None
        return True

    # -------------------------------------------------
    # Public interface
    # -------------------------------------------------
    def can_trade(self) -> tuple:
        """Check if trading is allowed."""
        if self.state == SystemState.ACTIVE:
            return True, "Trading enabled"
        
        reason = self.triggered_reason.value if self.triggered_reason else "Unknown"
        return False, f"Trading disabled: {reason}"

    def is_disabled(self) -> bool:
        """Check if system is disabled."""
        return self.state != SystemState.ACTIVE

    def get_severity(self) -> Optional[DisableSeverity]:
        """Get current severity level."""
        return self.current_severity

    def get_status(self) -> Dict:
        """Get current status for monitoring."""
        return {
            "state": self.state.value,
            "triggered_reason": self.triggered_reason.value if self.triggered_reason else None,
            "severity": self.current_severity.value if self.current_severity else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "total_events": len(self.events),
        }

    def record_trade_result(self, is_win: bool, metrics: SystemMetrics):
        """Update metrics after trade."""
        if is_win:
            metrics.consecutive_losses = 0
        else:
            metrics.consecutive_losses += 1
        self.evaluate(metrics)

    def record_model_error(self, metrics: SystemMetrics):
        """Record model prediction error."""
        metrics.model_error_count += 1
        self.evaluate(metrics)


# Alias for backward compatibility
AutoDisableManager = AutoDisableGuard
