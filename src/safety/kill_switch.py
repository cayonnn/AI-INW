# src/safety/kill_switch.py
"""
Kill Switch - Emergency Trading Halt
=====================================

Monitors critical metrics and halts trading when:
- Daily DD exceeds threshold
- Consecutive errors occur
- System instability detected

Auto-closes all positions and notifies dashboard.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List
from datetime import datetime
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("KILL_SWITCH")


class KillReason(Enum):
    """Reasons for triggering kill switch."""
    DD_LIMIT = "Daily drawdown limit exceeded"
    TOTAL_DD = "Total drawdown limit exceeded"
    CONSECUTIVE_ERRORS = "Too many consecutive errors"
    SYSTEM_ERROR = "Critical system error"
    MANUAL = "Manual intervention"
    MODEL_FAILURE = "Model produced invalid output"
    MT5_DISCONNECT = "MT5 connection lost"


@dataclass
class SafetyMetrics:
    """Current safety metrics."""
    dd_today: float
    dd_total: float
    consecutive_errors: int
    last_error: Optional[str]
    open_positions: int
    mt5_connected: bool
    model_healthy: bool


class KillSwitch:
    """
    Emergency Kill Switch.
    
    Monitors:
    - Daily drawdown
    - Total drawdown
    - Consecutive errors
    - System health
    
    Actions:
    - Close all positions
    - Disable trading
    - Notify dashboard
    - Log event
    """
    
    # Default thresholds
    DD_DAILY_LIMIT = 3.0       # 3% daily DD
    DD_TOTAL_LIMIT = 8.0       # 8% total DD
    MAX_CONSECUTIVE_ERRORS = 3
    
    def __init__(
        self,
        dd_daily_limit: float = DD_DAILY_LIMIT,
        dd_total_limit: float = DD_TOTAL_LIMIT,
        max_errors: int = MAX_CONSECUTIVE_ERRORS,
        log_dir: str = "logs/safety"
    ):
        """
        Initialize kill switch.
        
        Args:
            dd_daily_limit: Daily DD threshold
            dd_total_limit: Total DD threshold
            max_errors: Max consecutive errors
            log_dir: Directory for safety logs
        """
        self.dd_daily_limit = dd_daily_limit
        self.dd_total_limit = dd_total_limit
        self.max_errors = max_errors
        self.log_dir = log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.armed = True
        self.triggered = False
        self.trigger_reason: Optional[KillReason] = None
        self.trigger_time: Optional[datetime] = None
        
        # Callbacks
        self._on_trigger_callbacks: List[Callable] = []
        
        logger.info(
            f"KillSwitch initialized: DD={dd_daily_limit}%, "
            f"errors={max_errors}"
        )
    
    def check(self, metrics: SafetyMetrics) -> Optional[KillReason]:
        """
        Check if kill switch should trigger.
        
        Returns:
            KillReason if triggered, else None
        """
        if not self.armed:
            return None
        
        # Daily DD check
        if metrics.dd_today > self.dd_daily_limit:
            return self._trigger(KillReason.DD_LIMIT, f"DD={metrics.dd_today:.1f}%")
        
        # Total DD check
        if metrics.dd_total > self.dd_total_limit:
            return self._trigger(KillReason.TOTAL_DD, f"DD={metrics.dd_total:.1f}%")
        
        # Consecutive errors
        if metrics.consecutive_errors >= self.max_errors:
            return self._trigger(KillReason.CONSECUTIVE_ERRORS, metrics.last_error)
        
        # MT5 disconnect
        if not metrics.mt5_connected:
            return self._trigger(KillReason.MT5_DISCONNECT)
        
        # Model failure
        if not metrics.model_healthy:
            return self._trigger(KillReason.MODEL_FAILURE)
        
        return None
    
    def _trigger(
        self,
        reason: KillReason,
        details: str = None
    ) -> KillReason:
        """Trigger the kill switch."""
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_time = datetime.now()
        
        logger.critical(
            f"🚨 KILL SWITCH TRIGGERED: {reason.value}"
            f"{f' - {details}' if details else ''}"
        )
        
        # Log event
        self._log_trigger(reason, details)
        
        # Execute callbacks
        for callback in self._on_trigger_callbacks:
            try:
                callback(reason, details)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return reason
    
    def trigger_manual(self, details: str = None) -> KillReason:
        """Manually trigger kill switch."""
        return self._trigger(KillReason.MANUAL, details)
    
    def reset(self) -> None:
        """Reset kill switch after review."""
        self.triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        logger.warning("Kill switch RESET")
    
    def arm(self) -> None:
        """Arm the kill switch."""
        self.armed = True
        logger.info("Kill switch ARMED")
    
    def disarm(self) -> None:
        """Disarm the kill switch (dangerous!)."""
        self.armed = False
        logger.warning("Kill switch DISARMED (DANGEROUS)")
    
    def on_trigger(self, callback: Callable) -> None:
        """Register callback for trigger event."""
        self._on_trigger_callbacks.append(callback)
    
    def get_status(self) -> Dict:
        """Get current kill switch status."""
        return {
            "armed": self.armed,
            "triggered": self.triggered,
            "reason": self.trigger_reason.value if self.trigger_reason else None,
            "trigger_time": self.trigger_time.isoformat() if self.trigger_time else None,
            "thresholds": {
                "dd_daily": self.dd_daily_limit,
                "dd_total": self.dd_total_limit,
                "max_errors": self.max_errors,
            }
        }
    
    def _log_trigger(self, reason: KillReason, details: str = None) -> None:
        """Log trigger event to file."""
        filename = f"kill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason.value,
            "details": details,
        }
        
        with open(filepath, 'w') as f:
            json.dump(event, f, indent=2)


def close_all_positions() -> int:
    """
    Close all open positions.
    
    Returns number of positions closed.
    """
    logger.warning("Closing all positions...")
    
    try:
        # Import MT5 interface
        from src.execution.mt5_executor import get_mt5_executor
        executor = get_mt5_executor()
        closed = executor.close_all()
        logger.info(f"Closed {closed} positions")
        return closed
    except Exception as e:
        logger.error(f"Failed to close positions: {e}")
        return 0


def disable_trading() -> None:
    """Disable all trading activities."""
    logger.warning("Trading DISABLED")
    # Set global flag
    os.environ["TRADING_DISABLED"] = "1"


def is_trading_disabled() -> bool:
    """Check if trading is disabled."""
    return os.environ.get("TRADING_DISABLED") == "1"


# Singleton
_kill_switch: Optional[KillSwitch] = None


def get_kill_switch() -> KillSwitch:
    """Get singleton kill switch."""
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = KillSwitch()
    return _kill_switch
