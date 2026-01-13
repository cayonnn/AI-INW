"""
capital_protection.py
======================
Capital Protection First Architecture

"อยู่รอดก่อน กำไรทีหลัง"

Core Principle:
- Capital > Alpha
- Drawdown = ศัตรู
- Survive → compound

Protection Layers:
1. Trade Level - Vol-adaptive SL, risk cap
2. Strategy Level - Max DD, auto freeze
3. Portfolio Level - DD trigger, correlation cap
4. System Level - Kill switch, crisis mode
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger("CAPITAL_PROTECTION")


class ProtectionLevel(str, Enum):
    TRADE = "TRADE"
    STRATEGY = "STRATEGY"
    PORTFOLIO = "PORTFOLIO"
    SYSTEM = "SYSTEM"


class ProtectionAction(str, Enum):
    NONE = "NONE"
    REDUCE_SIZE = "REDUCE_SIZE"
    TIGHTEN_SL = "TIGHTEN_SL"
    FREEZE_STRATEGY = "FREEZE_STRATEGY"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    FORCE_RISK_OFF = "FORCE_RISK_OFF"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class ProtectionTrigger:
    level: ProtectionLevel
    action: ProtectionAction
    reason: str
    severity: float  # 0-1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProtectionState:
    is_risk_off: bool = False
    is_emergency: bool = False
    active_triggers: List[ProtectionTrigger] = field(default_factory=list)
    size_multiplier: float = 1.0
    sl_multiplier: float = 1.0
    frozen_strategies: List[str] = field(default_factory=list)


class CapitalProtectionSystem:
    """Capital Protection First Architecture."""

    def __init__(self):
        self.state = ProtectionState()
        self.trigger_history: List[ProtectionTrigger] = []
        
        # Hard limits (non-negotiable)
        self.limits = {
            # Trade level
            "max_risk_per_trade": 0.01,
            "max_sl_multiplier": 2.0,
            
            # Strategy level
            "max_strategy_dd": -0.05,
            "max_consecutive_losses": 5,
            
            # Portfolio level
            "max_portfolio_dd": -0.15,
            "max_correlation": 0.7,
            "max_exposure_pct": 0.8,
            
            # System level
            "emergency_dd": -0.20,
            "min_liquidity": 0.3,
        }
        
        # Recovery requirements
        self.recovery = {
            "min_stable_days": 5,
            "min_confidence": 0.6,
            "ramp_up_rate": 0.1,  # 10% per day
        }

    # -------------------------------------------------
    # Main check
    # -------------------------------------------------
    def check(self, metrics: Dict) -> ProtectionState:
        """Check all protection layers."""
        triggers = []
        
        # Layer 1: Trade level
        trade_trigger = self._check_trade_level(metrics)
        if trade_trigger:
            triggers.append(trade_trigger)
        
        # Layer 2: Strategy level
        strategy_triggers = self._check_strategy_level(metrics)
        triggers.extend(strategy_triggers)
        
        # Layer 3: Portfolio level
        portfolio_trigger = self._check_portfolio_level(metrics)
        if portfolio_trigger:
            triggers.append(portfolio_trigger)
        
        # Layer 4: System level
        system_trigger = self._check_system_level(metrics)
        if system_trigger:
            triggers.append(system_trigger)
        
        # Update state
        self._update_state(triggers)
        
        # Log significant triggers
        for t in triggers:
            if t.severity > 0.5:
                logger.warning(f"Protection: {t.level.value} - {t.action.value}")
        
        return self.state

    def _check_trade_level(self, metrics: Dict) -> Optional[ProtectionTrigger]:
        """Check trade-level protection."""
        volatility = metrics.get("current_volatility", 0.01)
        avg_vol = metrics.get("average_volatility", 0.01)
        
        # Vol-adaptive SL
        if volatility > avg_vol * 2:
            return ProtectionTrigger(
                level=ProtectionLevel.TRADE,
                action=ProtectionAction.TIGHTEN_SL,
                reason=f"Vol spike: {volatility/avg_vol:.1f}x normal",
                severity=0.5,
            )
        return None

    def _check_strategy_level(self, metrics: Dict) -> List[ProtectionTrigger]:
        """Check strategy-level protection."""
        triggers = []
        
        strategy_dds = metrics.get("strategy_drawdowns", {})
        for strategy, dd in strategy_dds.items():
            if dd < self.limits["max_strategy_dd"]:
                triggers.append(ProtectionTrigger(
                    level=ProtectionLevel.STRATEGY,
                    action=ProtectionAction.FREEZE_STRATEGY,
                    reason=f"{strategy} DD={dd:.1%}",
                    severity=0.7,
                ))
                if strategy not in self.state.frozen_strategies:
                    self.state.frozen_strategies.append(strategy)
        
        return triggers

    def _check_portfolio_level(self, metrics: Dict) -> Optional[ProtectionTrigger]:
        """Check portfolio-level protection."""
        portfolio_dd = metrics.get("portfolio_drawdown", 0)
        correlation = metrics.get("portfolio_correlation", 0)
        
        if portfolio_dd < self.limits["max_portfolio_dd"]:
            return ProtectionTrigger(
                level=ProtectionLevel.PORTFOLIO,
                action=ProtectionAction.FORCE_RISK_OFF,
                reason=f"Portfolio DD={portfolio_dd:.1%}",
                severity=0.9,
            )
        
        if correlation > self.limits["max_correlation"]:
            return ProtectionTrigger(
                level=ProtectionLevel.PORTFOLIO,
                action=ProtectionAction.REDUCE_EXPOSURE,
                reason=f"Correlation={correlation:.2f}",
                severity=0.6,
            )
        
        return None

    def _check_system_level(self, metrics: Dict) -> Optional[ProtectionTrigger]:
        """Check system-level protection."""
        portfolio_dd = metrics.get("portfolio_drawdown", 0)
        liquidity = metrics.get("liquidity_score", 1.0)
        
        if portfolio_dd < self.limits["emergency_dd"]:
            return ProtectionTrigger(
                level=ProtectionLevel.SYSTEM,
                action=ProtectionAction.EMERGENCY_STOP,
                reason=f"Emergency DD={portfolio_dd:.1%}",
                severity=1.0,
            )
        
        if liquidity < self.limits["min_liquidity"]:
            return ProtectionTrigger(
                level=ProtectionLevel.SYSTEM,
                action=ProtectionAction.REDUCE_EXPOSURE,
                reason=f"Low liquidity={liquidity:.2f}",
                severity=0.8,
            )
        
        return None

    def _update_state(self, triggers: List[ProtectionTrigger]):
        """Update protection state based on triggers."""
        self.state.active_triggers = triggers
        self.trigger_history.extend(triggers)
        
        # Calculate multipliers
        max_severity = max((t.severity for t in triggers), default=0)
        
        if max_severity > 0.9:
            self.state.is_emergency = True
            self.state.size_multiplier = 0.0
        elif max_severity > 0.7:
            self.state.is_risk_off = True
            self.state.size_multiplier = 0.3
        elif max_severity > 0.5:
            self.state.size_multiplier = 0.5
        else:
            self.state.size_multiplier = min(1.0, self.state.size_multiplier + 0.1)

    # -------------------------------------------------
    # Recovery
    # -------------------------------------------------
    def attempt_recovery(self, metrics: Dict) -> bool:
        """Attempt to recover from risk-off state."""
        if not (self.state.is_risk_off or self.state.is_emergency):
            return True
        
        stable_days = metrics.get("drawdown_stable_days", 0)
        confidence = metrics.get("avg_confidence", 0)
        
        if (stable_days >= self.recovery["min_stable_days"] and
            confidence >= self.recovery["min_confidence"]):
            
            self.state.is_emergency = False
            self.state.is_risk_off = False
            self.state.size_multiplier = 0.3  # Start low
            
            logger.info("Recovery initiated - gradual ramp-up")
            return True
        
        return False

    def get_position_multiplier(self) -> float:
        """Get position size multiplier."""
        return self.state.size_multiplier

    def can_trade(self, strategy: str = None) -> tuple:
        """Check if trading is allowed."""
        if self.state.is_emergency:
            return False, "Emergency stop active"
        
        if strategy and strategy in self.state.frozen_strategies:
            return False, f"Strategy {strategy} frozen"
        
        if self.state.is_risk_off:
            return False, "Risk-off mode"
        
        return True, "OK"

    def get_status(self) -> Dict:
        """Get protection status."""
        return {
            "is_risk_off": self.state.is_risk_off,
            "is_emergency": self.state.is_emergency,
            "size_multiplier": self.state.size_multiplier,
            "frozen_strategies": self.state.frozen_strategies,
            "active_triggers": len(self.state.active_triggers),
        }
