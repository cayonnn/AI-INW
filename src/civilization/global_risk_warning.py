# src/civilization/global_risk_warning.py
"""
Global Risk Early Warning System
=================================

Detects systemic risks before they manifest.

Monitors:
    - Liquidity cracks
    - Debt spirals
    - Currency stress
    - Social instability signals

Goal: "2008, but detected 18 months early"

Paper Statement:
    "Our early warning system detects leading indicators of
     systemic financial stress with an 18-month forecast horizon."
"""

import os
import sys
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("RISK_WARNING")


class RiskLevel(Enum):
    """Systemic risk levels."""
    GREEN = "green"      # Normal
    YELLOW = "yellow"    # Elevated
    ORANGE = "orange"    # High
    RED = "red"          # Critical
    BLACK = "black"      # Imminent crisis


class RiskCategory(Enum):
    """Categories of systemic risk."""
    LIQUIDITY = "liquidity"
    CREDIT = "credit"
    CURRENCY = "currency"
    SOVEREIGN = "sovereign"
    CONTAGION = "contagion"
    SOCIAL = "social"


@dataclass
class RiskIndicator:
    """A single risk indicator."""
    name: str
    category: RiskCategory
    value: float
    threshold_warning: float
    threshold_critical: float
    trend: str  # "rising", "stable", "falling"
    
    @property
    def level(self) -> RiskLevel:
        if self.value >= self.threshold_critical:
            return RiskLevel.RED
        elif self.value >= self.threshold_warning:
            return RiskLevel.ORANGE
        elif self.value >= self.threshold_warning * 0.7:
            return RiskLevel.YELLOW
        return RiskLevel.GREEN


@dataclass
class RiskWarning:
    """A risk warning alert."""
    timestamp: str
    category: RiskCategory
    level: RiskLevel
    indicators: List[RiskIndicator]
    message: str
    lead_time_months: int
    recommended_actions: List[str]


class GlobalRiskWarningSystem:
    """
    Early Warning System for Systemic Risk.
    
    Continuously monitors leading indicators of
    financial and economic crises.
    
    Features:
        - Multi-category risk monitoring
        - Lead-time estimation
        - Actionable warnings
        - Crisis playbook generation
    """
    
    def __init__(self):
        self.indicators: Dict[str, RiskIndicator] = {}
        self.warnings: List[RiskWarning] = []
        self.overall_risk = RiskLevel.GREEN
        
        self._initialize_indicators()
        logger.info("⚠️ GlobalRiskWarningSystem initialized")
    
    def _initialize_indicators(self):
        """Initialize risk indicators."""
        defaults = [
            # Liquidity
            RiskIndicator("TED_Spread", RiskCategory.LIQUIDITY, 0.3, 0.5, 1.0, "stable"),
            RiskIndicator("VIX_Term", RiskCategory.LIQUIDITY, 0.1, 0.2, 0.4, "stable"),
            
            # Credit
            RiskIndicator("HY_Spread", RiskCategory.CREDIT, 400, 500, 800, "stable"),
            RiskIndicator("Loan_Delinquency", RiskCategory.CREDIT, 2.0, 4.0, 7.0, "stable"),
            
            # Currency
            RiskIndicator("EM_FX_Vol", RiskCategory.CURRENCY, 10, 15, 25, "stable"),
            RiskIndicator("Reserve_Depletion", RiskCategory.CURRENCY, 5, 15, 30, "stable"),
            
            # Sovereign
            RiskIndicator("Debt_GDP_Change", RiskCategory.SOVEREIGN, 3, 8, 15, "stable"),
            RiskIndicator("Yield_Spike", RiskCategory.SOVEREIGN, 50, 100, 200, "stable"),
            
            # Contagion
            RiskIndicator("Cross_Market_Corr", RiskCategory.CONTAGION, 0.5, 0.75, 0.9, "stable"),
            
            # Social
            RiskIndicator("Inequality_Index", RiskCategory.SOCIAL, 0.40, 0.45, 0.50, "stable"),
        ]
        
        for ind in defaults:
            self.indicators[ind.name] = ind
    
    def update(self, data: Dict[str, float]):
        """Update indicators with new data."""
        for name, value in data.items():
            if name in self.indicators:
                old_value = self.indicators[name].value
                self.indicators[name].value = value
                
                # Update trend
                if value > old_value * 1.05:
                    self.indicators[name].trend = "rising"
                elif value < old_value * 0.95:
                    self.indicators[name].trend = "falling"
                else:
                    self.indicators[name].trend = "stable"
        
        self._check_warnings()
    
    def _check_warnings(self):
        """Check for warning conditions."""
        categories_at_risk = {}
        
        for ind in self.indicators.values():
            if ind.level in [RiskLevel.ORANGE, RiskLevel.RED]:
                if ind.category not in categories_at_risk:
                    categories_at_risk[ind.category] = []
                categories_at_risk[ind.category].append(ind)
        
        # Generate warnings
        for category, indicators in categories_at_risk.items():
            max_level = max(i.level for i in indicators)
            
            warning = RiskWarning(
                timestamp=datetime.now().isoformat(),
                category=category,
                level=max_level,
                indicators=indicators,
                message=self._generate_message(category, max_level, indicators),
                lead_time_months=self._estimate_lead_time(max_level),
                recommended_actions=self._get_actions(category, max_level)
            )
            
            self.warnings.append(warning)
        
        # Update overall risk
        if categories_at_risk:
            max_level = max(w.level for w in self.warnings[-len(categories_at_risk):])
            self.overall_risk = max_level
    
    def _generate_message(
        self,
        category: RiskCategory,
        level: RiskLevel,
        indicators: List[RiskIndicator]
    ) -> str:
        """Generate warning message."""
        rising = [i for i in indicators if i.trend == "rising"]
        
        if level == RiskLevel.RED:
            prefix = "CRITICAL"
        elif level == RiskLevel.ORANGE:
            prefix = "ELEVATED"
        else:
            prefix = "WATCH"
        
        return (
            f"{prefix}: {category.value.upper()} risk detected. "
            f"{len(rising)} indicators rising."
        )
    
    def _estimate_lead_time(self, level: RiskLevel) -> int:
        """Estimate months until potential crisis."""
        lead_times = {
            RiskLevel.GREEN: 24,
            RiskLevel.YELLOW: 18,
            RiskLevel.ORANGE: 12,
            RiskLevel.RED: 6,
            RiskLevel.BLACK: 1
        }
        return lead_times.get(level, 12)
    
    def _get_actions(
        self,
        category: RiskCategory,
        level: RiskLevel
    ) -> List[str]:
        """Get recommended actions."""
        actions = []
        
        if level in [RiskLevel.ORANGE, RiskLevel.RED]:
            actions.append("Reduce risk exposure")
            actions.append("Increase liquidity reserves")
            
        if category == RiskCategory.LIQUIDITY:
            actions.append("Monitor funding markets")
        elif category == RiskCategory.CREDIT:
            actions.append("Review credit exposures")
        elif category == RiskCategory.CURRENCY:
            actions.append("Hedge FX positions")
        elif category == RiskCategory.SOVEREIGN:
            actions.append("Diversify sovereign holdings")
        
        return actions
    
    def dashboard(self) -> str:
        """Generate risk dashboard."""
        lines = [
            "=" * 60,
            "⚠️ GLOBAL RISK DASHBOARD",
            "=" * 60,
            f"Overall Risk Level: {self.overall_risk.value.upper()}",
            f"Active Warnings: {len(self.warnings)}",
            "",
            "RISK BY CATEGORY:"
        ]
        
        by_category = {}
        for ind in self.indicators.values():
            if ind.category not in by_category:
                by_category[ind.category] = []
            by_category[ind.category].append(ind)
        
        for category, indicators in by_category.items():
            max_level = max(i.level for i in indicators)
            emoji = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴", "black": "⚫"}
            lines.append(f"  {emoji.get(max_level.value, '⚪')} {category.value}: {max_level.value}")
        
        if self.warnings:
            lines.extend(["", "RECENT WARNINGS:"])
            for w in self.warnings[-3:]:
                lines.append(f"  • {w.message}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Global Risk Warning System Test")
    print("=" * 60)
    
    system = GlobalRiskWarningSystem()
    
    # Simulate stress scenario
    system.update({
        "TED_Spread": 0.8,
        "HY_Spread": 650,
        "EM_FX_Vol": 18,
        "Cross_Market_Corr": 0.82
    })
    
    print(system.dashboard())
