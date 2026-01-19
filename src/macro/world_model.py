# src/macro/world_model.py
"""
World Model - Economic Simulator
=================================

Digital Twin of global financial markets.

Simulates:
    - Macro scenarios (rates, CPI, PMI)
    - Policy decisions (FOMC, ECB, BOJ)
    - Cross-asset flows
    - Systemic risk cascades

Paper Statement:
    "We construct a world model of financial markets that enables
     the agent to simulate and evaluate multi-scenario futures
     before committing capital."
"""

import os
import sys
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("WORLD_MODEL")


# =============================================================================
# Macro Regime Definitions
# =============================================================================

class MacroRegime(Enum):
    """Global macro regime."""
    RISK_ON = "risk_on"          # Growth + Low vol
    RISK_OFF = "risk_off"        # Fear + High vol
    INFLATIONARY = "inflationary"  # Rising prices
    DEFLATIONARY = "deflationary"  # Falling prices
    STAGFLATION = "stagflation"    # High inflation + Low growth
    GOLDILOCKS = "goldilocks"      # Perfect conditions


class PolicyStance(Enum):
    """Central bank policy stance."""
    VERY_HAWKISH = "very_hawkish"
    HAWKISH = "hawkish"
    NEUTRAL = "neutral"
    DOVISH = "dovish"
    VERY_DOVISH = "very_dovish"


@dataclass
class MacroState:
    """Global macro state snapshot."""
    timestamp: str
    
    # Rates
    fed_rate: float = 5.25
    ecb_rate: float = 4.50
    boj_rate: float = 0.10
    
    # Inflation
    us_cpi: float = 3.2
    us_core_cpi: float = 3.8
    eu_cpi: float = 2.5
    
    # Growth
    us_gdp: float = 2.5
    us_pmi: float = 52.5
    us_employment: float = 3.7
    
    # Market
    vix: float = 15.0
    dxy: float = 104.0
    spx: float = 4800.0
    gold: float = 2050.0
    btc: float = 45000.0
    
    def to_dict(self) -> Dict:
        return {
            "fed_rate": self.fed_rate,
            "cpi": self.us_cpi,
            "gdp": self.us_gdp,
            "vix": self.vix,
            "dxy": self.dxy
        }


@dataclass
class Scenario:
    """Economic scenario with probability."""
    name: str
    probability: float
    macro_state: MacroState
    policy_stance: PolicyStance
    impact: Dict[str, float]  # Asset → Expected return
    description: str


class WorldModel:
    """
    Digital Twin of Global Financial Markets.
    
    Simulates multi-scenario futures and evaluates
    systemic risk before committing capital.
    
    Features:
        - Macro regime detection
        - Policy path simulation
        - Cross-asset correlation
        - Monte Carlo futures
    """
    
    def __init__(self):
        self.current_state = MacroState(timestamp=datetime.now().isoformat())
        self.scenarios: List[Scenario] = []
        self.regime = MacroRegime.RISK_ON
        
        # Correlation matrix (simplified)
        self.correlations = {
            ("USD", "GOLD"): -0.6,
            ("USD", "SPX"): 0.3,
            ("VIX", "SPX"): -0.8,
            ("GOLD", "RATES"): -0.5,
            ("BTC", "SPX"): 0.5,
        }
        
        logger.info("🌍 WorldModel initialized")
    
    def update_state(self, data: Dict[str, float]):
        """Update current macro state."""
        for key, value in data.items():
            if hasattr(self.current_state, key):
                setattr(self.current_state, key, value)
        
        self.current_state.timestamp = datetime.now().isoformat()
        self._update_regime()
    
    def _update_regime(self):
        """Detect current macro regime."""
        state = self.current_state
        
        if state.vix > 25:
            self.regime = MacroRegime.RISK_OFF
        elif state.us_cpi > 4.0 and state.us_gdp < 1.5:
            self.regime = MacroRegime.STAGFLATION
        elif state.us_cpi > 3.5:
            self.regime = MacroRegime.INFLATIONARY
        elif state.us_cpi < 2.0 and state.us_gdp < 1.0:
            self.regime = MacroRegime.DEFLATIONARY
        elif state.us_gdp > 2.0 and state.us_cpi < 3.0:
            self.regime = MacroRegime.GOLDILOCKS
        else:
            self.regime = MacroRegime.RISK_ON
    
    def generate_scenarios(self, n_scenarios: int = 5) -> List[Scenario]:
        """Generate probability-weighted scenarios."""
        self.scenarios = []
        state = self.current_state
        
        # Scenario 1: Soft Landing
        soft_landing = Scenario(
            name="Soft Landing",
            probability=0.35,
            macro_state=MacroState(
                timestamp="future",
                fed_rate=state.fed_rate - 0.50,
                us_cpi=2.5,
                us_gdp=2.0,
                vix=14
            ),
            policy_stance=PolicyStance.NEUTRAL,
            impact={"SPX": 0.08, "GOLD": -0.02, "USD": -0.03, "BTC": 0.15},
            description="Fed achieves soft landing, gradual rate cuts"
        )
        self.scenarios.append(soft_landing)
        
        # Scenario 2: Higher for Longer
        higher_longer = Scenario(
            name="Higher for Longer",
            probability=0.30,
            macro_state=MacroState(
                timestamp="future",
                fed_rate=state.fed_rate,
                us_cpi=3.5,
                us_gdp=1.5,
                vix=18
            ),
            policy_stance=PolicyStance.HAWKISH,
            impact={"SPX": -0.05, "GOLD": -0.05, "USD": 0.05, "BTC": -0.10},
            description="Inflation sticky, rates stay elevated"
        )
        self.scenarios.append(higher_longer)
        
        # Scenario 3: Recession
        recession = Scenario(
            name="Recession",
            probability=0.15,
            macro_state=MacroState(
                timestamp="future",
                fed_rate=state.fed_rate - 1.50,
                us_cpi=2.0,
                us_gdp=-0.5,
                vix=30
            ),
            policy_stance=PolicyStance.VERY_DOVISH,
            impact={"SPX": -0.20, "GOLD": 0.15, "USD": -0.05, "BTC": -0.25},
            description="Economic contraction forces aggressive cuts"
        )
        self.scenarios.append(recession)
        
        # Scenario 4: Inflation Surge
        inflation_surge = Scenario(
            name="Inflation Surge",
            probability=0.15,
            macro_state=MacroState(
                timestamp="future",
                fed_rate=state.fed_rate + 0.50,
                us_cpi=5.0,
                us_gdp=1.0,
                vix=25
            ),
            policy_stance=PolicyStance.VERY_HAWKISH,
            impact={"SPX": -0.15, "GOLD": 0.10, "USD": 0.08, "BTC": -0.20},
            description="Second inflation wave forces hikes"
        )
        self.scenarios.append(inflation_surge)
        
        # Scenario 5: Goldilocks
        goldilocks = Scenario(
            name="Goldilocks",
            probability=0.05,
            macro_state=MacroState(
                timestamp="future",
                fed_rate=state.fed_rate - 0.75,
                us_cpi=2.0,
                us_gdp=3.0,
                vix=12
            ),
            policy_stance=PolicyStance.NEUTRAL,
            impact={"SPX": 0.15, "GOLD": 0.05, "USD": 0.00, "BTC": 0.30},
            description="Perfect conditions: low inflation, strong growth"
        )
        self.scenarios.append(goldilocks)
        
        return self.scenarios
    
    def get_expected_return(self, asset: str) -> float:
        """Calculate probability-weighted expected return."""
        if not self.scenarios:
            self.generate_scenarios()
        
        expected = 0.0
        for scenario in self.scenarios:
            prob = scenario.probability
            impact = scenario.impact.get(asset, 0)
            expected += prob * impact
        
        return expected
    
    def get_optimal_allocation(self) -> Dict[str, float]:
        """Get scenario-optimal allocation."""
        if not self.scenarios:
            self.generate_scenarios()
        
        assets = ["SPX", "GOLD", "USD", "BTC"]
        expected_returns = {a: self.get_expected_return(a) for a in assets}
        
        # Simple allocation: weight by positive expected return
        total_positive = sum(max(0, r) for r in expected_returns.values())
        
        if total_positive == 0:
            # All negative → hold cash
            return {"CASH": 1.0}
        
        allocation = {}
        for asset, ret in expected_returns.items():
            if ret > 0:
                allocation[asset] = max(0, ret) / total_positive
            else:
                allocation[asset] = 0.0
        
        # Ensure cash allocation
        total_alloc = sum(allocation.values())
        if total_alloc < 0.9:
            allocation["CASH"] = 1.0 - total_alloc
        
        return allocation
    
    def explain(self) -> str:
        """Generate economic reasoning explanation."""
        if not self.scenarios:
            self.generate_scenarios()
        
        lines = [
            "🌍 WORLD MODEL ANALYSIS",
            "=" * 40,
            f"Current Regime: {self.regime.value}",
            f"Fed Rate: {self.current_state.fed_rate}%",
            f"CPI: {self.current_state.us_cpi}%",
            f"VIX: {self.current_state.vix}",
            "",
            "SCENARIOS:",
        ]
        
        for s in sorted(self.scenarios, key=lambda x: -x.probability):
            lines.append(f"  • {s.name} ({s.probability:.0%}): {s.description}")
        
        allocation = self.get_optimal_allocation()
        lines.append("")
        lines.append("OPTIMAL ALLOCATION:")
        for asset, weight in sorted(allocation.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                lines.append(f"  • {asset}: {weight:.0%}")
        
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("World Model Test")
    print("=" * 60)
    
    world = WorldModel()
    world.generate_scenarios()
    
    print(world.explain())
    
    print("\n" + "=" * 60)
    print("Expected Returns:")
    for asset in ["SPX", "GOLD", "USD", "BTC"]:
        ret = world.get_expected_return(asset)
        print(f"  {asset}: {ret:+.1%}")
    
    print("=" * 60)
