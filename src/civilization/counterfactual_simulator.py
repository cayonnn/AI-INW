# src/civilization/counterfactual_simulator.py
"""
Counterfactual World Simulator
===============================

Simulates "What if?" scenarios at civilization scale.

Questions:
    - "What if rates rose faster?"
    - "What if QE ended earlier?"
    - "What if Country X was sanctioned?"

This is Policy Simulation, not Forecasting.

Paper Statement:
    "We develop a counterfactual reasoning engine that enables
     analysis of alternative policy trajectories and their
     long-term systemic consequences."
"""

import os
import sys
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("COUNTERFACTUAL")


class PolicyAction(Enum):
    """Policy actions to simulate."""
    RATE_HIKE = "rate_hike"
    RATE_CUT = "rate_cut"
    QE_START = "qe_start"
    QE_TAPER = "qe_taper"
    SANCTION = "sanction"
    TRADE_WAR = "trade_war"
    FISCAL_STIMULUS = "fiscal_stimulus"
    AUSTERITY = "austerity"


@dataclass
class CounterfactualScenario:
    """A counterfactual scenario definition."""
    name: str
    action: PolicyAction
    magnitude: float  # -1 to 1 scale
    timing: str  # "early", "on_time", "late"
    description: str


@dataclass
class WorldState:
    """State of the world at a point in time."""
    year: int
    gdp_growth: float
    inflation: float
    unemployment: float
    debt_to_gdp: float
    inequality_index: float
    stability_index: float  # 0-1, higher = more stable
    
    def to_dict(self) -> Dict:
        return {
            "year": self.year,
            "gdp": f"{self.gdp_growth:.1f}%",
            "inflation": f"{self.inflation:.1f}%",
            "unemployment": f"{self.unemployment:.1f}%",
            "debt_gdp": f"{self.debt_to_gdp:.0f}%",
            "stability": f"{self.stability_index:.2f}"
        }


@dataclass
class CounterfactualResult:
    """Result of counterfactual simulation."""
    scenario: CounterfactualScenario
    baseline_trajectory: List[WorldState]
    counterfactual_trajectory: List[WorldState]
    divergence_point: int
    max_deviation: Dict[str, float]
    stability_impact: float
    recommendation: str


class CounterfactualSimulator:
    """
    Counterfactual World Simulator.
    
    Simulates alternative policy paths and their
    long-term systemic consequences.
    
    Not forecasting - exploring "what could have been"
    or "what could happen if".
    """
    
    def __init__(self, initial_state: Optional[WorldState] = None):
        self.current_state = initial_state or WorldState(
            year=2024,
            gdp_growth=2.5,
            inflation=3.2,
            unemployment=3.8,
            debt_to_gdp=125,
            inequality_index=0.42,
            stability_index=0.75
        )
        
        self.history: List[CounterfactualResult] = []
        
        logger.info("🌏 CounterfactualSimulator initialized")
    
    def simulate(
        self,
        scenario: CounterfactualScenario,
        years: int = 10
    ) -> CounterfactualResult:
        """
        Simulate counterfactual scenario.
        
        Args:
            scenario: The counterfactual to simulate
            years: Years to project
            
        Returns:
            CounterfactualResult with both trajectories
        """
        baseline = self._simulate_trajectory(
            self.current_state, years, None
        )
        
        counterfactual = self._simulate_trajectory(
            self.current_state, years, scenario
        )
        
        # Calculate divergence
        divergence = self._calculate_divergence(baseline, counterfactual)
        
        # Stability impact
        baseline_stability = np.mean([s.stability_index for s in baseline])
        cf_stability = np.mean([s.stability_index for s in counterfactual])
        stability_impact = cf_stability - baseline_stability
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            scenario, stability_impact, divergence
        )
        
        result = CounterfactualResult(
            scenario=scenario,
            baseline_trajectory=baseline,
            counterfactual_trajectory=counterfactual,
            divergence_point=2,
            max_deviation=divergence,
            stability_impact=stability_impact,
            recommendation=recommendation
        )
        
        self.history.append(result)
        return result
    
    def _simulate_trajectory(
        self,
        initial: WorldState,
        years: int,
        scenario: Optional[CounterfactualScenario]
    ) -> List[WorldState]:
        """Simulate economic trajectory."""
        trajectory = [initial]
        state = WorldState(**initial.__dict__)
        
        for year in range(1, years + 1):
            state = WorldState(
                year=initial.year + year,
                gdp_growth=state.gdp_growth,
                inflation=state.inflation,
                unemployment=state.unemployment,
                debt_to_gdp=state.debt_to_gdp,
                inequality_index=state.inequality_index,
                stability_index=state.stability_index
            )
            
            # Apply scenario effects
            if scenario:
                self._apply_scenario_effect(state, scenario, year)
            
            # Natural evolution
            self._apply_natural_dynamics(state)
            
            trajectory.append(state)
        
        return trajectory
    
    def _apply_scenario_effect(
        self,
        state: WorldState,
        scenario: CounterfactualScenario,
        year: int
    ):
        """Apply scenario effects to state."""
        mag = scenario.magnitude
        
        if scenario.action == PolicyAction.RATE_HIKE:
            state.inflation -= 0.3 * mag
            state.gdp_growth -= 0.2 * mag
            state.unemployment += 0.1 * mag
            
        elif scenario.action == PolicyAction.RATE_CUT:
            state.inflation += 0.2 * mag
            state.gdp_growth += 0.3 * mag
            state.debt_to_gdp += 2 * mag
            
        elif scenario.action == PolicyAction.SANCTION:
            state.stability_index -= 0.1 * mag
            state.inflation += 0.5 * mag
            
        elif scenario.action == PolicyAction.FISCAL_STIMULUS:
            state.gdp_growth += 0.5 * mag
            state.debt_to_gdp += 5 * mag
            state.inequality_index -= 0.02 * mag
    
    def _apply_natural_dynamics(self, state: WorldState):
        """Apply natural economic dynamics."""
        # Mean reversion + noise
        state.gdp_growth += np.random.normal(0, 0.3)
        state.gdp_growth = 0.95 * state.gdp_growth + 0.05 * 2.0
        
        state.inflation += np.random.normal(0, 0.2)
        state.inflation = max(0, state.inflation)
        
        state.stability_index = np.clip(
            state.stability_index + np.random.normal(0, 0.02),
            0, 1
        )
    
    def _calculate_divergence(
        self,
        baseline: List[WorldState],
        counterfactual: List[WorldState]
    ) -> Dict[str, float]:
        """Calculate max divergence between trajectories."""
        divergence = {}
        
        for attr in ["gdp_growth", "inflation", "stability_index"]:
            max_diff = max(
                abs(getattr(b, attr) - getattr(c, attr))
                for b, c in zip(baseline, counterfactual)
            )
            divergence[attr] = max_diff
        
        return divergence
    
    def _generate_recommendation(
        self,
        scenario: CounterfactualScenario,
        stability_impact: float,
        divergence: Dict
    ) -> str:
        """Generate policy recommendation."""
        if stability_impact > 0.05:
            return f"RECOMMENDED: {scenario.name} improves stability"
        elif stability_impact < -0.10:
            return f"WARNING: {scenario.name} risks destabilization"
        else:
            return f"NEUTRAL: {scenario.name} has limited systemic impact"
    
    def explain(self, result: CounterfactualResult) -> str:
        """Generate explanation of counterfactual result."""
        lines = [
            "=" * 60,
            f"🌏 COUNTERFACTUAL ANALYSIS: {result.scenario.name}",
            "=" * 60,
            "",
            f"Action: {result.scenario.action.value}",
            f"Magnitude: {result.scenario.magnitude:+.1f}",
            f"Timing: {result.scenario.timing}",
            "",
            "DIVERGENCE FROM BASELINE:",
        ]
        
        for metric, deviation in result.max_deviation.items():
            lines.append(f"  • {metric}: ±{deviation:.2f}")
        
        lines.extend([
            "",
            f"STABILITY IMPACT: {result.stability_impact:+.3f}",
            "",
            f"📌 {result.recommendation}",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Counterfactual Simulator Test")
    print("=" * 60)
    
    sim = CounterfactualSimulator()
    
    # Test scenarios
    scenarios = [
        CounterfactualScenario(
            name="Aggressive Rate Hikes 2022",
            action=PolicyAction.RATE_HIKE,
            magnitude=0.8,
            timing="early",
            description="What if Fed raised rates faster in 2022?"
        ),
        CounterfactualScenario(
            name="Major Sanction Event",
            action=PolicyAction.SANCTION,
            magnitude=0.7,
            timing="on_time",
            description="What if major economy was sanctioned?"
        ),
    ]
    
    for scenario in scenarios:
        result = sim.simulate(scenario, years=5)
        print(sim.explain(result))
