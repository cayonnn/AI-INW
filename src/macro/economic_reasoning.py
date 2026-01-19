# src/macro/economic_reasoning.py
"""
Explainable Economic Reasoning (X-Econ AI)
==========================================

AI that reasons economically and explains decisions.

Features:
    - Natural language macro thesis
    - Cross-asset reasoning chains
    - Policy impact analysis
    - Investment memo generation

Paper Statement:
    "Our system generates explainable economic reasoning,
     articulating investment decisions through macro-logical
     chains rather than opaque model outputs."
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("ECON_REASONING")


@dataclass
class ReasoningChain:
    """Economic reasoning chain."""
    premise: str
    implication: str
    conclusion: str
    confidence: float
    
    def __str__(self) -> str:
        return f"IF {self.premise} → THEN {self.implication} → THEREFORE {self.conclusion}"


@dataclass
class InvestmentThesis:
    """Complete investment thesis."""
    title: str
    summary: str
    chains: List[ReasoningChain]
    allocation: Dict[str, float]
    risks: List[str]
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EconomicReasoningEngine:
    """
    Generates explainable economic reasoning.
    
    Capabilities:
        - Build causal reasoning chains
        - Generate macro thesis
        - Create investment memos
        - Explain decisions to non-experts
    """
    
    # Knowledge base of economic relationships
    CAUSAL_RULES = [
        # Inflation → Rates
        {"if": "inflation_high", "then": "rates_up", "impact": {"USD": 0.05, "GOLD": -0.03}},
        {"if": "inflation_low", "then": "rates_down", "impact": {"USD": -0.03, "GOLD": 0.02}},
        
        # Rates → Assets
        {"if": "rates_up", "then": "usd_strong", "impact": {"USD": 0.05}},
        {"if": "rates_down", "then": "growth_assets_up", "impact": {"SPX": 0.05, "BTC": 0.10}},
        
        # Growth → Risk
        {"if": "growth_strong", "then": "risk_on", "impact": {"SPX": 0.08, "VIX": -0.20}},
        {"if": "growth_weak", "then": "risk_off", "impact": {"GOLD": 0.05, "VIX": 0.30}},
        
        # VIX → Fear
        {"if": "vix_high", "then": "flight_to_safety", "impact": {"GOLD": 0.08, "USD": 0.03}},
        
        # Geopolitics
        {"if": "geopolitical_risk", "then": "safe_haven_bid", "impact": {"GOLD": 0.10, "USD": 0.05}},
    ]
    
    def __init__(self):
        self.current_thesis: Optional[InvestmentThesis] = None
        logger.info("🧠 EconomicReasoningEngine initialized")
    
    def analyze(
        self,
        inflation: float,
        growth: float,
        rates: float,
        vix: float,
        geopolitical_risk: bool = False
    ) -> InvestmentThesis:
        """
        Analyze macro conditions and generate thesis.
        """
        chains = []
        combined_impact: Dict[str, float] = {}
        
        # Inflation reasoning
        if inflation > 3.5:
            chain = ReasoningChain(
                premise=f"Inflation is elevated ({inflation}%)",
                implication="Central banks will maintain/raise rates",
                conclusion="USD strength, pressure on gold",
                confidence=0.75
            )
            chains.append(chain)
            self._merge_impact(combined_impact, {"USD": 0.05, "GOLD": -0.03})
        
        elif inflation < 2.5:
            chain = ReasoningChain(
                premise=f"Inflation is contained ({inflation}%)",
                implication="Room for rate cuts",
                conclusion="Support for risk assets",
                confidence=0.70
            )
            chains.append(chain)
            self._merge_impact(combined_impact, {"SPX": 0.05, "BTC": 0.08})
        
        # Growth reasoning
        if growth > 2.0:
            chain = ReasoningChain(
                premise=f"GDP growth is strong ({growth}%)",
                implication="Corporate earnings positive",
                conclusion="Equity bullish bias",
                confidence=0.75
            )
            chains.append(chain)
            self._merge_impact(combined_impact, {"SPX": 0.08})
        
        elif growth < 1.0:
            chain = ReasoningChain(
                premise=f"GDP growth is weak ({growth}%)",
                implication="Recession risk rising",
                conclusion="Defensive positioning recommended",
                confidence=0.70
            )
            chains.append(chain)
            self._merge_impact(combined_impact, {"GOLD": 0.05, "CASH": 0.20})
        
        # VIX reasoning
        if vix > 25:
            chain = ReasoningChain(
                premise=f"VIX is elevated ({vix})",
                implication="Market stress / uncertainty high",
                conclusion="Reduce risk exposure, favor safe havens",
                confidence=0.80
            )
            chains.append(chain)
            self._merge_impact(combined_impact, {"GOLD": 0.08, "CASH": 0.15})
        
        # Geopolitical reasoning
        if geopolitical_risk:
            chain = ReasoningChain(
                premise="Geopolitical tensions elevated",
                implication="Flight to safety likely",
                conclusion="Gold + USD outperformance",
                confidence=0.70
            )
            chains.append(chain)
            self._merge_impact(combined_impact, {"GOLD": 0.10, "USD": 0.05})
        
        # Build thesis
        title = self._generate_title(chains)
        summary = self._generate_summary(chains)
        allocation = self._calculate_allocation(combined_impact)
        risks = self._identify_risks(chains)
        
        thesis = InvestmentThesis(
            title=title,
            summary=summary,
            chains=chains,
            allocation=allocation,
            risks=risks
        )
        
        self.current_thesis = thesis
        return thesis
    
    def _merge_impact(self, base: Dict, new: Dict):
        """Merge impact dictionaries."""
        for asset, impact in new.items():
            base[asset] = base.get(asset, 0) + impact
    
    def _calculate_allocation(self, impact: Dict) -> Dict[str, float]:
        """Calculate allocation from combined impact."""
        if not impact:
            return {"CASH": 1.0}
        
        # Normalize positive impacts
        positive = {k: max(v, 0) for k, v in impact.items()}
        total = sum(positive.values())
        
        if total < 0.1:
            return {"CASH": 1.0}
        
        allocation = {}
        for asset, val in positive.items():
            if val > 0:
                allocation[asset] = val / total
        
        # Ensure minimum cash
        total_alloc = sum(allocation.values())
        if total_alloc < 1.0:
            allocation["CASH"] = 1.0 - total_alloc
        
        return allocation
    
    def _generate_title(self, chains: List[ReasoningChain]) -> str:
        """Generate thesis title."""
        if not chains:
            return "Neutral Market Outlook"
        
        # Find dominant theme
        themes = []
        for chain in chains:
            if "risk" in chain.conclusion.lower():
                themes.append("risk")
            if "gold" in chain.conclusion.lower():
                themes.append("gold")
            if "equity" in chain.conclusion.lower():
                themes.append("equity")
        
        if "risk" in themes:
            return "Defensive Positioning Thesis"
        if "equity" in themes:
            return "Growth-Oriented Thesis"
        if "gold" in themes:
            return "Safe Haven Thesis"
        
        return "Balanced Market Thesis"
    
    def _generate_summary(self, chains: List[ReasoningChain]) -> str:
        """Generate thesis summary."""
        if not chains:
            return "No strong directional conviction. Maintain neutral allocation."
        
        points = [str(c) for c in chains[:3]]
        return " | ".join(points)
    
    def _identify_risks(self, chains: List[ReasoningChain]) -> List[str]:
        """Identify thesis risks."""
        risks = []
        
        for chain in chains:
            if chain.confidence < 0.7:
                risks.append(f"Low confidence in: {chain.premise}")
        
        risks.append("Unexpected policy shifts")
        risks.append("Geopolitical escalation")
        
        return risks
    
    def generate_memo(self) -> str:
        """Generate investment memo."""
        if not self.current_thesis:
            return "No thesis available. Run analyze() first."
        
        t = self.current_thesis
        
        lines = [
            "=" * 60,
            f"📄 INVESTMENT MEMO: {t.title}",
            "=" * 60,
            f"Generated: {t.timestamp}",
            "",
            "## EXECUTIVE SUMMARY",
            t.summary,
            "",
            "## REASONING CHAINS",
        ]
        
        for i, chain in enumerate(t.chains, 1):
            lines.append(f"{i}. {chain}")
        
        lines.append("")
        lines.append("## RECOMMENDED ALLOCATION")
        for asset, weight in sorted(t.allocation.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                lines.append(f"  • {asset}: {weight:.0%}")
        
        lines.append("")
        lines.append("## RISKS")
        for risk in t.risks:
            lines.append(f"  ⚠️ {risk}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Economic Reasoning Engine Test")
    print("=" * 60)
    
    engine = EconomicReasoningEngine()
    
    # Analyze current conditions
    thesis = engine.analyze(
        inflation=3.2,
        growth=2.5,
        rates=5.25,
        vix=15,
        geopolitical_risk=False
    )
    
    print(engine.generate_memo())
    
    print("\n--- High Risk Scenario ---\n")
    
    thesis = engine.analyze(
        inflation=4.5,
        growth=0.8,
        rates=5.50,
        vix=28,
        geopolitical_risk=True
    )
    
    print(engine.generate_memo())
