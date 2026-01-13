"""
explainability.py
==================
Regulator-Grade Explainability (XAI for Trading)

ตอบได้ทั้ง:
- นักลงทุน
- Risk Committee
- Regulator
- ตัวคุณเอง

ถ้าอธิบายไม่ได้ = ห้ามเพิ่มทุน

Requirements:
- ทำไมเข้าดีลนี้
- ใช้ข้อมูลอะไร
- ถ้าไม่เข้า จะเกิดอะไร
- ความเสี่ยงคืออะไร
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
from src.utils.logger import get_logger

logger = get_logger("EXPLAINABILITY")


class ExplanationLevel(str, Enum):
    """Explanation detail levels."""
    SIGNAL = "SIGNAL"           # Signal-level explanation
    MODEL = "MODEL"             # Model-level (SHAP, etc.)
    PORTFOLIO = "PORTFOLIO"     # Portfolio contribution
    FULL = "FULL"               # All levels


@dataclass
class SignalExplanation:
    """Signal-level explanation."""
    indicators_used: List[str]
    conditions_met: List[str]
    regime: str
    confidence: float
    primary_driver: str


@dataclass
class ModelExplanation:
    """Model-level explanation (XAI)."""
    model_name: str
    feature_importance: Dict[str, float]
    decision_path: List[str]
    shap_values: Dict[str, float] = field(default_factory=dict)
    counterfactual: str = ""    # "If X was Y, decision would be Z"


@dataclass
class PortfolioExplanation:
    """Portfolio-level explanation."""
    contribution_to_alpha: float
    contribution_to_risk: float
    correlation_with_others: float
    crisis_exposure: float


@dataclass
class RiskExplanation:
    """Risk explanation."""
    risk_flags: List[str]
    max_loss_potential: float
    probability_of_loss: float
    regime_risk: str
    crowding_risk: str


@dataclass
class TradeExplanation:
    """Complete trade explanation record."""
    trade_id: str
    timestamp: datetime
    symbol: str
    direction: str
    
    # Decision components
    signal_explanation: SignalExplanation
    model_explanation: Optional[ModelExplanation]
    portfolio_explanation: Optional[PortfolioExplanation]
    risk_explanation: RiskExplanation
    
    # Summary
    decision_reasons: List[str]
    expected_alpha: float
    overall_confidence: float
    
    # Audit
    audit_trail: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/display."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "decision_reasons": self.decision_reasons,
            "expected_alpha": self.expected_alpha,
            "confidence": self.overall_confidence,
            "risk_flags": self.risk_explanation.risk_flags,
            "signal": {
                "indicators": self.signal_explanation.indicators_used,
                "conditions": self.signal_explanation.conditions_met,
                "regime": self.signal_explanation.regime,
                "primary_driver": self.signal_explanation.primary_driver,
            },
        }


class ExplainabilityEngine:
    """
    Regulator-Grade Explainability Engine.
    
    Provides complete audit trail for every trading decision.
    """

    def __init__(self):
        self.explanations: Dict[str, TradeExplanation] = {}
        self.feature_importances: Dict[str, Dict] = {}

    # -------------------------------------------------
    # Main explanation generation
    # -------------------------------------------------
    def explain_decision(self, 
                        trade_id: str,
                        signal_data: Dict,
                        model_outputs: Dict,
                        portfolio_state: Dict,
                        risk_metrics: Dict) -> TradeExplanation:
        """
        Generate complete explanation for a trading decision.
        
        Args:
            trade_id: Unique trade identifier
            signal_data: Signal generation data
            model_outputs: Model prediction outputs
            portfolio_state: Current portfolio state
            risk_metrics: Risk assessment
            
        Returns:
            Complete TradeExplanation
        """
        # Signal-level explanation
        signal_exp = self._explain_signal(signal_data)
        
        # Model-level explanation
        model_exp = self._explain_model(model_outputs) if model_outputs else None
        
        # Portfolio-level explanation
        portfolio_exp = self._explain_portfolio(portfolio_state) if portfolio_state else None
        
        # Risk explanation
        risk_exp = self._explain_risk(risk_metrics)
        
        # Generate decision reasons
        reasons = self._generate_reasons(signal_exp, model_exp, risk_exp)
        
        # Create explanation
        explanation = TradeExplanation(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=signal_data.get("symbol", "UNKNOWN"),
            direction=signal_data.get("direction", "UNKNOWN"),
            signal_explanation=signal_exp,
            model_explanation=model_exp,
            portfolio_explanation=portfolio_exp,
            risk_explanation=risk_exp,
            decision_reasons=reasons,
            expected_alpha=signal_data.get("expected_alpha", 0.0),
            overall_confidence=signal_data.get("confidence", 0.5),
            audit_trail=[
                f"Signal generated at {datetime.now().isoformat()}",
                f"Decision validated by explainability engine",
            ],
        )
        
        self.explanations[trade_id] = explanation
        
        logger.info(f"Explanation generated for {trade_id}")
        
        return explanation

    # -------------------------------------------------
    # Component explanations
    # -------------------------------------------------
    def _explain_signal(self, data: Dict) -> SignalExplanation:
        """Generate signal-level explanation."""
        return SignalExplanation(
            indicators_used=data.get("indicators", []),
            conditions_met=data.get("conditions", []),
            regime=data.get("regime", "UNKNOWN"),
            confidence=data.get("confidence", 0.5),
            primary_driver=data.get("primary_driver", "Unknown"),
        )

    def _explain_model(self, outputs: Dict) -> ModelExplanation:
        """Generate model-level explanation."""
        model_name = outputs.get("model_name", "ensemble")
        
        # Feature importance
        importance = outputs.get("feature_importance", {})
        
        # Decision path (simplified)
        path = outputs.get("decision_path", [])
        
        # SHAP values if available
        shap = outputs.get("shap_values", {})
        
        # Generate counterfactual
        counterfactual = self._generate_counterfactual(outputs)
        
        return ModelExplanation(
            model_name=model_name,
            feature_importance=importance,
            decision_path=path,
            shap_values=shap,
            counterfactual=counterfactual,
        )

    def _explain_portfolio(self, state: Dict) -> PortfolioExplanation:
        """Generate portfolio-level explanation."""
        return PortfolioExplanation(
            contribution_to_alpha=state.get("alpha_contribution", 0.0),
            contribution_to_risk=state.get("risk_contribution", 0.0),
            correlation_with_others=state.get("correlation", 0.0),
            crisis_exposure=state.get("crisis_exposure", 0.0),
        )

    def _explain_risk(self, metrics: Dict) -> RiskExplanation:
        """Generate risk explanation."""
        flags = []
        
        if metrics.get("volatility_high", False):
            flags.append("HIGH_VOLATILITY")
        if metrics.get("drawdown_risk", False):
            flags.append("DRAWDOWN_RISK")
        if metrics.get("crowding", 0) > 0.6:
            flags.append("CROWDING_RISK")
        if metrics.get("liquidity_low", False):
            flags.append("LOW_LIQUIDITY")
        
        return RiskExplanation(
            risk_flags=flags or ["NONE"],
            max_loss_potential=metrics.get("max_loss", 0.0),
            probability_of_loss=metrics.get("loss_probability", 0.0),
            regime_risk=metrics.get("regime_risk", "LOW"),
            crowding_risk="HIGH" if metrics.get("crowding", 0) > 0.6 else "LOW",
        )

    # -------------------------------------------------
    # Reason generation
    # -------------------------------------------------
    def _generate_reasons(self, signal: SignalExplanation, 
                         model: Optional[ModelExplanation],
                         risk: RiskExplanation) -> List[str]:
        """Generate human-readable decision reasons."""
        reasons = []
        
        # Signal reasons
        reasons.append(f"Primary driver: {signal.primary_driver}")
        reasons.append(f"Regime: {signal.regime}")
        
        if signal.conditions_met:
            reasons.append(f"Conditions: {', '.join(signal.conditions_met[:3])}")
        
        # Model reasons
        if model and model.feature_importance:
            top_features = sorted(model.feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            reasons.append(f"Top features: {', '.join(f[0] for f in top_features)}")
        
        # Risk reasons
        if risk.risk_flags and risk.risk_flags[0] != "NONE":
            reasons.append(f"Risk flags: {', '.join(risk.risk_flags)}")
        
        return reasons

    def _generate_counterfactual(self, outputs: Dict) -> str:
        """Generate counterfactual explanation."""
        # Simple counterfactual based on top feature
        importance = outputs.get("feature_importance", {})
        if not importance:
            return "N/A"
        
        top_feature = max(importance.items(), key=lambda x: x[1])
        return f"If {top_feature[0]} was reversed, decision would likely change"

    # -------------------------------------------------
    # Retrieval
    # -------------------------------------------------
    def get_explanation(self, trade_id: str) -> Optional[TradeExplanation]:
        """Get explanation for a trade."""
        return self.explanations.get(trade_id)

    def get_explanation_text(self, trade_id: str) -> str:
        """Get human-readable explanation text."""
        exp = self.explanations.get(trade_id)
        if not exp:
            return f"No explanation found for {trade_id}"
        
        lines = [
            f"TRADE EXPLANATION: {trade_id}",
            "=" * 50,
            f"Symbol: {exp.symbol}",
            f"Direction: {exp.direction}",
            f"Confidence: {exp.overall_confidence:.1%}",
            f"Expected Alpha: {exp.expected_alpha:.4f}",
            "",
            "DECISION REASONS:",
        ]
        
        for reason in exp.decision_reasons:
            lines.append(f"  • {reason}")
        
        lines.extend([
            "",
            "RISK:",
            f"  Flags: {', '.join(exp.risk_explanation.risk_flags)}",
            f"  Max Loss: {exp.risk_explanation.max_loss_potential:.1%}",
        ])
        
        return "\n".join(lines)

    def export_audit_trail(self, trade_id: str) -> Dict:
        """Export full audit trail for regulators."""
        exp = self.explanations.get(trade_id)
        if not exp:
            return {}
        
        return exp.to_dict()

    # -------------------------------------------------
    # Batch operations
    # -------------------------------------------------
    def explain_batch(self, decisions: List[Dict]) -> List[TradeExplanation]:
        """Explain multiple decisions."""
        explanations = []
        for d in decisions:
            exp = self.explain_decision(
                trade_id=d.get("id", f"trade_{len(self.explanations)}"),
                signal_data=d.get("signal", {}),
                model_outputs=d.get("model", {}),
                portfolio_state=d.get("portfolio", {}),
                risk_metrics=d.get("risk", {}),
            )
            explanations.append(exp)
        return explanations

    def get_status(self) -> Dict:
        """Get engine status."""
        return {
            "total_explanations": len(self.explanations),
            "recent_trades": list(self.explanations.keys())[-5:],
        }
