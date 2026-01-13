"""
signal_fusion.py
================
Combines multiple heterogeneous signals into a single
actionable meta-signal.

Implements hedge-fund-grade signal fusion:
- Context-aware weighting
- Direction vs Timing separation
- Confidence & uncertainty normalization
- Explainable decision trail
"""

from collections import defaultdict
from typing import List, Optional
from statistics import mean

from src.utils.logger import get_logger
from src.models.signal_schema import Signal, AISignal, SignalDirection, MarketRegime, VolatilityState

logger = get_logger("SIGNAL_FUSION")


class ModelOutput:
    """Individual model prediction output."""
    def __init__(self, model_name: str, direction: SignalDirection, 
                 probability: float, confidence: float, weight: float = 1.0):
        self.model_name = model_name
        self.direction = direction
        self.probability = probability
        self.confidence = confidence
        self.weight = weight


class SignalFusionEngine:
    """
    Hedge-fund grade signal fusion.
    
    Key principles:
    - HTF has more weight than LTF
    - Direction ≠ Timing (separate models)
    - Regime changes weights dynamically
    - Uncertainty penalizes scores
    """

    def __init__(self):
        # Base weights (can be config-driven)
        self.model_weights = {
            "lstm": 0.6,
            "xgboost": 0.4,
            "xgb": 0.4,
            "pa": 0.3,
            "rule": 0.2,
        }

        self.timeframe_weights = {
            "HTF": 0.6,
            "MTF": 0.3,
            "LTF": 0.1,
            "H4": 0.6,
            "H1": 0.4,
            "M15": 0.2,
            "M5": 0.1,
        }

        self.regime_adjustments = {
            "TRENDING": {"lstm": 1.2, "pa": 1.1},
            "RANGING": {"xgboost": 1.3, "xgb": 1.3},
            "BREAKOUT": {"pa": 1.4, "lstm": 1.1},
            "VOLATILE": {"lstm": 0.7, "xgboost": 0.8},
            "QUIET": {"lstm": 1.0, "xgboost": 1.0},
        }

    # -------------------------------------------------
    def fuse(self, signals: List[Signal]) -> Signal:
        """
        Main fusion entry point for legacy Signal format.
        """
        logger.debug("Fusing signals")

        if not signals:
            return self._hold_signal("No signals")

        action_scores = defaultdict(float)
        confidences = []
        uncertainties = []
        rr_ratios = []

        # Assume regime consistency (validated earlier)
        regime = signals[0].regime

        for sig in signals:
            for model in sig.source_models:
                weight = self._compute_weight(model, sig.timeframe_bias, regime)

                signed_conf = self._signed_confidence(sig)
                action_scores[sig.action] += signed_conf * weight

            confidences.append(sig.confidence)
            uncertainties.append(sig.uncertainty)
            rr_ratios.append(sig.rr_ratio)

        final_action = self._resolve_action(action_scores)
        final_confidence = self._aggregate_confidence(confidences, uncertainties)
        final_rr = mean(rr_ratios) if rr_ratios else 1.5
        final_uncertainty = mean(uncertainties) if uncertainties else 0.5

        fused_signal = Signal(
            action=final_action,
            confidence=final_confidence,
            rr_ratio=final_rr,
            regime=regime,
            uncertainty=final_uncertainty,
            htf_bias=self._extract_bias(signals, "HTF"),
            ltf_bias=self._extract_bias(signals, "LTF"),
            timeframe_bias="MTF",
            source_models=self._collect_sources(signals)
        )

        logger.info(f"Fused signal: {final_action} conf={final_confidence:.2f}")
        return fused_signal

    # -------------------------------------------------
    def fuse_ai_signals(self, outputs: List[ModelOutput], symbol: str, timeframe: str,
                       regime: MarketRegime, volatility: VolatilityState) -> AISignal:
        """
        Fuse multiple ModelOutput into single AISignal.
        """
        if not outputs:
            return AISignal.neutral(symbol, timeframe, regime)

        # Apply weights
        for output in outputs:
            base_weight = self.model_weights.get(output.model_name.lower(), 0.3)
            regime_adj = self.regime_adjustments.get(regime.value, {}).get(output.model_name.lower(), 1.0)
            output.weight = base_weight * regime_adj

        # Count votes by direction
        votes = {SignalDirection.LONG: 0.0, SignalDirection.SHORT: 0.0, SignalDirection.NEUTRAL: 0.0}
        weighted_probs = {SignalDirection.LONG: [], SignalDirection.SHORT: [], SignalDirection.NEUTRAL: []}
        confidences = []
        uncertainties = []

        for output in outputs:
            votes[output.direction] += output.weight * output.confidence
            weighted_probs[output.direction].append(output.probability * output.weight)
            confidences.append(output.confidence * output.weight)
            uncertainties.append((1 - output.confidence) * output.weight)

        # Determine direction by weighted vote
        winning_direction = max(votes, key=votes.get)
        vote_count = sum(1 for o in outputs if o.direction == winning_direction)

        # Check agreement threshold (min 2 models)
        if vote_count < 2 and len(outputs) >= 2:
            logger.debug(f"Insufficient agreement: {vote_count} < 2")
            return AISignal.neutral(symbol, timeframe, regime)

        # Calculate fused probability with uncertainty penalty
        direction_probs = weighted_probs[winning_direction]
        if direction_probs:
            total_weight = sum(o.weight for o in outputs if o.direction == winning_direction)
            base_probability = sum(direction_probs) / total_weight if total_weight > 0 else 0.5
            uncertainty_penalty = mean(uncertainties) if uncertainties else 0.0
            fused_probability = base_probability * (1 - uncertainty_penalty * 0.5)
        else:
            fused_probability = 0.5

        # Calculate fused confidence
        total_weight = sum(o.weight for o in outputs)
        fused_confidence = sum(confidences) / total_weight if total_weight > 0 else 0.5

        # Expected R:R based on confidence
        expected_rr = 1.5 + (fused_confidence * 1.5)  # 1.5 to 3.0

        return AISignal(
            direction=winning_direction,
            probability=min(fused_probability, 0.95),
            confidence=min(fused_confidence, 0.95),
            volatility_state=volatility,
            regime=regime,
            expected_rr=expected_rr,
            symbol=symbol,
            timeframe=timeframe,
            model_name="fusion",
            source_models=[o.model_name for o in outputs],
            metadata={"model_count": len(outputs), "agreement": vote_count}
        )

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------
    def _compute_weight(self, model: str, tf_bias: str, regime: str) -> float:
        base = self.model_weights.get(model.lower(), 0.1)
        tf_weight = self.timeframe_weights.get(tf_bias, 0.1)
        regime_adj = self.regime_adjustments.get(regime, {}).get(model.lower(), 1.0)

        return base * tf_weight * regime_adj

    def _signed_confidence(self, signal: Signal) -> float:
        if signal.action == "BUY":
            return signal.confidence
        if signal.action == "SELL":
            return -signal.confidence
        return 0.0

    def _resolve_action(self, scores: dict) -> str:
        if not scores:
            return "HOLD"

        buy_score = scores.get("BUY", 0)
        sell_score = scores.get("SELL", 0)

        # Tie-breaker: require clear winner
        if abs(buy_score - sell_score) < 0.05:
            return "HOLD"

        return "BUY" if buy_score > abs(sell_score) else "SELL"

    def _aggregate_confidence(self, confs: List[float], uncs: List[float]) -> float:
        if not confs:
            return 0.0
        base_conf = mean(confs)
        penalty = mean(uncs) if uncs else 0.0
        return max(0.0, min(1.0, base_conf * (1 - penalty)))

    def _extract_bias(self, signals: List[Signal], tf: str) -> str:
        biases = [s.action for s in signals if s.timeframe_bias == tf]
        if not biases:
            return "NEUTRAL"
        return max(set(biases), key=biases.count)

    def _collect_sources(self, signals: List[Signal]) -> list:
        sources = set()
        for s in signals:
            sources.update(s.source_models)
        return list(sources)

    def _hold_signal(self, reason: str) -> Signal:
        logger.info(f"HOLD signal issued: {reason}")
        return Signal(
            action="HOLD",
            confidence=0.0,
            rr_ratio=0.0,
            regime="UNKNOWN",
            uncertainty=1.0,
            htf_bias="NEUTRAL",
            ltf_bias="NEUTRAL",
            timeframe_bias="NONE",
            source_models=[]
        )


# Alias for backward compatibility
SignalFusion = SignalFusionEngine
