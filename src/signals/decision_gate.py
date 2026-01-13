"""
decision_gate.py
=================
Final decision authority before capital is committed.

Implements hedge-fund-grade decision filtering:
- Confidence thresholding
- Regime filtering
- Signal agreement checks
- Risk/Reward validation
- No-trade bias by default
"""

from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger
from src.models.signal_schema import AISignal, SignalDirection, MarketRegime

logger = get_logger("DECISION_GATE")


@dataclass
class TradeDecision:
    """
    Final trade decision output.
    
    Required fields for StopLossEngine:
    - entry_price
    - structure_high
    - structure_low
    - atr
    - confidence
    - rr_ratio
    - action
    """
    symbol: str
    action: str                           # BUY / SELL / NO_TRADE
    signal: Optional[AISignal]
    confidence: float
    reason: str
    timeframe_bias: str = "H1"
    
    # Fields required by StopLossEngine
    entry_price: float = 0.0
    structure_high: float = 0.0
    structure_low: float = 0.0
    atr: float = 0.0
    rr_ratio: float = 1.5



class DecisionGate:
    """
    Hedge-fund grade decision filter.
    
    Philosophy:
    - AI proposes, Decision Gate disposes
    - No-trade is the default
    - Every rejection is logged with reason
    """

    def __init__(self):
        # --- Hard rules (can be config-driven later) ---
        self.min_confidence = 0.60
        self.min_probability = 0.55
        self.min_rr_ratio = 1.5
        self.allowed_regimes = {MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.QUIET}
        self.blocked_regimes = {MarketRegime.VOLATILE}
        self.max_uncertainty = 0.40

    # -------------------------------------------------
    def evaluate(self, signal: AISignal) -> TradeDecision:
        """
        Evaluate whether a fused signal is worthy of capital.
        
        Args:
            signal: AISignal from fusion engine
            
        Returns:
            TradeDecision with action and reason
        """
        symbol = signal.symbol
        logger.debug(f"Evaluating signal for {symbol}")

        # 0️⃣ No signal or NEUTRAL → no trade
        if signal is None:
            return self._reject(symbol, None, "No signal provided")
            
        if signal.direction == SignalDirection.NEUTRAL:
            return self._reject(symbol, signal, "Signal is NEUTRAL - no directional bias")

        # 1️⃣ Probability gate
        if signal.probability < self.min_probability:
            return self._reject(
                symbol, signal,
                f"Probability {signal.probability:.2%} < {self.min_probability:.2%}"
            )

        # 2️⃣ Confidence gate
        if signal.confidence < self.min_confidence:
            return self._reject(
                symbol, signal,
                f"Confidence {signal.confidence:.2%} < {self.min_confidence:.2%}"
            )

        # 3️⃣ Regime filter
        if signal.regime in self.blocked_regimes:
            return self._reject(
                symbol, signal,
                f"Regime {signal.regime.value} is blocked"
            )

        # 4️⃣ Risk / Reward validation
        if signal.expected_rr < self.min_rr_ratio:
            return self._reject(
                symbol, signal,
                f"Expected R:R {signal.expected_rr:.2f} < {self.min_rr_ratio}"
            )

        # 5️⃣ Uncertainty guard (1 - confidence)
        uncertainty = 1 - signal.confidence
        if uncertainty > self.max_uncertainty:
            return self._reject(
                symbol, signal,
                f"Uncertainty {uncertainty:.2%} > {self.max_uncertainty:.2%}"
            )

        # ✅ Trade approved
        action = "BUY" if signal.direction == SignalDirection.LONG else "SELL"
        
        decision = TradeDecision(
            symbol=symbol,
            action=action,
            signal=signal,
            confidence=signal.confidence,
            reason=self._decision_reason(signal),
            timeframe_bias=signal.timeframe,
        )

        logger.info(f"{symbol} APPROVED: {action} conf={signal.confidence:.2%}")
        return decision

    # -------------------------------------------------
    # Helper methods
    # -------------------------------------------------
    def _reject(self, symbol: str, signal: Optional[AISignal], reason: str) -> TradeDecision:
        """Create rejection decision with logging."""
        logger.info(f"{symbol} REJECTED: {reason}")
        
        return TradeDecision(
            symbol=symbol,
            action="NO_TRADE",
            signal=signal,
            confidence=signal.confidence if signal else 0.0,
            reason=reason,
        )

    def _decision_reason(self, signal: AISignal) -> str:
        """Human-readable explanation for audit & debugging."""
        return (
            f"Direction={signal.direction.value}, "
            f"Prob={signal.probability:.2%}, "
            f"Conf={signal.confidence:.2%}, "
            f"RR={signal.expected_rr:.2f}, "
            f"Regime={signal.regime.value}"
        )

    def update_thresholds(self, min_probability: float = None, 
                         min_confidence: float = None,
                         min_rr: float = None):
        """Update decision thresholds (for regime-adaptive trading)."""
        if min_probability:
            self.min_probability = min_probability
        if min_confidence:
            self.min_confidence = min_confidence
        if min_rr:
            self.min_rr_ratio = min_rr
        
        logger.info(f"Thresholds updated: prob={self.min_probability}, conf={self.min_confidence}, rr={self.min_rr_ratio}")
