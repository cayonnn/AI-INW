"""
signal_schema.py
=================
Standardized AI signal output format for all models.

This is the CORE CONTRACT between:
- AI Models (producers)
- Signal Fusion (aggregator)
- Decision Gate (consumer)

A rich signal enables hedge-fund-grade decision making.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class SignalDirection(str, Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    
    # Aliases for compatibility
    BUY = "LONG"
    SELL = "SHORT"
    HOLD = "NEUTRAL"


class VolatilityState(str, Enum):
    """Volatility classification."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"
    BREAKOUT = "BREAKOUT"


class TimeframeBias(str, Enum):
    """Timeframe alignment."""
    HTF = "HTF"  # Higher timeframe dominant
    LTF = "LTF"  # Lower timeframe dominant
    MTF = "MTF"  # Multi-timeframe aligned


@dataclass
class Signal:
    """
    Competition-grade signal schema.
    
    This rich signal is required for DecisionGate to work properly.
    All fields must be populated for proper decision making.
    """
    
    # Core action
    action: str                           # BUY / SELL / HOLD
    confidence: float                     # 0-1: model confidence
    rr_ratio: float                       # Expected Risk:Reward ratio
    
    # Market context
    regime: str                           # TREND / RANGE / BREAKOUT / VOLATILE
    uncertainty: float                    # 0-1: inverse of confidence
    
    # Multi-timeframe bias
    htf_bias: str                         # BUY / SELL / NEUTRAL (Higher TF)
    ltf_bias: str                         # BUY / SELL / NEUTRAL (Lower TF)
    timeframe_bias: str                   # HTF / LTF / MTF
    
    # Source tracking
    source_models: List[str] = field(default_factory=list)  # ["LSTM", "XGB", "PA"]
    
    # Metadata
    symbol: str = ""
    timeframe: str = "H1"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.source_models:
            self.source_models = []
        if self.uncertainty == 0:
            self.uncertainty = 1 - self.confidence


@dataclass
class AISignal:
    """
    Standardized AI signal output.
    
    This schema is used by all AI models to output predictions.
    NO raw price predictions - only probabilities, confidence, and regime.
    """
    
    # Core signal
    direction: SignalDirection
    probability: float              # 0.0 to 1.0 - probability of predicted direction
    confidence: float               # 0.0 to 1.0 - model confidence in prediction
    
    # Market context
    volatility_state: VolatilityState
    regime: MarketRegime
    expected_rr: float              # Expected risk:reward ratio
    
    # Multi-timeframe bias
    htf_bias: SignalDirection = SignalDirection.NEUTRAL
    ltf_bias: SignalDirection = SignalDirection.NEUTRAL
    timeframe_bias: TimeframeBias = TimeframeBias.MTF
    
    # Uncertainty (computed)
    uncertainty: float = 0.0
    
    # Metadata
    symbol: str = ""
    timeframe: str = "H1"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model_version: str = "1.0.0"
    model_name: str = "ensemble"
    
    # Source tracking
    source_models: List[str] = field(default_factory=list)
    
    # Optional details
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Validate ranges
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be 0-1, got {self.probability}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if self.expected_rr < 0:
            raise ValueError(f"Expected R:R must be positive, got {self.expected_rr}")
        
        # Compute uncertainty
        self.uncertainty = 1 - self.confidence
        
        # Initialize source models if empty
        if not self.source_models:
            self.source_models = [self.model_name]
    
    @property
    def action(self) -> str:
        """Convert direction to action string for compatibility."""
        if self.direction == SignalDirection.LONG:
            return "BUY"
        elif self.direction == SignalDirection.SHORT:
            return "SELL"
        return "HOLD"
    
    @property
    def rr_ratio(self) -> float:
        """Alias for expected_rr."""
        return self.expected_rr
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not neutral with sufficient confidence)."""
        return self.direction != SignalDirection.NEUTRAL and self.probability >= 0.55
    
    @property
    def strength(self) -> float:
        """Combined signal strength (0-1)."""
        return self.probability * self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "direction": self.direction.value,
            "probability": round(self.probability, 4),
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "volatility_state": self.volatility_state.value,
            "regime": self.regime.value,
            "rr_ratio": round(self.expected_rr, 2),
            "htf_bias": self.htf_bias.value if isinstance(self.htf_bias, SignalDirection) else self.htf_bias,
            "ltf_bias": self.ltf_bias.value if isinstance(self.ltf_bias, SignalDirection) else self.ltf_bias,
            "timeframe_bias": self.timeframe_bias.value if isinstance(self.timeframe_bias, TimeframeBias) else self.timeframe_bias,
            "source_models": self.source_models,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "model_name": self.model_name,
            "strength": round(self.strength, 4),
            "is_actionable": self.is_actionable,
        }
    
    def to_legacy_signal(self) -> Signal:
        """Convert to legacy Signal format for backward compatibility."""
        return Signal(
            action=self.action,
            confidence=self.confidence,
            rr_ratio=self.expected_rr,
            regime=self.regime.value,
            uncertainty=self.uncertainty,
            htf_bias=self.htf_bias.value if isinstance(self.htf_bias, SignalDirection) else self.htf_bias,
            ltf_bias=self.ltf_bias.value if isinstance(self.ltf_bias, SignalDirection) else self.ltf_bias,
            timeframe_bias=self.timeframe_bias.value if isinstance(self.timeframe_bias, TimeframeBias) else self.timeframe_bias,
            source_models=self.source_models,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )
    
    @classmethod
    def neutral(cls, symbol: str, timeframe: str, regime: MarketRegime = MarketRegime.RANGING) -> "AISignal":
        """Create a neutral (no action) signal."""
        return cls(
            direction=SignalDirection.NEUTRAL,
            probability=0.5,
            confidence=0.5,
            volatility_state=VolatilityState.NORMAL,
            regime=regime,
            expected_rr=0.0,
            symbol=symbol,
            timeframe=timeframe,
        )
    
    @classmethod
    def from_prediction(cls, pred: Dict, symbol: str, timeframe: str, 
                       model_name: str, regime: MarketRegime = MarketRegime.RANGING) -> "AISignal":
        """Create AISignal from model prediction dict."""
        direction_map = {
            "LONG": SignalDirection.LONG,
            "SHORT": SignalDirection.SHORT,
            "NEUTRAL": SignalDirection.NEUTRAL,
            "BUY": SignalDirection.LONG,
            "SELL": SignalDirection.SHORT,
            "HOLD": SignalDirection.NEUTRAL,
            "LONG_ENTRY": SignalDirection.LONG,
            "SHORT_ENTRY": SignalDirection.SHORT,
            "NO_ENTRY": SignalDirection.NEUTRAL,
        }
        
        direction_str = pred.get("direction", pred.get("prediction", "NEUTRAL"))
        direction = direction_map.get(direction_str, SignalDirection.NEUTRAL)
        
        return cls(
            direction=direction,
            probability=pred.get("probability", 0.5),
            confidence=pred.get("confidence", 0.5),
            volatility_state=VolatilityState.NORMAL,
            regime=regime,
            expected_rr=pred.get("expected_rr", 1.5),
            symbol=symbol,
            timeframe=timeframe,
            model_name=model_name,
            source_models=[model_name],
        )
