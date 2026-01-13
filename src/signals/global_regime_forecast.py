"""
global_regime_forecast.py
===========================
Global Market Regime Forecast

รู้ก่อนว่าโลกกำลังจะ "เปลี่ยนโหมด"

Regime ไม่เปลี่ยนทันที
แต่คนที่ "รอ confirmation" จะสายเสมอ

Usage:
- Pre-empt crisis mode
- Reduce leverage early
- Rotate strategy pool
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
from src.utils.logger import get_logger

logger = get_logger("GLOBAL_REGIME")


class GlobalRegime(str, Enum):
    """Global market regime states."""
    RISK_ON = "RISK_ON"           # Bull mode, aggressive
    NEUTRAL = "NEUTRAL"           # Balanced
    RISK_OFF = "RISK_OFF"         # Defensive
    CRISIS = "CRISIS"             # Capital preservation only


class RegimeConfidence(str, Enum):
    """Confidence in regime forecast."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


@dataclass
class MacroIndicators:
    """Macro economic indicators."""
    # Interest rates
    fed_funds_rate: float = 0.0
    rate_change_3m: float = 0.0
    
    # Yield curve
    yield_curve_slope: float = 0.0   # 10Y - 2Y
    yield_curve_inverted: bool = False
    
    # Inflation
    inflation_yoy: float = 0.0
    inflation_surprise: float = 0.0  # Actual - Expected
    
    # Central bank
    cb_hawkish_score: float = 0.5    # 0 = dovish, 1 = hawkish


@dataclass
class MarketIndicators:
    """Market-based indicators."""
    # Volatility
    vix_level: float = 20.0
    vix_term_structure: float = 0.0  # Contango = positive
    realized_vol_20d: float = 0.0
    
    # Correlation
    cross_asset_correlation: float = 0.0
    equity_fx_correlation: float = 0.0
    
    # Risk appetite
    high_yield_spread: float = 0.0
    em_spread: float = 0.0
    
    # Flows
    fx_risk_on_score: float = 0.5    # 0 = risk-off, 1 = risk-on
    equity_breadth: float = 0.5      # Market breadth
    
    # Liquidity
    bid_ask_spread_avg: float = 0.0
    market_depth_score: float = 1.0


@dataclass
class RegimeForecast:
    """Regime forecast output."""
    regime: GlobalRegime
    confidence: float                 # 0-1
    confidence_level: RegimeConfidence
    expected_duration_weeks: Tuple[int, int]  # Range
    transition_probability: Dict[GlobalRegime, float]
    key_drivers: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class GlobalRegimeForecaster:
    """
    Global Market Regime Forecaster.
    
    Uses HMM-inspired logic + Bayesian updates to forecast regime shifts.
    """

    def __init__(self):
        self.current_regime = GlobalRegime.NEUTRAL
        self.regime_start = datetime.now()
        self.forecast_history: List[RegimeForecast] = []
        
        # Regime transition probabilities (simplified HMM)
        self.transition_probs = {
            GlobalRegime.RISK_ON: {
                GlobalRegime.RISK_ON: 0.85,
                GlobalRegime.NEUTRAL: 0.12,
                GlobalRegime.RISK_OFF: 0.02,
                GlobalRegime.CRISIS: 0.01,
            },
            GlobalRegime.NEUTRAL: {
                GlobalRegime.RISK_ON: 0.30,
                GlobalRegime.NEUTRAL: 0.50,
                GlobalRegime.RISK_OFF: 0.15,
                GlobalRegime.CRISIS: 0.05,
            },
            GlobalRegime.RISK_OFF: {
                GlobalRegime.RISK_ON: 0.10,
                GlobalRegime.NEUTRAL: 0.25,
                GlobalRegime.RISK_OFF: 0.55,
                GlobalRegime.CRISIS: 0.10,
            },
            GlobalRegime.CRISIS: {
                GlobalRegime.RISK_ON: 0.05,
                GlobalRegime.NEUTRAL: 0.15,
                GlobalRegime.RISK_OFF: 0.40,
                GlobalRegime.CRISIS: 0.40,
            },
        }
        
        # Thresholds
        self.vix_thresholds = {
            "low": 15,
            "normal": 20,
            "elevated": 30,
            "crisis": 40,
        }
        
        # Weights for scoring
        self.weights = {
            "vix": 0.25,
            "correlation": 0.20,
            "yield_curve": 0.15,
            "spreads": 0.15,
            "flows": 0.15,
            "cb_tone": 0.10,
        }

    # -------------------------------------------------
    # Main forecast
    # -------------------------------------------------
    def forecast(self, macro: MacroIndicators, 
                market: MarketIndicators) -> RegimeForecast:
        """
        Generate regime forecast.
        
        Args:
            macro: Macro economic indicators
            market: Market-based indicators
            
        Returns:
            RegimeForecast with regime and confidence
        """
        # Calculate regime scores
        scores = self._calculate_regime_scores(macro, market)
        
        # Determine most likely regime
        regime = self._determine_regime(scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(scores, regime)
        conf_level = self._confidence_to_level(confidence)
        
        # Get transition probabilities
        transitions = self._get_transitions(regime, scores)
        
        # Expected duration
        duration = self._estimate_duration(regime, confidence)
        
        # Key drivers
        drivers = self._identify_drivers(macro, market, regime)
        
        # Update current regime if high confidence
        if confidence > 0.7 and regime != self.current_regime:
            old_regime = self.current_regime
            self.current_regime = regime
            self.regime_start = datetime.now()
            logger.warning(f"Regime shift: {old_regime.value} → {regime.value}")
        
        forecast = RegimeForecast(
            regime=regime,
            confidence=confidence,
            confidence_level=conf_level,
            expected_duration_weeks=duration,
            transition_probability=transitions,
            key_drivers=drivers,
        )
        
        self.forecast_history.append(forecast)
        
        logger.info(f"Regime forecast: {regime.value} ({confidence:.0%})")
        
        return forecast

    # -------------------------------------------------
    # Scoring
    # -------------------------------------------------
    def _calculate_regime_scores(self, macro: MacroIndicators,
                                market: MarketIndicators) -> Dict[GlobalRegime, float]:
        """Calculate score for each regime."""
        scores = {r: 0.0 for r in GlobalRegime}
        
        # VIX-based scoring
        vix = market.vix_level
        if vix < self.vix_thresholds["low"]:
            scores[GlobalRegime.RISK_ON] += 1.0 * self.weights["vix"]
        elif vix < self.vix_thresholds["normal"]:
            scores[GlobalRegime.NEUTRAL] += 1.0 * self.weights["vix"]
        elif vix < self.vix_thresholds["elevated"]:
            scores[GlobalRegime.RISK_OFF] += 1.0 * self.weights["vix"]
        else:
            scores[GlobalRegime.CRISIS] += 1.0 * self.weights["vix"]
        
        # Correlation scoring
        corr = market.cross_asset_correlation
        if corr > 0.8:
            scores[GlobalRegime.CRISIS] += 1.0 * self.weights["correlation"]
        elif corr > 0.6:
            scores[GlobalRegime.RISK_OFF] += 1.0 * self.weights["correlation"]
        elif corr > 0.3:
            scores[GlobalRegime.NEUTRAL] += 1.0 * self.weights["correlation"]
        else:
            scores[GlobalRegime.RISK_ON] += 1.0 * self.weights["correlation"]
        
        # Yield curve scoring
        if macro.yield_curve_inverted:
            scores[GlobalRegime.RISK_OFF] += 1.0 * self.weights["yield_curve"]
            scores[GlobalRegime.CRISIS] += 0.5 * self.weights["yield_curve"]
        elif macro.yield_curve_slope > 1.0:
            scores[GlobalRegime.RISK_ON] += 1.0 * self.weights["yield_curve"]
        else:
            scores[GlobalRegime.NEUTRAL] += 1.0 * self.weights["yield_curve"]
        
        # Spreads scoring
        spread_score = (market.high_yield_spread + market.em_spread) / 10
        if spread_score > 0.8:
            scores[GlobalRegime.CRISIS] += 1.0 * self.weights["spreads"]
        elif spread_score > 0.5:
            scores[GlobalRegime.RISK_OFF] += 1.0 * self.weights["spreads"]
        else:
            scores[GlobalRegime.RISK_ON] += 1.0 * self.weights["spreads"]
        
        # Risk flows scoring
        fx_risk = market.fx_risk_on_score
        if fx_risk > 0.7:
            scores[GlobalRegime.RISK_ON] += 1.0 * self.weights["flows"]
        elif fx_risk > 0.4:
            scores[GlobalRegime.NEUTRAL] += 1.0 * self.weights["flows"]
        else:
            scores[GlobalRegime.RISK_OFF] += 1.0 * self.weights["flows"]
        
        # CB tone scoring
        if macro.cb_hawkish_score > 0.7:
            scores[GlobalRegime.RISK_OFF] += 0.5 * self.weights["cb_tone"]
        elif macro.cb_hawkish_score < 0.3:
            scores[GlobalRegime.RISK_ON] += 0.5 * self.weights["cb_tone"]
        
        return scores

    def _determine_regime(self, scores: Dict[GlobalRegime, float]) -> GlobalRegime:
        """Determine most likely regime."""
        return max(scores, key=scores.get)

    def _calculate_confidence(self, scores: Dict[GlobalRegime, float],
                             regime: GlobalRegime) -> float:
        """Calculate confidence in forecast."""
        total = sum(scores.values())
        if total == 0:
            return 0.5
        
        # Confidence based on how much higher top score is
        top_score = scores[regime]
        sorted_scores = sorted(scores.values(), reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        
        # Gap between top and second
        gap = top_score - second_score
        
        # Normalize to 0-1
        confidence = min(1.0, 0.5 + gap)
        
        return confidence

    def _confidence_to_level(self, confidence: float) -> RegimeConfidence:
        """Convert confidence to level."""
        if confidence >= 0.85:
            return RegimeConfidence.VERY_HIGH
        elif confidence >= 0.7:
            return RegimeConfidence.HIGH
        elif confidence >= 0.5:
            return RegimeConfidence.MEDIUM
        return RegimeConfidence.LOW

    def _get_transitions(self, regime: GlobalRegime,
                        scores: Dict[GlobalRegime, float]) -> Dict[GlobalRegime, float]:
        """Get transition probabilities adjusted by scores."""
        base_probs = self.transition_probs.get(regime, {})
        
        # Adjust based on current scores
        adjusted = {}
        total_score = sum(scores.values())
        
        for r in GlobalRegime:
            base = base_probs.get(r, 0.1)
            score_weight = scores.get(r, 0) / total_score if total_score > 0 else 0.25
            adjusted[r] = base * 0.7 + score_weight * 0.3
        
        # Normalize
        total = sum(adjusted.values())
        return {r: p / total for r, p in adjusted.items()}

    def _estimate_duration(self, regime: GlobalRegime,
                          confidence: float) -> Tuple[int, int]:
        """Estimate regime duration in weeks."""
        # Base durations
        base = {
            GlobalRegime.RISK_ON: (8, 24),
            GlobalRegime.NEUTRAL: (4, 12),
            GlobalRegime.RISK_OFF: (4, 16),
            GlobalRegime.CRISIS: (2, 8),
        }
        
        min_weeks, max_weeks = base.get(regime, (4, 12))
        
        # Adjust by confidence
        if confidence > 0.8:
            # High confidence = longer expected duration
            min_weeks = int(min_weeks * 1.2)
            max_weeks = int(max_weeks * 1.2)
        
        return (min_weeks, max_weeks)

    def _identify_drivers(self, macro: MacroIndicators,
                         market: MarketIndicators,
                         regime: GlobalRegime) -> List[str]:
        """Identify key drivers for the regime."""
        drivers = []
        
        if market.vix_level > 30:
            drivers.append(f"VIX elevated at {market.vix_level:.1f}")
        
        if market.cross_asset_correlation > 0.7:
            drivers.append(f"High cross-asset correlation ({market.cross_asset_correlation:.2f})")
        
        if macro.yield_curve_inverted:
            drivers.append("Yield curve inverted")
        
        if macro.inflation_surprise > 0.5:
            drivers.append("Inflation surprise to upside")
        
        if market.fx_risk_on_score < 0.3:
            drivers.append("FX flows showing risk-off")
        
        if market.market_depth_score < 0.5:
            drivers.append("Low market liquidity")
        
        return drivers[:5]  # Top 5

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def get_current_regime(self) -> Tuple[GlobalRegime, float]:
        """Get current regime and days in regime."""
        days = (datetime.now() - self.regime_start).days
        return self.current_regime, days

    def get_status(self) -> Dict:
        """Get forecaster status."""
        days_in_regime = (datetime.now() - self.regime_start).days
        
        return {
            "current_regime": self.current_regime.value,
            "days_in_regime": days_in_regime,
            "forecast_count": len(self.forecast_history),
            "last_forecast": self.forecast_history[-1].regime.value if self.forecast_history else None,
        }
