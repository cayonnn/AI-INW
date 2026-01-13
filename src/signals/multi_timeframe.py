"""
AI Trading System - Multi-Timeframe Analyzer
==============================================
Combines signals from multiple timeframes for confirmation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.models.signal_schema import AISignal, SignalDirection, MarketRegime
from src.features.trend_momentum import TrendMomentumFeatures
from src.features.regime_detector import RegimeDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimeframeAnalysis:
    """Analysis result for single timeframe."""
    timeframe: str
    trend_direction: str  # BULLISH, BEARISH, NEUTRAL
    trend_strength: float  # 0-1
    regime: MarketRegime
    key_levels: Dict[str, float]  # support, resistance


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes for confluence.
    
    Strategy:
    - Higher TF determines overall bias
    - Lower TF determines entry timing
    - All TFs must agree for high-confidence trades
    """
    
    TIMEFRAME_WEIGHTS = {
        "MN1": 1.0,
        "W1": 0.9,
        "D1": 0.8,
        "H4": 0.6,
        "H1": 0.4,
        "M15": 0.2,
        "M5": 0.1,
    }
    
    def __init__(self, timeframes: List[str] = None, min_confluence: float = 0.6):
        self.timeframes = timeframes or ["H4", "H1", "M15"]
        self.min_confluence = min_confluence
        self.tm_features = TrendMomentumFeatures()
        self.regime_detector = RegimeDetector()
    
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> TimeframeAnalysis:
        """Analyze single timeframe."""
        # Add trend features
        df_features = self.tm_features.add_ema(df)
        df_features = self.tm_features.add_adx(df_features)
        
        latest = df_features.iloc[-1]
        
        # Determine trend direction
        ema_8 = latest.get("ema_8", latest["close"])
        ema_21 = latest.get("ema_21", latest["close"])
        ema_55 = latest.get("ema_55", latest["close"])
        
        if ema_8 > ema_21 > ema_55:
            trend_direction = "BULLISH"
        elif ema_8 < ema_21 < ema_55:
            trend_direction = "BEARISH"
        else:
            trend_direction = "NEUTRAL"
        
        # Trend strength from ADX
        adx = latest.get("adx", 20)
        trend_strength = min(adx / 50, 1.0)
        
        # Regime
        regime, _ = self.regime_detector.detect_regime(df)
        
        # Key levels
        recent = df.tail(50)
        key_levels = {
            "resistance": recent["high"].max(),
            "support": recent["low"].min(),
            "current": latest["close"],
        }
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            regime=regime,
            key_levels=key_levels
        )
    
    def analyze_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, TimeframeAnalysis]:
        """Analyze all timeframes."""
        results = {}
        for tf, df in data.items():
            if tf in self.timeframes and df is not None and len(df) > 50:
                results[tf] = self.analyze_timeframe(df, tf)
        return results
    
    def get_confluence(self, analyses: Dict[str, TimeframeAnalysis]) -> Tuple[str, float]:
        """
        Calculate trend confluence across timeframes.
        
        Returns:
            Tuple of (direction, confluence_score)
        """
        if not analyses:
            return "NEUTRAL", 0.0
        
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        
        for tf, analysis in analyses.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.3) * analysis.trend_strength
            total_weight += weight
            
            if analysis.trend_direction == "BULLISH":
                bullish_score += weight
            elif analysis.trend_direction == "BEARISH":
                bearish_score += weight
        
        if total_weight == 0:
            return "NEUTRAL", 0.0
        
        bullish_pct = bullish_score / total_weight
        bearish_pct = bearish_score / total_weight
        
        if bullish_pct > bearish_pct and bullish_pct >= self.min_confluence:
            return "BULLISH", bullish_pct
        elif bearish_pct > bullish_pct and bearish_pct >= self.min_confluence:
            return "BEARISH", bearish_pct
        else:
            return "NEUTRAL", max(bullish_pct, bearish_pct)
    
    def filter_signal(self, signal: AISignal, analyses: Dict[str, TimeframeAnalysis]) -> Tuple[bool, str]:
        """
        Filter signal based on MTF confluence.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if signal.direction == SignalDirection.NEUTRAL:
            return True, "Neutral signal - no filter"
        
        confluence_dir, confluence_score = self.get_confluence(analyses)
        
        # Check alignment
        signal_dir = "BULLISH" if signal.direction == SignalDirection.LONG else "BEARISH"
        
        if confluence_dir == "NEUTRAL":
            return False, f"No clear MTF confluence ({confluence_score:.2%})"
        
        if signal_dir != confluence_dir:
            return False, f"Signal conflicts with MTF bias ({confluence_dir})"
        
        if confluence_score < self.min_confluence:
            return False, f"Insufficient confluence ({confluence_score:.2%} < {self.min_confluence:.2%})"
        
        return True, f"MTF aligned: {confluence_dir} ({confluence_score:.2%})"
