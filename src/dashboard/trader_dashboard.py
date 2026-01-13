# src/dashboard/trader_dashboard.py
"""
Trader Dashboard - Fund-Grade Explainability
==============================================

เป้าหมาย: รู้ว่า "ทำไมบอทเข้าออเดอร์" ไม่ใช่แค่ดูผล

Sections:
1. Live Trades - Symbol, Lot, SL/TP, Model version, Confidence
2. Explainability Panel - Feature impact (Top 5), Signal sources
3. Risk Monitor - Daily DD %, Exposure heatmap, Margin safety
4. Model Health - Accuracy (7/30 days), Drift detection, Last retrain
"""

import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("DASHBOARD")


@dataclass
class LiveTrade:
    """Live trade information."""
    ticket: int
    symbol: str
    direction: str
    lot: float
    entry_price: float
    sl: float
    tp: float
    profit: float
    model_version: str
    confidence: float
    open_time: str


@dataclass
class FeatureImpact:
    """Feature impact on decision."""
    feature: str
    value: float
    impact: float  # -1 to 1
    contribution: str  # "bullish", "bearish", "neutral"


@dataclass
class SignalSource:
    """Signal source breakdown."""
    source: str  # RSI, BB, EMA, Momentum
    value: float
    signal: str  # BUY, SELL, NEUTRAL


@dataclass
class RiskMetrics:
    """Risk monitoring metrics."""
    daily_dd_pct: float
    weekly_dd_pct: float
    exposure_by_symbol: Dict[str, float]
    margin_used_pct: float
    margin_safety: str  # SAFE, WARNING, DANGER


@dataclass
class ModelHealth:
    """Model health metrics."""
    accuracy_7d: float
    accuracy_30d: float
    drift_detected: bool
    drift_score: float
    last_retrain: str
    model_version: str
    predictions_today: int


class TraderDashboard:
    """
    Fund-Grade Trader Dashboard.
    
    Provides:
    - Live trade monitoring
    - Decision explainability
    - Risk monitoring
    - Model health tracking
    """

    def __init__(self):
        self._live_trades: Dict[int, LiveTrade] = {}
        self._trade_history: List[Dict] = []
        self._decision_log: List[Dict] = []
        self._model_predictions: List[Dict] = []
        
        # Cache for performance
        self._cached_metrics: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 60  # seconds
        
        logger.info("TraderDashboard initialized")

    # =========================================================
    # SECTION 1: LIVE TRADES
    # =========================================================
    
    def register_trade(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        lot: float,
        entry_price: float,
        sl: float,
        tp: float,
        model_version: str = "unknown",
        confidence: float = 0.0
    ):
        """Register a new live trade."""
        trade = LiveTrade(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            lot=lot,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            profit=0.0,
            model_version=model_version,
            confidence=confidence,
            open_time=datetime.now().isoformat()
        )
        self._live_trades[ticket] = trade
        logger.debug(f"Registered trade #{ticket}")

    def update_trade_profit(self, ticket: int, profit: float):
        """Update trade profit."""
        if ticket in self._live_trades:
            self._live_trades[ticket].profit = profit

    def close_trade(self, ticket: int, profit: float):
        """Close a trade and move to history."""
        if ticket in self._live_trades:
            trade = self._live_trades.pop(ticket)
            trade.profit = profit
            self._trade_history.append({
                **asdict(trade),
                "close_time": datetime.now().isoformat()
            })

    def get_live_trades(self) -> List[Dict]:
        """Get all live trades."""
        return [asdict(t) for t in self._live_trades.values()]

    # =========================================================
    # SECTION 2: EXPLAINABILITY
    # =========================================================
    
    def log_decision(
        self,
        symbol: str,
        action: str,
        features: Dict[str, float],
        feature_importance: Dict[str, float],
        signal_sources: List[SignalSource],
        confidence: float,
        reason: str
    ):
        """Log a trading decision with full explainability."""
        # Calculate feature impacts
        impacts = []
        for feat, value in features.items():
            importance = feature_importance.get(feat, 0)
            
            # Determine contribution direction
            if feat in ['rsi14']:
                contrib = "bullish" if value < 30 else "bearish" if value > 70 else "neutral"
            elif feat in ['ema_spread']:
                contrib = "bullish" if value > 0 else "bearish" if value < 0 else "neutral"
            else:
                contrib = "neutral"
            
            impacts.append(FeatureImpact(
                feature=feat,
                value=value,
                impact=importance,
                contribution=contrib
            ))
        
        # Sort by impact
        impacts.sort(key=lambda x: abs(x.impact), reverse=True)
        
        decision = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "feature_impacts": [asdict(i) for i in impacts[:5]],
            "signal_sources": [asdict(s) for s in signal_sources]
        }
        
        self._decision_log.append(decision)
        
        # Keep only last 100 decisions
        if len(self._decision_log) > 100:
            self._decision_log = self._decision_log[-100:]

    def get_latest_explanation(self) -> Optional[Dict]:
        """Get the latest decision explanation."""
        if self._decision_log:
            return self._decision_log[-1]
        return None

    def get_feature_impact_summary(self, n: int = 5) -> List[Dict]:
        """Get summary of top feature impacts."""
        if not self._decision_log:
            return []
        
        # Aggregate impacts
        impact_sum = defaultdict(float)
        impact_count = defaultdict(int)
        
        for decision in self._decision_log[-20:]:  # Last 20 decisions
            for impact in decision.get("feature_impacts", []):
                feat = impact["feature"]
                impact_sum[feat] += abs(impact["impact"])
                impact_count[feat] += 1
        
        # Calculate averages
        avg_impacts = []
        for feat in impact_sum:
            avg_impacts.append({
                "feature": feat,
                "avg_impact": impact_sum[feat] / impact_count[feat]
            })
        
        avg_impacts.sort(key=lambda x: x["avg_impact"], reverse=True)
        return avg_impacts[:n]

    # =========================================================
    # SECTION 3: RISK MONITOR
    # =========================================================
    
    def calculate_risk_metrics(
        self,
        equity: float,
        balance: float,
        margin_used: float,
        start_equity_today: float,
        start_equity_week: float
    ) -> RiskMetrics:
        """Calculate current risk metrics."""
        # Daily drawdown
        if start_equity_today > 0:
            daily_dd = (start_equity_today - equity) / start_equity_today
        else:
            daily_dd = 0.0
        
        # Weekly drawdown
        if start_equity_week > 0:
            weekly_dd = (start_equity_week - equity) / start_equity_week
        else:
            weekly_dd = 0.0
        
        # Exposure by symbol
        exposure = {}
        for trade in self._live_trades.values():
            if trade.symbol not in exposure:
                exposure[trade.symbol] = 0.0
            exposure[trade.symbol] += trade.lot * trade.entry_price
        
        # Margin safety
        margin_pct = (margin_used / equity) * 100 if equity > 0 else 0
        if margin_pct < 30:
            safety = "SAFE"
        elif margin_pct < 60:
            safety = "WARNING"
        else:
            safety = "DANGER"
        
        return RiskMetrics(
            daily_dd_pct=daily_dd * 100,
            weekly_dd_pct=weekly_dd * 100,
            exposure_by_symbol=exposure,
            margin_used_pct=margin_pct,
            margin_safety=safety
        )

    def get_exposure_heatmap(self) -> Dict[str, float]:
        """Get exposure heatmap by symbol."""
        exposure = defaultdict(float)
        for trade in self._live_trades.values():
            exposure[trade.symbol] += trade.lot
        return dict(exposure)

    # =========================================================
    # SECTION 4: MODEL HEALTH
    # =========================================================
    
    def log_prediction(
        self,
        prediction: str,
        actual: Optional[str],
        model_version: str
    ):
        """Log a model prediction for accuracy tracking."""
        self._model_predictions.append({
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "actual": actual,
            "model_version": model_version,
            "correct": prediction == actual if actual else None
        })
        
        # Keep only last 1000
        if len(self._model_predictions) > 1000:
            self._model_predictions = self._model_predictions[-1000:]

    def calculate_model_health(self, model_version: str, last_retrain: str) -> ModelHealth:
        """Calculate model health metrics."""
        now = datetime.now()
        
        # Filter predictions by time
        preds_7d = [p for p in self._model_predictions 
                    if p.get("correct") is not None and 
                    datetime.fromisoformat(p["timestamp"]) > now - timedelta(days=7)]
        
        preds_30d = [p for p in self._model_predictions 
                     if p.get("correct") is not None and 
                     datetime.fromisoformat(p["timestamp"]) > now - timedelta(days=30)]
        
        # Calculate accuracy
        acc_7d = sum(1 for p in preds_7d if p["correct"]) / len(preds_7d) if preds_7d else 0
        acc_30d = sum(1 for p in preds_30d if p["correct"]) / len(preds_30d) if preds_30d else 0
        
        # Drift detection (compare 7d vs 30d accuracy)
        drift_score = acc_30d - acc_7d if acc_30d > 0 else 0
        drift_detected = drift_score > 0.1  # >10% drop
        
        # Predictions today
        today = now.date()
        preds_today = len([p for p in self._model_predictions 
                          if datetime.fromisoformat(p["timestamp"]).date() == today])
        
        return ModelHealth(
            accuracy_7d=acc_7d * 100,
            accuracy_30d=acc_30d * 100,
            drift_detected=drift_detected,
            drift_score=drift_score * 100,
            last_retrain=last_retrain,
            model_version=model_version,
            predictions_today=preds_today
        )

    # =========================================================
    # FULL DASHBOARD
    # =========================================================
    
    def get_full_dashboard(
        self,
        equity: float = 0,
        balance: float = 0,
        margin_used: float = 0,
        start_equity_today: float = 0,
        start_equity_week: float = 0,
        model_version: str = "unknown",
        last_retrain: str = "unknown"
    ) -> Dict:
        """Get complete dashboard data."""
        return {
            "timestamp": datetime.now().isoformat(),
            
            # Section 1: Live Trades
            "live_trades": self.get_live_trades(),
            "total_positions": len(self._live_trades),
            
            # Section 2: Explainability
            "latest_decision": self.get_latest_explanation(),
            "top_features": self.get_feature_impact_summary(5),
            
            # Section 3: Risk Monitor
            "risk": asdict(self.calculate_risk_metrics(
                equity, balance, margin_used,
                start_equity_today, start_equity_week
            )),
            "exposure_heatmap": self.get_exposure_heatmap(),
            
            # Section 4: Model Health
            "model_health": asdict(self.calculate_model_health(
                model_version, last_retrain
            ))
        }

    def to_json(self, **kwargs) -> str:
        """Export dashboard as JSON."""
        return json.dumps(self.get_full_dashboard(**kwargs), indent=2, default=str)
