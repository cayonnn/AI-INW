# src/dashboard/dashboard_api.py
"""
Dashboard API - PRODUCTION READY
=================================

FastAPI-style Dashboard API with:
- Live trades endpoint
- Model health endpoint
- Explainability payload
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("DASHBOARD_API")


@dataclass
class TradeRecord:
    """Trade record for database."""
    id: int
    symbol: str
    side: str
    lot: float
    sl: float
    tp: float
    confidence: float
    model_version: str
    profit: float
    opened_at: str
    closed_at: Optional[str] = None


@dataclass
class ModelMetric:
    """Model metrics record."""
    model_version: str
    winrate: float
    drawdown: float
    sharpe: float
    trained_at: str


@dataclass
class ExplainabilityPayload:
    """Explainability data for a signal."""
    signal: str
    confidence: float
    top_features: Dict[str, float]
    model_version: str
    timestamp: str


class DashboardAPI:
    """
    Dashboard API (FastAPI-compatible).
    
    Can be used standalone or wrapped in FastAPI.
    """

    def __init__(self):
        self._trades: Dict[int, TradeRecord] = {}
        self._closed_trades: List[TradeRecord] = []
        self._model_metrics: List[ModelMetric] = []
        self._explanations: List[ExplainabilityPayload] = []
        self._next_id = 1
        
        logger.info("DashboardAPI initialized")

    # =========================================================
    # TRADE ENDPOINTS
    # =========================================================
    
    def add_trade(
        self,
        symbol: str,
        side: str,
        lot: float,
        sl: float,
        tp: float,
        confidence: float,
        model_version: str
    ) -> int:
        """Add a new trade."""
        trade_id = self._next_id
        self._next_id += 1
        
        trade = TradeRecord(
            id=trade_id,
            symbol=symbol,
            side=side,
            lot=lot,
            sl=sl,
            tp=tp,
            confidence=confidence,
            model_version=model_version,
            profit=0.0,
            opened_at=datetime.now().isoformat()
        )
        
        self._trades[trade_id] = trade
        return trade_id

    def close_trade(self, trade_id: int, profit: float):
        """Close a trade."""
        if trade_id in self._trades:
            trade = self._trades.pop(trade_id)
            trade.profit = profit
            trade.closed_at = datetime.now().isoformat()
            self._closed_trades.append(trade)

    def update_profit(self, trade_id: int, profit: float):
        """Update trade profit."""
        if trade_id in self._trades:
            self._trades[trade_id].profit = profit

    def get_live_trades(self) -> List[Dict]:
        """GET /dashboard/live"""
        return [asdict(t) for t in self._trades.values()]

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """GET /dashboard/history"""
        return [asdict(t) for t in self._closed_trades[-limit:]]

    # =========================================================
    # MODEL HEALTH ENDPOINTS
    # =========================================================
    
    def add_model_metric(
        self,
        model_version: str,
        winrate: float,
        drawdown: float,
        sharpe: float
    ):
        """Add model metrics."""
        metric = ModelMetric(
            model_version=model_version,
            winrate=winrate,
            drawdown=drawdown,
            sharpe=sharpe,
            trained_at=datetime.now().isoformat()
        )
        self._model_metrics.append(metric)

    def get_model_health(self, limit: int = 5) -> List[Dict]:
        """GET /dashboard/model-health"""
        return [asdict(m) for m in self._model_metrics[-limit:]]

    def get_latest_model(self) -> Optional[Dict]:
        """GET /dashboard/model-latest"""
        if self._model_metrics:
            return asdict(self._model_metrics[-1])
        return None

    # =========================================================
    # EXPLAINABILITY ENDPOINTS
    # =========================================================
    
    def log_explanation(
        self,
        signal: str,
        confidence: float,
        top_features: Dict[str, float],
        model_version: str
    ):
        """Log explainability data."""
        payload = ExplainabilityPayload(
            signal=signal,
            confidence=confidence,
            top_features=top_features,
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )
        self._explanations.append(payload)
        
        # Keep last 100
        if len(self._explanations) > 100:
            self._explanations = self._explanations[-100:]

    def get_latest_explanation(self) -> Optional[Dict]:
        """GET /dashboard/explain"""
        if self._explanations:
            return asdict(self._explanations[-1])
        return None

    def get_explanations(self, limit: int = 10) -> List[Dict]:
        """GET /dashboard/explain/history"""
        return [asdict(e) for e in self._explanations[-limit:]]

    # =========================================================
    # FULL DASHBOARD
    # =========================================================
    
    def get_full_dashboard(self) -> Dict:
        """GET /dashboard/full"""
        return {
            "timestamp": datetime.now().isoformat(),
            "live_trades": self.get_live_trades(),
            "open_positions": len(self._trades),
            "model_health": self.get_model_health(),
            "latest_explanation": self.get_latest_explanation(),
            "stats": {
                "total_trades": len(self._closed_trades) + len(self._trades),
                "closed_trades": len(self._closed_trades),
                "open_trades": len(self._trades)
            }
        }

    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps(self.get_full_dashboard(), indent=2, default=str)


# =========================================================
# FASTAPI INTEGRATION (Optional)
# =========================================================

def create_fastapi_app():
    """
    Create FastAPI app with dashboard routes.
    
    Usage:
        from src.dashboard.dashboard_api import create_fastapi_app
        app = create_fastapi_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    try:
        from fastapi import FastAPI
        
        app = FastAPI(title="Trading Dashboard API")
        dashboard = DashboardAPI()
        
        @app.get("/dashboard/live")
        def live_trades():
            return dashboard.get_live_trades()
        
        @app.get("/dashboard/model-health")
        def model_health():
            return dashboard.get_model_health()
        
        @app.get("/dashboard/explain")
        def explain():
            return dashboard.get_latest_explanation()
        
        @app.get("/dashboard/full")
        def full():
            return dashboard.get_full_dashboard()
        
        return app, dashboard
        
    except ImportError:
        logger.warning("FastAPI not installed - API endpoints unavailable")
        return None, DashboardAPI()
