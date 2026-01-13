"""
AI Trading System - Analytics Dashboard
=========================================
Performance analytics and monitoring dashboard.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class SymbolStats:
    """Per-symbol statistics."""
    symbol: str
    trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0


class AnalyticsDashboard:
    """
    Trading performance analytics.
    
    Tracks:
    - Daily P&L and win rate
    - Per-symbol performance
    - Model accuracy
    - Risk metrics
    """
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.daily_stats: Dict[str, DailyStats] = {}
        self.symbol_stats: Dict[str, SymbolStats] = {}
        self.model_predictions: List[Dict] = []
        self.equity_curve: List[float] = []
    
    def record_trade(self, trade: Dict):
        """Record completed trade."""
        self.trades.append(trade)
        
        # Update daily stats
        date_str = trade.get("exit_time", datetime.now()).strftime("%Y-%m-%d")
        if date_str not in self.daily_stats:
            self.daily_stats[date_str] = DailyStats(date=date_str)
        
        daily = self.daily_stats[date_str]
        daily.trades += 1
        pnl = trade.get("pnl", 0)
        daily.pnl += pnl
        
        if pnl > 0:
            daily.wins += 1
        else:
            daily.losses += 1
        
        # Update symbol stats
        symbol = trade.get("symbol", "UNKNOWN")
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = SymbolStats(symbol=symbol)
        
        sym = self.symbol_stats[symbol]
        sym.trades += 1
        sym.total_pnl += pnl
        sym.avg_pnl = sym.total_pnl / sym.trades
        sym.best_trade = max(sym.best_trade, pnl)
        sym.worst_trade = min(sym.worst_trade, pnl)
        sym.win_rate = sum(1 for t in self.trades if t.get("symbol") == symbol and t.get("pnl", 0) > 0) / sym.trades
    
    def record_prediction(self, symbol: str, predicted: str, actual: str, confidence: float):
        """Record model prediction for accuracy tracking."""
        self.model_predictions.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "predicted": predicted,
            "actual": actual,
            "confidence": confidence,
            "correct": predicted == actual
        })
    
    def update_equity(self, equity: float):
        """Update equity curve."""
        self.equity_curve.append(equity)
    
    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics."""
        if not self.trades:
            return {"message": "No trades recorded"}
        
        pnls = [t.get("pnl", 0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0,
            "total_pnl": sum(pnls),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses else float('inf'),
            "expectancy": np.mean(pnls) if pnls else 0,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0,
        }
    
    def get_model_accuracy(self) -> Dict:
        """Get model prediction accuracy."""
        if not self.model_predictions:
            return {"message": "No predictions recorded"}
        
        correct = sum(1 for p in self.model_predictions if p["correct"])
        total = len(self.model_predictions)
        
        by_symbol = {}
        for symbol in set(p["symbol"] for p in self.model_predictions):
            symbol_preds = [p for p in self.model_predictions if p["symbol"] == symbol]
            by_symbol[symbol] = sum(1 for p in symbol_preds if p["correct"]) / len(symbol_preds)
        
        return {
            "overall_accuracy": correct / total,
            "total_predictions": total,
            "correct_predictions": correct,
            "by_symbol": by_symbol,
        }
    
    def get_drawdown_stats(self) -> Dict:
        """Calculate drawdown statistics."""
        if len(self.equity_curve) < 2:
            return {"message": "Insufficient equity data"}
        
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        
        return {
            "current_drawdown_pct": float(drawdown[-1]),
            "max_drawdown_pct": float(np.max(drawdown)),
            "avg_drawdown_pct": float(np.mean(drawdown)),
        }
    
    def get_daily_summary(self, days: int = 7) -> List[Dict]:
        """Get last N days summary."""
        dates = sorted(self.daily_stats.keys(), reverse=True)[:days]
        return [
            {
                "date": d,
                "trades": self.daily_stats[d].trades,
                "wins": self.daily_stats[d].wins,
                "losses": self.daily_stats[d].losses,
                "pnl": round(self.daily_stats[d].pnl, 2),
                "win_rate": round(self.daily_stats[d].wins / self.daily_stats[d].trades * 100, 1) 
                           if self.daily_stats[d].trades > 0 else 0,
            }
            for d in dates
        ]
    
    def get_symbol_ranking(self) -> List[Dict]:
        """Get symbols ranked by profitability."""
        return sorted(
            [
                {
                    "symbol": s.symbol,
                    "trades": s.trades,
                    "win_rate": round(s.win_rate * 100, 1),
                    "total_pnl": round(s.total_pnl, 2),
                    "avg_pnl": round(s.avg_pnl, 2),
                }
                for s in self.symbol_stats.values()
            ],
            key=lambda x: x["total_pnl"],
            reverse=True
        )
    
    def generate_report(self) -> str:
        """Generate text performance report."""
        stats = self.get_overall_stats()
        dd = self.get_drawdown_stats()
        model = self.get_model_accuracy()
        
        report = []
        report.append("=" * 50)
        report.append("AI TRADING SYSTEM - PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("\n--- OVERALL PERFORMANCE ---")
        report.append(f"Total Trades: {stats.get('total_trades', 0)}")
        report.append(f"Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
        report.append(f"Total P&L: ${stats.get('total_pnl', 0):.2f}")
        report.append(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
        report.append(f"Expectancy: ${stats.get('expectancy', 0):.2f}")
        
        if "max_drawdown_pct" in dd:
            report.append(f"\n--- RISK METRICS ---")
            report.append(f"Max Drawdown: {dd['max_drawdown_pct']:.2f}%")
            report.append(f"Current Drawdown: {dd['current_drawdown_pct']:.2f}%")
        
        if "overall_accuracy" in model:
            report.append(f"\n--- MODEL ACCURACY ---")
            report.append(f"Overall: {model['overall_accuracy']*100:.1f}%")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)


# Global dashboard instance
_dashboard: Optional[AnalyticsDashboard] = None


def get_dashboard() -> AnalyticsDashboard:
    """Get global dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = AnalyticsDashboard()
    return _dashboard
