# src/retrain/data_collector.py
"""
Retrain Data Collector - Competition Grade
==========================================

Collects trading data for daily retraining:
- Live trades from trade_history.db
- Shadow trades from shadow simulation
- Regime statistics
- Score timeline

All data is normalized for meta-parameter optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import os

from src.utils.logger import get_logger

logger = get_logger("DATA_COLLECTOR")


@dataclass
class TradeRecord:
    """Single trade record for retraining."""
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    entry_time: datetime
    exit_time: datetime
    regime: str
    mode: str
    confidence: float
    streak_level: int
    pyramid_entry: int


@dataclass
class DailyData:
    """Collected data for one day."""
    date: str
    trades: List[TradeRecord] = field(default_factory=list)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    mode_distribution: Dict[str, int] = field(default_factory=dict)
    score_timeline: List[Dict] = field(default_factory=list)
    starting_equity: float = 0
    ending_equity: float = 0
    max_drawdown: float = 0
    total_pnl: float = 0
    win_count: int = 0
    loss_count: int = 0
    
    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0
    
    @property
    def profit_pct(self) -> float:
        if self.starting_equity == 0:
            return 0
        return ((self.ending_equity - self.starting_equity) / 
                self.starting_equity) * 100


class RetrainDataCollector:
    """
    Collects and stores trading data for retraining.
    
    Data sources:
    - Live trades (from MT5 history)
    - Shadow trades (from shadow simulator)
    - Regime logs
    - Score logs
    """
    
    def __init__(self, data_dir: str = "data/retrain"):
        """
        Initialize Data Collector.
        
        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.current_day_data: Optional[DailyData] = None
        self.trade_buffer: List[TradeRecord] = []
        
        logger.info(f"RetrainDataCollector initialized: {data_dir}")
    
    def start_day(self, date: str = None) -> None:
        """Start collecting for a new day."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self.current_day_data = DailyData(date=date)
        self.trade_buffer = []
        
        logger.info(f"Started collecting data for {date}")
    
    def record_trade(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        volume: float,
        pnl: float,
        entry_time: datetime,
        exit_time: datetime,
        regime: str = "NEUTRAL",
        mode: str = "NEUTRAL",
        confidence: float = 0.5,
        streak_level: int = 0,
        pyramid_entry: int = 0
    ) -> None:
        """Record a completed trade."""
        record = TradeRecord(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            volume=volume,
            pnl=pnl,
            entry_time=entry_time,
            exit_time=exit_time,
            regime=regime,
            mode=mode,
            confidence=confidence,
            streak_level=streak_level,
            pyramid_entry=pyramid_entry
        )
        
        self.trade_buffer.append(record)
        
        if self.current_day_data:
            self.current_day_data.trades.append(record)
            self.current_day_data.total_pnl += pnl
            
            if pnl > 0:
                self.current_day_data.win_count += 1
            else:
                self.current_day_data.loss_count += 1
            
            # Update distributions
            self.current_day_data.regime_distribution[regime] = \
                self.current_day_data.regime_distribution.get(regime, 0) + 1
            self.current_day_data.mode_distribution[mode] = \
                self.current_day_data.mode_distribution.get(mode, 0) + 1
    
    def record_score(self, score: float, timestamp: datetime = None) -> None:
        """Record score snapshot."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.current_day_data:
            self.current_day_data.score_timeline.append({
                "timestamp": timestamp.isoformat(),
                "score": score
            })
    
    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        if self.current_day_data:
            if self.current_day_data.starting_equity == 0:
                self.current_day_data.starting_equity = equity
            self.current_day_data.ending_equity = equity
    
    def update_drawdown(self, drawdown: float) -> None:
        """Update max drawdown."""
        if self.current_day_data:
            self.current_day_data.max_drawdown = max(
                self.current_day_data.max_drawdown, drawdown
            )
    
    def end_day(self) -> Optional[DailyData]:
        """
        End day collection and save data.
        
        Returns:
            Collected day data
        """
        if not self.current_day_data:
            return None
        
        data = self.current_day_data
        
        # Save to file
        filename = f"{self.data_dir}/daily_{data.date}.json"
        self._save_to_file(data, filename)
        
        logger.info(
            f"Day ended: {data.date} | "
            f"Trades: {len(data.trades)} | "
            f"PnL: ${data.total_pnl:.2f} | "
            f"WinRate: {data.win_rate:.1%}"
        )
        
        self.current_day_data = None
        return data
    
    def _save_to_file(self, data: DailyData, filename: str) -> None:
        """Save daily data to JSON file."""
        output = {
            "date": data.date,
            "starting_equity": data.starting_equity,
            "ending_equity": data.ending_equity,
            "total_pnl": data.total_pnl,
            "max_drawdown": data.max_drawdown,
            "win_count": data.win_count,
            "loss_count": data.loss_count,
            "win_rate": data.win_rate,
            "profit_pct": data.profit_pct,
            "regime_distribution": data.regime_distribution,
            "mode_distribution": data.mode_distribution,
            "score_timeline": data.score_timeline,
            "trades": [
                {
                    "ticket": t.ticket,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "volume": t.volume,
                    "pnl": t.pnl,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "regime": t.regime,
                    "mode": t.mode,
                    "confidence": t.confidence,
                    "streak_level": t.streak_level,
                    "pyramid_entry": t.pyramid_entry,
                }
                for t in data.trades
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
    
    def load_day(self, date: str) -> Optional[Dict]:
        """Load data for a specific day."""
        filename = f"{self.data_dir}/daily_{date}.json"
        
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'r') as f:
            return json.load(f)
    
    def load_recent_days(self, n_days: int = 7) -> List[Dict]:
        """Load data for recent N days."""
        results = []
        
        for i in range(n_days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            data = self.load_day(date)
            if data:
                results.append(data)
        
        return results
    
    def get_aggregate_stats(self, n_days: int = 7) -> Dict[str, Any]:
        """Get aggregate statistics for recent days."""
        days = self.load_recent_days(n_days)
        
        if not days:
            return {}
        
        total_trades = sum(d.get("win_count", 0) + d.get("loss_count", 0) for d in days)
        total_wins = sum(d.get("win_count", 0) for d in days)
        total_pnl = sum(d.get("total_pnl", 0) for d in days)
        max_dd = max(d.get("max_drawdown", 0) for d in days)
        
        # Regime stats
        regime_stats = {}
        mode_stats = {}
        
        for d in days:
            for regime, count in d.get("regime_distribution", {}).items():
                regime_stats[regime] = regime_stats.get(regime, 0) + count
            for mode, count in d.get("mode_distribution", {}).items():
                mode_stats[mode] = mode_stats.get(mode, 0) + count
        
        return {
            "days_analyzed": len(days),
            "total_trades": total_trades,
            "total_wins": total_wins,
            "win_rate": total_wins / total_trades if total_trades > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "regime_distribution": regime_stats,
            "mode_distribution": mode_stats,
        }


# Singleton instance
_collector: Optional[RetrainDataCollector] = None


def get_data_collector() -> RetrainDataCollector:
    """Get or create singleton DataCollector."""
    global _collector
    if _collector is None:
        _collector = RetrainDataCollector()
    return _collector
