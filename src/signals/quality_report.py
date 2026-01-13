"""
quality_report.py
=================
Signal Quality Report (Win Rate / MAE / MFE Analysis)

Tracks trade outcomes for:
- Performance measurement
- Alpha attribution
- Strategy improvement
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json


@dataclass
class TradeRecord:
    """Individual trade record for quality analysis."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float           # Profit/Loss in points
    mae: float           # Maximum Adverse Excursion
    mfe: float           # Maximum Favorable Excursion
    confidence: float    # Signal confidence at entry
    reason: str          # Signal reason
    indicators: Dict     # Indicator snapshot


@dataclass
class QualitySummary:
    """Summary statistics for signal quality."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl: float
    avg_mae: float
    avg_mfe: float
    avg_confidence: float
    profit_factor: float
    expectancy: float


class SignalQualityReport:
    """
    Signal quality tracker and analyzer.
    
    Tracks:
    - Win rate
    - MAE (Maximum Adverse Excursion) - how far goes against you
    - MFE (Maximum Favorable Excursion) - how far goes in your favor
    - Profit factor
    - Expectancy
    """
    
    def __init__(self):
        self.records: List[TradeRecord] = []
    
    def log_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        high_since_entry: float,
        low_since_entry: float,
        confidence: float = 0,
        reason: str = "",
        indicators: Dict = None
    ):
        """Log a completed trade for quality analysis."""
        # Calculate PnL
        if direction == "BUY":
            pnl = exit_price - entry_price
            mae = entry_price - low_since_entry  # How far dropped
            mfe = high_since_entry - entry_price  # How far rose
        else:
            pnl = entry_price - exit_price
            mae = high_since_entry - entry_price
            mfe = entry_price - low_since_entry
        
        record = TradeRecord(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=pnl,
            mae=mae,
            mfe=mfe,
            confidence=confidence,
            reason=reason,
            indicators=indicators or {}
        )
        
        self.records.append(record)
    
    def get_summary(self) -> QualitySummary:
        """Get quality summary statistics."""
        if not self.records:
            return QualitySummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_pnl=0,
                avg_mae=0,
                avg_mfe=0,
                avg_confidence=0,
                profit_factor=0,
                expectancy=0
            )
        
        n = len(self.records)
        
        wins = [r for r in self.records if r.pnl > 0]
        losses = [r for r in self.records if r.pnl <= 0]
        
        total_wins = sum(r.pnl for r in wins) if wins else 0
        total_losses = abs(sum(r.pnl for r in losses)) if losses else 0
        
        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0
        
        return QualitySummary(
            total_trades=n,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / n,
            avg_pnl=sum(r.pnl for r in self.records) / n,
            avg_mae=sum(r.mae for r in self.records) / n,
            avg_mfe=sum(r.mfe for r in self.records) / n,
            avg_confidence=sum(r.confidence for r in self.records) / n,
            profit_factor=total_wins / max(total_losses, 0.01),
            expectancy=(len(wins)/n * avg_win) - (len(losses)/n * avg_loss)
        )
    
    def get_by_confidence(self, min_confidence: float = 0.5) -> List[TradeRecord]:
        """Get trades filtered by confidence threshold."""
        return [r for r in self.records if r.confidence >= min_confidence]
    
    def export_json(self, filepath: str):
        """Export records to JSON file."""
        data = []
        for r in self.records:
            data.append({
                "symbol": r.symbol,
                "direction": r.direction,
                "entry_price": r.entry_price,
                "exit_price": r.exit_price,
                "entry_time": r.entry_time.isoformat(),
                "exit_time": r.exit_time.isoformat(),
                "pnl": r.pnl,
                "mae": r.mae,
                "mfe": r.mfe,
                "confidence": r.confidence,
                "reason": r.reason,
                "indicators": r.indicators
            })
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def clear(self):
        """Clear all records."""
        self.records.clear()
