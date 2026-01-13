"""
dataset_recorder.py
====================
AI Imitation Learning Dataset Recorder

Records every decision from SignalEngine V2 for AI training:
- State/Context (indicators at decision time)
- Action (BUY / SELL / HOLD)
- Optional outcome data (for future RL)

Fund-Grade Rules:
- Record HOLD decisions (AI must learn when NOT to trade)
- No future data leakage
- Time-balanced sampling
"""

import csv
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
import json


# =========================
# Feature Schema
# =========================

STATE_FEATURES = [
    "ema_fast",
    "ema_slow", 
    "ema_spread",
    "ema_slope",
    "atr",
    "atr_ratio",
    "atr_threshold",
    "htf_trend",      # -1, 0, +1
    "volatility_ok",
    "hour",
    "day_of_week",
]

ACTION_MAP = {
    "BUY": 1,
    "SELL": -1,
    "HOLD": 0
}


@dataclass
class DatasetRecord:
    """Single training record."""
    timestamp: str
    symbol: str
    
    # State features
    ema_fast: float
    ema_slow: float
    ema_spread: float
    ema_slope: float
    atr: float
    atr_ratio: float
    atr_threshold: float
    htf_trend: int        # 1=BULL, -1=BEAR, 0=FLAT
    volatility_ok: int    # 1=True, 0=False
    hour: int
    day_of_week: int
    
    # Action (label)
    action: str           # BUY / SELL / HOLD
    action_code: int      # 1 / -1 / 0
    
    # Signal metadata
    confidence: float
    reason: str
    blocked_by: str
    
    # Optional outcome (filled later for RL)
    future_return: Optional[float] = None
    mae: Optional[float] = None
    mfe: Optional[float] = None


# =========================
# Dataset Recorder
# =========================

class ImitationDatasetRecorder:
    """
    Records SignalEngine decisions for AI training.
    
    Usage:
        recorder = ImitationDatasetRecorder()
        recorder.record(symbol, signal_result, indicators)
    """
    
    def __init__(
        self,
        csv_path: str = "data/imitation_dataset.csv",
        json_path: str = "data/imitation_dataset.json"
    ):
        """
        Args:
            csv_path: Path for CSV dataset
            json_path: Path for JSON dataset (optional backup)
        """
        self.csv_path = csv_path
        self.json_path = json_path
        self.records: List[DatasetRecord] = []
        
        self._ensure_dirs()
        self._init_csv()
    
    def _ensure_dirs(self):
        """Create data directory if needed."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
    
    def _init_csv(self):
        """Initialize CSV with headers if empty."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "symbol",
                    "ema_fast",
                    "ema_slow",
                    "ema_spread",
                    "ema_slope",
                    "atr",
                    "atr_ratio",
                    "atr_threshold",
                    "htf_trend",
                    "volatility_ok",
                    "hour",
                    "day_of_week",
                    "action",
                    "action_code",
                    "confidence",
                    "reason",
                    "blocked_by",
                    "future_return",
                    "mae",
                    "mfe"
                ])
    
    def record(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reason: str,
        indicators: Dict,
        blocked_by: str = "",
        htf_trend: str = "FLAT"
    ) -> DatasetRecord:
        """
        Record a decision for AI training.
        
        Args:
            symbol: Trading symbol
            action: BUY / SELL / HOLD
            confidence: Signal confidence 0-1
            reason: Human-readable reason
            indicators: Dict of indicator values
            blocked_by: What blocked the signal (if HOLD)
            htf_trend: HTF trend string
            
        Returns:
            DatasetRecord that was recorded
        """
        now = datetime.utcnow()
        
        # Calculate slope (if we have previous records)
        ema_slope = self._calc_ema_slope(symbol, indicators.get("ema_fast", 0))
        
        # Convert HTF to numeric
        htf_numeric = 1 if htf_trend == "BULL" else -1 if htf_trend == "BEAR" else 0
        
        # Calculate ATR ratio
        close = indicators.get("close", 1)
        atr = indicators.get("atr", 0)
        atr_ratio = atr / close if close > 0 else 0
        
        record = DatasetRecord(
            timestamp=now.isoformat(),
            symbol=symbol,
            ema_fast=round(indicators.get("ema_fast", 0), 2),
            ema_slow=round(indicators.get("ema_slow", 0), 2),
            ema_spread=round(indicators.get("ema_spread", 0), 2),
            ema_slope=round(ema_slope, 4),
            atr=round(atr, 2),
            atr_ratio=round(atr_ratio, 6),
            atr_threshold=round(indicators.get("atr_threshold", 0), 2),
            htf_trend=htf_numeric,
            volatility_ok=1 if indicators.get("volatility_ok", False) else 0,
            hour=now.hour,
            day_of_week=now.weekday(),
            action=action,
            action_code=ACTION_MAP.get(action, 0),
            confidence=round(confidence, 3),
            reason=reason,
            blocked_by=blocked_by
        )
        
        # Store in memory
        self.records.append(record)
        
        # Write to CSV
        self._write_csv(record)
        
        return record
    
    def _calc_ema_slope(self, symbol: str, current_ema: float) -> float:
        """Calculate EMA slope from recent records."""
        symbol_records = [r for r in self.records[-10:] if r.symbol == symbol]
        if len(symbol_records) < 2:
            return 0.0
        
        prev_ema = symbol_records[-1].ema_fast
        return current_ema - prev_ema
    
    def _write_csv(self, record: DatasetRecord):
        """Append record to CSV."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.timestamp,
                record.symbol,
                record.ema_fast,
                record.ema_slow,
                record.ema_spread,
                record.ema_slope,
                record.atr,
                record.atr_ratio,
                record.atr_threshold,
                record.htf_trend,
                record.volatility_ok,
                record.hour,
                record.day_of_week,
                record.action,
                record.action_code,
                record.confidence,
                record.reason,
                record.blocked_by,
                record.future_return,
                record.mae,
                record.mfe
            ])
    
    def update_outcome(
        self,
        symbol: str,
        entry_time: str,
        future_return: float,
        mae: float,
        mfe: float
    ):
        """
        Update a record with outcome data (for RL training).
        
        Call this after trade is closed.
        """
        for record in reversed(self.records):
            if record.symbol == symbol and record.timestamp == entry_time:
                record.future_return = future_return
                record.mae = mae
                record.mfe = mfe
                break
    
    def save_json(self):
        """Save all records to JSON file."""
        data = [asdict(r) for r in self.records]
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.records:
            return {"total": 0}
        
        buys = sum(1 for r in self.records if r.action == "BUY")
        sells = sum(1 for r in self.records if r.action == "SELL")
        holds = sum(1 for r in self.records if r.action == "HOLD")
        
        return {
            "total": len(self.records),
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "buy_pct": round(buys / len(self.records) * 100, 1),
            "sell_pct": round(sells / len(self.records) * 100, 1),
            "hold_pct": round(holds / len(self.records) * 100, 1),
            "avg_confidence": round(sum(r.confidence for r in self.records) / len(self.records), 3)
        }
    
    def clear(self):
        """Clear in-memory records."""
        self.records.clear()


# =========================
# Singleton Instance
# =========================

_recorder_instance: Optional[ImitationDatasetRecorder] = None


def get_dataset_recorder() -> ImitationDatasetRecorder:
    """Get singleton recorder instance."""
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = ImitationDatasetRecorder()
    return _recorder_instance
