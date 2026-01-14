# src/analytics/score_tracker.py
"""
Score Tracker - Shadow vs Live Comparison
==========================================

Tracks scores for both live and shadow profiles to enable:
- Performance comparison
- Promotion decisions
- Historical analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, date
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("SCORE_TRACKER")


@dataclass
class ScoreSnapshot:
    """Single score snapshot."""
    date: str
    profile_id: str
    mode: str  # "live" or "shadow"
    score: float
    max_dd: float
    sharpe: float = 0.0
    winrate: float = 0.0
    profit: float = 0.0
    trades: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "profile": self.profile_id,
            "mode": self.mode,
            "score": self.score,
            "max_dd": self.max_dd,
            "sharpe": self.sharpe,
            "winrate": self.winrate,
            "profit": self.profit,
            "trades": self.trades,
        }


class ScoreTracker:
    """
    Tracks scores for live and shadow profiles.
    
    Usage:
        tracker = ScoreTracker()
        tracker.record("alpha_v2", "live", metrics)
        tracker.record("alpha_v3", "shadow", metrics)
        
        if tracker.shadow_beats_live("alpha_v3", "alpha_v2", n=3):
            promote("alpha_v3")
    """
    
    def __init__(self, data_dir: str = "data/scores"):
        """
        Initialize Score Tracker.
        
        Args:
            data_dir: Directory to store score history
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.records: List[ScoreSnapshot] = []
        self._load_history()
        
        logger.info(f"ScoreTracker initialized: {len(self.records)} records")
    
    def record(
        self,
        profile_id: str,
        mode: str,
        score: float,
        max_dd: float,
        sharpe: float = 0.0,
        winrate: float = 0.0,
        profit: float = 0.0,
        trades: int = 0,
        record_date: str = None
    ) -> ScoreSnapshot:
        """
        Record a score snapshot.
        
        Args:
            profile_id: Profile identifier
            mode: "live" or "shadow"
            score: Competition score
            max_dd: Maximum drawdown
            sharpe: Sharpe ratio
            winrate: Win rate
            profit: Total profit
            trades: Number of trades
            record_date: Date string (default: today)
            
        Returns:
            Created snapshot
        """
        if record_date is None:
            record_date = date.today().isoformat()
        
        snapshot = ScoreSnapshot(
            date=record_date,
            profile_id=profile_id,
            mode=mode,
            score=score,
            max_dd=max_dd,
            sharpe=sharpe,
            winrate=winrate,
            profit=profit,
            trades=trades,
        )
        
        self.records.append(snapshot)
        self._save_snapshot(snapshot)
        
        logger.info(f"Recorded {mode} score for {profile_id}: {score:.2f}")
        
        return snapshot
    
    def last_n(
        self,
        profile_id: str,
        mode: str,
        n: int = 3
    ) -> List[ScoreSnapshot]:
        """Get last N records for a profile/mode."""
        filtered = [
            r for r in self.records
            if r.profile_id == profile_id and r.mode == mode
        ]
        return filtered[-n:] if len(filtered) >= n else filtered
    
    def get_latest(
        self,
        profile_id: str,
        mode: str
    ) -> Optional[ScoreSnapshot]:
        """Get latest record for a profile/mode."""
        records = self.last_n(profile_id, mode, 1)
        return records[0] if records else None
    
    def shadow_beats_live(
        self,
        shadow_profile: str,
        live_profile: str,
        n: int = 3,
        epsilon: float = 0.05
    ) -> bool:
        """
        Check if shadow beats live for N consecutive days.
        
        Args:
            shadow_profile: Shadow profile ID
            live_profile: Live profile ID
            n: Number of consecutive days required
            epsilon: Minimum improvement margin
            
        Returns:
            True if shadow is eligible for promotion
        """
        shadow = self.last_n(shadow_profile, "shadow", n)
        live = self.last_n(live_profile, "live", n)
        
        if len(shadow) < n or len(live) < n:
            return False
        
        for s, l in zip(shadow, live):
            # Shadow score must exceed live by epsilon
            if s.score <= l.score * (1 + epsilon):
                return False
            # Shadow DD must be lower than live
            if s.max_dd >= l.max_dd:
                return False
        
        return True
    
    def get_comparison(
        self,
        shadow_profile: str,
        live_profile: str,
        n: int = 3
    ) -> Dict[str, Any]:
        """Get side-by-side comparison."""
        shadow = self.last_n(shadow_profile, "shadow", n)
        live = self.last_n(live_profile, "live", n)
        
        shadow_avg = sum(s.score for s in shadow) / len(shadow) if shadow else 0
        live_avg = sum(l.score for l in live) / len(live) if live else 0
        
        return {
            "shadow_profile": shadow_profile,
            "live_profile": live_profile,
            "shadow_avg_score": round(shadow_avg, 2),
            "live_avg_score": round(live_avg, 2),
            "delta": round(shadow_avg - live_avg, 2),
            "shadow_beats": self.shadow_beats_live(shadow_profile, live_profile, n),
            "days_compared": min(len(shadow), len(live)),
        }
    
    def _save_snapshot(self, snapshot: ScoreSnapshot) -> None:
        """Save snapshot to file."""
        filename = f"scores_{snapshot.date}.jsonl"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(snapshot.to_dict()) + "\n")
    
    def _load_history(self) -> None:
        """Load historical scores."""
        if not os.path.exists(self.data_dir):
            return
        
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            self.records.append(ScoreSnapshot(
                                date=data["date"],
                                profile_id=data["profile"],
                                mode=data["mode"],
                                score=data["score"],
                                max_dd=data["max_dd"],
                                sharpe=data.get("sharpe", 0),
                                winrate=data.get("winrate", 0),
                                profit=data.get("profit", 0),
                                trades=data.get("trades", 0),
                            ))
                        except:
                            pass


# Singleton
_tracker: Optional[ScoreTracker] = None


def get_score_tracker() -> ScoreTracker:
    """Get singleton ScoreTracker."""
    global _tracker
    if _tracker is None:
        _tracker = ScoreTracker()
    return _tracker
