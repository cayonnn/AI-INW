# src/shadow/tournament.py
"""
Multi-Shadow Tournament Engine
===============================

Runs multiple shadow strategies in parallel and compares performance.
Only the winner gets promoted after meeting strict criteria.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("TOURNAMENT")


@dataclass
class TournamentEntry:
    """Entry in the tournament."""
    name: str
    profile_id: str
    score: float = 0.0
    max_dd: float = 0.0
    trades: int = 0
    win_days: int = 0
    total_days: int = 0
    is_live: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "profile": self.profile_id,
            "score": self.score,
            "max_dd": self.max_dd,
            "trades": self.trades,
            "win_days": self.win_days,
            "total_days": self.total_days,
            "is_live": self.is_live,
        }


@dataclass
class TournamentResult:
    """Result of a tournament round."""
    date: str
    live: TournamentEntry
    shadows: List[TournamentEntry]
    winner: str
    eligible_for_promotion: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "live": self.live.to_dict(),
            "shadows": [s.to_dict() for s in self.shadows],
            "winner": self.winner,
            "eligible": self.eligible_for_promotion,
        }


class ShadowTournament:
    """
    Multi-Shadow Tournament Engine.
    
    Runs multiple shadow strategies:
    - shadow_bayes_latest: Latest Bayesian optimization result
    - shadow_best_7d: Best performer from last 7 days
    - shadow_defensive: Conservative low-risk strategy
    
    Promotion criteria:
    - Score > Live score * 1.05 (5% improvement)
    - Max DD <= Live max DD
    - Win days >= 3 consecutive
    """
    
    PROMOTION_SCORE_MARGIN = 1.05  # 5% better than live
    PROMOTION_MIN_WIN_DAYS = 3
    
    def __init__(self, data_dir: str = "data/tournament"):
        """Initialize tournament."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.live: Optional[TournamentEntry] = None
        self.shadows: Dict[str, TournamentEntry] = {}
        self.history: List[TournamentResult] = []
        
        logger.info("ShadowTournament initialized")
    
    def register_live(self, profile_id: str) -> TournamentEntry:
        """Register the live strategy."""
        self.live = TournamentEntry(
            name="live",
            profile_id=profile_id,
            is_live=True,
        )
        return self.live
    
    def register_shadow(
        self,
        name: str,
        profile_id: str
    ) -> TournamentEntry:
        """Register a shadow strategy."""
        entry = TournamentEntry(
            name=name,
            profile_id=profile_id,
            is_live=False,
        )
        self.shadows[name] = entry
        logger.info(f"Registered shadow: {name}")
        return entry
    
    def update_scores(
        self,
        live_score: float,
        live_dd: float,
        shadow_scores: Dict[str, Dict]
    ) -> None:
        """Update scores for all entries."""
        if self.live:
            self.live.score = live_score
            self.live.max_dd = live_dd
            self.live.total_days += 1
        
        for name, metrics in shadow_scores.items():
            if name in self.shadows:
                entry = self.shadows[name]
                entry.score = metrics.get("score", entry.score)
                entry.max_dd = metrics.get("max_dd", entry.max_dd)
                entry.trades = metrics.get("trades", entry.trades)
                entry.total_days += 1
                
                # Track win days
                if entry.score > self.live.score:
                    entry.win_days += 1
                else:
                    entry.win_days = 0  # Reset on loss
    
    def run_round(self, run_date: date = None) -> TournamentResult:
        """
        Run a tournament round.
        
        Returns comparison results and promotion eligibility.
        """
        if run_date is None:
            run_date = date.today()
        
        if not self.live:
            raise ValueError("No live strategy registered")
        
        results = {}
        for name, shadow in self.shadows.items():
            results[name] = self._compare(self.live, shadow)
        
        # Find winner
        shadow_list = list(self.shadows.values())
        if shadow_list:
            winner = max(shadow_list, key=lambda x: x.score)
            winner_name = winner.name
        else:
            winner_name = "live"
        
        # Check promotion eligibility
        eligible = []
        for name, shadow in self.shadows.items():
            if self._is_eligible(shadow, self.live):
                eligible.append(name)
        
        result = TournamentResult(
            date=run_date.isoformat(),
            live=self.live,
            shadows=shadow_list,
            winner=winner_name,
            eligible_for_promotion=eligible,
        )
        
        self.history.append(result)
        self._save_result(result)
        
        logger.info(
            f"Tournament round: winner={winner_name}, "
            f"eligible={eligible}"
        )
        
        return result
    
    def _compare(
        self,
        live: TournamentEntry,
        shadow: TournamentEntry
    ) -> Dict[str, Any]:
        """Compare shadow to live."""
        return {
            "score_delta": shadow.score - live.score,
            "dd_delta": shadow.max_dd - live.max_dd,
            "shadow_wins": shadow.score > live.score,
            "dd_better": shadow.max_dd <= live.max_dd,
        }
    
    def _is_eligible(
        self,
        shadow: TournamentEntry,
        live: TournamentEntry
    ) -> bool:
        """Check if shadow is eligible for promotion."""
        # Score must be 5% better
        if shadow.score <= live.score * self.PROMOTION_SCORE_MARGIN:
            return False
        
        # DD must be equal or better
        if shadow.max_dd > live.max_dd:
            return False
        
        # Must have 3+ consecutive win days
        if shadow.win_days < self.PROMOTION_MIN_WIN_DAYS:
            return False
        
        return True
    
    def get_promotion_candidate(self) -> Optional[str]:
        """Get the best promotion candidate if any."""
        eligible = [
            (name, shadow) 
            for name, shadow in self.shadows.items()
            if self._is_eligible(shadow, self.live)
        ]
        
        if not eligible:
            return None
        
        # Return highest scoring eligible
        best = max(eligible, key=lambda x: x[1].score)
        return best[0]
    
    def _save_result(self, result: TournamentResult) -> None:
        """Save tournament result."""
        filename = f"tournament_{result.date}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def get_leaderboard(self) -> List[Dict]:
        """Get current leaderboard."""
        entries = [self.live] + list(self.shadows.values()) if self.live else list(self.shadows.values())
        entries.sort(key=lambda x: x.score, reverse=True)
        
        return [
            {
                "rank": i + 1,
                "name": e.name,
                "score": e.score,
                "max_dd": e.max_dd,
                "win_days": e.win_days,
                "is_live": e.is_live,
            }
            for i, e in enumerate(entries)
        ]


# Singleton
_tournament: Optional[ShadowTournament] = None


def get_tournament() -> ShadowTournament:
    """Get singleton tournament."""
    global _tournament
    if _tournament is None:
        _tournament = ShadowTournament()
    return _tournament
