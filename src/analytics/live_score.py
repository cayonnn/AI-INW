# src/analytics/live_score.py
"""
Live Score Estimator - Leaderboard Brain
=========================================

Real-time competition scoring to estimate leaderboard position.

Scoring Formula:
  Score = (Net Profit × 0.4)
        + (Sharpe × 0.25)
        + (Win Rate × 0.2)
        - (Max DD × 0.3)
        + (Consistency × 0.15)

Features:
- Estimate percentile ranking
- Auto mode switching based on score
- Dashboard integration
"""

from dataclasses import dataclass
from typing import Optional
import math

from src.utils.logger import get_logger

logger = get_logger("LIVE_SCORE")


@dataclass
class ScoreComponents:
    """Individual score components."""
    profit_score: float
    sharpe_score: float
    winrate_score: float
    dd_penalty: float
    consistency_score: float


@dataclass
class LiveScore:
    """Live score result."""
    total_score: float
    percentile_estimate: str
    rank_simulation: int
    components: ScoreComponents
    recommended_mode: str


@dataclass
class TradingStats:
    """Stats for scoring calculation."""
    net_profit: float          # Total P/L in $
    net_profit_pct: float      # Return %
    sharpe: float              # Sharpe ratio
    win_rate: float            # Win rate 0-1
    max_dd: float              # Max drawdown %
    consistency: float         # Consistency score 0-1
    total_trades: int
    equity: float


class LiveScoreEstimator:
    """
    Competition Score Estimator.
    
    Score Weights:
    ┌─────────────────┬────────┐
    │ Component       │ Weight │
    ├─────────────────┼────────┤
    │ Net Profit      │ 0.40   │
    │ Sharpe Ratio    │ 0.25   │
    │ Win Rate        │ 0.20   │
    │ Max Drawdown    │ -0.30  │
    │ Consistency     │ 0.15   │
    └─────────────────┴────────┘
    """
    
    # Score weights
    WEIGHTS = {
        "profit": 0.40,
        "sharpe": 0.25,
        "winrate": 0.20,
        "drawdown": 0.30,  # Penalty (subtracted)
        "consistency": 0.15,
    }
    
    # Percentile thresholds
    PERCENTILE_THRESHOLDS = [
        (90, "Top 1%", 5),
        (80, "Top 5%", 25),
        (70, "Top 10%", 50),
        (60, "Top 20%", 100),
        (50, "Top 30%", 150),
        (40, "Top 50%", 250),
        (0, "Bottom 50%", 400),
    ]
    
    # Mode recommendations
    MODE_THRESHOLDS = {
        "ultra_aggressive": 85,
        "aggressive": 70,
        "standard": 50,
        "defensive": 30,
    }
    
    def __init__(
        self,
        target_profit: float = 500.0,      # Target profit for scaling
        target_sharpe: float = 2.0,         # Target Sharpe for scaling
        total_participants: int = 500       # Estimated competition size
    ):
        """
        Initialize Live Score Estimator.
        
        Args:
            target_profit: Target profit for 100% score
            target_sharpe: Target Sharpe for 100% score
            total_participants: Estimated competition size
        """
        self.target_profit = target_profit
        self.target_sharpe = target_sharpe
        self.total_participants = total_participants
        
        logger.info(
            f"LiveScoreEstimator initialized: "
            f"target_profit=${target_profit}, target_sharpe={target_sharpe}"
        )
    
    def estimate(self, stats: TradingStats) -> LiveScore:
        """
        Calculate live competition score.
        
        Args:
            stats: Current trading statistics
            
        Returns:
            LiveScore with total score and estimates
        """
        # Calculate component scores (0-100 each)
        profit_score = self._score_profit(stats.net_profit_pct)
        sharpe_score = self._score_sharpe(stats.sharpe)
        winrate_score = self._score_winrate(stats.win_rate)
        dd_penalty = self._score_drawdown(stats.max_dd)
        consistency_score = self._score_consistency(stats.consistency)
        
        # Calculate weighted total
        total = (
            profit_score * self.WEIGHTS["profit"]
            + sharpe_score * self.WEIGHTS["sharpe"]
            + winrate_score * self.WEIGHTS["winrate"]
            - dd_penalty * self.WEIGHTS["drawdown"]
            + consistency_score * self.WEIGHTS["consistency"]
        )
        
        # Clamp to 0-100
        total = max(0, min(100, total))
        
        # Get percentile estimate
        percentile, rank = self._estimate_percentile(total)
        
        # Get recommended mode
        mode = self._recommend_mode(total)
        
        components = ScoreComponents(
            profit_score=profit_score,
            sharpe_score=sharpe_score,
            winrate_score=winrate_score,
            dd_penalty=dd_penalty,
            consistency_score=consistency_score
        )
        
        return LiveScore(
            total_score=round(total, 1),
            percentile_estimate=percentile,
            rank_simulation=rank,
            components=components,
            recommended_mode=mode
        )
    
    def _score_profit(self, profit_pct: float) -> float:
        """Score profit component (0-100)."""
        # Target: 50% return = 100 score
        target_pct = 50.0
        score = (profit_pct / target_pct) * 100
        return max(0, min(100, score))
    
    def _score_sharpe(self, sharpe: float) -> float:
        """Score Sharpe ratio (0-100)."""
        # Target: Sharpe 2.0 = 100 score
        score = (sharpe / self.target_sharpe) * 100
        return max(0, min(100, score))
    
    def _score_winrate(self, win_rate: float) -> float:
        """Score win rate (0-100)."""
        # Win rate is already 0-1, scale to 0-100
        # 50% = 50, 70% = 100
        score = min(100, (win_rate - 0.3) / 0.4 * 100)
        return max(0, score)
    
    def _score_drawdown(self, max_dd: float) -> float:
        """Score drawdown penalty (0-100, higher = more penalty)."""
        # 0% DD = 0 penalty, 20% DD = 100 penalty
        penalty = (max_dd / 20.0) * 100
        return max(0, min(100, penalty))
    
    def _score_consistency(self, consistency: float) -> float:
        """Score consistency (0-100)."""
        return consistency * 100
    
    def _estimate_percentile(self, score: float) -> tuple[str, int]:
        """Estimate percentile and rank from score."""
        for threshold, percentile, rank in self.PERCENTILE_THRESHOLDS:
            if score >= threshold:
                return percentile, rank
        return "Bottom 50%", self.total_participants
    
    def _recommend_mode(self, score: float) -> str:
        """Recommend trading mode based on score."""
        for mode, threshold in sorted(
            self.MODE_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score >= threshold:
                return mode
        return "defensive"
    
    def to_dict(self, score: LiveScore) -> dict:
        """Convert score to dictionary for dashboard."""
        return {
            "live_score": score.total_score,
            "percentile_estimate": score.percentile_estimate,
            "rank_simulation": score.rank_simulation,
            "recommended_mode": score.recommended_mode,
            "components": {
                "profit": score.components.profit_score,
                "sharpe": score.components.sharpe_score,
                "winrate": score.components.winrate_score,
                "drawdown": score.components.dd_penalty,
                "consistency": score.components.consistency_score,
            }
        }


# Singleton instance
_estimator: Optional[LiveScoreEstimator] = None


def get_live_score_estimator() -> LiveScoreEstimator:
    """Get or create singleton LiveScoreEstimator."""
    global _estimator
    if _estimator is None:
        _estimator = LiveScoreEstimator()
    return _estimator


def calculate_consistency(daily_returns: list[float]) -> float:
    """
    Calculate consistency score from daily returns.
    
    Higher score = more consistent returns
    """
    if len(daily_returns) < 5:
        return 0.5  # Not enough data
    
    # Count positive days
    positive_days = sum(1 for r in daily_returns if r > 0)
    positive_ratio = positive_days / len(daily_returns)
    
    # Calculate variance
    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)
    std_dev = math.sqrt(variance) if variance > 0 else 0
    
    # Lower variance = higher consistency
    variance_score = max(0, 1 - (std_dev / 5))  # Normalize
    
    # Combine positive ratio and variance
    return positive_ratio * 0.6 + variance_score * 0.4


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LiveScoreEstimator Test")
    print("=" * 60)
    
    estimator = LiveScoreEstimator()
    
    # Test different scenarios
    scenarios = [
        TradingStats(
            net_profit=150, net_profit_pct=15, sharpe=1.5,
            win_rate=0.65, max_dd=5.0, consistency=0.7,
            total_trades=50, equity=1150
        ),
        TradingStats(
            net_profit=350, net_profit_pct=35, sharpe=2.2,
            win_rate=0.72, max_dd=8.0, consistency=0.8,
            total_trades=100, equity=1350
        ),
        TradingStats(
            net_profit=-50, net_profit_pct=-5, sharpe=-0.5,
            win_rate=0.40, max_dd=15.0, consistency=0.3,
            total_trades=30, equity=950
        ),
    ]
    
    for i, stats in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        print(f"Net Profit: ${stats.net_profit} ({stats.net_profit_pct}%)")
        print(f"Sharpe: {stats.sharpe}, Win Rate: {stats.win_rate:.0%}")
        print(f"Max DD: {stats.max_dd}%")
        
        score = estimator.estimate(stats)
        
        print(f"\nScore: {score.total_score}")
        print(f"Percentile: {score.percentile_estimate}")
        print(f"Est. Rank: #{score.rank_simulation}")
        print(f"Recommended Mode: {score.recommended_mode}")
