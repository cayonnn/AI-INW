# src/utils/alpha_comparison_dashboard.py
"""
Alpha Comparison Dashboard
==========================

Compare Rule Alpha vs PPO Alpha performance in Shadow Mode.

Features:
- Real-time agreement tracking
- Hypothetical PnL comparison
- Action distribution analysis
- Promotion gate evaluation
"""

import os
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("ALPHA_COMPARE")


@dataclass
class AlphaComparisonMetrics:
    """Metrics for Alpha comparison."""
    timestamp: str
    
    # Agreement
    total_comparisons: int
    agreement_rate: float
    
    # Action rates
    rule_buy_rate: float
    rule_sell_rate: float
    rule_hold_rate: float
    ppo_buy_rate: float
    ppo_sell_rate: float
    ppo_hold_rate: float
    
    # Shadow PnL
    shadow_pnl: float
    shadow_trades: int
    shadow_win_rate: float
    
    # Promotion gate status
    promotion_ready: bool
    promotion_blockers: List[str]


@dataclass 
class PromotionGateCriteria:
    """Criteria for promoting PPO Alpha to live."""
    min_comparisons: int = 500
    min_agreement_rate: float = 0.40  # PPO should agree at least 40%
    min_shadow_win_rate: float = 0.50
    max_shadow_dd: float = 0.15
    min_shadow_pnl: float = 0.0
    require_positive_pnl: bool = True


class AlphaComparisonDashboard:
    """
    Dashboard for comparing Rule Alpha vs PPO Alpha.
    
    Reads from alpha_shadow logs and calculates metrics.
    """
    
    def __init__(
        self,
        shadow_log_path: str = "logs/alpha_shadow.csv",
        output_path: str = "logs/alpha_comparison.json"
    ):
        self.shadow_log_path = shadow_log_path
        self.output_path = output_path
        self.criteria = PromotionGateCriteria()
        
        # Live stats (from alpha_shadow runner)
        self.live_stats: Optional[Dict] = None
    
    def update_from_shadow(self, shadow_stats: Dict):
        """Update with live stats from AlphaShadowRunner."""
        self.live_stats = shadow_stats
    
    def load_log_data(self) -> List[Dict]:
        """Load shadow comparison log."""
        if not Path(self.shadow_log_path).exists():
            return []
        
        data = []
        try:
            with open(self.shadow_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
        except Exception as e:
            logger.error(f"Failed to load shadow log: {e}")
        
        return data
    
    def calculate_metrics(self) -> AlphaComparisonMetrics:
        """Calculate comparison metrics from log data."""
        data = self.load_log_data()
        
        if not data and not self.live_stats:
            return self._empty_metrics()
        
        # Use live stats if available (more accurate)
        if self.live_stats:
            return self._metrics_from_live()
        
        # Otherwise calculate from CSV
        return self._metrics_from_csv(data)
    
    def _metrics_from_live(self) -> AlphaComparisonMetrics:
        """Build metrics from live shadow stats."""
        stats = self.live_stats
        
        # Calculate action rates
        total = max(stats.get("total_comparisons", 1), 1)
        
        # Estimate action rates (if available from PPO stats)
        ppo_stats = stats.get("ppo_stats", {})
        
        # Promotion gate check
        promotion_ready, blockers = self._check_promotion_gate(
            comparisons=total,
            agreement_rate=stats.get("agreement_rate", 0),
            shadow_win_rate=stats.get("shadow_win_rate", 0),
            shadow_pnl=stats.get("shadow_pnl", 0)
        )
        
        return AlphaComparisonMetrics(
            timestamp=datetime.now().isoformat(),
            total_comparisons=total,
            agreement_rate=stats.get("agreement_rate", 0),
            rule_buy_rate=stats.get("rule_trade_rate", 0) / 2,
            rule_sell_rate=stats.get("rule_trade_rate", 0) / 2,
            rule_hold_rate=1 - stats.get("rule_trade_rate", 0),
            ppo_buy_rate=ppo_stats.get("buy_rate", 0),
            ppo_sell_rate=ppo_stats.get("sell_rate", 0),
            ppo_hold_rate=ppo_stats.get("hold_rate", 0),
            shadow_pnl=stats.get("shadow_pnl", 0),
            shadow_trades=stats.get("shadow_trades_closed", 0),
            shadow_win_rate=stats.get("shadow_win_rate", 0),
            promotion_ready=promotion_ready,
            promotion_blockers=blockers
        )
    
    def _metrics_from_csv(self, data: List[Dict]) -> AlphaComparisonMetrics:
        """Build metrics from CSV data."""
        total = len(data)
        if total == 0:
            return self._empty_metrics()
        
        # Count agreements
        agreed = sum(1 for row in data if row.get("agreed") == "True")
        agreement_rate = agreed / total
        
        # Count action types
        rule_buy = sum(1 for row in data if row.get("rule_action") == "BUY")
        rule_sell = sum(1 for row in data if row.get("rule_action") == "SELL")
        ppo_buy = sum(1 for row in data if row.get("ppo_action") == "BUY")
        ppo_sell = sum(1 for row in data if row.get("ppo_action") == "SELL")
        
        # Shadow PnL
        shadow_pnls = [float(row.get("shadow_pnl", 0) or 0) for row in data]
        total_pnl = sum(shadow_pnls)
        shadow_wins = sum(1 for pnl in shadow_pnls if pnl > 0)
        shadow_trades = sum(1 for pnl in shadow_pnls if pnl != 0)
        
        # Promotion gate
        promotion_ready, blockers = self._check_promotion_gate(
            comparisons=total,
            agreement_rate=agreement_rate,
            shadow_win_rate=shadow_wins / max(shadow_trades, 1),
            shadow_pnl=total_pnl
        )
        
        return AlphaComparisonMetrics(
            timestamp=datetime.now().isoformat(),
            total_comparisons=total,
            agreement_rate=agreement_rate,
            rule_buy_rate=rule_buy / total,
            rule_sell_rate=rule_sell / total,
            rule_hold_rate=(total - rule_buy - rule_sell) / total,
            ppo_buy_rate=ppo_buy / total,
            ppo_sell_rate=ppo_sell / total,
            ppo_hold_rate=(total - ppo_buy - ppo_sell) / total,
            shadow_pnl=total_pnl,
            shadow_trades=shadow_trades,
            shadow_win_rate=shadow_wins / max(shadow_trades, 1),
            promotion_ready=promotion_ready,
            promotion_blockers=blockers
        )
    
    def _check_promotion_gate(
        self,
        comparisons: int,
        agreement_rate: float,
        shadow_win_rate: float,
        shadow_pnl: float
    ) -> tuple:
        """Check if PPO Alpha meets promotion criteria."""
        blockers = []
        
        if comparisons < self.criteria.min_comparisons:
            blockers.append(f"Need {self.criteria.min_comparisons} comparisons (have {comparisons})")
        
        if agreement_rate < self.criteria.min_agreement_rate:
            blockers.append(f"Agreement rate {agreement_rate:.1%} < {self.criteria.min_agreement_rate:.1%}")
        
        if shadow_win_rate < self.criteria.min_shadow_win_rate:
            blockers.append(f"Shadow win rate {shadow_win_rate:.1%} < {self.criteria.min_shadow_win_rate:.1%}")
        
        if self.criteria.require_positive_pnl and shadow_pnl < 0:
            blockers.append(f"Shadow PnL negative: ${shadow_pnl:.2f}")
        
        return len(blockers) == 0, blockers
    
    def _empty_metrics(self) -> AlphaComparisonMetrics:
        """Return empty metrics when no data."""
        return AlphaComparisonMetrics(
            timestamp=datetime.now().isoformat(),
            total_comparisons=0,
            agreement_rate=0,
            rule_buy_rate=0,
            rule_sell_rate=0,
            rule_hold_rate=1.0,
            ppo_buy_rate=0,
            ppo_sell_rate=0,
            ppo_hold_rate=1.0,
            shadow_pnl=0,
            shadow_trades=0,
            shadow_win_rate=0,
            promotion_ready=False,
            promotion_blockers=["No comparison data available"]
        )
    
    def generate_report(self) -> str:
        """Generate text report for logging."""
        metrics = self.calculate_metrics()
        
        lines = [
            "=" * 60,
            "📊 ALPHA COMPARISON REPORT",
            "=" * 60,
            f"Timestamp: {metrics.timestamp}",
            f"Total Comparisons: {metrics.total_comparisons}",
            "",
            "--- Agreement ---",
            f"Agreement Rate: {metrics.agreement_rate:.1%}",
            "",
            "--- Action Distribution ---",
            f"Rule: BUY={metrics.rule_buy_rate:.1%} SELL={metrics.rule_sell_rate:.1%} HOLD={metrics.rule_hold_rate:.1%}",
            f"PPO:  BUY={metrics.ppo_buy_rate:.1%} SELL={metrics.ppo_sell_rate:.1%} HOLD={metrics.ppo_hold_rate:.1%}",
            "",
            "--- Shadow Performance ---",
            f"Shadow Trades: {metrics.shadow_trades}",
            f"Shadow PnL: ${metrics.shadow_pnl:.2f}",
            f"Shadow Win Rate: {metrics.shadow_win_rate:.1%}",
            "",
            "--- Promotion Gate ---",
            f"Ready for Live: {'✅ YES' if metrics.promotion_ready else '❌ NO'}",
        ]
        
        if metrics.promotion_blockers:
            lines.append("Blockers:")
            for blocker in metrics.promotion_blockers:
                lines.append(f"  • {blocker}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_snapshot(self):
        """Save current metrics to JSON."""
        metrics = self.calculate_metrics()
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        logger.info(f"Saved comparison snapshot to {self.output_path}")


# =============================================================================
# Singleton
# =============================================================================

_dashboard: Optional[AlphaComparisonDashboard] = None


def get_alpha_comparison_dashboard() -> AlphaComparisonDashboard:
    """Get singleton dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = AlphaComparisonDashboard()
    return _dashboard


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha Comparison Dashboard")
    print("=" * 60)
    
    dashboard = AlphaComparisonDashboard()
    
    # Try to load existing data
    data = dashboard.load_log_data()
    print(f"Loaded {len(data)} comparison records")
    
    # Generate report
    print(dashboard.generate_report())
    
    # Save snapshot
    dashboard.save_snapshot()
    print(f"\nSnapshot saved to {dashboard.output_path}")
