# src/rl/alpha_shadow.py
"""
Alpha Shadow Mode Runner
========================

Run Alpha PPO alongside the live loop without affecting real trading.

Features:
- Compare Alpha Rule vs Alpha PPO decisions
- Track agreement rate
- Log shadow trades for analysis
- Calculate hypothetical PnL

Usage in live_loop_v3.py:
    from src.rl.alpha_shadow import AlphaShadowRunner
    
    shadow = AlphaShadowRunner()
    shadow.compare(
        rule_action=alpha_decision,
        market_state=df.iloc[-1],
        account_state={"open_positions": 2, "floating_dd": 0.03},
        guardian_state=0
    )
"""

import os
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rl.alpha_ppo_v1 import get_alpha_ppo, AlphaPPOInference
from src.rl.alpha_env import AlphaStateNormalizer
from src.utils.logger import get_logger

logger = get_logger("ALPHA_SHADOW")


@dataclass
class ShadowComparison:
    """Single comparison between Rule Alpha and PPO Alpha."""
    timestamp: str
    cycle: int
    
    # Rule Alpha
    rule_action: str
    rule_ema20: float
    rule_ema50: float
    
    # PPO Alpha
    ppo_action: str
    ppo_confidence: float
    
    # Agreement
    agreed: bool
    
    # Market context
    price: float
    atr: float
    guardian_state: int
    floating_dd: float
    
    # Shadow outcome (filled later)
    shadow_entry: Optional[float] = None
    shadow_exit: Optional[float] = None
    shadow_pnl: Optional[float] = None


class AlphaShadowRunner:
    """
    Shadow mode comparison engine for Alpha Rule vs Alpha PPO.
    
    Tracks all decisions and calculates what-if scenarios.
    """
    
    def __init__(
        self,
        log_path: str = "logs/alpha_shadow.csv",
        enabled: bool = True
    ):
        self.log_path = log_path
        self.enabled = enabled
        self.normalizer = AlphaStateNormalizer()
        
        # Get PPO instance
        self.ppo = get_alpha_ppo(enabled=True)
        if not self.ppo.enabled:
            logger.warning("Alpha PPO not available, shadow mode limited")
        
        # Stats
        self.total_comparisons = 0
        self.agreements = 0
        self.ppo_would_trade = 0
        self.rule_would_trade = 0
        
        # Shadow trades (for PnL tracking)
        self.open_shadow_trades: Dict[str, Dict] = {}
        self.closed_shadow_trades = []
        
        # Initialize log file
        self._init_log()
        
        logger.info("👻 Alpha Shadow Runner initialized")
    
    def _init_log(self):
        """Initialize CSV log file with headers."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        if not Path(self.log_path).exists():
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "cycle",
                    "rule_action", "ppo_action", "ppo_confidence",
                    "agreed", "price", "atr", "guardian_state", "floating_dd",
                    "shadow_pnl"
                ])
    
    def compare(
        self,
        rule_action: str,
        cycle: int,
        market_row,  # DataFrame row
        open_positions: int,
        floating_dd: float,
        guardian_state: int,
        current_price: float
    ) -> Tuple[str, float, bool]:
        """
        Compare Rule Alpha action with PPO Alpha action.
        
        Args:
            rule_action: Current rule-based Alpha decision (BUY/SELL/HOLD)
            cycle: Current loop cycle
            market_row: DataFrame row with market data
            open_positions: Current open position count
            floating_dd: Current floating drawdown
            guardian_state: 0=OK, 1=WARNING, 2=BLOCK
            current_price: Current market price
            
        Returns:
            (ppo_action, ppo_confidence, agreed)
        """
        if not self.enabled:
            return "HOLD", 0.0, True
        
        self.total_comparisons += 1
        
        # Get PPO decision
        ppo_action, ppo_confidence = self.ppo.predict_from_df(
            row=market_row,
            open_positions=open_positions,
            floating_dd=floating_dd,
            guardian_state=guardian_state
        )
        
        # Check agreement
        agreed = (rule_action == ppo_action)
        if agreed:
            self.agreements += 1
        
        # Track trading intent
        if rule_action in ["BUY", "SELL"]:
            self.rule_would_trade += 1
        if ppo_action in ["BUY", "SELL"]:
            self.ppo_would_trade += 1
        
        # Get market data for logging
        ema20 = market_row.get('ema20', market_row.get('EMA20', 0))
        ema50 = market_row.get('ema50', market_row.get('EMA50', 0))
        atr = market_row.get('atr14', market_row.get('atr', 2.5))
        
        # Create comparison record
        comparison = ShadowComparison(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cycle=cycle,
            rule_action=rule_action,
            rule_ema20=float(ema20),
            rule_ema50=float(ema50),
            ppo_action=ppo_action,
            ppo_confidence=ppo_confidence,
            agreed=agreed,
            price=current_price,
            atr=float(atr),
            guardian_state=guardian_state,
            floating_dd=floating_dd
        )
        
        # Log comparison
        self._log_comparison(comparison)
        
        # Track shadow trades (if PPO would trade and Rule wouldn't)
        if ppo_action in ["BUY", "SELL"] and rule_action == "HOLD":
            self._open_shadow_trade(ppo_action, current_price, atr, cycle)
        
        # Update existing shadow trades
        self._update_shadow_trades(current_price)
        
        # Log decision
        agree_icon = "✅" if agreed else "❌"
        logger.info(
            f"👻 SHADOW | Rule={rule_action:4s} | "
            f"PPO={ppo_action:4s} (conf={ppo_confidence:.2f}) | "
            f"{agree_icon} {'Agree' if agreed else 'Differ'}"
        )
        
        return ppo_action, ppo_confidence, agreed
    
    def _log_comparison(self, comparison: ShadowComparison):
        """Append comparison to CSV log."""
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    comparison.timestamp,
                    comparison.cycle,
                    comparison.rule_action,
                    comparison.ppo_action,
                    f"{comparison.ppo_confidence:.3f}",
                    comparison.agreed,
                    f"{comparison.price:.2f}",
                    f"{comparison.atr:.2f}",
                    comparison.guardian_state,
                    f"{comparison.floating_dd:.4f}",
                    comparison.shadow_pnl or ""
                ])
        except Exception as e:
            logger.error(f"Shadow log write failed: {e}")
    
    def _open_shadow_trade(
        self,
        direction: str,
        price: float,
        atr: float,
        cycle: int
    ):
        """Open a shadow trade (hypothetical)."""
        trade_id = f"shadow_{cycle}_{direction}"
        
        # Simple SL/TP based on ATR
        sl_dist = atr * 2.0
        tp_dist = atr * 4.0
        
        if direction == "BUY":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
        
        self.open_shadow_trades[trade_id] = {
            "direction": direction,
            "entry_price": price,
            "sl": sl,
            "tp": tp,
            "cycle": cycle,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"👻 Shadow trade opened: {trade_id} @ {price:.2f}")
    
    def _update_shadow_trades(self, current_price: float):
        """Update and close shadow trades based on current price."""
        to_close = []
        
        for trade_id, trade in self.open_shadow_trades.items():
            direction = trade["direction"]
            entry = trade["entry_price"]
            sl = trade["sl"]
            tp = trade["tp"]
            
            closed = False
            pnl = 0.0
            
            if direction == "BUY":
                if current_price >= tp:
                    pnl = tp - entry
                    closed = True
                elif current_price <= sl:
                    pnl = sl - entry
                    closed = True
            else:  # SELL
                if current_price <= tp:
                    pnl = entry - tp
                    closed = True
                elif current_price >= sl:
                    pnl = entry - sl
                    closed = True
            
            if closed:
                trade["exit_price"] = current_price
                trade["pnl"] = pnl
                self.closed_shadow_trades.append(trade)
                to_close.append(trade_id)
                
                win_icon = "💚" if pnl > 0 else "❤️"
                logger.info(
                    f"👻 Shadow trade closed: {trade_id} | "
                    f"PnL={pnl:+.2f} {win_icon}"
                )
        
        for trade_id in to_close:
            del self.open_shadow_trades[trade_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get shadow comparison statistics."""
        total = max(self.total_comparisons, 1)
        
        # Shadow PnL
        total_shadow_pnl = sum(t.get("pnl", 0) for t in self.closed_shadow_trades)
        shadow_wins = sum(1 for t in self.closed_shadow_trades if t.get("pnl", 0) > 0)
        shadow_trades = len(self.closed_shadow_trades)
        
        return {
            "total_comparisons": self.total_comparisons,
            "agreement_rate": self.agreements / total,
            "rule_trade_rate": self.rule_would_trade / total,
            "ppo_trade_rate": self.ppo_would_trade / total,
            "shadow_trades_closed": shadow_trades,
            "shadow_pnl": total_shadow_pnl,
            "shadow_win_rate": shadow_wins / max(shadow_trades, 1),
            "open_shadow_trades": len(self.open_shadow_trades)
        }
    
    def summary(self) -> str:
        """Generate summary string for logging."""
        stats = self.get_stats()
        return (
            f"📊 Alpha Shadow Summary: "
            f"Agree={stats['agreement_rate']:.1%} | "
            f"Rule Trade={stats['rule_trade_rate']:.1%} | "
            f"PPO Trade={stats['ppo_trade_rate']:.1%} | "
            f"Shadow PnL=${stats['shadow_pnl']:.2f}"
        )


# =============================================================================
# Singleton Access
# =============================================================================

_shadow_runner: Optional[AlphaShadowRunner] = None


def get_alpha_shadow(enabled: bool = True) -> AlphaShadowRunner:
    """Get singleton Alpha Shadow Runner."""
    global _shadow_runner
    if _shadow_runner is None:
        _shadow_runner = AlphaShadowRunner(enabled=enabled)
    return _shadow_runner


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha Shadow Mode Test")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    
    shadow = AlphaShadowRunner()
    
    # Simulate comparisons
    for i in range(10):
        # Mock market data
        mock_row = pd.Series({
            'ema20': 2030 + np.random.randn() * 5,
            'ema50': 2028 + np.random.randn() * 3,
            'rsi14': 50 + np.random.randn() * 20,
            'atr14': 2.5 + np.random.rand(),
            'spread': 0.1 + np.random.rand() * 0.1,
            'hour': np.random.randint(0, 24)
        })
        
        # Mock rule action
        rule_actions = ["HOLD", "BUY", "SELL"]
        rule_action = np.random.choice(rule_actions, p=[0.6, 0.2, 0.2])
        
        ppo_action, conf, agreed = shadow.compare(
            rule_action=rule_action,
            cycle=i + 1,
            market_row=mock_row,
            open_positions=np.random.randint(0, 3),
            floating_dd=np.random.uniform(0, 0.05),
            guardian_state=0,
            current_price=2030 + np.random.randn() * 10
        )
        
    print("\n" + shadow.summary())
    print(f"\nFull stats: {shadow.get_stats()}")
    print("=" * 60)
