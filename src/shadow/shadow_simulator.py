# src/shadow/shadow_simulator.py
"""
Shadow Simulator
=================

Simulates "what Guardian blocked" to calculate:
    - DD Avoided
    - Missed Profit
    - Freeze Cost

Paper Statement:
    "We demonstrate that Guardian interventions prevented X% of
     potential drawdown while sacrificing only Y% of potential profit."
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("SHADOW_SIM")


@dataclass
class ShadowTrade:
    """A simulated trade."""
    timestamp: datetime
    signal: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    pnl: float
    equity_after: float
    dd_pct: float
    was_blocked: bool
    blocked_reason: str = ""


class ShadowSimulator:
    """
    Simulates trades that Guardian would have blocked.
    
    Answers the question:
        "What would have happened if Guardian didn't exist?"
    
    Features:
        - Lightweight vector backtest
        - DD tracking
        - Profit/loss attribution
        - Export for paper figures
    """
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.max_equity = initial_balance
        
        self.trades: List[ShadowTrade] = []
        self.blocked_trades: List[ShadowTrade] = []
        
        # Metrics
        self.dd_avoided = 0.0
        self.missed_profit = 0.0
        self.freeze_cost = 0.0
        
        logger.info(f"🧪 ShadowSimulator initialized (balance=${initial_balance})")
    
    def simulate_trade(
        self,
        signal: str,
        price: float,
        sl: float,
        tp: float,
        direction: str,
        timestamp: datetime = None,
        was_blocked: bool = False,
        blocked_reason: str = ""
    ) -> ShadowTrade:
        """
        Simulate a single trade.
        
        Args:
            signal: Trade signal source
            price: Entry price
            sl: Stop loss price
            tp: Take profit price
            direction: "LONG" or "SHORT"
            timestamp: Trade time
            was_blocked: Whether Guardian blocked this
            blocked_reason: Why it was blocked
            
        Returns:
            ShadowTrade result
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate R-multiple
        risk = abs(price - sl)
        reward = abs(tp - price)
        
        if risk <= 0:
            r_multiple = 0
        else:
            r_multiple = reward / risk
        
        # Simulate outcome (simplified: 50% win rate)
        # In real use, this would use historical data
        win = np.random.random() > 0.45
        
        if win:
            pnl = r_multiple * 0.01 * self.balance
        else:
            pnl = -1.0 * 0.01 * self.balance
        
        # Update equity if not blocked
        if not was_blocked:
            self.equity += pnl
            self.max_equity = max(self.max_equity, self.equity)
        else:
            # Track what would have happened
            if pnl < 0:
                self.dd_avoided += abs(pnl)
            else:
                self.missed_profit += pnl
        
        # Calculate DD
        dd_pct = (self.max_equity - self.equity) / self.max_equity * 100
        
        trade = ShadowTrade(
            timestamp=timestamp,
            signal=signal,
            direction=direction,
            entry_price=price,
            sl=sl,
            tp=tp,
            pnl=pnl,
            equity_after=self.equity,
            dd_pct=dd_pct,
            was_blocked=was_blocked,
            blocked_reason=blocked_reason
        )
        
        if was_blocked:
            self.blocked_trades.append(trade)
        else:
            self.trades.append(trade)
        
        return trade
    
    def summary(self) -> Dict:
        """Get simulation summary."""
        all_trades = self.trades + self.blocked_trades
        
        if not all_trades:
            return {
                "total_trades": 0,
                "executed": 0,
                "blocked": 0,
                "net_pnl": 0,
                "max_dd": 0,
                "dd_avoided": 0,
                "missed_profit": 0
            }
        
        df = pd.DataFrame([{
            "pnl": t.pnl,
            "dd": t.dd_pct,
            "blocked": t.was_blocked
        } for t in all_trades])
        
        return {
            "total_trades": len(all_trades),
            "executed": len(self.trades),
            "blocked": len(self.blocked_trades),
            "net_pnl": sum(t.pnl for t in self.trades),
            "max_dd": max(t.dd_pct for t in self.trades) if self.trades else 0,
            "winrate": sum(1 for t in self.trades if t.pnl > 0) / max(len(self.trades), 1),
            "dd_avoided": self.dd_avoided,
            "missed_profit": self.missed_profit,
            "guardian_value": self.dd_avoided - self.missed_profit
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export all trades as DataFrame."""
        all_trades = self.trades + self.blocked_trades
        return pd.DataFrame([{
            "time": t.timestamp,
            "signal": t.signal,
            "direction": t.direction,
            "pnl": t.pnl,
            "equity": t.equity_after,
            "dd": t.dd_pct,
            "blocked": t.was_blocked,
            "reason": t.blocked_reason
        } for t in sorted(all_trades, key=lambda x: x.timestamp)])
    
    def export_csv(self, path: str = "logs/shadow_trades.csv"):
        """Export to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"📁 Exported shadow trades: {path}")
        return path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Shadow Simulator Test")
    print("=" * 60)
    
    sim = ShadowSimulator(initial_balance=1000)
    
    # Simulate some trades
    for i in range(20):
        blocked = i % 4 == 0  # Block every 4th trade
        sim.simulate_trade(
            signal="RULE",
            price=2000 + np.random.uniform(-10, 10),
            sl=1990,
            tp=2020,
            direction="LONG",
            was_blocked=blocked,
            blocked_reason="High DD" if blocked else ""
        )
    
    print(f"\nSummary: {sim.summary()}")
    
    # Export
    sim.export_csv("logs/shadow_trades.csv")
    
    print("=" * 60)
