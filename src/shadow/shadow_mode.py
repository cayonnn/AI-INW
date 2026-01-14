# src/shadow/shadow_mode.py
"""
Shadow Mode - Multi-Mode Simulation Engine
==========================================

Simulates multiple trading modes simultaneously:
- Live trades use one mode (e.g., NEUTRAL)
- Shadow simulates ALPHA and DEFENSIVE
- Compares performance to inform mode switching

Benefits:
- See which mode would have performed best
- Use as input for auto mode switch
- No risk - pure simulation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("SHADOW_MODE")


class ShadowMode(str, Enum):
    """Shadow simulation modes."""
    ALPHA = "ALPHA"
    NEUTRAL = "NEUTRAL"
    DEFENSIVE = "DEFENSIVE"


@dataclass
class ShadowTrade:
    """Simulated shadow trade."""
    mode: ShadowMode
    direction: str          # BUY or SELL
    entry_price: float
    entry_time: datetime
    volume: float
    sl_price: float
    tp_price: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    is_open: bool = True


@dataclass
class ShadowAccountState:
    """Shadow account state for a mode."""
    mode: ShadowMode
    starting_balance: float
    current_balance: float
    equity: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    open_trades: List[ShadowTrade] = field(default_factory=list)
    closed_trades: List[ShadowTrade] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return self.winning_trades / self.total_trades
    
    @property
    def profit_pct(self) -> float:
        if self.starting_balance == 0:
            return 0
        return ((self.current_balance - self.starting_balance) / 
                self.starting_balance) * 100


# Mode risk multipliers (matches MODE_PROFILES)
MODE_RISK_MULT = {
    ShadowMode.ALPHA: 1.3,
    ShadowMode.NEUTRAL: 1.0,
    ShadowMode.DEFENSIVE: 0.6,
}


class ShadowModeSimulator:
    """
    Shadow Mode Simulator.
    
    Runs parallel simulations for each mode to compare performance.
    
    Usage:
    1. On each signal, simulate trades for all modes
    2. Update prices to close/update trades
    3. Compare performance between modes
    """
    
    def __init__(self, starting_balance: float = 1000.0):
        """
        Initialize Shadow Mode Simulator.
        
        Args:
            starting_balance: Starting balance for each shadow account
        """
        self.starting_balance = starting_balance
        
        # Create shadow accounts for each mode
        self.accounts: Dict[ShadowMode, ShadowAccountState] = {}
        for mode in ShadowMode:
            self.accounts[mode] = ShadowAccountState(
                mode=mode,
                starting_balance=starting_balance,
                current_balance=starting_balance,
                equity=starting_balance,
                peak_balance=starting_balance
            )
        
        logger.info(f"ShadowModeSimulator initialized: balance=${starting_balance}")
    
    def simulate_signal(
        self,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        base_volume: float = 0.01
    ) -> Dict[ShadowMode, ShadowTrade]:
        """
        Simulate a signal across all modes.
        
        Args:
            direction: BUY or SELL
            entry_price: Entry price
            sl_price: Stop loss price
            tp_price: Take profit price
            base_volume: Base volume (will be scaled by mode)
            
        Returns:
            Dictionary of shadow trades by mode
        """
        trades = {}
        
        for mode in ShadowMode:
            account = self.accounts[mode]
            
            # Scale volume by mode risk multiplier
            volume = base_volume * MODE_RISK_MULT[mode]
            
            trade = ShadowTrade(
                mode=mode,
                direction=direction,
                entry_price=entry_price,
                entry_time=datetime.now(),
                volume=volume,
                sl_price=sl_price,
                tp_price=tp_price
            )
            
            account.open_trades.append(trade)
            trades[mode] = trade
        
        return trades
    
    def update_prices(
        self,
        current_price: float,
        high_price: Optional[float] = None,
        low_price: Optional[float] = None
    ) -> Dict[ShadowMode, List[ShadowTrade]]:
        """
        Update all shadow trades with current prices.
        
        Args:
            current_price: Current market price
            high_price: High since last update (for SL/TP check)
            low_price: Low since last update (for SL/TP check)
            
        Returns:
            Dictionary of closed trades by mode
        """
        if high_price is None:
            high_price = current_price
        if low_price is None:
            low_price = current_price
        
        closed = {}
        
        for mode, account in self.accounts.items():
            closed[mode] = []
            
            for trade in account.open_trades[:]:  # Copy list for iteration
                hit_sl = False
                hit_tp = False
                
                if trade.direction == "BUY":
                    # Check SL
                    if low_price <= trade.sl_price:
                        hit_sl = True
                        trade.exit_price = trade.sl_price
                    # Check TP
                    elif high_price >= trade.tp_price:
                        hit_tp = True
                        trade.exit_price = trade.tp_price
                else:  # SELL
                    # Check SL
                    if high_price >= trade.sl_price:
                        hit_sl = True
                        trade.exit_price = trade.sl_price
                    # Check TP
                    elif low_price <= trade.tp_price:
                        hit_tp = True
                        trade.exit_price = trade.tp_price
                
                if hit_sl or hit_tp:
                    self._close_trade(trade, account)
                    closed[mode].append(trade)
        
        # Update equity for open positions
        self._update_equity(current_price)
        
        return closed
    
    def _close_trade(self, trade: ShadowTrade, account: ShadowAccountState):
        """Close a shadow trade and update account."""
        trade.exit_time = datetime.now()
        trade.is_open = False
        
        # Calculate PnL
        if trade.direction == "BUY":
            pnl = (trade.exit_price - trade.entry_price) * trade.volume * 100
        else:
            pnl = (trade.entry_price - trade.exit_price) * trade.volume * 100
        
        trade.pnl = round(pnl, 2)
        
        # Update account
        account.current_balance += trade.pnl
        account.total_pnl += trade.pnl
        account.total_trades += 1
        
        if trade.pnl > 0:
            account.winning_trades += 1
        else:
            account.losing_trades += 1
        
        # Update peak and drawdown
        if account.current_balance > account.peak_balance:
            account.peak_balance = account.current_balance
        
        dd = ((account.peak_balance - account.current_balance) / 
              account.peak_balance) * 100 if account.peak_balance > 0 else 0
        if dd > account.max_drawdown:
            account.max_drawdown = dd
        
        # Move to closed
        account.open_trades.remove(trade)
        account.closed_trades.append(trade)
    
    def _update_equity(self, current_price: float):
        """Update equity for all accounts based on open positions."""
        for mode, account in self.accounts.items():
            unrealized_pnl = 0
            
            for trade in account.open_trades:
                if trade.direction == "BUY":
                    unrealized_pnl += (current_price - trade.entry_price) * trade.volume * 100
                else:
                    unrealized_pnl += (trade.entry_price - current_price) * trade.volume * 100
            
            account.equity = account.current_balance + unrealized_pnl
    
    def get_best_mode(self) -> ShadowMode:
        """Get the best performing mode based on profit %."""
        best_mode = ShadowMode.NEUTRAL
        best_profit = -float('inf')
        
        for mode, account in self.accounts.items():
            if account.profit_pct > best_profit:
                best_profit = account.profit_pct
                best_mode = mode
        
        return best_mode
    
    def get_mode_comparison(self) -> Dict[str, Dict]:
        """Get comparison of all modes for dashboard."""
        comparison = {}
        
        for mode, account in self.accounts.items():
            comparison[mode.value] = {
                "balance": round(account.current_balance, 2),
                "equity": round(account.equity, 2),
                "profit_pct": round(account.profit_pct, 2),
                "total_trades": account.total_trades,
                "win_rate": round(account.win_rate * 100, 1),
                "max_drawdown": round(account.max_drawdown, 2),
                "open_positions": len(account.open_trades),
            }
        
        return comparison
    
    def get_recommendation(self) -> Dict:
        """Get trading mode recommendation based on shadow performance."""
        best_mode = self.get_best_mode()
        best_account = self.accounts[best_mode]
        
        return {
            "recommended_mode": best_mode.value,
            "profit_pct": round(best_account.profit_pct, 2),
            "win_rate": round(best_account.win_rate * 100, 1),
            "confidence": self._calc_recommendation_confidence(),
            "comparison": self.get_mode_comparison()
        }
    
    def _calc_recommendation_confidence(self) -> float:
        """Calculate confidence in recommendation."""
        profits = [acc.profit_pct for acc in self.accounts.values()]
        
        if max(profits) == min(profits):
            return 0.5  # All same
        
        # Higher confidence if best mode is clearly better
        diff = max(profits) - sorted(profits)[-2]  # Diff from second best
        confidence = min(1.0, 0.5 + diff / 10)
        
        return round(confidence, 2)


# Singleton instance
_simulator: Optional[ShadowModeSimulator] = None


def get_shadow_simulator(starting_balance: float = 1000.0) -> ShadowModeSimulator:
    """Get or create singleton ShadowModeSimulator."""
    global _simulator
    if _simulator is None:
        _simulator = ShadowModeSimulator(starting_balance)
    return _simulator


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ShadowModeSimulator Test")
    print("=" * 60)
    
    simulator = ShadowModeSimulator(starting_balance=1000)
    
    # Simulate some trades
    print("\n--- Simulating Trades ---")
    
    # Trade 1: BUY at 2650, SL 2640, TP 2670
    simulator.simulate_signal("BUY", 2650, 2640, 2670, 0.02)
    print("Trade 1: BUY @ 2650")
    
    # Price moves to 2670 (hits TP)
    closed = simulator.update_prices(2670, 2670, 2648)
    print(f"Closed trades: {sum(len(t) for t in closed.values())}")
    
    # Trade 2: SELL at 2680, SL 2695, TP 2650
    simulator.simulate_signal("SELL", 2680, 2695, 2650, 0.02)
    print("Trade 2: SELL @ 2680")
    
    # Price moves to 2695 (hits SL)
    closed = simulator.update_prices(2695, 2695, 2678)
    print(f"Closed trades: {sum(len(t) for t in closed.values())}")
    
    # Show results
    print("\n--- Mode Comparison ---")
    comparison = simulator.get_mode_comparison()
    for mode, stats in comparison.items():
        print(f"{mode}: ${stats['balance']} ({stats['profit_pct']:+.1f}%), "
              f"WR={stats['win_rate']}%, Trades={stats['total_trades']}")
    
    print("\n--- Recommendation ---")
    rec = simulator.get_recommendation()
    print(f"Recommended: {rec['recommended_mode']} (conf={rec['confidence']})")
