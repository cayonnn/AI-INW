"""
AI Trading System - Backtesting Engine
========================================
Professional backtesting with realistic execution simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str
    lot_size: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "OPEN"
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Backtest performance results."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Features:
    - Realistic spread and slippage simulation
    - Walk-forward support
    - Detailed performance metrics
    """
    
    def __init__(self, initial_balance: float = 10000.0, spread_pips: float = 1.0,
                 slippage_pips: float = 0.5, commission_per_lot: float = 7.0):
        self.initial_balance = initial_balance
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_per_lot = commission_per_lot
        
        self.balance = initial_balance
        self.equity = initial_balance
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_balance]
    
    def reset(self):
        """Reset for new backtest."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = [self.initial_balance]
    
    def run(self, df: pd.DataFrame, signals: pd.DataFrame,
            pip_value: float = 10.0, pip_size: float = 0.0001) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: OHLCV DataFrame
            signals: DataFrame with columns [direction, sl_pips, tp_pips, lot_size]
            pip_value: Value per pip per lot
            pip_size: Price per pip
        
        Returns:
            BacktestResult with performance metrics
        """
        self.reset()
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = df.index[i]
            
            # Check existing positions
            self._check_exits(row, pip_size, pip_value, current_time)
            
            # Process new signal
            if i < len(signals):
                signal = signals.iloc[i]
                if signal.get("direction") in ["LONG", "SHORT"]:
                    self._open_trade(row, signal, pip_size, pip_value, current_time)
            
            # Update equity
            self._update_equity(row, pip_size, pip_value)
        
        # Close any remaining positions
        self._close_all(df.iloc[-1], pip_size, pip_value, df.index[-1])
        
        return self._calculate_metrics()
    
    def _open_trade(self, bar: pd.Series, signal: pd.Series,
                   pip_size: float, pip_value: float, time: datetime):
        """Open new trade."""
        direction = signal["direction"]
        lot_size = signal.get("lot_size", 0.1)
        sl_pips = signal.get("sl_pips", 50)
        tp_pips = signal.get("tp_pips", 100)
        
        # Apply spread and slippage
        spread_cost = self.spread_pips * pip_size
        slippage_cost = self.slippage_pips * pip_size
        
        if direction == "LONG":
            entry_price = bar["close"] + spread_cost + slippage_cost
            sl = entry_price - (sl_pips * pip_size)
            tp = entry_price + (tp_pips * pip_size)
        else:
            entry_price = bar["close"] - slippage_cost
            sl = entry_price + (sl_pips * pip_size)
            tp = entry_price - (tp_pips * pip_size)
        
        # Commission
        commission = self.commission_per_lot * lot_size
        self.balance -= commission
        
        trade = Trade(
            entry_time=time,
            exit_time=None,
            symbol="BACKTEST",
            direction=direction,
            lot_size=lot_size,
            entry_price=entry_price,
            exit_price=None,
            stop_loss=sl,
            take_profit=tp,
            status="OPEN"
        )
        
        self.open_trades.append(trade)
    
    def _check_exits(self, bar: pd.Series, pip_size: float,
                    pip_value: float, time: datetime):
        """Check SL/TP hits for open trades."""
        for trade in self.open_trades[:]:
            hit_sl = False
            hit_tp = False
            
            if trade.direction == "LONG":
                if bar["low"] <= trade.stop_loss:
                    hit_sl = True
                    exit_price = trade.stop_loss
                elif bar["high"] >= trade.take_profit:
                    hit_tp = True
                    exit_price = trade.take_profit
            else:
                if bar["high"] >= trade.stop_loss:
                    hit_sl = True
                    exit_price = trade.stop_loss
                elif bar["low"] <= trade.take_profit:
                    hit_tp = True
                    exit_price = trade.take_profit
            
            if hit_sl or hit_tp:
                self._close_trade(trade, exit_price, pip_size, pip_value, time,
                                "SL Hit" if hit_sl else "TP Hit")
    
    def _close_trade(self, trade: Trade, exit_price: float, pip_size: float,
                    pip_value: float, time: datetime, reason: str):
        """Close trade and calculate PnL."""
        trade.exit_time = time
        trade.exit_price = exit_price
        trade.status = "CLOSED"
        trade.exit_reason = reason
        
        # Calculate PnL
        if trade.direction == "LONG":
            pips = (exit_price - trade.entry_price) / pip_size
        else:
            pips = (trade.entry_price - exit_price) / pip_size
        
        trade.pnl = pips * pip_value * trade.lot_size
        trade.pnl_pct = (trade.pnl / self.initial_balance) * 100
        
        self.balance += trade.pnl
        
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)
    
    def _close_all(self, bar: pd.Series, pip_size: float,
                  pip_value: float, time: datetime):
        """Close all open positions."""
        for trade in self.open_trades[:]:
            exit_price = bar["close"]
            self._close_trade(trade, exit_price, pip_size, pip_value, time, "End of Backtest")
    
    def _update_equity(self, bar: pd.Series, pip_size: float, pip_value: float):
        """Update equity curve."""
        unrealized = 0
        for trade in self.open_trades:
            if trade.direction == "LONG":
                pips = (bar["close"] - trade.entry_price) / pip_size
            else:
                pips = (trade.entry_price - bar["close"]) / pip_size
            unrealized += pips * pip_value * trade.lot_size
        
        self.equity = self.balance + unrealized
        self.equity_curve.append(self.equity)
    
    def _calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics."""
        result = BacktestResult()
        result.trades = self.closed_trades
        result.equity_curve = self.equity_curve
        result.total_trades = len(self.closed_trades)
        
        if result.total_trades == 0:
            return result
        
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl < 0]
        
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        
        result.total_pnl = sum(t.pnl for t in self.closed_trades)
        result.total_pnl_pct = (result.total_pnl / self.initial_balance) * 100
        
        # Drawdown
        equity_arr = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = peak - equity_arr
        result.max_drawdown = np.max(drawdown)
        result.max_drawdown_pct = (result.max_drawdown / np.max(peak)) * 100 if np.max(peak) > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Win/Loss averages
        result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        result.largest_win = max([t.pnl for t in wins]) if wins else 0
        result.largest_loss = min([t.pnl for t in losses]) if losses else 0
        
        # Expectancy
        result.expectancy = (result.win_rate * result.avg_win) - ((1 - result.win_rate) * abs(result.avg_loss))
        
        # Sharpe ratio
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return result
