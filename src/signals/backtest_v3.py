# src/signals/backtest_v3.py
"""
Backtest V3 + Signal Quality Report
====================================
Backtest SignalEngine V3 (Rule + AI) with quality metrics.

Features:
- Run backtest on H1 + H4 data
- Use SignalEngine V3 hybrid signals
- Calculate Win Rate, Profit Factor, PnL, Max Drawdown
- Export dataset for AI training

Usage:
    python src/signals/backtest_v3.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.signals.signal_engine_v3 import SignalEngineV3


# =========================
# REPORT STRUCTURE
# =========================

@dataclass
class BacktestReportV3:
    """Signal Quality Report V3."""
    symbol: str
    total_trades: int
    buy_count: int
    sell_count: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    avg_pnl: float
    best_trade: float
    worst_trade: float


# =========================
# BACKTEST V3
# =========================

class BacktestV3:
    """
    Backtest SignalEngine V3.
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        bar_count: int = 2000,
        initial_balance: float = 10000.0,
        backtest_mode: bool = True
    ):
        self.symbol = symbol
        self.bar_count = bar_count
        self.initial_balance = initial_balance
        
        # Backtest mode: disable cooldown and duplicate filter
        if backtest_mode:
            self.engine = SignalEngineV3(
                max_positions=100,   # No limit
                cooldown_sec=0       # No cooldown
            )
        else:
            self.engine = SignalEngineV3()
        
        self.trades: List[Dict] = []
        self.balance = initial_balance
    
    def load_data_mt5(self) -> tuple:
        """Load data from MT5."""
        try:
            from src.data.mt5_connector import MT5Connector
            
            mt5 = MT5Connector()
            mt5.connect()
            
            rates_h1 = mt5.get_rates(self.symbol, "H1", self.bar_count)
            rates_h4 = mt5.get_rates(self.symbol, "H4", self.bar_count // 4)
            
            df_h1 = pd.DataFrame(rates_h1)
            df_h4 = pd.DataFrame(rates_h4)
            
            print(f"✅ Loaded H1: {len(df_h1)} bars, H4: {len(df_h4)} bars")
            return df_h1, df_h4
        except Exception as e:
            print(f"⚠️ MT5 not available: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self) -> tuple:
        """Create sample data."""
        np.random.seed(42)
        n = self.bar_count
        price = 2000 + np.cumsum(np.random.randn(n) * 10)
        
        df_h1 = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": price,
            "high": price + np.abs(np.random.randn(n) * 5),
            "low": price - np.abs(np.random.randn(n) * 5),
            "close": price + np.random.randn(n) * 3,
            "volume": np.random.randint(100, 1000, n)
        })
        
        # H4 = resample from H1
        df_h4 = df_h1.copy()
        
        print(f"📊 Created sample data: {n} bars")
        return df_h1, df_h4
    
    def run_backtest(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame):
        """Run backtest on data."""
        print("\n🔄 Running Backtest V3...")
        
        self.trades = []
        self.balance = self.initial_balance
        
        lookback = 60  # Min bars for indicators
        
        for i in range(lookback, len(df_h1) - 1):
            # Slice data up to current bar
            df_h1_slice = df_h1.iloc[:i+1].copy()
            df_h4_slice = df_h4.iloc[:min(i//4 + 1, len(df_h4))].copy()
            
            # Generate signal
            signal, info = self.engine.generate_signal(df_h1_slice, df_h4_slice, self.symbol)
            
            # Execute trade
            if signal in ["BUY", "SELL"]:
                open_price = df_h1.iloc[i]["close"]
                close_price = df_h1.iloc[i + 1]["close"]
                
                direction = 1 if signal == "BUY" else -1
                pnl = direction * (close_price - open_price)
                
                self.trades.append({
                    "bar": i,
                    "signal": signal,
                    "open_price": open_price,
                    "close_price": close_price,
                    "pnl": pnl,
                    "info": info
                })
                
                self.balance += pnl
        
        print(f"✅ Backtest complete: {len(self.trades)} trades")
    
    def generate_report(self) -> BacktestReportV3:
        """Generate quality report."""
        if not self.trades:
            return BacktestReportV3(
                symbol=self.symbol,
                total_trades=0,
                buy_count=0,
                sell_count=0,
                win_rate=0,
                profit_factor=0,
                total_pnl=0,
                max_drawdown=0,
                avg_pnl=0,
                best_trade=0,
                worst_trade=0
            )
        
        df = pd.DataFrame(self.trades)
        
        # Counts
        total = len(df)
        buy_count = len(df[df["signal"] == "BUY"])
        sell_count = len(df[df["signal"] == "SELL"])
        
        # Win rate
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        win_rate = len(wins) / total if total > 0 else 0
        
        # Profit factor
        total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
        total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.01
        profit_factor = total_wins / total_losses
        
        # PnL
        total_pnl = df["pnl"].sum()
        avg_pnl = df["pnl"].mean()
        best_trade = df["pnl"].max()
        worst_trade = df["pnl"].min()
        
        # Drawdown
        cumsum = df["pnl"].cumsum()
        running_max = cumsum.cummax()
        drawdown = cumsum - running_max
        max_drawdown = abs(drawdown.min())
        
        return BacktestReportV3(
            symbol=self.symbol,
            total_trades=total,
            buy_count=buy_count,
            sell_count=sell_count,
            win_rate=round(win_rate, 3),
            profit_factor=round(profit_factor, 2),
            total_pnl=round(total_pnl, 2),
            max_drawdown=round(max_drawdown, 2),
            avg_pnl=round(avg_pnl, 2),
            best_trade=round(best_trade, 2),
            worst_trade=round(worst_trade, 2)
        )
    
    def print_report(self):
        """Print formatted report."""
        report = self.generate_report()
        
        print("\n" + "=" * 50)
        print(f"🧪 SIGNAL QUALITY REPORT V3: {report.symbol}")
        print("=" * 50)
        print(f"Total Trades:     {report.total_trades}")
        print(f"   BUY:           {report.buy_count}")
        print(f"   SELL:          {report.sell_count}")
        print(f"Win Rate:         {report.win_rate:.1%}")
        print(f"Profit Factor:    {report.profit_factor:.2f}")
        print(f"Total PnL:        {report.total_pnl:.2f}")
        print(f"Avg PnL/Trade:    {report.avg_pnl:.2f}")
        print(f"Best Trade:       {report.best_trade:.2f}")
        print(f"Worst Trade:      {report.worst_trade:.2f}")
        print(f"Max Drawdown:     {report.max_drawdown:.2f}")
        print("=" * 50)
        
        return report
    
    def plot_equity_curve(self, save_path: str = "data/equity_curve.png"):
        """Plot equity curve."""
        if not self.trades:
            print("⚠️ No trades to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame(self.trades)
            df["cumulative_pnl"] = df["pnl"].cumsum()
            df["equity"] = self.initial_balance + df["cumulative_pnl"]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            axes[0].plot(df.index, df["equity"], 'b-', linewidth=1.5)
            axes[0].fill_between(df.index, self.initial_balance, df["equity"], 
                               where=df["equity"] >= self.initial_balance, 
                               alpha=0.3, color='green', label='Profit')
            axes[0].fill_between(df.index, self.initial_balance, df["equity"], 
                               where=df["equity"] < self.initial_balance, 
                               alpha=0.3, color='red', label='Loss')
            axes[0].axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.5)
            axes[0].set_title(f"Equity Curve - {self.symbol}")
            axes[0].set_ylabel("Balance")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # PnL per trade
            colors = ['green' if x > 0 else 'red' for x in df["pnl"]]
            axes[1].bar(df.index, df["pnl"], color=colors, alpha=0.7)
            axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[1].set_title("PnL per Trade")
            axes[1].set_xlabel("Trade #")
            axes[1].set_ylabel("PnL")
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=100)
            print(f"📊 Equity curve saved: {save_path}")
            
            plt.close()
            
        except ImportError:
            print("⚠️ matplotlib not installed - skipping plot")
    
    def save_to_database(self, db_path: str = "data/backtest_v3.db"):
        """Save results to SQLite database."""
        import sqlite3
        
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_reports_v3 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                total_pnl REAL,
                max_drawdown REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades_v3 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER,
                bar INTEGER,
                signal TEXT,
                open_price REAL,
                close_price REAL,
                pnl REAL,
                info TEXT
            )
        """)
        
        # Save report
        report = self.generate_report()
        cursor.execute("""
            INSERT INTO backtest_reports_v3
            (timestamp, symbol, total_trades, win_rate, profit_factor, total_pnl, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            report.symbol,
            report.total_trades,
            report.win_rate,
            report.profit_factor,
            report.total_pnl,
            report.max_drawdown
        ))
        
        report_id = cursor.lastrowid
        
        # Save trades
        for trade in self.trades:
            cursor.execute("""
                INSERT INTO backtest_trades_v3
                (report_id, bar, signal, open_price, close_price, pnl, info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                trade["bar"],
                trade["signal"],
                trade["open_price"],
                trade["close_price"],
                trade["pnl"],
                trade["info"]
            ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved to database: {db_path}")
    
    def export_csv(self, path: str = "data/backtest_v3_trades.csv"):
        """Export trades to CSV."""
        if not self.trades:
            print("⚠️ No trades to export")
            return
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        df = pd.DataFrame(self.trades)
        df.to_csv(path, index=False)
        print(f"✅ Exported to: {path}")


# =========================
# MAIN
# =========================

def run_backtest_v3(use_mt5: bool = True, bar_count: int = 2000):
    """Run full backtest pipeline."""
    bt = BacktestV3(symbol="XAUUSD", bar_count=bar_count, backtest_mode=True)
    
    if use_mt5:
        df_h1, df_h4 = bt.load_data_mt5()
    else:
        df_h1, df_h4 = bt.create_sample_data()
    
    bt.run_backtest(df_h1, df_h4)
    bt.print_report()
    bt.plot_equity_curve()
    bt.save_to_database()
    bt.export_csv()
    
    return bt


if __name__ == "__main__":
    print("🧪 Backtest V3 - SignalEngine V3")
    print("=" * 50)
    
    bt = run_backtest_v3(use_mt5=True, bar_count=2000)
