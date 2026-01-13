"""
backtest_adapter.py
====================
Backtest Adapter for Signal Quality Report

Features:
- Use historical OHLC data
- Generate signals using Rule-Based or AI
- Track PnL, MAE (Maximum Adverse Excursion), MFE (Maximum Favorable Excursion)
- Generate quality report for signal evaluation

Usage:
    adapter = BacktestAdapter(df, "XAUUSD")
    adapter.generate_signals()
    results = adapter.run_backtest()
    report = adapter.generate_report()
"""

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =========================
# INDICATOR FUNCTIONS (inline to avoid import issues)
# =========================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# =========================
# REPORT STRUCTURE
# =========================

@dataclass
class BacktestReport:
    """Signal quality report."""
    symbol: str
    total_trades: int
    win_rate: float
    avg_mae: float
    avg_mfe: float
    total_pnl: float
    profit_factor: float
    avg_trade_duration: float  # in bars
    max_drawdown: float


# =========================
# BACKTEST ADAPTER
# =========================

class BacktestAdapter:
    """
    Adapter for backtesting Rule-Based or AI signals.
    
    Tracks:
    - Win Rate
    - MAE (Maximum Adverse Excursion) - how far goes against you
    - MFE (Maximum Favorable Excursion) - how far goes in your favor
    - PnL per trade
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str = "XAUUSD",
        signal_generator: Callable = None
    ):
        """
        Args:
            df: OHLC DataFrame with columns: open, high, low, close, volume, time
            symbol: Trading symbol
            signal_generator: Optional custom signal generator function
        """
        self.df = df.copy()
        self.symbol = symbol
        self.signal_generator = signal_generator
        
        self.signals: List[str] = []
        self.trades: List[Dict] = []
    
    def generate_signals(self, lookback: int = 50):
        """
        Generate signals for each bar using SignalEngine V2.
        
        Args:
            lookback: Number of bars to use for indicators
        """
        signals = []
        
        for i in range(len(self.df)):
            if i < lookback:
                signals.append("HOLD")
                continue
            
            # Get historical data up to current bar
            hist = self.df.iloc[max(0, i - lookback):i + 1].copy()
            close = hist['close']
            
            # Calculate indicators
            ema_fast = ema(close, 20).iloc[-1] if len(close) >= 20 else close.iloc[-1]
            ema_slow = ema(close, 50).iloc[-1] if len(close) >= 50 else close.iloc[-1]
            atr_val = atr(hist, 14).iloc[-1] if len(hist) >= 14 else 0
            atr_mean = atr(hist, 14).rolling(50).mean().iloc[-1] if len(hist) >= 50 else atr_val
            
            # Handle NaN
            if pd.isna(atr_mean):
                atr_mean = atr_val
            
            vol_threshold = atr_mean * 0.7
            vol_ok = atr_val > vol_threshold
            
            # Generate signal
            if ema_fast > ema_slow and vol_ok:
                signals.append("BUY")
            elif ema_fast < ema_slow and vol_ok:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        self.df["signal"] = signals
        self.signals = signals
        
        print(f"📊 Generated {len(signals)} signals")
        print(f"   BUY:  {signals.count('BUY')}")
        print(f"   SELL: {signals.count('SELL')}")
        print(f"   HOLD: {signals.count('HOLD')}")
    
    def run_backtest(self) -> pd.DataFrame:
        """
        Run backtest on generated signals.
        
        Returns:
            DataFrame with trade results
        """
        if "signal" not in self.df.columns:
            self.generate_signals()
        
        # Initialize columns
        self.df["position"] = 0
        self.df["entry_price"] = np.nan
        self.df["exit_price"] = np.nan
        self.df["pnl"] = 0.0
        self.df["mae"] = 0.0
        self.df["mfe"] = 0.0
        
        pos = 0  # Current position: 0 = flat, 1 = long, -1 = short
        entry = 0  # Entry price
        entry_bar = 0  # Entry bar index
        mae = 0  # Maximum Adverse Excursion
        mfe = 0  # Maximum Favorable Excursion
        
        trades = []
        
        for i in range(len(self.df)):
            sig = self.df["signal"].iloc[i]
            high = self.df["high"].iloc[i]
            low = self.df["low"].iloc[i]
            close = self.df["close"].iloc[i]
            
            # Entry Logic
            if sig == "BUY" and pos == 0:
                pos = 1
                entry = close
                entry_bar = i
                mae = 0
                mfe = 0
            elif sig == "SELL" and pos == 0:
                pos = -1
                entry = close
                entry_bar = i
                mae = 0
                mfe = 0
            
            # Update MAE/MFE while in position
            if pos != 0:
                if pos == 1:  # Long
                    mae = min(mae, low - entry)
                    mfe = max(mfe, high - entry)
                else:  # Short
                    mae = min(mae, entry - high)
                    mfe = max(mfe, entry - low)
            
            # Exit Logic (reverse signal or opposite signal)
            if (sig == "SELL" and pos == 1) or (sig == "BUY" and pos == -1):
                exit_price = close
                pnl = (exit_price - entry) * pos
                duration = i - entry_bar
                
                self.df.at[self.df.index[i], "position"] = pos
                self.df.at[self.df.index[i], "entry_price"] = entry
                self.df.at[self.df.index[i], "exit_price"] = exit_price
                self.df.at[self.df.index[i], "pnl"] = pnl
                self.df.at[self.df.index[i], "mae"] = mae
                self.df.at[self.df.index[i], "mfe"] = mfe
                
                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "direction": "BUY" if pos == 1 else "SELL",
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "mae": mae,
                    "mfe": mfe,
                    "duration": duration,
                })
                
                pos = 0
        
        self.trades = trades
        print(f"\n✅ Backtest complete: {len(trades)} trades")
        
        return self.df
    
    def generate_report(self) -> BacktestReport:
        """
        Generate signal quality report.
        
        Returns:
            BacktestReport with key metrics
        """
        if not self.trades:
            return BacktestReport(
                symbol=self.symbol,
                total_trades=0,
                win_rate=0,
                avg_mae=0,
                avg_mfe=0,
                total_pnl=0,
                profit_factor=0,
                avg_trade_duration=0,
                max_drawdown=0
            )
        
        trades_df = pd.DataFrame(self.trades)
        
        # Win rate
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / len(trades_df)
        
        # Profit factor
        total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
        total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / max(total_losses, 0.01)
        
        # Drawdown
        cumsum = trades_df["pnl"].cumsum()
        running_max = cumsum.cummax()
        drawdown = cumsum - running_max
        max_drawdown = abs(drawdown.min())
        
        report = BacktestReport(
            symbol=self.symbol,
            total_trades=len(trades_df),
            win_rate=round(win_rate, 3),
            avg_mae=round(trades_df["mae"].mean(), 2),
            avg_mfe=round(trades_df["mfe"].mean(), 2),
            total_pnl=round(trades_df["pnl"].sum(), 2),
            profit_factor=round(profit_factor, 2),
            avg_trade_duration=round(trades_df["duration"].mean(), 1),
            max_drawdown=round(max_drawdown, 2)
        )
        
        return report
    
    def print_report(self):
        """Print formatted report."""
        report = self.generate_report()
        
        print("\n" + "=" * 50)
        print(f"📊 SIGNAL QUALITY REPORT: {report.symbol}")
        print("=" * 50)
        print(f"Total Trades:     {report.total_trades}")
        print(f"Win Rate:         {report.win_rate:.1%}")
        print(f"Profit Factor:    {report.profit_factor:.2f}")
        print(f"Total PnL:        {report.total_pnl:.2f}")
        print(f"Max Drawdown:     {report.max_drawdown:.2f}")
        print(f"Avg MAE:          {report.avg_mae:.2f}")
        print(f"Avg MFE:          {report.avg_mfe:.2f}")
        print(f"Avg Duration:     {report.avg_trade_duration:.1f} bars")
        print("=" * 50)
        
        return report
    
    def export_dataset(self, path: str = "data/backtest_signals.csv"):
        """Export signals and trades for AI training."""
        self.df.to_csv(path, index=False)
        print(f"✅ Dataset exported: {path}")
    
    def save_to_database(self, db_path: str = "data/backtest.db"):
        """
        Save backtest results to SQLite database.
        
        Tables:
        - backtest_reports: Summary metrics
        - backtest_trades: Individual trades
        """
        import sqlite3
        from datetime import datetime
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                total_trades INTEGER,
                win_rate REAL,
                avg_mae REAL,
                avg_mfe REAL,
                total_pnl REAL,
                profit_factor REAL,
                avg_trade_duration REAL,
                max_drawdown REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER,
                entry_bar INTEGER,
                exit_bar INTEGER,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                mae REAL,
                mfe REAL,
                duration INTEGER,
                FOREIGN KEY (report_id) REFERENCES backtest_reports(id)
            )
        """)
        
        # Save report
        report = self.generate_report()
        cursor.execute("""
            INSERT INTO backtest_reports 
            (timestamp, symbol, total_trades, win_rate, avg_mae, avg_mfe, 
             total_pnl, profit_factor, avg_trade_duration, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            report.symbol,
            report.total_trades,
            report.win_rate,
            report.avg_mae,
            report.avg_mfe,
            report.total_pnl,
            report.profit_factor,
            report.avg_trade_duration,
            report.max_drawdown
        ))
        
        report_id = cursor.lastrowid
        
        # Save trades
        for trade in self.trades:
            cursor.execute("""
                INSERT INTO backtest_trades
                (report_id, entry_bar, exit_bar, direction, entry_price, 
                 exit_price, pnl, mae, mfe, duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                trade["entry_bar"],
                trade["exit_bar"],
                trade["direction"],
                trade["entry_price"],
                trade["exit_price"],
                trade["pnl"],
                trade["mae"],
                trade["mfe"],
                trade["duration"]
            ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved to database: {db_path}")
        print(f"   Report ID: {report_id}")
        print(f"   Trades: {len(self.trades)}")
    
    @staticmethod
    def get_backtest_history(db_path: str = "data/backtest.db") -> pd.DataFrame:
        """Get all backtest reports from database."""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM backtest_reports ORDER BY timestamp DESC", conn)
        conn.close()
        
        return df
    
    @staticmethod
    def get_trades_by_report(report_id: int, db_path: str = "data/backtest.db") -> pd.DataFrame:
        """Get trades for specific report."""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM backtest_trades WHERE report_id = {report_id}", 
            conn
        )
        conn.close()
        
        return df


# =========================
# QUICK BACKTEST FUNCTION
# =========================

def run_quick_backtest(
    csv_path: str = None,
    mt5_connector = None,
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    bars: int = 500
) -> BacktestReport:
    """
    Quick backtest function.
    
    Args:
        csv_path: Path to OHLC CSV file
        mt5_connector: MT5 connector to get live data
        symbol: Trading symbol
        timeframe: Timeframe
        bars: Number of bars
        
    Returns:
        BacktestReport
    """
    # Load data
    if csv_path:
        df = pd.read_csv(csv_path)
    elif mt5_connector:
        rates = mt5_connector.get_rates(symbol, timeframe, bars)
        df = pd.DataFrame(rates)
    else:
        raise ValueError("Provide csv_path or mt5_connector")
    
    # Run backtest
    adapter = BacktestAdapter(df, symbol)
    adapter.generate_signals()
    adapter.run_backtest()
    
    return adapter.print_report()


# =========================
# TEST
# =========================

if __name__ == "__main__":
    print("🧪 Testing Backtest Adapter")
    print("=" * 50)
    
    # Try to connect to MT5
    try:
        from src.data.mt5_connector import MT5Connector
        
        mt5 = MT5Connector()
        mt5.connect()
        
        report = run_quick_backtest(
            mt5_connector=mt5,
            symbol="XAUUSD",
            bars=500
        )
    except Exception as e:
        print(f"⚠️ MT5 not available: {e}")
        print("   Create sample data for testing...")
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="H")
        price = 2000 + np.cumsum(np.random.randn(200) * 10)
        
        df = pd.DataFrame({
            "time": dates,
            "open": price,
            "high": price + np.abs(np.random.randn(200) * 5),
            "low": price - np.abs(np.random.randn(200) * 5),
            "close": price + np.random.randn(200) * 3,
            "volume": np.random.randint(100, 1000, 200)
        })
        
        adapter = BacktestAdapter(df, "SAMPLE")
        adapter.generate_signals()
        adapter.run_backtest()
        adapter.print_report()
