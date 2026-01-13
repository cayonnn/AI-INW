# main_v3.py
"""
SignalEngine V3 - Live Loop
============================
Hybrid Rule+AI Trading Loop with Risk Guard and HTF Filter

Usage:
    python main_v3.py
    python main_v3.py --live
    python main_v3.py --interval 60 --cycles 50
"""

import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.signals.signal_engine_v3 import SignalEngineV3


# =========================
# CONFIG
# =========================

DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_INTERVAL = 30
DEFAULT_CYCLES = 20


# =========================
# DATA LOADER
# =========================

def load_mt5_data(symbol: str, timeframe: str, bars: int = 100) -> pd.DataFrame:
    """Load data from MT5."""
    try:
        from src.data.mt5_connector import MT5Connector
        
        mt5 = MT5Connector()
        mt5.connect()
        
        rates = mt5.get_rates(symbol, timeframe, bars)
        return pd.DataFrame(rates)
    except Exception as e:
        print(f"⚠️ MT5 error: {e}")
        return pd.DataFrame()


def create_sample_data(n: int = 100) -> pd.DataFrame:
    """Create sample data for sandbox."""
    np.random.seed(int(time.time()) % 1000)
    price = 2650 + np.cumsum(np.random.randn(n) * 5)
    
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 3),
        "low": price - np.abs(np.random.randn(n) * 3),
        "close": price + np.random.randn(n) * 2,
        "volume": np.random.randint(100, 1000, n)
    })


# =========================
# LIVE LOOP
# =========================

def run_live_loop(
    symbol: str = DEFAULT_SYMBOL,
    interval_sec: int = DEFAULT_INTERVAL,
    max_cycles: int = DEFAULT_CYCLES,
    sandbox: bool = True
):
    """
    Run SignalEngine V3 live loop.
    
    Args:
        symbol: Trading symbol
        interval_sec: Seconds between cycles
        max_cycles: Maximum number of cycles
        sandbox: If True, don't execute real trades
    """
    print("=" * 60)
    print("🧪 SignalEngine V3 - Hybrid Rule+AI Live Loop")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval_sec}s")
    print(f"Max Cycles: {max_cycles}")
    print(f"Mode: {'SANDBOX' if sandbox else '🔴 LIVE'}")
    print("=" * 60)
    
    # Initialize
    engine = SignalEngineV3()
    
    # Command writer for MT5
    command_writer = None
    if not sandbox:
        try:
            from src.execution.mt5_command_writer import get_command_writer
            command_writer = get_command_writer()
            print("✅ Command writer initialized")
        except:
            print("⚠️ Command writer not available")
    
    # Stats
    cycle = 0
    signals_count = {"BUY": 0, "SELL": 0, "HOLD": 0}
    trades_sent = 0
    
    print("\n" + "-" * 60)
    
    try:
        while cycle < max_cycles:
            cycle += 1
            
            # Fetch data
            if sandbox:
                df_h1 = create_sample_data(100)
                df_h4 = create_sample_data(50)
            else:
                df_h1 = load_mt5_data(symbol, "H1", 100)
                df_h4 = load_mt5_data(symbol, "H4", 50)
                
                if df_h1.empty:
                    print(f"Cycle {cycle}: ⚠️ No data available")
                    time.sleep(interval_sec)
                    continue
            
            # Generate signal
            signal, info = engine.generate_signal(df_h1, df_h4, symbol)
            
            # Update stats
            signals_count[signal] = signals_count.get(signal, 0) + 1
            
            # Log
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Cycle {cycle}/{max_cycles}")
            
            if signal == "HOLD":
                print(f"   ⏸️  {info}")
            else:
                print(f"   🎯 {info}")
                
                # Execute trade
                if not sandbox and command_writer:
                    trade = {
                        "action": "OPEN",
                        "symbol": symbol,
                        "direction": signal,
                        "volume": 0.01,
                        "magic": 900005,
                        "comment": f"V3_{signal}"
                    }
                    
                    success = command_writer.send_trade(trade)
                    if success:
                        trades_sent += 1
                        print(f"   ✅ Trade sent to MT5")
                    else:
                        print(f"   ⚠️ Failed to send trade")
                else:
                    print(f"   📝 Sandbox - no trade executed")
            
            # Wait
            if cycle < max_cycles:
                print(f"   ⏳ Next in {interval_sec}s...")
                time.sleep(interval_sec)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"   Total Cycles: {cycle}")
    print(f"   BUY signals:  {signals_count.get('BUY', 0)}")
    print(f"   SELL signals: {signals_count.get('SELL', 0)}")
    print(f"   HOLD signals: {signals_count.get('HOLD', 0)}")
    print(f"   Trades Sent:  {trades_sent}")
    print("=" * 60)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalEngine V3 Live Loop")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL,
                       help="Trading symbol")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                       help="Seconds between cycles")
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES,
                       help="Max cycles")
    parser.add_argument("--live", action="store_true",
                       help="Enable live trading (default: sandbox)")
    
    args = parser.parse_args()
    
    run_live_loop(
        symbol=args.symbol,
        interval_sec=args.interval,
        max_cycles=args.cycles,
        sandbox=not args.live
    )
