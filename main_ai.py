"""
main_ai.py
==========
AI Trading System - Main Loop with SignalEngine V2 (AI + Rule + HTF + Risk Guard)

Usage:
    python main_ai.py
    python main_ai.py --live        # Live mode with MT5
    python main_ai.py --cycles 50   # Custom cycle count
    python main_ai.py --interval 60 # Custom interval
    python main_ai.py --mode hybrid # Signal mode
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

from src.signals.signal_engine_ai import SignalEngineAI, get_signal_engine_ai
from src.execution.mt5_command_writer import get_command_writer
from src.utils.logger import setup_logging, get_logger


# =========================
# CONFIGURATION
# =========================

DEFAULT_CONFIG = {
    "symbol": "XAUUSD",
    "timeframe": "H1",
    "htf": "H4",
    "interval": 30,
    "max_cycles": 100,
    "mode": "hybrid",  # teacher, student, consensus, hybrid
    "verbose": True,
}


# =========================
# DATA LOADER
# =========================

def load_market_data(symbol: str, timeframe: str, bars: int = 100) -> pd.DataFrame:
    """
    Load market data from MT5.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        bars: Number of bars
        
    Returns:
        OHLC DataFrame with indicators
    """
    try:
        from src.data.mt5_connector import MT5Connector
        
        mt5 = MT5Connector()
        mt5.connect()
        
        # Get rates
        rates = mt5.get_rates(symbol, timeframe, bars)
        df = pd.DataFrame(rates)
        
        # Add indicators
        from src.signals.signal_engine_ai import ema, atr, rsi
        
        df["ema12"] = ema(df["close"], 12)
        df["ema26"] = ema(df["close"], 26)
        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        df["atr14"] = atr(df, 14)
        df["rsi14"] = rsi(df["close"], 14)
        
        return df
        
    except Exception as e:
        print(f"⚠️ MT5 not available: {e}")
        return pd.DataFrame()


def create_sample_data(n: int = 100) -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(int(time.time()) % 1000)
    price = 2650 + np.cumsum(np.random.randn(n) * 5)
    
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="H"),
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 3),
        "low": price - np.abs(np.random.randn(n) * 3),
        "close": price + np.random.randn(n) * 2,
        "volume": np.random.randint(100, 1000, n)
    })
    
    from src.signals.signal_engine_ai import ema, atr, rsi
    
    df["ema12"] = ema(df["close"], 12)
    df["ema26"] = ema(df["close"], 26)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr14"] = atr(df, 14)
    df["rsi14"] = rsi(df["close"], 14)
    
    return df


# =========================
# SANDBOX LOOP
# =========================

def run_sandbox_loop(
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    htf: str = "H4",
    interval: int = 30,
    max_cycles: int = 100,
    mode: str = "hybrid",
    live: bool = False,
    verbose: bool = True
):
    """
    Run sandbox loop with AI Signal Engine.
    """
    # Setup logging
    setup_logging()
    logger = get_logger("MAIN_AI")
    
    logger.info("=" * 60)
    logger.info("🤖 AI Trading System - SignalEngine V2")
    logger.info(f"   Symbol: {symbol} | TF: {timeframe} | HTF: {htf}")
    logger.info(f"   Mode: {mode.upper()}")
    logger.info(f"   Interval: {interval}s | Max Cycles: {max_cycles}")
    logger.info("=" * 60)
    
    # Initialize Signal Engine
    engine = SignalEngineAI(
        symbol=symbol,
        timeframe=timeframe,
        htf=htf,
        mode=mode
    )
    
    # Command writer for MT5
    command_writer = get_command_writer()
    
    # Load initial data
    if live:
        logger.info("📡 Loading live data from MT5...")
        df_h1 = load_market_data(symbol, timeframe, 100)
        df_h4 = load_market_data(symbol, htf, 50)
    else:
        logger.info("🧪 Using sample data for sandbox...")
        df_h1 = create_sample_data(100)
        df_h4 = create_sample_data(50)
    
    if df_h1.empty:
        logger.error("❌ No data available")
        return
    
    logger.info(f"✅ Data loaded: H1={len(df_h1)} bars, H4={len(df_h4)} bars")
    logger.info("-" * 60)
    
    # Main loop
    cycle = 0
    signals_generated = 0
    signals_executed = 0
    
    try:
        while cycle < max_cycles:
            cycle += 1
            
            # Refresh data in live mode
            if live and cycle % 5 == 0:
                df_h1 = load_market_data(symbol, timeframe, 100)
                df_h4 = load_market_data(symbol, htf, 50)
            
            # Generate signal
            result = engine.generate_signal(df_h1, df_h4)
            
            # Log signal details
            logger.info(f"\n--- Cycle {cycle}/{max_cycles} --- {datetime.now().strftime('%H:%M:%S')} ---")
            
            if result.final_signal == "HOLD":
                logger.info(f"⏸️  HOLD | Reason: {result.reason}")
                if verbose:
                    logger.info(f"   Teacher: {result.teacher_signal} | Student: {result.student_signal}")
                    logger.info(f"   HTF: {result.htf_trend} | Blocked: {result.blocked}")
            else:
                signals_generated += 1
                
                # Get current indicators
                last = df_h1.iloc[-1]
                
                logger.info(f"🎯 {result.final_signal} | Confidence: {result.confidence:.0%}")
                logger.info(f"   Reason: {result.reason}")
                
                if verbose:
                    logger.info(f"   Teacher: {result.teacher_signal} | Student: {result.student_signal}")
                    logger.info(f"   HTF Trend: {result.htf_trend}")
                    logger.info(f"   EMA20: {last.get('ema20', 0):.2f} | EMA50: {last.get('ema50', 0):.2f}")
                    logger.info(f"   ATR14: {last.get('atr14', 0):.2f} | RSI: {last.get('rsi14', 0):.1f}")
                    logger.info(f"   Positions: {engine.positions}/{engine.max_positions}")
                
                # Execute trade (if live mode)
                if live and not result.blocked:
                    trade = {
                        "action": "OPEN",
                        "symbol": symbol,
                        "direction": result.final_signal,
                        "volume": 0.01,
                        "sl": 0,
                        "tp": 0,
                        "magic": 900005,
                        "comment": f"AI_{result.final_signal}_{mode}",
                        "confidence": result.confidence,
                    }
                    
                    success = command_writer.send_trade(trade)
                    if success:
                        signals_executed += 1
                        logger.info(f"   ✅ Trade sent to MT5")
                    else:
                        logger.warning(f"   ⚠️ Failed to send trade")
            
            # Wait for next cycle
            logger.info(f"   Next cycle in {interval}s...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Total Cycles: {cycle}")
    logger.info(f"   Signals Generated: {signals_generated}")
    logger.info(f"   Signals Executed: {signals_executed}")
    logger.info(f"   Final Positions: {engine.positions}")
    logger.info("=" * 60)


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="AI Trading System - Main Loop")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="H1", help="Main timeframe")
    parser.add_argument("--htf", type=str, default="H4", help="Higher timeframe")
    parser.add_argument("--interval", type=int, default=30, help="Loop interval (seconds)")
    parser.add_argument("--cycles", type=int, default=100, help="Max cycles")
    parser.add_argument("--mode", type=str, default="hybrid",
                       choices=["teacher", "student", "consensus", "hybrid"],
                       help="Signal mode")
    parser.add_argument("--live", action="store_true", help="Enable live mode with MT5")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose logging")
    
    args = parser.parse_args()
    
    run_sandbox_loop(
        symbol=args.symbol,
        timeframe=args.timeframe,
        htf=args.htf,
        interval=args.interval,
        max_cycles=args.cycles,
        mode=args.mode,
        live=args.live,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
