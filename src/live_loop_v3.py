# src/live_loop_v3.py
"""
Live Loop V3 - Complete AI Trading System
==========================================
Full integration of all components:

- SignalEngine V3: Rule + AI hybrid + H1/H4 filter
- Auto AI Training: Dataset → Train → Update model
- MT5 Integration: Real-time trade execution
- Logging & Metrics: Full observability
- Sandbox Mode: Testing without real trades

Usage:
    python src/live_loop_v3.py
    python src/live_loop_v3.py --live
    python src/live_loop_v3.py --auto-train
    python src/live_loop_v3.py --interval 60 --cycles 100
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.signal_engine_v3 import SignalEngineV3
from src.signals.imitation_dataset import ImitationDataset

# 🔥 Position Management
from src.risk.position_manager import PositionManager, PositionAction, get_position_manager
from src.execution.position_executor import PositionExecutor, get_position_executor
from src.config.trading_profiles import get_active_profile, TradingMode


# =========================
# CONFIGURATION
# =========================

CONFIG = {
    "symbol": "XAUUSD",
    "interval_sec": 30,
    "max_cycles": 1000,
    "auto_train_threshold": 200,  # Min samples for auto-train
    "auto_train_interval": 50,    # Train every N cycles
    "model_path": "models/xgb_imitation.pkl",
    "dataset_path": "data/live_dataset.csv",
    "log_file": "logs/live_loop_v3.log",
}


# =========================
# SETUP LOGGING
# =========================

def setup_logging(log_file: str = None):
    """Setup logging configuration."""
    os.makedirs("logs", exist_ok=True)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    
    return logging.getLogger("LiveLoopV3")


# =========================
# DATA LOADER
# =========================

def load_market_data(symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
    """Load data from MT5."""
    try:
        from src.data.mt5_connector import MT5Connector
        
        mt5 = MT5Connector()
        mt5.connect()
        
        rates = mt5.get_rates(symbol, timeframe, bars)
        return pd.DataFrame(rates)
    except Exception as e:
        logging.warning(f"MT5 error: {e}")
        return pd.DataFrame()


def create_sample_data(n: int = 500) -> pd.DataFrame:
    """Create sample data for sandbox."""
    np.random.seed(int(time.time()) % 10000)
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
# AUTO AI TRAINING
# =========================

def auto_train_ai(dataset_path: str, model_path: str, min_samples: int = 200) -> bool:
    """
    Auto-train AI model when enough data available.
    
    Returns:
        True if training successful
    """
    logger = logging.getLogger("LiveLoopV3")
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found: {dataset_path}")
        return False
    
    df = pd.read_csv(dataset_path)
    
    if len(df) < min_samples:
        logger.info(f"Not enough data for training: {len(df)}/{min_samples}")
        return False
    
    try:
        from src.train.xgb_train import main as train_main
        
        logger.info(f"🔄 Auto-training AI with {len(df)} samples...")
        train_main(dataset_path, model_path)
        logger.info("✅ AI Model updated")
        
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


# =========================
# RECORD DATASET
# =========================

def record_to_dataset(
    dataset_path: str,
    df_h1: pd.DataFrame,
    signal: str,
    info: str
):
    """Record signal data to dataset for future training."""
    if df_h1.empty:
        return
    
    last = df_h1.iloc[-1]
    
    record = {
        "timestamp": datetime.now().isoformat(),
        "close": last.get("close", 0),
        "ema20": last.get("ema20", 0) if "ema20" in df_h1.columns else 0,
        "ema50": last.get("ema50", 0) if "ema50" in df_h1.columns else 0,
        "atr14": last.get("atr14", 0) if "atr14" in df_h1.columns else 0,
        "signal": signal,
        "info": info
    }
    
    # Compute features if not present
    close = df_h1["close"]
    if "ema20" not in df_h1.columns:
        record["ema20"] = close.ewm(span=20).mean().iloc[-1]
        record["ema50"] = close.ewm(span=50).mean().iloc[-1]
    
    # Append to CSV
    df = pd.DataFrame([record])
    
    os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)
    
    if os.path.exists(dataset_path):
        df.to_csv(dataset_path, mode='a', header=False, index=False)
    else:
        df.to_csv(dataset_path, index=False)


# =========================
# LIVE LOOP
# =========================

def run_live_loop(
    symbol: str = "XAUUSD",
    interval_sec: int = 30,
    max_cycles: int = 1000,
    sandbox: bool = True,
    auto_train: bool = False,
    auto_train_interval: int = 50
):
    """
    Run Live Loop V3.
    
    Args:
        symbol: Trading symbol
        interval_sec: Seconds between cycles
        max_cycles: Maximum cycles
        sandbox: If True, don't execute real trades
        auto_train: If True, auto-train AI periodically
        auto_train_interval: Cycles between training
    """
    logger = setup_logging(CONFIG["log_file"])
    
    logger.info("=" * 60)
    logger.info("🚀 LIVE LOOP V3 - AI Trading System")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Interval: {interval_sec}s")
    logger.info(f"Max Cycles: {max_cycles}")
    logger.info(f"Mode: {'SANDBOX' if sandbox else '🔴 LIVE'}")
    logger.info(f"Auto-Train: {'ON' if auto_train else 'OFF'}")
    logger.info("=" * 60)
    
    # Initialize Signal Engine
    engine = SignalEngineV3()
    
    # Command writer for MT5
    command_writer = None
    if not sandbox:
        try:
            from src.execution.mt5_command_writer import get_command_writer
            command_writer = get_command_writer()
            logger.info("✅ MT5 Command Writer initialized")
        except Exception as e:
            logger.warning(f"MT5 Command Writer not available: {e}")
    
    # Stats
    stats = {
        "cycles": 0,
        "signals": {"BUY": 0, "SELL": 0, "HOLD": 0},
        "trades_sent": 0,
        "trades_success": 0,
        "last_train": 0,
        # 🔥 Position Management Stats
        "pm_decisions": 0,
        "pm_be_moves": 0,
        "pm_partials": 0,
        "pm_exits": 0,
    }
    
    logger.info("\n" + "-" * 60)
    
    try:
        while stats["cycles"] < max_cycles:
            start_time = time.time()
            stats["cycles"] += 1
            cycle = stats["cycles"]
            
            # 1. Fetch Data
            if sandbox:
                df_h1 = create_sample_data(500)
                df_h4 = create_sample_data(125)
            else:
                df_h1 = load_market_data(symbol, "H1", 500)
                df_h4 = load_market_data(symbol, "H4", 125)
                
                if df_h1.empty:
                    logger.warning(f"Cycle {cycle}: No data available")
                    time.sleep(interval_sec)
                    continue
            
            # 2. Generate Signal
            signal, info = engine.generate_signal(df_h1, df_h4, symbol)
            stats["signals"][signal] = stats["signals"].get(signal, 0) + 1
            
            # 3. Log Signal
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if signal == "HOLD":
                logger.info(f"[{timestamp}] Cycle {cycle}: ⏸️  HOLD | {info}")
            else:
                logger.info(f"[{timestamp}] Cycle {cycle}: 🎯 {signal} | {info}")
                
                # 4. Execute Trade
                if not sandbox and command_writer:
                    trade = {
                        "action": "OPEN",
                        "symbol": symbol,
                        "direction": signal,
                        "volume": 0.01,
                        "magic": 900005,
                        "comment": f"V3_{signal}"
                    }
                    
                    stats["trades_sent"] += 1
                    success = command_writer.send_trade(trade)
                    
                    if success:
                        stats["trades_success"] += 1
                        logger.info(f"   ✅ Trade sent to MT5")
                    else:
                        logger.warning(f"   ⚠️ Failed to send trade")
                else:
                    logger.info(f"   📝 Sandbox - trade simulated")
            
            # 5. Record Dataset
            record_to_dataset(CONFIG["dataset_path"], df_h1, signal, info)
            
            # 6. Auto-Train AI
            if auto_train and cycle % auto_train_interval == 0:
                if auto_train_ai(
                    CONFIG["dataset_path"],
                    CONFIG["model_path"],
                    CONFIG["auto_train_threshold"]
                ):
                    # Reload model in engine
                    engine._load_ai_model()
                    stats["last_train"] = cycle
            
            # 7. Wait for next cycle
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_sec - elapsed)
            
            if sleep_time > 0 and cycle < max_cycles:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user")
    
    # Final Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Cycles: {stats['cycles']}")
    logger.info(f"Signals:")
    logger.info(f"   BUY:  {stats['signals'].get('BUY', 0)}")
    logger.info(f"   SELL: {stats['signals'].get('SELL', 0)}")
    logger.info(f"   HOLD: {stats['signals'].get('HOLD', 0)}")
    logger.info(f"Trades Sent: {stats['trades_sent']}")
    logger.info(f"Trades Success: {stats['trades_success']}")
    if auto_train:
        logger.info(f"Last Training: Cycle {stats['last_train']}")
    logger.info("=" * 60)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Loop V3 - AI Trading System")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between cycles")
    parser.add_argument("--cycles", type=int, default=1000, help="Max cycles")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--auto-train", action="store_true", help="Enable auto AI training")
    parser.add_argument("--train-interval", type=int, default=50, help="Cycles between training")
    
    args = parser.parse_args()
    
    run_live_loop(
        symbol=args.symbol,
        interval_sec=args.interval,
        max_cycles=args.cycles,
        sandbox=not args.live,
        auto_train=args.auto_train,
        auto_train_interval=args.train_interval
    )
