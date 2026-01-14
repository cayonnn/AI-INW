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

# Position Management
from src.risk.position_manager import PositionManager, PositionAction, get_position_manager
from src.execution.position_executor import PositionExecutor, get_position_executor
from src.config.trading_profiles import get_active_profile, TradingMode

# Win Streak Booster
from src.risk.win_streak_booster import WinStreakRiskBooster, get_win_streak_booster, TradeResult

# Risk Manager for lot sizing
from src.risk.risk_manager import RiskManager

# Competition Alpha Stack
from src.execution.pyramid_manager import PyramidManager, get_pyramid_manager
from src.analytics.live_score import LiveScoreEstimator, TradingStats, get_live_score_estimator
from src.ai.confidence_engine import ConfidenceEngine, get_confidence_engine, calculate_effective_risk

# Dynamic Mode System
from src.core.trading_mode import TradingMode, MODE_PROFILES, get_mode_profile
from src.core.mode_controller import ModeController, get_mode_controller
from src.analytics.score_risk_cap import ScoreRiskCap, get_score_risk_cap
from src.analytics.risk_stack import EffectiveRiskStack, get_effective_risk_stack

# Regime-Aware System
from src.analytics.market_regime import MarketRegimeDetector, get_regime_detector, Regime, REGIME_MODE_MAP
from src.ai.score_optimizer import ScoreOptimizer, get_score_optimizer, optimize_full_stack

# Safety Systems (Progressive Guard + Kill Switch)
from src.safety.progressive_guard import ProgressiveGuard, get_progressive_guard, AlertLevel
from src.safety.kill_switch import get_kill_switch, is_trading_disabled


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
    """Setup logging configuration with UTF-8 support."""
    import sys
    os.makedirs("logs", exist_ok=True)
    
    # Create UTF-8 stream handler for Windows
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    
    handlers = [stream_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True  # Reset any existing handlers
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
    logger.info("LIVE LOOP V3 - AI Trading System")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Interval: {interval_sec}s")
    logger.info(f"Max Cycles: {max_cycles}")
    logger.info(f"Mode: {'SANDBOX' if sandbox else '[LIVE]'}")
    logger.info(f"Auto-Train: {'ON' if auto_train else 'OFF'}")
    logger.info("=" * 60)
    
    # Initialize Signal Engine
    engine = SignalEngineV3()
    
    # Initialize Win Streak Booster
    profile = get_active_profile()
    booster = get_win_streak_booster(profile)
    logger.info(f"WinStreakBooster: Base Risk={booster.base_risk:.1f}%, Max={booster.max_risk:.1f}%")
    
    # Initialize Risk Manager (will use dynamic risk from booster)
    risk_manager = RiskManager(
        risk_per_trade=booster.base_risk / 100,  # Convert % to decimal
        max_positions_per_symbol=3,
        max_daily_loss_pct=0.05  # 5% max daily loss
    )
    logger.info(f"RiskManager: max_positions=3, max_daily_loss=5%")
    
    # Competition Alpha Stack
    pyramid_manager = get_pyramid_manager(mode="smart")
    confidence_engine = get_confidence_engine()
    score_estimator = get_live_score_estimator()
    logger.info("Alpha Stack: Pyramid + Confidence + LiveScore initialized")
    
    # Dynamic Mode System
    mode_controller = get_mode_controller()
    score_cap = get_score_risk_cap()
    risk_stack = get_effective_risk_stack()
    logger.info("Mode System: Controller + ScoreCap + RiskStack initialized")
    
    # Regime-Aware System
    regime_detector = get_regime_detector()
    score_optimizer = get_score_optimizer()
    logger.info("Regime System: Detector + Optimizer initialized")
    
    # 🛡️ Safety Systems
    progressive_guard = get_progressive_guard()
    kill_switch = get_kill_switch()
    logger.info("Safety Systems: Progressive Guard + Kill Switch initialized")
    
    # Command writer for MT5
    command_writer = None
    if not sandbox:
        try:
            from src.execution.mt5_command_writer import get_command_writer
            command_writer = get_command_writer()
            logger.info("MT5 Command Writer initialized")
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
        # 🔥 Win Streak Stats
        "streak_max": 0,
        "streak_wins": 0,
        "streak_losses": 0,
    }
    
    logger.info("\n" + "-" * 60)
    
    try:
        while stats["cycles"] < max_cycles:
            start_time = time.time()
            stats["cycles"] += 1
            cycle = stats["cycles"]
            
            # 🛡️ PROGRESSIVE GUARD CHECK - HARD STOP
            guard_status = progressive_guard.get_status()
            if guard_status.get("kill_latched"):
                logger.critical("🔒 SYSTEM HALTED - PROGRESSIVE GUARD KILL LATCHED")
                logger.critical(f"   Reason: {guard_status.get('latch_reason')}")
                logger.critical(f"   Time: {guard_status.get('latch_time')}")
                break  # EXIT PROCESS - ห้าม catch แล้ว continue
            
            # Check if trading is disabled by kill switch
            if is_trading_disabled():
                logger.warning("Trading disabled by kill switch, waiting...")
                time.sleep(interval_sec)
                continue
            
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
                logger.info(f"[{timestamp}] Cycle {cycle}: HOLD | {info}")
            else:
                # Get base components
                base_risk = booster.base_risk
                streak_level = booster.get_risk_level()
                streak_mult = booster.current_risk() / booster.base_risk
                
                # Get AI confidence from signal
                signal_confidence = 0.7
                try:
                    if "Conf=" in info:
                        conf_str = info.split("Conf=")[1].split("%")[0]
                        signal_confidence = float(conf_str) / 100
                except:
                    pass
                
                # Get confidence multiplier
                conf_result = confidence_engine.compute(signal_confidence, 0.2, 0.7)
                conf_mult = conf_result.risk_multiplier
                
                # Detect market regime
                regime_result = regime_detector.detect(df_h1)
                regime_name = regime_result.regime.value
                recommended_mode = REGIME_MODE_MAP.get(regime_result.regime, "NEUTRAL")
                
                # Get current drawdown
                current_dd = 0.0
                try:
                    import MetaTrader5 as mt5
                    if mt5.initialize():
                        acc = mt5.account_info()
                        if acc and acc.balance > 0:
                            current_dd = ((acc.balance - acc.equity) / acc.balance) * 100
                except:
                    pass
                
                # Get live score (simplified for now)
                live_score = 60.0  # Default mid-range score
                
                # Optimize based on score and regime
                opt_result = optimize_full_stack(live_score, regime_name, recommended_mode)
                
                # Decide trading mode (with regime influence)
                mode_decision = mode_controller.decide(live_score, current_dd, regime_result.confidence)
                mode_mult = mode_decision.profile["risk_mult"]
                max_pyramid_mode = mode_decision.profile["max_pyramid"]
                
                # Apply score cap
                cap_result = score_cap.cap(live_score)
                cap_mult = cap_result.risk_mult
                max_pyramid_cap = cap_result.max_pyramid if cap_result.max_pyramid is not None else 99
                
                # Calculate effective risk with full stack
                snapshot = risk_stack.calculate(
                    base_risk=base_risk,
                    win_streak_mult=streak_mult,
                    confidence_mult=conf_mult,
                    mode_mult=mode_mult,
                    score_cap_mult=cap_mult,
                    pyramid_mult=1.0,
                    max_pyramid=min(3, max_pyramid_mode, max_pyramid_cap),
                    mode_name=mode_decision.mode.value,
                    score=live_score,
                    drawdown=current_dd,
                    confidence=signal_confidence,
                    streak_level=streak_level
                )
                current_risk = snapshot.effective_risk
                
                # Update risk_manager
                risk_manager.risk_per_trade = current_risk / 100
                
                # Log with mode
                mode_icon = f"[{mode_decision.mode.value}]"
                logger.info(f"[{timestamp}] Cycle {cycle}: {signal} {mode_icon} | Risk: {current_risk:.2f}% | {info}")
                
                # 4. Execute Trade with dynamic risk
                if not sandbox and command_writer:
                    # Calculate SL/TP based on ATR first
                    try:
                        import MetaTrader5 as mt5
                        if mt5.initialize():
                            tick = mt5.symbol_info_tick(symbol)
                            current_price = tick.ask if signal == "BUY" else tick.bid
                            
                            # Calculate ATR from H1 data
                            atr = df_h1["high"].tail(14).max() - df_h1["low"].tail(14).min()
                            atr = max(atr / 14, 5.0)  # Min 5 points for XAUUSD
                            
                            # SL = 1.5 ATR (minimum 500 pips = 5.0 for XAUUSD)
                            sl_distance = max(atr * 1.5, 5.0)
                            # TP = 2x SL (1:2 RR ratio)
                            tp_distance = sl_distance * 2
                            
                            if signal == "BUY":
                                sl_price = round(current_price - sl_distance, 2)
                                tp_price = round(current_price + tp_distance, 2)
                            else:  # SELL
                                sl_price = round(current_price + sl_distance, 2)
                                tp_price = round(current_price - tp_distance, 2)
                            
                            # Use RiskManager for lot sizing (proper formula)
                            risk_result = risk_manager.check_and_size(symbol, sl_distance)
                            
                            if not risk_result.can_trade:
                                logger.warning(f"   RiskManager blocked: {risk_result.rejection_reason}")
                                continue
                            
                            lot_size = risk_result.lot_size
                            logger.info(f"   SL: {sl_price} | TP: {tp_price} | Lot: {lot_size} (SL dist: {sl_distance:.2f})")
                            
                    except Exception as e:
                        logger.warning(f"   SL/TP calc failed: {e}, using defaults")
                        sl_price = 0
                        tp_price = 0
                        lot_size = 0.01
                    
                    trade = {
                        "action": "OPEN",
                        "symbol": symbol,
                        "direction": signal,
                        "volume": lot_size,
                        "sl": sl_price,
                        "tp": tp_price,
                        "magic": 900005,
                        "comment": f"V3_{signal}_R{current_risk:.1f}"
                    }
                    
                    stats["trades_sent"] += 1
                    success = command_writer.send_trade(trade)
                    
                    if success:
                        stats["trades_success"] += 1
                        logger.info(f"   Trade sent: {lot_size} lot @ Risk {current_risk:.1f}%")
                    else:
                        logger.warning(f"   Failed to send trade")
                else:
                    logger.info(f"   Sandbox - trade simulated (Risk: {current_risk:.1f}%)")
            
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
            
            # 7. Check for closed trades and update booster (live mode)
            if not sandbox and command_writer:
                try:
                    import MetaTrader5 as mt5
                    if mt5.initialize():
                        from datetime import timedelta
                        deals = mt5.history_deals_get(
                            datetime.now() - timedelta(minutes=5),
                            datetime.now()
                        )
                        if deals:
                            for deal in deals:
                                if deal.magic == 900005 and deal.entry == 1:  # Our closed trades
                                    if deal.profit > 0:
                                        booster.on_trade_result(TradeResult.WIN)
                                        stats["streak_wins"] += 1
                                        logger.info(f"WIN +${deal.profit:.2f} | Streak: {booster.win_streak}")
                                    elif deal.profit < 0:
                                        booster.on_trade_result(TradeResult.LOSS)
                                        stats["streak_losses"] += 1
                                        logger.info(f"LOSS -${abs(deal.profit):.2f} | Streak reset")
                                    else:
                                        booster.on_trade_result(TradeResult.BREAKEVEN)
                                        logger.info(f"BE | Streak: {booster.win_streak}")
                                    
                                    stats["streak_max"] = max(stats["streak_max"], booster.win_streak)
                except Exception as e:
                    pass  # Silent fail for MT5 check
            
            # 8. Wait for next cycle
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_sec - elapsed)
            
            if sleep_time > 0 and cycle < max_cycles:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    
    # Final Summary
    logger.info("\n" + "=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Cycles: {stats['cycles']}")
    logger.info(f"Signals:")
    logger.info(f"   BUY:  {stats['signals'].get('BUY', 0)}")
    logger.info(f"   SELL: {stats['signals'].get('SELL', 0)}")
    logger.info(f"   HOLD: {stats['signals'].get('HOLD', 0)}")
    logger.info(f"Trades Sent: {stats['trades_sent']}")
    logger.info(f"Trades Success: {stats['trades_success']}")
    logger.info(f"Win Streak Stats:")
    logger.info(f"   Max Streak: {stats['streak_max']}")
    logger.info(f"   Wins: {stats['streak_wins']}")
    logger.info(f"   Losses: {stats['streak_losses']}")
    logger.info(f"   Final Streak: {booster.win_streak}")
    logger.info(f"   Final Risk: {booster.current_risk():.2f}%")
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
