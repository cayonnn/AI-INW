"""
main.py
========
AI Trading System - Main Entry Point

Usage:
    python main.py              # Run single cycle
    python main.py --loop       # Run continuous loop
    python main.py --test       # Test mode (no trading)
    python main.py --inject-test # Inject test signal to verify execution
    python main.py --sandbox    # Sandbox mode with rule-based signals
    python main.py --train      # Train AI from collected data

Signal Modes (--mode):
    teacher   - Rule-based only (default, safe)
    student   - AI only (requires trained model)
    consensus - Trade only when both agree
"""

import sys
import time
import argparse
from datetime import datetime
import numpy as np

from src.fund_grade_orchestrator import get_fund_orchestrator
from src.execution.mt5_command_writer import get_command_writer
from src.utils.logger import setup_logging, get_logger


def create_test_signal():
    """Create a test signal to verify execution pipeline."""
    return {
        "action": "OPEN",
        "symbol": "XAUUSD",
        "direction": "BUY",
        "volume": 0.01,  # Minimum lot
        "sl": 0,         # No stop loss for test
        "tp": 0,         # No take profit for test
        "magic": 900001,
        "comment": f"TEST_INJECT_{datetime.now().strftime('%H%M%S')}",
        "confidence": 0.95,
        "regime": "NEUTRAL",
    }


def create_sandbox_signal(symbol: str = "XAUUSD", mt5_connector = None, mode: str = "teacher") -> dict | None:
    """
    Create signal using specified mode.
    
    Modes:
        teacher   - Rule-based only (safe)
        student   - AI only (requires model)
        consensus - Trade only when both agree
    
    Returns None if HOLD.
    """
    from src.signals.adapters import MT5Adapter
    from src.signals.signal_engine_v2 import get_signal_engine_v2
    
    # If no MT5 connector, return None (HOLD)
    if mt5_connector is None:
        return None
    
    try:
        # Create adapter and engine
        adapter = MT5Adapter(mt5_connector)
        engine = get_signal_engine_v2(adapter, {
            "use_htf_filter": False,
            "cooldown_seconds": 30
        })
        
        # Get Teacher (Rule-Based) signal
        teacher_result = engine.compute(symbol)
        
        # Mode: Teacher only
        if mode == "teacher":
            if teacher_result.action == "HOLD":
                return None
            
            engine.record_signal(symbol, teacher_result.action, teacher_result.indicators)
            
            return {
                "action": "OPEN",
                "symbol": symbol,
                "direction": teacher_result.action,
                "volume": 0.01,
                "sl": 0,
                "tp": 0,
                "magic": 900002,
                "comment": f"TEACHER_{teacher_result.action}_{datetime.now().strftime('%H%M%S')}",
                "confidence": teacher_result.confidence,
                "reason": teacher_result.reason,
                "indicators": teacher_result.indicators,
                "htf_trend": teacher_result.htf_trend,
                "source": "teacher",
            }
        
        # Mode: Student (AI) or Consensus
        try:
            from src.ai.inference_xgb import get_inference_engine
            ai_engine = get_inference_engine()
            
            if ai_engine.loaded:
                ai_result = ai_engine.infer(teacher_result.indicators, symbol)
                
                # Mode: Student only
                if mode == "student":
                    if ai_result.signal == "HOLD":
                        return None
                    
                    ai_engine.record_trade(symbol, ai_result.signal)
                    
                    return {
                        "action": "OPEN",
                        "symbol": symbol,
                        "direction": ai_result.signal,
                        "volume": 0.01,
                        "sl": 0,
                        "tp": 0,
                        "magic": 900003,
                        "comment": f"AI_{ai_result.signal}_{datetime.now().strftime('%H%M%S')}",
                        "confidence": ai_result.confidence,
                        "reason": ai_result.reason,
                        "indicators": teacher_result.indicators,
                        "htf_trend": teacher_result.htf_trend,
                        "source": "ai",
                    }
                
                # Mode: Consensus (both must agree)
                if mode == "consensus":
                    if teacher_result.action == ai_result.signal and teacher_result.action != "HOLD":
                        combined_conf = (teacher_result.confidence + ai_result.confidence) / 2
                        engine.record_signal(symbol, teacher_result.action, teacher_result.indicators)
                        ai_engine.record_trade(symbol, ai_result.signal)
                        
                        return {
                            "action": "OPEN",
                            "symbol": symbol,
                            "direction": teacher_result.action,
                            "volume": 0.01,
                            "sl": 0,
                            "tp": 0,
                            "magic": 900004,
                            "comment": f"CONSENSUS_{teacher_result.action}_{datetime.now().strftime('%H%M%S')}",
                            "confidence": combined_conf,
                            "reason": f"Teacher={teacher_result.action}, AI={ai_result.signal} (AGREE)",
                            "indicators": teacher_result.indicators,
                            "htf_trend": teacher_result.htf_trend,
                            "source": "consensus",
                        }
                    else:
                        return None  # No consensus → HOLD
            else:
                # AI not available, fallback to teacher
                if teacher_result.action == "HOLD":
                    return None
                return create_sandbox_signal(symbol, mt5_connector, mode="teacher")
                
        except Exception as ai_error:
            # AI failed, fallback to teacher
            print(f"AI fallback: {ai_error}")
            if teacher_result.action == "HOLD":
                return None
            return create_sandbox_signal(symbol, mt5_connector, mode="teacher")
        
        return None
        
    except Exception as e:
        print(f"Signal error: {e}")
        import traceback
        traceback.print_exc()
        return None


class SandboxRiskGuard:
    """Minimal risk guard for sandbox testing (Fund-Grade requirement)."""
    
    def __init__(self):
        self.max_positions_per_symbol = 3      # Max open positions per symbol
        self.cooldown_seconds = 30             # Min seconds between orders
        self.max_errors_per_hour = 10          # Kill-switch trigger
        self.max_daily_trades = 50             # Max trades per day
        self.prevent_duplicate_direction = True # Don't allow BUY-BUY-BUY
        
        # State tracking
        self.positions = {}              # {symbol: count}
        self.position_directions = {}    # {symbol: {"BUY": 2, "SELL": 1}}
        self.last_action = {}            # {symbol: "BUY" or "SELL"}
        self.last_order_time = {}        # {symbol: datetime}
        self.error_count = 0
        self.trade_count = 0
        self.error_reset_time = datetime.now()
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        self.killed = False
        self.kill_reason = ""
    
    def reset_daily(self):
        """Reset daily counters."""
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.trade_count = 0
            self.positions = {}
            self.position_directions = {}
            self.last_action = {}
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
    
    def reset_error_counter(self):
        """Reset error counter every hour."""
        now = datetime.now()
        if (now - self.error_reset_time).total_seconds() > 3600:
            self.error_count = 0
            self.error_reset_time = now
    
    def can_trade(self, symbol: str, direction: str = None) -> tuple[bool, str]:
        """Check if trading is allowed."""
        if self.killed:
            return False, f"KILLED: {self.kill_reason}"
        
        self.reset_daily()
        self.reset_error_counter()
        
        # Check daily trade limit
        if self.trade_count >= self.max_daily_trades:
            return False, f"Daily limit reached ({self.max_daily_trades})"
        
        # Check max positions per symbol
        current_pos = self.positions.get(symbol, 0)
        if current_pos >= self.max_positions_per_symbol:
            return False, f"Max positions for {symbol} ({self.max_positions_per_symbol})"
        
        # Check cooldown
        last_time = self.last_order_time.get(symbol)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False, f"Cooldown: {self.cooldown_seconds - elapsed:.0f}s remaining"
        
        # Check duplicate direction (anti BUY-BUY-BUY)
        if direction and self.prevent_duplicate_direction:
            last = self.last_action.get(symbol)
            if last == direction:
                return False, f"Duplicate direction blocked (last={last})"
        
        return True, "OK"
    
    def record_trade(self, symbol: str, direction: str, success: bool):
        """Record trade result."""
        if success:
            self.positions[symbol] = self.positions.get(symbol, 0) + 1
            self.trade_count += 1
            self.last_order_time[symbol] = datetime.now()
            self.last_action[symbol] = direction
            
            # Track direction counts
            if symbol not in self.position_directions:
                self.position_directions[symbol] = {"BUY": 0, "SELL": 0}
            self.position_directions[symbol][direction] += 1
        else:
            self.error_count += 1
            if self.error_count >= self.max_errors_per_hour:
                self.killed = True
                self.kill_reason = f"Error spike: {self.error_count} errors/hour"
    
    def record_close(self, symbol: str, direction: str = None):
        """Record position close."""
        if symbol in self.positions:
            self.positions[symbol] = max(0, self.positions.get(symbol, 0) - 1)
            
            # Update direction counts
            if direction and symbol in self.position_directions:
                self.position_directions[symbol][direction] = max(
                    0, self.position_directions[symbol].get(direction, 0) - 1
                )
    
    def get_status(self) -> dict:
        """Get current risk guard status."""
        return {
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "trade_count": self.trade_count,
            "error_count": self.error_count,
            "positions": self.positions,
            "directions": self.position_directions,
            "last_action": self.last_action,
        }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="AI Trading System")
    parser.add_argument("--loop", action="store_true", help="Run continuous loop")
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--inject-test", action="store_true", help="Inject test signal")
    parser.add_argument("--sandbox", action="store_true", help="Sandbox loop with simple signals")
    parser.add_argument("--train", action="store_true", help="Train AI from collected data")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval (seconds)")
    parser.add_argument("--mode", type=str, default="teacher", 
                       choices=["teacher", "student", "consensus"],
                       help="Signal mode (teacher/student/consensus)")
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = get_logger("MAIN")

    # ---------------------------
    # TRAIN AI MODE
    # ---------------------------
    if args.train:
        logger.info("=" * 50)
        logger.info("🤖 AI TRAINING MODE")
        logger.info("=" * 50)
        
        import subprocess
        result = subprocess.run(
            [sys.executable, "src/ai/train_xgboost.py"],
            cwd="f:\\Mobile App\\AI trade แขงขัน"
        )
        
        if result.returncode == 0:
            logger.info("✅ Training complete!")
        else:
            logger.error("❌ Training failed")
        
        return

    logger.info("=" * 60)
    logger.info("🏦 AI Trading System - Fund Grade")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Mode: {'INJECT-TEST' if args.inject_test else 'SANDBOX' if args.sandbox else 'TEST' if args.test else 'LIVE'}")
    logger.info(f"Signal Mode: {args.mode.upper()}")
    logger.info("=" * 60)

    # Initialize
    orchestrator = get_fund_orchestrator()
    command_writer = get_command_writer()

    # ---------------------------
    # INJECT TEST SIGNAL MODE
    # ---------------------------
    if args.inject_test:
        logger.info("=" * 50)
        logger.info("🧪 INJECT TEST MODE - Verifying Execution Pipeline")
        logger.info("=" * 50)
        
        test_trade = create_test_signal()
        
        logger.info(f"Test Signal: {test_trade}")
        logger.info("-" * 40)
        
        # Write command to MT5
        success = command_writer.send_trade(test_trade)
        
        if success:
            logger.info("✅ Command written to MT5 successfully")
            logger.info(f"   File: {command_writer.command_path}")
            
            # Wait for response (if EA is running)
            logger.info("⏳ Waiting for EA response...")
            response = command_writer.wait_for_response(timeout_sec=10)
            
            if response:
                logger.info(f"✅ EA Response: {response}")
            else:
                logger.warning("⚠️ No EA response (EA may not be running)")
        else:
            logger.error("❌ Failed to write command")
        
        logger.info("=" * 50)
        logger.info("Test injection complete")
        logger.info("=" * 50)
        return

    # ---------------------------
    # SANDBOX MODE - Rule-based signal loop
    # ---------------------------
    if args.sandbox:
        logger.info("=" * 50)
        logger.info("🏖️ SANDBOX MODE - Rule-Based Signal Testing")
        logger.info(f"   Strategy: EMA(12/26) Crossover + ATR Filter")
        logger.info(f"   Interval: {args.interval}s")
        logger.info("=" * 50)
        
        risk_guard = SandboxRiskGuard()
        logger.info(f"Risk Guard: max_pos={risk_guard.max_positions_per_symbol}, cooldown={risk_guard.cooldown_seconds}s, anti-dup={risk_guard.prevent_duplicate_direction}")
        
        # Get MT5 connector from orchestrator
        mt5 = orchestrator.mt5 if hasattr(orchestrator, 'mt5') else None
        if mt5 is None:
            logger.error("MT5 connector not available")
            return
        
        cycle = 0
        while True:
            try:
                cycle += 1
                symbol = "XAUUSD"
                
                # Generate signal using specified mode
                trade = create_sandbox_signal(symbol, mt5, mode=args.mode)
                
                # No clear signal = HOLD
                if trade is None:
                    logger.info(f"--- Cycle {cycle} HOLD: No clear signal ({args.mode}) ---")
                    time.sleep(args.interval)
                    continue
                
                direction = trade['direction']
                indicators = trade.get('indicators', {})
                reason = trade.get('reason', '')
                confidence = trade.get('confidence', 0)
                source = trade.get('source', 'unknown')
                
                # Risk Guard Check (with direction)
                can_trade, guard_reason = risk_guard.can_trade(symbol, direction)
                
                if not can_trade:
                    logger.warning(f"--- Cycle {cycle} BLOCKED: {guard_reason} ---")
                    time.sleep(args.interval)
                    continue
                
                logger.info(f"--- Sandbox Cycle {cycle} [{source.upper()}] ---")
                logger.info(f"Signal: {direction} {symbol} | Confidence: {confidence:.0%}")
                logger.info(f"   Reason: {reason}")
                logger.info(f"   EMA: fast={indicators.get('ema_fast')}, slow={indicators.get('ema_slow')}, spread={indicators.get('ema_spread')}")
                logger.info(f"   ATR: {indicators.get('atr')} (threshold: {indicators.get('atr_threshold')})")
                
                # Send to EA
                start_time = time.time()
                success = command_writer.send_trade(trade)
                
                trade_success = False
                if success:
                    response = command_writer.wait_for_response(timeout_sec=5)
                    latency = (time.time() - start_time) * 1000
                    
                    if response:
                        status = response.get('status', 'UNKNOWN')
                        trade_success = (status == 'OK')
                        logger.info(f"{'✅' if trade_success else '❌'} {status} | Latency: {latency:.0f}ms")
                    else:
                        logger.warning("⚠️ No response")
                
                # Record result (with direction)
                risk_guard.record_trade(symbol, direction, trade_success)
                
                # Show status
                guard_status = risk_guard.get_status()
                logger.info(f"   Positions: {guard_status['positions']} | Last: {guard_status['last_action']} | Trades: {guard_status['trade_count']}")
                
                if risk_guard.killed:
                    logger.critical(f"🛑 KILL SWITCH: {guard_status['kill_reason']}")
                    break
                
                # Wait for next cycle
                logger.info(f"Next cycle in {args.interval}s...")
                time.sleep(args.interval)
                
            except KeyboardInterrupt:
                logger.info("Sandbox stopped")
                break
            except Exception as e:
                logger.exception(f"Error: {e}")
                risk_guard.record_trade(symbol, "UNKNOWN", False)
                time.sleep(5)
        return

    # ---------------------------
    # NORMAL OPERATION
    # ---------------------------
    if args.loop:
        # Continuous loop
        logger.info(f"Running continuous loop (interval: {args.interval}s)")
        
        while True:
            try:
                # Run cycle
                trades = orchestrator.run_cycle()
                
                # Send to MT5
                if not args.test:
                    for trade in trades:
                        success = command_writer.send_trade(trade)
                        if success:
                            response = command_writer.wait_for_response(timeout_sec=5)
                            if response:
                                logger.info(f"EA Response: {response.get('status')}")
                else:
                    for trade in trades:
                        logger.info(f"[TEST] Would send: {trade}")

                # Wait for next cycle
                logger.info(f"Sleeping {args.interval}s...")
                time.sleep(args.interval)

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.exception(f"Cycle error: {e}")
                time.sleep(10)
    else:
        # Single cycle
        trades = orchestrator.run_cycle()
        
        for trade in trades:
            if args.test:
                logger.info(f"[TEST] Trade: {trade}")
            else:
                command_writer.send_trade(trade)

        logger.info("Single cycle complete")

    logger.info("AI Trading System stopped")


if __name__ == "__main__":
    main()

