# fund_grade_orchestrator.py
"""
Fund-Grade Orchestrator - Single Entry Point
=============================================

🏦 This is the UNIFIED entry point for all trading modes.

Architecture:
    fund_grade_orchestrator.py
            ↓
       LiveLoopV3 (Single Source of Truth)
            ↓
    ┌───────────────────────────────────────┐
    │  RiskManager                          │
    │  TrailingManager                      │
    │  AI SL/TP Models                      │
    │  SignalEngineV3                       │
    └───────────────────────────────────────┘

Usage:
    python fund_grade_orchestrator.py --mode LIVE
    python fund_grade_orchestrator.py --mode SANDBOX
    python fund_grade_orchestrator.py --mode BACKTEST --data data.csv

NO TRADING LOGIC IN THIS FILE.
This is a SHELL only.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from live_loop_v3 import live_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("FUND_ORCHESTRATOR")


# =========================================================
# ORCHESTRATOR SHELL
# =========================================================

def run_orchestrator(
    mode: str = "SANDBOX",
    interval: int = 30,
    max_cycles: int = 1000,
    symbol: str = "XAUUSD"
):
    """
    Fund-Grade Orchestrator Entry Point.
    
    This is a SHELL that delegates to LiveLoopV3.
    NO trading logic should be here.
    
    Args:
        mode: LIVE, SANDBOX, or BACKTEST
        interval: Seconds between cycles
        max_cycles: Maximum trading cycles
        symbol: Trading symbol
    """
    logger.info("=" * 60)
    logger.info("🏦 FUND-GRADE ORCHESTRATOR")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode}")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Interval: {interval}s")
    logger.info(f"Max Cycles: {max_cycles}")
    logger.info("=" * 60)
    
    # Validate mode
    if mode not in ["LIVE", "SANDBOX", "BACKTEST"]:
        logger.error(f"Invalid mode: {mode}")
        return
    
    # Convert mode to sandbox flag
    sandbox = mode != "LIVE"
    
    # Delegate to LiveLoopV3 (Single Source of Truth)
    logger.info("🚀 Delegating to LiveLoopV3...")
    logger.info("-" * 60)
    
    live_loop(
        interval=interval,
        max_cycles=max_cycles,
        sandbox=sandbox,
        auto_train_enabled=True
    )
    
    logger.info("-" * 60)
    logger.info("🏁 Fund-Grade Orchestrator completed")


# =========================================================
# CLI
# =========================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fund-Grade Trading Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fund_grade_orchestrator.py --mode LIVE
  python fund_grade_orchestrator.py --mode SANDBOX --interval 60
  python fund_grade_orchestrator.py --mode BACKTEST
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["LIVE", "SANDBOX", "BACKTEST"],
        default="SANDBOX",
        help="Trading mode (default: SANDBOX)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between trading cycles (default: 30)"
    )
    
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Maximum trading cycles (default: 1000)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD",
        help="Trading symbol (default: XAUUSD)"
    )
    
    args = parser.parse_args()
    
    run_orchestrator(
        mode=args.mode,
        interval=args.interval,
        max_cycles=args.cycles,
        symbol=args.symbol
    )


if __name__ == "__main__":
    main()
