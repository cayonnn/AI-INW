# src/utils/decision_logger.py
"""
Decision Logger
================

Logs all trading decisions for paper and debugging.

Features:
    - Alpha action logging
    - Guardian decision logging
    - CSV export
    - Paper-ready format
"""

import os
import csv
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("DECISION_LOG")

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# CSV file path
DECISION_LOG_PATH = "logs/decision_trace.csv"

# CSV headers
HEADERS = [
    "timestamp",
    "symbol",
    "alpha_action",
    "alpha_confidence",
    "guardian_decision",
    "guardian_reason",
    "final_action",
    "price"
]

# Initialize CSV if not exists
if not os.path.exists(DECISION_LOG_PATH):
    with open(DECISION_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)


def log_decision_trace(
    symbol: str,
    alpha_action: str,
    alpha_confidence: float,
    guardian_decision: str,
    guardian_reason: str,
    final_action: Optional[str] = None,
    price: Optional[float] = None
):
    """
    Log a decision trace.
    
    Args:
        symbol: Trading symbol
        alpha_action: Alpha's proposed action
        alpha_confidence: Alpha's confidence
        guardian_decision: Guardian's decision (ALLOW/BLOCK)
        guardian_reason: Guardian's reason
        final_action: Final executed action
        price: Current price
    """
    timestamp = datetime.now().isoformat()
    
    if final_action is None:
        final_action = alpha_action if guardian_decision == "ALLOW" else "HOLD"
    
    # Log to console
    status = "✅" if guardian_decision == "ALLOW" else "❌"
    logger.info(
        f"📊 {status} {symbol} | Alpha={alpha_action}({alpha_confidence:.0%}) "
        f"| Guardian={guardian_decision} | Final={final_action}"
    )
    
    # Log to CSV
    try:
        with open(DECISION_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                symbol,
                alpha_action,
                f"{alpha_confidence:.4f}",
                guardian_decision,
                guardian_reason,
                final_action,
                f"{price:.2f}" if price else ""
            ])
    except Exception as e:
        logger.error(f"Failed to write decision log: {e}")


def get_decision_summary(limit: int = 100) -> dict:
    """Get summary of recent decisions."""
    try:
        import pandas as pd
        df = pd.read_csv(DECISION_LOG_PATH)
        
        if df.empty:
            return {}
        
        recent = df.tail(limit)
        
        return {
            "total": len(recent),
            "alpha_actions": recent["alpha_action"].value_counts().to_dict(),
            "guardian_allowed": len(recent[recent["guardian_decision"] == "ALLOW"]),
            "guardian_blocked": len(recent[recent["guardian_decision"] == "BLOCK"]),
            "avg_confidence": recent["alpha_confidence"].astype(float).mean()
        }
    except:
        return {}


if __name__ == "__main__":
    # Test logging
    log_decision_trace(
        symbol="XAUUSD",
        alpha_action="BUY",
        alpha_confidence=0.72,
        guardian_decision="ALLOW",
        guardian_reason="ALL_CHECKS_PASSED",
        price=2650.50
    )
    
    log_decision_trace(
        symbol="XAUUSD",
        alpha_action="SELL",
        alpha_confidence=0.58,
        guardian_decision="BLOCK",
        guardian_reason="HIGH_DD_RISK",
        price=2651.20
    )
    
    print(f"\nSummary: {get_decision_summary()}")
