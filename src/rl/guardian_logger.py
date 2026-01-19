# src/rl/guardian_logger.py
"""
Guardian CSV Logger
====================

Export Guardian metrics for analysis and RL training.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_LOGGER")


@dataclass
class GuardianMetric:
    """Single Guardian metric entry."""
    timestamp: str = ""
    margin_ratio: float = 0.0
    daily_dd: float = 0.0
    action: str = ""
    block_reason: str = ""
    escalation: bool = False
    equity: float = 0.0
    block_count: int = 0
    cycle: int = 0
    # Competition Metrics
    source: str = "Guardian"
    event: str = ""
    potential_dd: float = 0.0
    missed_profit: float = 0.0
    # PPO Attention
    att_equity: float = 0.0
    att_margin: float = 0.0
    att_dd: float = 0.0
    att_error: float = 0.0
    att_latency: float = 0.0


class GuardianCSVLogger:
    """
    CSV Logger for Guardian metrics.
    
    Usage:
        logger = GuardianCSVLogger("guardian_metrics.csv")
        logger.log({
            "margin_ratio": 0.5,
            "daily_dd": 0.03,
            "action": "ALLOW",
            ...
        })
    """
    
    FIELDNAMES = [
        "timestamp",
        "cycle",
        "margin_ratio",
        "daily_dd",
        "equity",
        "action",
        "block_reason",
        "escalation",
        "block_count",
        # New Competition Metrics
        "source",          # Alpha vs Guardian
        "event",           # High level event (e.g. ACCOUNT_DEAD)
        "potential_dd",    # Estimated DD prevented
        "missed_profit",   # Estimated profit lost during freeze
        # PPO Attention Weights
        "att_equity",
        "att_margin",
        "att_dd",
        "att_error",
        "att_latency"
    ]
    
    def __init__(self, path: str = "logs/guardian_metrics.csv"):
        """Initialize logger with file path."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_file()
        logger.info(f"GuardianCSVLogger initialized: {self.path}")
    
    def _init_file(self) -> None:
        """Initialize CSV file with headers."""
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
    
    def log(self, data: Dict) -> None:
        """
        Log a single metric entry.
        
        Args:
            data: Dictionary with metric fields
        """
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat()
        
        # Filter to known fields only
        filtered = {k: data.get(k, "") for k in self.FIELDNAMES}
        
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(filtered)
    
    def log_metric(self, metric: GuardianMetric) -> None:
        """Log a GuardianMetric dataclass."""
        metric.timestamp = datetime.utcnow().isoformat()
        self.log(asdict(metric))


# Singleton instance
_logger_instance: Optional[GuardianCSVLogger] = None


def get_guardian_logger(path: str = "logs/guardian_metrics.csv") -> GuardianCSVLogger:
    """Get singleton Guardian logger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = GuardianCSVLogger(path)
    return _logger_instance


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("\n=== Guardian CSV Logger Test ===\n")
    
    logger = GuardianCSVLogger("test_guardian_metrics.csv")
    
    # Log some test entries
    for i in range(5):
        logger.log({
            "cycle": i + 1,
            "margin_ratio": 0.8 - i * 0.1,
            "daily_dd": 0.02 + i * 0.02,
            "equity": 1000 - i * 50,
            "action": "ALLOW" if i < 3 else "FORCE_HOLD",
            "block_reason": "" if i < 3 else "MARGIN_CRITICAL",
            "escalation": i >= 4,
            "block_count": max(0, i - 2),
        })
        print(f"  Logged cycle {i + 1}")
    
    print(f"\n✅ Log file created: test_guardian_metrics.csv")
