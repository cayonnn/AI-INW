# src/chaos/chaos_monkey.py
"""
ChaosMonkey - MT5 Failure Injection
====================================

Test Guardian's resilience to infrastructure failures:
- MT5 disconnections
- Latency spikes
- Order execution failures
- Data feed errors

Guardian learns to handle "infra risk", not just market risk.
"""

import random
import time
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("CHAOS_MONKEY")


class ChaosType(Enum):
    """Types of chaos events."""
    DISCONNECT = "MT5_DISCONNECT"
    LATENCY = "LATENCY_SPIKE"
    ORDER_FAIL = "ORDER_FAILURE"
    DATA_STALE = "STALE_DATA"
    MEMORY_SPIKE = "MEMORY_SPIKE"


@dataclass
class ChaosEvent:
    """Record of a chaos event."""
    timestamp: datetime
    chaos_type: ChaosType
    duration: float = 0.0
    message: str = ""
    recovered: bool = False


@dataclass
class ChaosStats:
    """Statistics for chaos testing."""
    total_events: int = 0
    disconnects: int = 0
    latencies: int = 0
    order_failures: int = 0
    avg_latency: float = 0.0
    guardian_responses: int = 0
    missed_responses: int = 0
    
    def __repr__(self):
        return (
            f"ChaosStats(events={self.total_events}, "
            f"disconnects={self.disconnects}, "
            f"latencies={self.latencies}, "
            f"guardian_resp={self.guardian_responses})"
        )


class ChaosMonkey:
    """
    Chaos injection for resilience testing.
    
    Usage:
        chaos = ChaosMonkey(p_disconnect=0.02, p_latency=0.05)
        
        try:
            chaos.inject()
            # ... normal operation
        except Exception as e:
            guardian.freeze(reason=str(e))
    """
    
    def __init__(
        self,
        p_disconnect: float = 0.02,
        p_latency: float = 0.05,
        p_order_fail: float = 0.03,
        p_stale_data: float = 0.01,
        enabled: bool = True
    ):
        """
        Initialize ChaosMonkey.
        
        Args:
            p_disconnect: Probability of MT5 disconnect
            p_latency: Probability of latency spike
            p_order_fail: Probability of order failure
            p_stale_data: Probability of stale data
            enabled: Whether chaos injection is active
        """
        self.p_disconnect = p_disconnect
        self.p_latency = p_latency
        self.p_order_fail = p_order_fail
        self.p_stale_data = p_stale_data
        self.enabled = enabled
        
        self.events: List[ChaosEvent] = []
        self.stats = ChaosStats()
        
        logger.info(
            f"ChaosMonkey initialized: disconnect={p_disconnect:.1%}, "
            f"latency={p_latency:.1%}, order_fail={p_order_fail:.1%}"
        )
    
    def inject(self) -> Optional[ChaosEvent]:
        """
        Inject chaos with configured probabilities.
        
        Returns:
            ChaosEvent if latency was injected, raises exception for disconnects
            
        Raises:
            ConnectionError: For MT5 disconnect simulation
            TimeoutError: For order failure simulation
            ValueError: For stale data simulation
        """
        if not self.enabled:
            return None
        
        r = random.random()
        cumulative = 0.0
        
        # Check disconnect
        cumulative += self.p_disconnect
        if r < cumulative:
            self.stats.total_events += 1
            self.stats.disconnects += 1
            
            event = ChaosEvent(
                timestamp=datetime.now(),
                chaos_type=ChaosType.DISCONNECT,
                message="💥 MT5 DISCONNECTED"
            )
            self.events.append(event)
            
            logger.critical("💥 CHAOS: MT5 DISCONNECTED")
            raise ConnectionError("💥 MT5 DISCONNECTED")
        
        # Check latency
        cumulative += self.p_latency
        if r < cumulative:
            delay = random.uniform(0.5, 2.5)
            self.stats.total_events += 1
            self.stats.latencies += 1
            self.stats.avg_latency = (
                (self.stats.avg_latency * (self.stats.latencies - 1) + delay) 
                / self.stats.latencies
            )
            
            event = ChaosEvent(
                timestamp=datetime.now(),
                chaos_type=ChaosType.LATENCY,
                duration=delay,
                message=f"⚠️ LATENCY {delay:.2f}s"
            )
            self.events.append(event)
            
            logger.warning(f"⚠️ CHAOS: LATENCY SPIKE {delay:.2f}s")
            time.sleep(delay)
            return event
        
        # Check order failure
        cumulative += self.p_order_fail
        if r < cumulative:
            self.stats.total_events += 1
            self.stats.order_failures += 1
            
            event = ChaosEvent(
                timestamp=datetime.now(),
                chaos_type=ChaosType.ORDER_FAIL,
                message="❌ ORDER EXECUTION FAILED"
            )
            self.events.append(event)
            
            logger.error("❌ CHAOS: ORDER EXECUTION FAILED")
            raise TimeoutError("❌ ORDER EXECUTION FAILED")
        
        # Check stale data
        cumulative += self.p_stale_data
        if r < cumulative:
            self.stats.total_events += 1
            
            event = ChaosEvent(
                timestamp=datetime.now(),
                chaos_type=ChaosType.DATA_STALE,
                message="📊 STALE MARKET DATA"
            )
            self.events.append(event)
            
            logger.warning("📊 CHAOS: STALE MARKET DATA")
            raise ValueError("📊 STALE MARKET DATA")
        
        return None
    
    def record_guardian_response(self, responded: bool):
        """Record whether Guardian responded to chaos."""
        if responded:
            self.stats.guardian_responses += 1
        else:
            self.stats.missed_responses += 1
    
    def get_recent_events(self, count: int = 10) -> List[ChaosEvent]:
        """Get most recent chaos events."""
        return self.events[-count:]
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = ChaosStats()
        self.events = []
    
    def get_report(self) -> str:
        """Generate chaos report."""
        report = [
            "=" * 50,
            "🐒 CHAOS MONKEY REPORT",
            "=" * 50,
            f"Total Events: {self.stats.total_events}",
            f"Disconnects: {self.stats.disconnects}",
            f"Latencies: {self.stats.latencies} (avg: {self.stats.avg_latency:.2f}s)",
            f"Order Failures: {self.stats.order_failures}",
            f"Guardian Responses: {self.stats.guardian_responses}",
            f"Missed Responses: {self.stats.missed_responses}",
        ]
        
        if self.stats.guardian_responses + self.stats.missed_responses > 0:
            response_rate = self.stats.guardian_responses / (
                self.stats.guardian_responses + self.stats.missed_responses
            )
            report.append(f"Response Rate: {response_rate:.1%}")
        
        report.append("=" * 50)
        return "\n".join(report)


# Singleton
_chaos_monkey: Optional[ChaosMonkey] = None


def get_chaos_monkey(enabled: bool = False) -> ChaosMonkey:
    """Get singleton ChaosMonkey instance."""
    global _chaos_monkey
    if _chaos_monkey is None:
        _chaos_monkey = ChaosMonkey(enabled=enabled)
    return _chaos_monkey


# =============================================================================
# Chaos Test Runner
# =============================================================================

def run_chaos_test(agent, cycles: int = 50) -> Dict:
    """
    Run chaos test against Guardian agent.
    
    Args:
        agent: Guardian agent with decide(state) method
        cycles: Number of test cycles
        
    Returns:
        Test results dictionary
    """
    chaos = ChaosMonkey(
        p_disconnect=0.05,
        p_latency=0.10,
        p_order_fail=0.05,
        enabled=True
    )
    
    results = {
        "cycles": cycles,
        "chaos_events": 0,
        "guardian_blocks": 0,
        "successful_trades": 0,
        "errors_caught": 0,
    }
    
    print("\n" + "=" * 60)
    print("🐒 CHAOS TEST RUNNING")
    print("=" * 60)
    
    error_count = 0
    
    for i in range(cycles):
        state = {
            "margin_ratio": random.uniform(0.3, 0.9),
            "daily_dd": random.uniform(0, 0.08),
            "open_positions": random.randint(0, 5),
            "margin_block_count": error_count,
            "error_detected": error_count > 0,
        }
        
        try:
            chaos_event = chaos.inject()
            if chaos_event:
                results["chaos_events"] += 1
            
            # Guardian decision
            action = agent.decide(state)
            action_name = action.name if hasattr(action, 'name') else str(action)
            
            if action_name in ["FORCE_HOLD", "EMERGENCY_FREEZE"]:
                results["guardian_blocks"] += 1
                chaos.record_guardian_response(True)
            else:
                results["successful_trades"] += 1
            
            error_count = max(0, error_count - 1)
            
        except Exception as e:
            results["errors_caught"] += 1
            error_count += 1
            
            # Guardian should respond to error
            state["error_detected"] = True
            state["margin_block_count"] = error_count
            action = agent.decide(state)
            action_name = action.name if hasattr(action, 'name') else str(action)
            
            if action_name in ["FORCE_HOLD", "EMERGENCY_FREEZE"]:
                chaos.record_guardian_response(True)
            else:
                chaos.record_guardian_response(False)
            
            print(f"  Cycle {i:3d} | 💥 ERROR: {type(e).__name__} | Guardian: {action_name}")
    
    print(chaos.get_report())
    
    results["chaos_stats"] = {
        "total_events": chaos.stats.total_events,
        "guardian_responses": chaos.stats.guardian_responses,
        "missed_responses": chaos.stats.missed_responses,
    }
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from src.rl.guardian_agent import get_guardian_agent
    
    print("\n🐒 Running ChaosMonkey test...\n")
    
    agent = get_guardian_agent()
    results = run_chaos_test(agent, cycles=30)
    
    print("\n📊 Test Results:")
    print(f"  Cycles: {results['cycles']}")
    print(f"  Chaos Events: {results['chaos_events']}")
    print(f"  Errors Caught: {results['errors_caught']}")
    print(f"  Guardian Blocks: {results['guardian_blocks']}")
