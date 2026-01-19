# src/chaos/chaos_level2.py
"""
Chaos Level 2 - Bankruptcy Simulation
======================================

Advanced chaos testing for Guardian resilience:
- Margin death spiral
- Broker attack (order rejections)
- Alpha gone wild
- Spread blowup

Use this to stress-test Guardian before competition.
"""

import random
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("CHAOS_L2")


class ChaosL2Event(Enum):
    """Level 2 chaos events."""
    NONE = "NONE"
    SPREAD_BLOWUP = "SPREAD_BLOWUP"
    MARGIN_DRAIN = "MARGIN_DRAIN"
    ORDER_REJECT = "ORDER_REJECT"
    EQUITY_JUMP = "EQUITY_JUMP"
    ALPHA_SPAM = "ALPHA_SPAM"
    REQUOTE_STORM = "REQUOTE_STORM"
    DISCONNECT = "DISCONNECT"


@dataclass
class ChaosL2State:
    """State for Level 2 chaos simulation."""
    equity: float = 1000.0
    margin: float = 1.0
    free_margin: float = 0.8
    spread: float = 1.0
    latency: float = 0.1
    consecutive_rejects: int = 0
    alpha_intent_count: int = 0


@dataclass
class ChaosL2Result:
    """Result from chaos test."""
    survived: bool
    cycles: int
    bankruptcy: bool
    kill_switch_triggered: bool
    events: List[str]
    guardian_responses: int
    missed_responses: int


class ChaosLevel2:
    """
    Level 2 Chaos Engine - Bankruptcy Simulation.
    
    Scenarios:
    A) Margin Death Spiral - continuous margin drain
    B) Broker Attack - random rejects + requotes
    C) Alpha Gone Wild - BUY spam every tick
    """
    
    def __init__(self, scenario: str = "mixed"):
        """
        Initialize chaos engine.
        
        Args:
            scenario: "margin_spiral", "broker_attack", "alpha_wild", or "mixed"
        """
        self.scenario = scenario
        self.state = ChaosL2State()
        self.events_log: List[str] = []
        
        # Event probabilities per scenario
        self.probabilities = self._get_probabilities()
        
        logger.info(f"ChaosLevel2 initialized: scenario={scenario}")
    
    def _get_probabilities(self) -> Dict[ChaosL2Event, float]:
        """Get event probabilities based on scenario."""
        if self.scenario == "margin_spiral":
            return {
                ChaosL2Event.MARGIN_DRAIN: 0.30,
                ChaosL2Event.SPREAD_BLOWUP: 0.15,
                ChaosL2Event.EQUITY_JUMP: 0.10,
            }
        elif self.scenario == "broker_attack":
            return {
                ChaosL2Event.ORDER_REJECT: 0.25,
                ChaosL2Event.REQUOTE_STORM: 0.20,
                ChaosL2Event.DISCONNECT: 0.10,
            }
        elif self.scenario == "alpha_wild":
            return {
                ChaosL2Event.ALPHA_SPAM: 0.40,
                ChaosL2Event.MARGIN_DRAIN: 0.15,
            }
        else:  # mixed
            return {
                ChaosL2Event.MARGIN_DRAIN: 0.15,
                ChaosL2Event.SPREAD_BLOWUP: 0.10,
                ChaosL2Event.ORDER_REJECT: 0.10,
                ChaosL2Event.REQUOTE_STORM: 0.08,
                ChaosL2Event.ALPHA_SPAM: 0.12,
                ChaosL2Event.DISCONNECT: 0.05,
            }
    
    def tick(self) -> ChaosL2Event:
        """
        Execute one chaos tick.
        
        Returns:
            Chaos event that occurred
            
        Raises:
            ConnectionError: For disconnect events
            TimeoutError: For order rejects
        """
        # Pick event based on probabilities
        r = random.random()
        cumulative = 0.0
        
        for event, prob in self.probabilities.items():
            cumulative += prob
            if r < cumulative:
                return self._apply_event(event)
        
        # No event
        self._natural_drift()
        return ChaosL2Event.NONE
    
    def _apply_event(self, event: ChaosL2Event) -> ChaosL2Event:
        """Apply chaos event effects."""
        self.events_log.append(event.value)
        
        if event == ChaosL2Event.SPREAD_BLOWUP:
            self.state.spread *= random.uniform(1.5, 3.0)
            self.state.equity -= self.state.equity * 0.02
            logger.warning(f"💥 SPREAD BLOWUP: spread={self.state.spread:.1f}")
            
        elif event == ChaosL2Event.MARGIN_DRAIN:
            drain = random.uniform(0.05, 0.15)
            self.state.margin -= drain
            self.state.free_margin -= drain * 1.5
            logger.warning(f"📉 MARGIN DRAIN: -{drain*100:.1f}%")
            
        elif event == ChaosL2Event.ORDER_REJECT:
            self.state.consecutive_rejects += 1
            logger.error("❌ ORDER REJECTED")
            raise TimeoutError("Order rejected by broker")
            
        elif event == ChaosL2Event.EQUITY_JUMP:
            jump = random.uniform(-0.05, -0.10)
            self.state.equity *= (1 + jump)
            logger.warning(f"📊 EQUITY JUMP: {jump*100:+.1f}%")
            
        elif event == ChaosL2Event.ALPHA_SPAM:
            self.state.alpha_intent_count += 10
            logger.warning(f"🤖 ALPHA SPAM: {self.state.alpha_intent_count} BUY signals")
            
        elif event == ChaosL2Event.REQUOTE_STORM:
            self.state.latency += random.uniform(1.0, 3.0)
            logger.warning(f"⚡ REQUOTE STORM: latency={self.state.latency:.1f}s")
            
        elif event == ChaosL2Event.DISCONNECT:
            logger.critical("💥 MT5 DISCONNECTED")
            raise ConnectionError("MT5 connection lost")
        
        return event
    
    def _natural_drift(self):
        """Apply natural market drift."""
        self.state.equity -= self.state.equity * random.uniform(0, 0.005)
        self.state.margin -= random.uniform(0, 0.02)
        self.state.free_margin -= random.uniform(0, 0.025)
        self.state.spread = max(1.0, self.state.spread * 0.95)  # Spread recovery
        self.state.latency = max(0.1, self.state.latency * 0.9)  # Latency recovery
    
    def is_bankrupt(self) -> bool:
        """Check if account is bankrupt."""
        return (
            self.state.equity <= 0 or
            self.state.margin <= 0 or
            self.state.free_margin <= 0.05
        )
    
    def get_guardian_state(self) -> Dict:
        """Get state dict for Guardian."""
        return {
            "daily_dd": max(0, (1000 - self.state.equity) / 1000),
            "margin_ratio": self.state.margin,
            "free_margin_ratio": self.state.free_margin,
            "error_detected": self.state.consecutive_rejects > 0,
            "margin_block_count": self.state.consecutive_rejects,
            "alpha_signal": 1 if self.state.alpha_intent_count > 5 else 0,
        }
    
    def reset(self):
        """Reset chaos state."""
        self.state = ChaosL2State()
        self.events_log = []


def run_chaos_level2_test(agent, scenario: str = "mixed", cycles: int = 50) -> ChaosL2Result:
    """
    Run Level 2 chaos test.
    
    Args:
        agent: Guardian agent with decide(state) method
        scenario: Chaos scenario to run
        cycles: Number of cycles
        
    Returns:
        Test results
    """
    chaos = ChaosLevel2(scenario=scenario)
    
    guardian_responses = 0
    missed_responses = 0
    kill_triggered = False
    
    print("\n" + "=" * 60)
    print(f"🔥 CHAOS LEVEL 2 TEST: {scenario.upper()}")
    print("=" * 60)
    
    for i in range(cycles):
        try:
            event = chaos.tick()
            
            # Get Guardian decision
            state = chaos.get_guardian_state()
            action = agent.decide(state)
            action_name = action.name if hasattr(action, 'name') else str(action)
            
            # Check response appropriateness
            if event != ChaosL2Event.NONE:
                if action_name in ["FORCE_HOLD", "EMERGENCY_FREEZE", "REDUCE_RISK"]:
                    guardian_responses += 1
                else:
                    missed_responses += 1
            
            if action_name == "EMERGENCY_FREEZE":
                kill_triggered = True
                print(f"  Cycle {i:2d} | ☠️ KILL SWITCH TRIGGERED")
                break
            
            # Status
            status = f"Cycle {i:2d} | Equity=${chaos.state.equity:.0f} | Margin={chaos.state.margin*100:.0f}%"
            if event != ChaosL2Event.NONE:
                status += f" | 💥 {event.value}"
            status += f" | Guardian: {action_name}"
            print(f"  {status}")
            
        except (ConnectionError, TimeoutError) as e:
            # Error event
            state = chaos.get_guardian_state()
            state["error_detected"] = True
            action = agent.decide(state)
            action_name = action.name if hasattr(action, 'name') else str(action)
            
            if action_name in ["FORCE_HOLD", "EMERGENCY_FREEZE"]:
                guardian_responses += 1
            else:
                missed_responses += 1
            
            print(f"  Cycle {i:2d} | 💥 ERROR: {type(e).__name__} | Guardian: {action_name}")
            chaos.state.consecutive_rejects = 0  # Reset after handling
        
        # Check bankruptcy
        if chaos.is_bankrupt():
            print(f"  💀 BANKRUPTCY at cycle {i}")
            break
    
    # Results
    result = ChaosL2Result(
        survived=not chaos.is_bankrupt(),
        cycles=i + 1,
        bankruptcy=chaos.is_bankrupt(),
        kill_switch_triggered=kill_triggered,
        events=chaos.events_log,
        guardian_responses=guardian_responses,
        missed_responses=missed_responses,
    )
    
    print("=" * 60)
    print(f"📊 RESULTS:")
    print(f"   Survived: {'✅' if result.survived else '❌'}")
    print(f"   Cycles: {result.cycles}")
    print(f"   Bankruptcy: {'❌ YES' if result.bankruptcy else '✅ NO'}")
    print(f"   Kill Switch: {'⚠️ YES' if result.kill_switch_triggered else 'NO'}")
    print(f"   Guardian Responses: {result.guardian_responses}")
    print(f"   Missed Responses: {result.missed_responses}")
    if result.guardian_responses + result.missed_responses > 0:
        rate = result.guardian_responses / (result.guardian_responses + result.missed_responses)
        print(f"   Response Rate: {rate:.1%}")
    print("=" * 60 + "\n")
    
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from src.rl.guardian_agent import get_guardian_agent
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="mixed", 
                       choices=["mixed", "margin_spiral", "broker_attack", "alpha_wild"])
    parser.add_argument("--cycles", type=int, default=50)
    args = parser.parse_args()
    
    agent = get_guardian_agent()
    run_chaos_level2_test(agent, scenario=args.scenario, cycles=args.cycles)
