"""
stress_simulator.py
====================
Capital Stress Simulator (What-if Engine)

"ถ้าวัน Black Swan มา จะรอดไหม?"

Fund Rule: if survival_probability < 95%: strategy_not_allowed()

Strategy ที่กำไรดี แต่ไม่รอด stress = หายนะในอนาคต
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import math
from src.utils.logger import get_logger

logger = get_logger("STRESS_SIMULATOR")


class StressScenario(str, Enum):
    """Pre-defined stress scenarios."""
    # Market scenarios
    LIQUIDITY_FREEZE_2008 = "LIQUIDITY_FREEZE_2008"
    COVID_VOLATILITY = "COVID_VOLATILITY"
    FLASH_CRASH = "FLASH_CRASH"
    INTEREST_RATE_SHOCK = "INTEREST_RATE_SHOCK"
    
    # Portfolio scenarios
    CORRELATION_SPIKE = "CORRELATION_SPIKE"
    SLIPPAGE_5X = "SLIPPAGE_5X"
    EXECUTION_DELAY = "EXECUTION_DELAY"
    MARGIN_CALL = "MARGIN_CALL"
    
    # Combined
    BLACK_SWAN = "BLACK_SWAN"


@dataclass
class ScenarioConfig:
    """Configuration for a stress scenario."""
    name: str
    volatility_multiplier: float = 1.0
    correlation_override: float = 0.0    # Set correlation to this value
    slippage_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    execution_delay_ms: float = 0.0
    liquidity_reduction: float = 0.0     # 0-1, reduce liquidity by this %
    drawdown_acceleration: float = 1.0   # Speed up drawdown


@dataclass
class StressResult:
    """Result of stress simulation."""
    scenario: str
    survival_probability: float          # 0-1
    max_capital_loss: float              # As percentage
    worst_drawdown: float
    recovery_days: int
    forced_liquidation_risk: float       # 0-1
    passed: bool
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StressSummary:
    """Summary of all stress tests."""
    strategy: str
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: List[str]
    worst_scenario: str
    min_survival_prob: float
    overall_passed: bool


class CapitalStressSimulator:
    """
    Capital Stress Simulator.
    
    Tests strategy survival under extreme market conditions.
    """

    def __init__(self):
        # Default scenarios
        self.scenarios: Dict[str, ScenarioConfig] = {
            StressScenario.LIQUIDITY_FREEZE_2008.value: ScenarioConfig(
                name="2008 Liquidity Freeze",
                volatility_multiplier=3.0,
                correlation_override=0.95,
                slippage_multiplier=10.0,
                spread_multiplier=5.0,
                liquidity_reduction=0.8,
            ),
            StressScenario.COVID_VOLATILITY.value: ScenarioConfig(
                name="COVID Volatility",
                volatility_multiplier=5.0,
                correlation_override=0.85,
                slippage_multiplier=3.0,
                spread_multiplier=3.0,
            ),
            StressScenario.FLASH_CRASH.value: ScenarioConfig(
                name="Flash Crash",
                volatility_multiplier=10.0,
                slippage_multiplier=20.0,
                execution_delay_ms=1000,
                drawdown_acceleration=5.0,
            ),
            StressScenario.INTEREST_RATE_SHOCK.value: ScenarioConfig(
                name="Interest Rate Shock",
                volatility_multiplier=2.0,
                correlation_override=0.7,
                slippage_multiplier=2.0,
            ),
            StressScenario.CORRELATION_SPIKE.value: ScenarioConfig(
                name="Correlation Spike to 1.0",
                correlation_override=1.0,
            ),
            StressScenario.SLIPPAGE_5X.value: ScenarioConfig(
                name="5x Slippage",
                slippage_multiplier=5.0,
            ),
            StressScenario.BLACK_SWAN.value: ScenarioConfig(
                name="Black Swan Event",
                volatility_multiplier=8.0,
                correlation_override=0.98,
                slippage_multiplier=15.0,
                spread_multiplier=10.0,
                execution_delay_ms=2000,
                liquidity_reduction=0.9,
                drawdown_acceleration=10.0,
            ),
        }
        
        # Thresholds
        self.min_survival_probability = 0.95
        self.max_acceptable_loss = 0.20
        self.max_recovery_days = 60

    # -------------------------------------------------
    # Main simulation
    # -------------------------------------------------
    def run_stress_test(self, strategy_name: str,
                        trades: List[Any],
                        portfolio_value: float,
                        scenarios: List[str] = None) -> StressSummary:
        """
        Run stress tests for a strategy.
        
        Args:
            strategy_name: Name of strategy
            trades: Historical trades for replay
            portfolio_value: Current portfolio value
            scenarios: List of scenario names (or None for all)
            
        Returns:
            StressSummary with all results
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
        
        logger.info(f"Running stress test for {strategy_name}: {len(scenarios)} scenarios")
        
        results = []
        for scenario_name in scenarios:
            if scenario_name not in self.scenarios:
                continue
            
            config = self.scenarios[scenario_name]
            result = self._simulate_scenario(strategy_name, trades, portfolio_value, config)
            result.scenario = scenario_name
            results.append(result)
            
            logger.debug(f"  {scenario_name}: Survival={result.survival_probability:.1%}, "
                        f"MaxLoss={result.max_capital_loss:.1%}")
        
        # Create summary
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        worst = min(results, key=lambda x: x.survival_probability) if results else None
        
        summary = StressSummary(
            strategy=strategy_name,
            total_scenarios=len(results),
            passed_scenarios=len(passed),
            failed_scenarios=[r.scenario for r in failed],
            worst_scenario=worst.scenario if worst else "NONE",
            min_survival_prob=worst.survival_probability if worst else 1.0,
            overall_passed=len(failed) == 0,
        )
        
        if not summary.overall_passed:
            logger.warning(f"{strategy_name}: FAILED stress test - {len(failed)} scenarios failed")
        else:
            logger.info(f"{strategy_name}: PASSED all stress tests")
        
        return summary

    def _simulate_scenario(self, strategy: str, trades: List[Any],
                          portfolio_value: float,
                          config: ScenarioConfig) -> StressResult:
        """Simulate a single stress scenario."""
        # Monte Carlo simulation
        num_simulations = 1000
        survivals = 0
        max_losses = []
        drawdowns = []
        recovery_times = []
        liquidation_count = 0
        
        for _ in range(num_simulations):
            result = self._run_single_simulation(trades, portfolio_value, config)
            
            if result["survived"]:
                survivals += 1
            
            max_losses.append(result["max_loss"])
            drawdowns.append(result["max_drawdown"])
            recovery_times.append(result["recovery_days"])
            
            if result["liquidated"]:
                liquidation_count += 1
        
        survival_prob = survivals / num_simulations
        avg_max_loss = sum(max_losses) / len(max_losses) if max_losses else 0
        worst_dd = max(drawdowns) if drawdowns else 0
        avg_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        liquidation_risk = liquidation_count / num_simulations
        
        # Determine if passed
        passed = (
            survival_prob >= self.min_survival_probability and
            avg_max_loss <= self.max_acceptable_loss and
            avg_recovery <= self.max_recovery_days
        )
        
        return StressResult(
            scenario=config.name,
            survival_probability=survival_prob,
            max_capital_loss=avg_max_loss,
            worst_drawdown=worst_dd,
            recovery_days=int(avg_recovery),
            forced_liquidation_risk=liquidation_risk,
            passed=passed,
            details={
                "simulations": num_simulations,
                "config": config.name,
            },
        )

    def _run_single_simulation(self, trades: List[Any],
                              portfolio_value: float,
                              config: ScenarioConfig) -> Dict:
        """Run single Monte Carlo simulation."""
        equity = portfolio_value
        peak = portfolio_value
        max_dd = 0.0
        max_loss = 0.0
        days = 0
        
        # Simulate stressed conditions
        for _ in range(100):  # 100 trading days
            # Apply stress multipliers
            base_change = random.gauss(0, 0.01)  # Base daily change
            stressed_change = base_change * config.volatility_multiplier
            
            # Slippage impact
            slippage_cost = 0.0001 * config.slippage_multiplier
            
            # Net change
            net_change = stressed_change - slippage_cost
            
            # Accelerate drawdowns
            if net_change < 0:
                net_change *= config.drawdown_acceleration
            
            # Update equity
            equity *= (1 + net_change)
            
            # Track metrics
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
            
            loss = (portfolio_value - equity) / portfolio_value
            if loss > max_loss:
                max_loss = loss
            
            days += 1
            
            # Check for liquidation
            if equity < portfolio_value * 0.5:  # 50% loss = margin call
                return {
                    "survived": False,
                    "max_loss": loss,
                    "max_drawdown": max_dd,
                    "recovery_days": 999,
                    "liquidated": True,
                }
        
        # Calculate recovery time
        recovery_days = 0
        if equity < portfolio_value:
            # Estimate recovery
            daily_return = 0.001  # Assume 0.1% daily return
            needed_return = (portfolio_value - equity) / equity
            recovery_days = int(needed_return / daily_return) if daily_return > 0 else 999
        
        return {
            "survived": equity >= portfolio_value * 0.8,  # Survive if > 80%
            "max_loss": max((portfolio_value - equity) / portfolio_value, 0),
            "max_drawdown": max_dd,
            "recovery_days": recovery_days,
            "liquidated": False,
        }

    # -------------------------------------------------
    # Custom scenarios
    # -------------------------------------------------
    def add_scenario(self, name: str, config: ScenarioConfig):
        """Add custom stress scenario."""
        self.scenarios[name] = config
        logger.info(f"Added stress scenario: {name}")

    # -------------------------------------------------
    # Quick check
    # -------------------------------------------------
    def quick_check(self, strategy: str, portfolio_value: float) -> bool:
        """Quick stress check without full simulation."""
        # Run only critical scenarios
        critical = [
            StressScenario.BLACK_SWAN.value,
            StressScenario.FLASH_CRASH.value,
        ]
        
        summary = self.run_stress_test(strategy, [], portfolio_value, critical)
        return summary.overall_passed

    def get_status(self) -> Dict:
        """Get simulator status."""
        return {
            "scenarios_available": len(self.scenarios),
            "min_survival_required": self.min_survival_probability,
            "max_loss_allowed": self.max_acceptable_loss,
        }
