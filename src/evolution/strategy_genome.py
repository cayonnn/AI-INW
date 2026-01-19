# src/evolution/strategy_genome.py
"""
Strategy Genome Engine
=======================

AI-driven strategy evolution through genetic algorithms.

Concept:
    Strategy ≠ Code
    Strategy = DNA (a genome that can mutate and evolve)

Features:
    - Strategy representation as genomes
    - Crossover and mutation operators
    - Fitness evaluation
    - Population-based evolution
    - Auto-discovery of new strategies

Paper Statement:
    "We represent trading strategies as evolvable genomes,
     enabling the autonomous discovery of novel alpha sources."
"""

import os
import sys
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("STRATEGY_GENOME")


# =============================================================================
# Genome Components
# =============================================================================

class EntryLogic(Enum):
    """Entry logic types."""
    EMA_CROSS = "ema_cross"
    RSI_REVERSAL = "rsi_reversal"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    PATTERN = "pattern"


class ExitLogic(Enum):
    """Exit logic types."""
    FIXED_TP_SL = "fixed_tp_sl"
    TRAILING_STOP = "trailing_stop"
    ATR_BASED = "atr_based"
    TIME_BASED = "time_based"
    SIGNAL_REVERSAL = "signal_reversal"


class RiskProfile(Enum):
    """Risk profile types."""
    CONSERVATIVE = "conservative"  # Small lots, tight stops
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"  # Larger lots, wider stops


class Timeframe(Enum):
    """Strategy timeframe."""
    SCALP = "scalp"  # < 15m
    INTRADAY = "intraday"  # 15m-4h
    SWING = "swing"  # 4h-1d
    POSITION = "position"  # > 1d


@dataclass
class StrategyGenome:
    """
    A trading strategy represented as DNA.
    
    Can be mutated, crossed over, and evolved.
    """
    id: str
    generation: int = 0
    
    # Core DNA
    entry_logic: EntryLogic = EntryLogic.EMA_CROSS
    exit_logic: ExitLogic = ExitLogic.TRAILING_STOP
    risk_profile: RiskProfile = RiskProfile.MODERATE
    timeframe: Timeframe = Timeframe.INTRADAY
    
    # Numeric genes
    entry_threshold: float = 0.5  # 0-1
    exit_threshold: float = 0.5  # 0-1
    position_size_gene: float = 0.5  # 0-1 (maps to lot size)
    stop_loss_gene: float = 0.5  # 0-1 (maps to SL pips)
    take_profit_gene: float = 0.5  # 0-1 (maps to TP pips)
    
    # Regime sensitivity
    trend_sensitivity: float = 0.5  # 0-1
    volatility_sensitivity: float = 0.5  # 0-1
    
    # Fitness
    fitness: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    
    # Lineage
    parent_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "entry": self.entry_logic.value,
            "exit": self.exit_logic.value,
            "risk": self.risk_profile.value,
            "timeframe": self.timeframe.value,
            "fitness": round(self.fitness, 4),
            "win_rate": round(self.win_rate, 3),
            "sharpe": round(self.sharpe, 2)
        }
    
    def get_parameters(self) -> Dict:
        """Get tradable parameters from genes."""
        return {
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "lot_size": 0.01 + self.position_size_gene * 0.09,  # 0.01-0.10
            "sl_pips": 10 + self.stop_loss_gene * 90,  # 10-100
            "tp_pips": 15 + self.take_profit_gene * 135,  # 15-150
            "trend_sensitivity": self.trend_sensitivity,
            "volatility_sensitivity": self.volatility_sensitivity
        }


class StrategyGenomeEngine:
    """
    Evolves trading strategies through genetic algorithms.
    
    Features:
        - Population management
        - Crossover (combine parent strategies)
        - Mutation (random changes)
        - Selection (survival of fittest)
    """
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.best_ever: Optional[StrategyGenome] = None
        
        # History
        self.generation_history: List[Dict] = []
        
        logger.info(f"🧬 StrategyGenomeEngine initialized (pop={population_size})")
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        
        for i in range(self.population_size):
            genome = StrategyGenome(
                id=f"G0_{i}",
                generation=0,
                entry_logic=random.choice(list(EntryLogic)),
                exit_logic=random.choice(list(ExitLogic)),
                risk_profile=random.choice(list(RiskProfile)),
                timeframe=random.choice(list(Timeframe)),
                entry_threshold=random.random(),
                exit_threshold=random.random(),
                position_size_gene=random.random(),
                stop_loss_gene=random.random(),
                take_profit_gene=random.random(),
                trend_sensitivity=random.random(),
                volatility_sensitivity=random.random()
            )
            self.population.append(genome)
        
        logger.info(f"🧬 Initialized population with {len(self.population)} genomes")
    
    def evaluate_fitness(self, genome: StrategyGenome, backtest_result: Dict):
        """Evaluate fitness from backtest results."""
        # Fitness = Sharpe * (1 - DD) * sqrt(trades)
        sharpe = backtest_result.get("sharpe", 0)
        max_dd = backtest_result.get("max_dd", 0.5)
        trades = backtest_result.get("trades", 0)
        win_rate = backtest_result.get("win_rate", 0)
        
        # Penalize high DD
        dd_penalty = max(0, 1 - max_dd * 2)
        
        # Reward more trades (statistically significant)
        trade_bonus = np.sqrt(max(trades, 1)) / 10
        
        fitness = sharpe * dd_penalty * trade_bonus
        
        genome.fitness = fitness
        genome.trades = trades
        genome.win_rate = win_rate
        genome.sharpe = sharpe
        genome.max_dd = max_dd
        
        return fitness
    
    def crossover(
        self,
        parent1: StrategyGenome,
        parent2: StrategyGenome
    ) -> StrategyGenome:
        """Create child genome from two parents."""
        child_id = f"G{self.generation}_{len(self.population)}"
        
        child = StrategyGenome(
            id=child_id,
            generation=self.generation,
            # Randomly inherit from parents
            entry_logic=random.choice([parent1.entry_logic, parent2.entry_logic]),
            exit_logic=random.choice([parent1.exit_logic, parent2.exit_logic]),
            risk_profile=random.choice([parent1.risk_profile, parent2.risk_profile]),
            timeframe=random.choice([parent1.timeframe, parent2.timeframe]),
            # Blend numeric genes
            entry_threshold=(parent1.entry_threshold + parent2.entry_threshold) / 2,
            exit_threshold=(parent1.exit_threshold + parent2.exit_threshold) / 2,
            position_size_gene=(parent1.position_size_gene + parent2.position_size_gene) / 2,
            stop_loss_gene=(parent1.stop_loss_gene + parent2.stop_loss_gene) / 2,
            take_profit_gene=(parent1.take_profit_gene + parent2.take_profit_gene) / 2,
            trend_sensitivity=(parent1.trend_sensitivity + parent2.trend_sensitivity) / 2,
            volatility_sensitivity=(parent1.volatility_sensitivity + parent2.volatility_sensitivity) / 2,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child
    
    def mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """Apply random mutations to genome."""
        # Mutate entry logic
        if random.random() < self.mutation_rate:
            genome.entry_logic = random.choice(list(EntryLogic))
        
        # Mutate exit logic
        if random.random() < self.mutation_rate:
            genome.exit_logic = random.choice(list(ExitLogic))
        
        # Mutate numeric genes
        for attr in ['entry_threshold', 'exit_threshold', 'position_size_gene',
                     'stop_loss_gene', 'take_profit_gene', 'trend_sensitivity',
                     'volatility_sensitivity']:
            if random.random() < self.mutation_rate:
                current = getattr(genome, attr)
                mutation = random.gauss(0, 0.1)
                new_value = np.clip(current + mutation, 0, 1)
                setattr(genome, attr, new_value)
        
        return genome
    
    def select_parents(self) -> Tuple[StrategyGenome, StrategyGenome]:
        """Tournament selection for parents."""
        tournament_size = 3
        
        def tournament():
            candidates = random.sample(self.population, tournament_size)
            return max(candidates, key=lambda g: g.fitness)
        
        return tournament(), tournament()
    
    def evolve_generation(self):
        """Evolve to next generation."""
        self.generation += 1
        
        # Keep top performers (elitism)
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        elite_count = max(2, self.population_size // 5)
        new_population = sorted_pop[:elite_count]
        
        # Update best ever
        if not self.best_ever or sorted_pop[0].fitness > self.best_ever.fitness:
            self.best_ever = sorted_pop[0]
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
            else:
                child = random.choice(sorted_pop[:10])  # Clone top 10
            
            child = self.mutate(child)
            child.id = f"G{self.generation}_{len(new_population)}"
            child.generation = self.generation
            new_population.append(child)
        
        self.population = new_population
        
        # Log
        avg_fitness = np.mean([g.fitness for g in self.population])
        best_fitness = max(g.fitness for g in self.population)
        
        self.generation_history.append({
            "generation": self.generation,
            "avg_fitness": avg_fitness,
            "best_fitness": best_fitness,
            "best_id": sorted_pop[0].id
        })
        
        logger.info(
            f"🧬 Generation {self.generation}: "
            f"Best={best_fitness:.3f} Avg={avg_fitness:.3f}"
        )
    
    def get_best_strategies(self, n: int = 5) -> List[StrategyGenome]:
        """Get top N strategies."""
        return sorted(self.population, key=lambda g: g.fitness, reverse=True)[:n]
    
    def summary(self) -> str:
        """Generate summary."""
        best = self.get_best_strategies(1)[0] if self.population else None
        return (
            f"🧬 Evolution | Gen={self.generation} | "
            f"Best={best.fitness:.3f}" if best else "No population"
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Strategy Genome Engine Test")
    print("=" * 60)
    
    engine = StrategyGenomeEngine(population_size=20)
    engine.initialize_population()
    
    # Simulate evolution
    for gen in range(5):
        # Simulate backtest for each genome
        for genome in engine.population:
            # Random backtest results (in real use, actual backtest)
            result = {
                "sharpe": random.uniform(-0.5, 2.0),
                "max_dd": random.uniform(0.05, 0.25),
                "trades": random.randint(10, 100),
                "win_rate": random.uniform(0.4, 0.7)
            }
            engine.evaluate_fitness(genome, result)
        
        engine.evolve_generation()
    
    print("\n--- Top 3 Strategies ---")
    for i, genome in enumerate(engine.get_best_strategies(3)):
        print(f"\n{i+1}. {genome.id}")
        print(f"   Entry: {genome.entry_logic.value}")
        print(f"   Exit: {genome.exit_logic.value}")
        print(f"   Fitness: {genome.fitness:.3f}")
        print(f"   Params: {genome.get_parameters()}")
    
    print("\n" + "=" * 60)
