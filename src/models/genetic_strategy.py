"""
genetic_strategy.py
====================
Auto Strategy Generation using Genetic Algorithms

Strategy = Genome
ตลาด = Environment
PnL + Risk = Fitness

ไม่รอมนุษย์คิดกลยุทธ์ → ให้ AI "วิวัฒน์" กลยุทธ์เอง
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger("GENETIC_STRATEGY")


@dataclass
class StrategyGenome:
    """
    Strategy represented as a genome.
    
    Each gene encodes a trading parameter.
    """
    # Indicator genes
    indicators: List[str] = field(default_factory=list)
    lookback_windows: List[int] = field(default_factory=list)
    
    # Entry logic genes
    entry_conditions: List[Dict] = field(default_factory=list)
    entry_threshold: float = 0.5
    
    # Exit logic genes
    exit_conditions: List[Dict] = field(default_factory=list)
    sl_multiplier: float = 1.5
    tp_multiplier: float = 2.0
    
    # Risk genes
    max_risk_pct: float = 0.01
    max_positions: int = 3
    
    # Time filter genes
    trading_hours: Tuple[int, int] = (8, 20)
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    
    # Metadata
    generation: int = 0
    fitness: float = 0.0
    alpha: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "indicators": self.indicators,
            "lookback_windows": self.lookback_windows,
            "entry_conditions": self.entry_conditions,
            "entry_threshold": self.entry_threshold,
            "exit_conditions": self.exit_conditions,
            "sl_multiplier": self.sl_multiplier,
            "tp_multiplier": self.tp_multiplier,
            "max_risk_pct": self.max_risk_pct,
            "max_positions": self.max_positions,
            "trading_hours": self.trading_hours,
            "fitness": self.fitness,
        }


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    elite_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    max_generations: int = 100
    min_fitness_threshold: float = 0.5
    
    # Available gene values
    available_indicators: List[str] = field(default_factory=lambda: [
        "RSI", "MACD", "EMA", "SMA", "BB", "ATR", "ADX", "STOCH"
    ])
    lookback_range: Tuple[int, int] = (5, 200)
    sl_range: Tuple[float, float] = (1.0, 3.0)
    tp_range: Tuple[float, float] = (1.5, 5.0)


class GeneticStrategyEngine:
    """
    Genetic Algorithm Strategy Generator.
    
    Evolves trading strategies through selection, crossover, and mutation.
    """

    def __init__(self, config: GeneticConfig = None, 
                 fitness_func: Callable = None):
        self.config = config or GeneticConfig()
        self.fitness_func = fitness_func or self._default_fitness
        
        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.best_genome: Optional[StrategyGenome] = None
        self.history: List[Dict] = []

    # -------------------------------------------------
    # Main evolution loop
    # -------------------------------------------------
    def evolve(self, market_data: Any, generations: int = None) -> StrategyGenome:
        """
        Run genetic evolution.
        
        Args:
            market_data: Historical data for backtesting
            generations: Number of generations to run
            
        Returns:
            Best evolved strategy genome
        """
        generations = generations or self.config.max_generations
        
        # Initialize population
        if not self.population:
            self._initialize_population()
        
        logger.info(f"Starting evolution: {generations} generations, {len(self.population)} population")
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            self._evaluate_population(market_data)
            
            # Track best
            self._update_best()
            
            # Log progress
            if gen % 10 == 0:
                logger.info(f"Gen {gen}: Best fitness={self.best_genome.fitness:.4f}, "
                           f"Alpha={self.best_genome.alpha:.4f}")
            
            # Check convergence
            if self.best_genome.fitness >= self.config.min_fitness_threshold:
                logger.info(f"Converged at generation {gen}")
                break
            
            # Selection
            parents = self._select_parents()
            
            # Create next generation
            offspring = self._create_offspring(parents)
            
            # Replace population (keep elites)
            self._replace_population(offspring)
        
        logger.info(f"Evolution complete: Best fitness={self.best_genome.fitness:.4f}")
        return self.best_genome

    # -------------------------------------------------
    # Population management
    # -------------------------------------------------
    def _initialize_population(self):
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.config.population_size):
            genome = self._random_genome()
            self.population.append(genome)
        
        logger.debug(f"Initialized population: {len(self.population)}")

    def _random_genome(self) -> StrategyGenome:
        """Create random genome."""
        cfg = self.config
        
        # Random indicators (2-4)
        num_indicators = random.randint(2, 4)
        indicators = random.sample(cfg.available_indicators, num_indicators)
        
        # Random lookbacks
        lookbacks = [
            random.randint(cfg.lookback_range[0], cfg.lookback_range[1])
            for _ in range(num_indicators)
        ]
        
        # Random entry conditions
        entry_conditions = [
            {"indicator": ind, "condition": random.choice(["above", "below", "cross_up", "cross_down"])}
            for ind in indicators[:2]
        ]
        
        # Random exit conditions
        exit_conditions = [
            {"indicator": indicators[0], "condition": random.choice(["reversal", "target", "time"])}
        ]
        
        return StrategyGenome(
            indicators=indicators,
            lookback_windows=lookbacks,
            entry_conditions=entry_conditions,
            entry_threshold=random.uniform(0.4, 0.8),
            exit_conditions=exit_conditions,
            sl_multiplier=random.uniform(cfg.sl_range[0], cfg.sl_range[1]),
            tp_multiplier=random.uniform(cfg.tp_range[0], cfg.tp_range[1]),
            max_risk_pct=random.uniform(0.005, 0.02),
            max_positions=random.randint(1, 5),
            generation=self.generation,
        )

    # -------------------------------------------------
    # Fitness evaluation
    # -------------------------------------------------
    def _evaluate_population(self, market_data: Any):
        """Evaluate fitness for all genomes."""
        for genome in self.population:
            genome.fitness = self.fitness_func(genome, market_data)

    def _default_fitness(self, genome: StrategyGenome, market_data: Any) -> float:
        """
        Default fitness function.
        
        fitness = (
            alpha_score * 0.4
            - drawdown_penalty * 0.3
            + regime_robustness * 0.2
            + execution_quality * 0.1
        )
        """
        # Simulate backtest (placeholder - would use actual backtester)
        alpha = random.uniform(-0.1, 0.3)  # Simulated alpha
        max_dd = random.uniform(0.02, 0.15)  # Simulated drawdown
        sharpe = random.uniform(-0.5, 2.0)  # Simulated Sharpe
        regime_score = random.uniform(0.3, 1.0)  # Simulated regime robustness
        
        genome.alpha = alpha
        genome.max_drawdown = max_dd
        genome.sharpe = sharpe
        
        # Calculate fitness
        alpha_score = max(0, alpha) / 0.3  # Normalize
        dd_penalty = max_dd / 0.15
        
        fitness = (
            alpha_score * 0.4 -
            dd_penalty * 0.3 +
            regime_score * 0.2 +
            0.7 * 0.1  # Assume good execution
        )
        
        return max(0, min(1, fitness))

    # -------------------------------------------------
    # Selection
    # -------------------------------------------------
    def _select_parents(self) -> List[StrategyGenome]:
        """Select parents using tournament selection."""
        parents = []
        
        # Keep elites
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents.extend(sorted_pop[:self.config.elite_size])
        
        # Tournament selection for rest
        while len(parents) < self.config.population_size // 2:
            tournament = random.sample(self.population, 3)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents

    # -------------------------------------------------
    # Crossover and Mutation
    # -------------------------------------------------
    def _create_offspring(self, parents: List[StrategyGenome]) -> List[StrategyGenome]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        while len(offspring) < self.config.population_size - self.config.elite_size:
            # Select two parents
            p1, p2 = random.sample(parents, 2)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child = self._crossover(p1, p2)
            else:
                child = self._copy_genome(p1)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child = self._mutate(child)
            
            child.generation = self.generation + 1
            offspring.append(child)
        
        return offspring

    def _crossover(self, p1: StrategyGenome, p2: StrategyGenome) -> StrategyGenome:
        """Crossover two parents."""
        child = StrategyGenome(
            indicators=p1.indicators if random.random() < 0.5 else p2.indicators,
            lookback_windows=p1.lookback_windows if random.random() < 0.5 else p2.lookback_windows,
            entry_conditions=p1.entry_conditions if random.random() < 0.5 else p2.entry_conditions,
            entry_threshold=(p1.entry_threshold + p2.entry_threshold) / 2,
            exit_conditions=p1.exit_conditions if random.random() < 0.5 else p2.exit_conditions,
            sl_multiplier=(p1.sl_multiplier + p2.sl_multiplier) / 2,
            tp_multiplier=(p1.tp_multiplier + p2.tp_multiplier) / 2,
            max_risk_pct=(p1.max_risk_pct + p2.max_risk_pct) / 2,
            max_positions=p1.max_positions if random.random() < 0.5 else p2.max_positions,
        )
        return child

    def _mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """Apply random mutation."""
        cfg = self.config
        
        # Random mutation type
        mutation_type = random.choice([
            "indicator", "lookback", "threshold", "sl_tp", "risk"
        ])
        
        if mutation_type == "indicator" and genome.indicators:
            idx = random.randint(0, len(genome.indicators) - 1)
            genome.indicators[idx] = random.choice(cfg.available_indicators)
        
        elif mutation_type == "lookback" and genome.lookback_windows:
            idx = random.randint(0, len(genome.lookback_windows) - 1)
            genome.lookback_windows[idx] = random.randint(cfg.lookback_range[0], cfg.lookback_range[1])
        
        elif mutation_type == "threshold":
            genome.entry_threshold = max(0.3, min(0.9, genome.entry_threshold + random.uniform(-0.1, 0.1)))
        
        elif mutation_type == "sl_tp":
            genome.sl_multiplier = max(1.0, min(3.0, genome.sl_multiplier + random.uniform(-0.3, 0.3)))
            genome.tp_multiplier = max(1.5, min(5.0, genome.tp_multiplier + random.uniform(-0.5, 0.5)))
        
        elif mutation_type == "risk":
            genome.max_risk_pct = max(0.005, min(0.03, genome.max_risk_pct + random.uniform(-0.005, 0.005)))
        
        return genome

    def _copy_genome(self, genome: StrategyGenome) -> StrategyGenome:
        """Create a copy of genome."""
        return StrategyGenome(
            indicators=genome.indicators.copy(),
            lookback_windows=genome.lookback_windows.copy(),
            entry_conditions=[c.copy() for c in genome.entry_conditions],
            entry_threshold=genome.entry_threshold,
            exit_conditions=[c.copy() for c in genome.exit_conditions],
            sl_multiplier=genome.sl_multiplier,
            tp_multiplier=genome.tp_multiplier,
            max_risk_pct=genome.max_risk_pct,
            max_positions=genome.max_positions,
        )

    # -------------------------------------------------
    # Population replacement
    # -------------------------------------------------
    def _replace_population(self, offspring: List[StrategyGenome]):
        """Replace population with elites and offspring."""
        # Keep elites
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:self.config.elite_size]
        
        self.population = elites + offspring

    def _update_best(self):
        """Update best genome."""
        current_best = max(self.population, key=lambda x: x.fitness)
        
        if self.best_genome is None or current_best.fitness > self.best_genome.fitness:
            self.best_genome = self._copy_genome(current_best)
            self.best_genome.fitness = current_best.fitness
            self.best_genome.alpha = current_best.alpha
            self.best_genome.max_drawdown = current_best.max_drawdown
            self.best_genome.sharpe = current_best.sharpe
            
            self.history.append({
                "generation": self.generation,
                "fitness": current_best.fitness,
                "alpha": current_best.alpha,
            })

    # -------------------------------------------------
    # Status
    # -------------------------------------------------
    def get_status(self) -> Dict:
        """Get engine status."""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_genome.fitness if self.best_genome else 0,
            "best_alpha": self.best_genome.alpha if self.best_genome else 0,
            "best_sharpe": self.best_genome.sharpe if self.best_genome else 0,
        }
