# src/retrain/meta_optimizer.py
"""
Meta-Parameter Optimizer - Competition Grade
=============================================

Optimizes meta-parameters (not signal logic) based on daily performance:
- ConfidenceEngine thresholds
- PyramidManager risk multipliers
- ModeController sensitivity
- ScoreOptimizer caps
- WinStreakBooster decay

Uses grid search + performance simulation to find optimal configs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os
import copy

from src.utils.logger import get_logger

logger = get_logger("META_OPTIMIZER")


@dataclass
class MetaConfig:
    """Meta-parameter configuration."""
    name: str
    version: int
    created: datetime
    
    # ConfidenceEngine
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.80, 0.65, 0.50, 0.35])
    confidence_multipliers: List[float] = field(default_factory=lambda: [1.4, 1.2, 1.0, 0.8, 0.6])
    
    # PyramidManager
    pyramid_risk_mults: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.4])
    pyramid_max_entries: int = 3
    pyramid_dd_threshold: float = 0.03
    
    # ModeController
    alpha_score_threshold: float = 70
    alpha_dd_threshold: float = 5.0
    defensive_score_threshold: float = 40
    defensive_dd_threshold: float = 8.0
    mode_hysteresis_cycles: int = 3
    
    # ScoreOptimizer
    score_critical: float = 40
    score_warning: float = 60
    score_good: float = 80
    
    # WinStreakBooster
    streak_base_risk: float = 2.0
    streak_max_risk: float = 3.0
    streak_ramp_speed: float = 0.2
    streak_decay_on_loss: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created": self.created.isoformat(),
            "confidence": {
                "thresholds": self.confidence_thresholds,
                "multipliers": self.confidence_multipliers,
            },
            "pyramid": {
                "risk_mults": self.pyramid_risk_mults,
                "max_entries": self.pyramid_max_entries,
                "dd_threshold": self.pyramid_dd_threshold,
            },
            "mode": {
                "alpha_score": self.alpha_score_threshold,
                "alpha_dd": self.alpha_dd_threshold,
                "defensive_score": self.defensive_score_threshold,
                "defensive_dd": self.defensive_dd_threshold,
                "hysteresis": self.mode_hysteresis_cycles,
            },
            "score": {
                "critical": self.score_critical,
                "warning": self.score_warning,
                "good": self.score_good,
            },
            "streak": {
                "base_risk": self.streak_base_risk,
                "max_risk": self.streak_max_risk,
                "ramp_speed": self.streak_ramp_speed,
                "decay_on_loss": self.streak_decay_on_loss,
            },
        }


@dataclass
class OptimizationResult:
    """Result of optimization run."""
    config: MetaConfig
    score: float
    metrics: Dict[str, float]
    improvements: Dict[str, float]
    recommended: bool


class MetaParameterOptimizer:
    """
    Meta-Parameter Optimizer.
    
    Optimization targets (not signal logic):
    - Confidence thresholds and multipliers
    - Pyramid entry rules
    - Mode switching sensitivity
    - Score-based caps
    - Win streak behavior
    
    Process:
    1. Load recent performance data
    2. Generate candidate configs
    3. Simulate each config
    4. Score and rank
    5. Return best + improvements
    """
    
    PARAM_RANGES = {
        "alpha_score_threshold": [65, 70, 75, 80],
        "alpha_dd_threshold": [4.0, 5.0, 6.0],
        "defensive_score_threshold": [35, 40, 45],
        "defensive_dd_threshold": [7.0, 8.0, 10.0],
        "streak_base_risk": [1.5, 2.0, 2.5],
        "streak_max_risk": [2.5, 3.0, 3.5],
        "pyramid_max_entries": [2, 3],
        "pyramid_dd_threshold": [0.02, 0.03, 0.04],
    }
    
    def __init__(
        self,
        profiles_dir: str = "profiles",
        max_configs: int = 50,
        min_improvement: float = 0.05  # 5% improvement to recommend
    ):
        """
        Initialize Meta Optimizer.
        
        Args:
            profiles_dir: Directory for profile storage
            max_configs: Max configs to test
            min_improvement: Min improvement to recommend
        """
        self.profiles_dir = profiles_dir
        self.max_configs = max_configs
        self.min_improvement = min_improvement
        
        os.makedirs(profiles_dir, exist_ok=True)
        
        logger.info(
            f"MetaParameterOptimizer initialized: "
            f"max_configs={max_configs}, min_improvement={min_improvement:.0%}"
        )
    
    def optimize(
        self,
        historical_data: List[Dict],
        current_config: MetaConfig
    ) -> OptimizationResult:
        """
        Run optimization on historical data.
        
        Args:
            historical_data: List of daily data dicts
            current_config: Current active config
            
        Returns:
            OptimizationResult with best config
        """
        if not historical_data:
            logger.warning("No historical data for optimization")
            return OptimizationResult(
                config=current_config,
                score=0,
                metrics={},
                improvements={},
                recommended=False
            )
        
        # Generate candidate configs
        candidates = self._generate_candidates(current_config)
        logger.info(f"Generated {len(candidates)} candidate configs")
        
        # Score current config
        current_score = self._simulate_config(current_config, historical_data)
        logger.info(f"Current config score: {current_score:.2f}")
        
        # Score all candidates
        results = []
        for i, candidate in enumerate(candidates):
            score = self._simulate_config(candidate, historical_data)
            results.append((candidate, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Get best
        best_config, best_score = results[0]
        
        # Calculate improvements
        improvements = {}
        if current_score > 0:
            improvements["score"] = (best_score - current_score) / current_score
        
        # Check if recommended
        recommended = improvements.get("score", 0) >= self.min_improvement
        
        return OptimizationResult(
            config=best_config,
            score=best_score,
            metrics=self._calc_metrics(best_config, historical_data),
            improvements=improvements,
            recommended=recommended
        )
    
    def _generate_candidates(self, base: MetaConfig) -> List[MetaConfig]:
        """Generate candidate configurations."""
        candidates = []
        
        # Grid search over key parameters
        for alpha_score in self.PARAM_RANGES["alpha_score_threshold"]:
            for defensive_score in self.PARAM_RANGES["defensive_score_threshold"]:
                for base_risk in self.PARAM_RANGES["streak_base_risk"]:
                    config = copy.copy(base)
                    config.alpha_score_threshold = alpha_score
                    config.defensive_score_threshold = defensive_score
                    config.streak_base_risk = base_risk
                    config.version = base.version + 1
                    candidates.append(config)
        
        # Limit to max
        return candidates[:self.max_configs]
    
    def _simulate_config(
        self,
        config: MetaConfig,
        historical_data: List[Dict]
    ) -> float:
        """
        Simulate config performance on historical data.
        
        Returns composite score.
        """
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        max_dd = 0
        
        for day in historical_data:
            total_pnl += day.get("total_pnl", 0)
            total_trades += day.get("win_count", 0) + day.get("loss_count", 0)
            total_wins += day.get("win_count", 0)
            max_dd = max(max_dd, day.get("max_drawdown", 0))
        
        # Calculate metrics
        win_rate = total_wins / total_trades if total_trades > 0 else 0
        
        # Composite score (same as LiveScoreEstimator)
        score = (
            total_pnl * 0.4
            + win_rate * 100 * 0.2
            - max_dd * 0.3
        )
        
        # Apply config-based adjustments
        # More aggressive configs get bonus if profitable
        if total_pnl > 0:
            aggression_bonus = (config.streak_max_risk / 3.0) * 0.1
            score *= (1 + aggression_bonus)
        
        # Defensive configs get bonus if low DD
        if max_dd < 5:
            defense_bonus = (1 - max_dd / 10) * 0.1
            score *= (1 + defense_bonus)
        
        return score
    
    def _calc_metrics(
        self,
        config: MetaConfig,
        historical_data: List[Dict]
    ) -> Dict[str, float]:
        """Calculate detailed metrics for a config."""
        total_pnl = sum(d.get("total_pnl", 0) for d in historical_data)
        total_trades = sum(
            d.get("win_count", 0) + d.get("loss_count", 0) 
            for d in historical_data
        )
        total_wins = sum(d.get("win_count", 0) for d in historical_data)
        max_dd = max(d.get("max_drawdown", 0) for d in historical_data)
        
        return {
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "win_rate": round(total_wins / total_trades if total_trades > 0 else 0, 3),
            "max_drawdown": round(max_dd, 2),
        }
    
    def save_config(self, config: MetaConfig, active: bool = False) -> str:
        """
        Save config to file.
        
        Args:
            config: Config to save
            active: Whether this is the active config
            
        Returns:
            File path
        """
        filename = f"{config.name}_v{config.version}.json"
        filepath = os.path.join(self.profiles_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        if active:
            active_path = os.path.join(self.profiles_dir, "active.json")
            with open(active_path, 'w') as f:
                json.dump({"active": filename}, f)
        
        logger.info(f"Saved config: {filepath} (active={active})")
        return filepath
    
    def load_active_config(self) -> Optional[MetaConfig]:
        """Load the currently active config."""
        active_path = os.path.join(self.profiles_dir, "active.json")
        
        if not os.path.exists(active_path):
            return self._get_default_config()
        
        with open(active_path, 'r') as f:
            active_info = json.load(f)
        
        filename = active_info.get("active")
        if not filename:
            return self._get_default_config()
        
        filepath = os.path.join(self.profiles_dir, filename)
        if not os.path.exists(filepath):
            return self._get_default_config()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self._dict_to_config(data)
    
    def _get_default_config(self) -> MetaConfig:
        """Get default configuration."""
        return MetaConfig(
            name="default",
            version=1,
            created=datetime.now()
        )
    
    def _dict_to_config(self, data: Dict) -> MetaConfig:
        """Convert dict to MetaConfig."""
        return MetaConfig(
            name=data.get("name", "loaded"),
            version=data.get("version", 1),
            created=datetime.fromisoformat(data.get("created", datetime.now().isoformat())),
            confidence_thresholds=data.get("confidence", {}).get("thresholds", [0.8, 0.65, 0.5, 0.35]),
            confidence_multipliers=data.get("confidence", {}).get("multipliers", [1.4, 1.2, 1.0, 0.8, 0.6]),
            pyramid_risk_mults=data.get("pyramid", {}).get("risk_mults", [1.0, 0.7, 0.4]),
            pyramid_max_entries=data.get("pyramid", {}).get("max_entries", 3),
            pyramid_dd_threshold=data.get("pyramid", {}).get("dd_threshold", 0.03),
            alpha_score_threshold=data.get("mode", {}).get("alpha_score", 70),
            alpha_dd_threshold=data.get("mode", {}).get("alpha_dd", 5.0),
            defensive_score_threshold=data.get("mode", {}).get("defensive_score", 40),
            defensive_dd_threshold=data.get("mode", {}).get("defensive_dd", 8.0),
            mode_hysteresis_cycles=data.get("mode", {}).get("hysteresis", 3),
            score_critical=data.get("score", {}).get("critical", 40),
            score_warning=data.get("score", {}).get("warning", 60),
            score_good=data.get("score", {}).get("good", 80),
            streak_base_risk=data.get("streak", {}).get("base_risk", 2.0),
            streak_max_risk=data.get("streak", {}).get("max_risk", 3.0),
            streak_ramp_speed=data.get("streak", {}).get("ramp_speed", 0.2),
            streak_decay_on_loss=data.get("streak", {}).get("decay_on_loss", True),
        )


# Singleton instance
_optimizer: Optional[MetaParameterOptimizer] = None


def get_meta_optimizer() -> MetaParameterOptimizer:
    """Get or create singleton MetaParameterOptimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = MetaParameterOptimizer()
    return _optimizer
