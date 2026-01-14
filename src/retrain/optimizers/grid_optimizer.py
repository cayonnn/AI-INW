# src/retrain/optimizers/grid_optimizer.py
"""
Grid Optimizer - Fallback Optimizer
===================================

Simple grid search optimizer as fallback when Bayesian not available.
"""

from typing import Dict, List, Any
import itertools

from src.utils.logger import get_logger

logger = get_logger("GRID_OPT")


# Grid values
GRID_VALUES = {
    "conf_high": [0.75, 0.80, 0.85],
    "conf_mid": [0.55, 0.60, 0.65],
    "pyramid_r2": [0.5, 0.6, 0.7],
    "streak_base": [1.5, 2.0, 2.5],
}


def optimize_grid(
    data: Dict,
    max_configs: int = 50
) -> List[Dict]:
    """
    Run grid search optimization.
    
    Args:
        data: Training data
        max_configs: Maximum configs to generate
        
    Returns:
        List of candidate configs
    """
    logger.info("Running grid search optimization")
    
    # Generate all combinations
    keys = list(GRID_VALUES.keys())
    values = [GRID_VALUES[k] for k in keys]
    
    combinations = list(itertools.product(*values))
    
    if len(combinations) > max_configs:
        # Sample subset
        import random
        combinations = random.sample(combinations, max_configs)
    
    candidates = []
    
    for combo in combinations:
        conf_high, conf_mid, pyramid_r2, streak_base = combo
        
        # Calculate simple score estimate
        score = _estimate_score(data, conf_high, pyramid_r2, streak_base)
        
        config = {
            "confidence": {
                "high": conf_high,
                "mid": conf_mid,
                "low": 0.50,
            },
            "pyramid": {
                "r1": 1.0,
                "r2": pyramid_r2,
                "r3": 0.4,
            },
            "streak": {
                "base": streak_base,
                "max": streak_base + 1.0,
            },
            "score": round(score, 2),
            "max_dd": 0.05,
        }
        
        candidates.append(config)
    
    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(f"Generated {len(candidates)} configs, best score: {candidates[0]['score']:.2f}")
    
    return candidates


def _estimate_score(
    data: Dict,
    conf_high: float,
    pyramid_r2: float,
    streak_base: float
) -> float:
    """Estimate score for a configuration."""
    base_score = 50
    
    # Get data-based score if available
    if data.get("score") is not None and len(data["score"]) > 0:
        score_df = data["score"]
        if "score" in score_df.columns:
            base_score = score_df["score"].mean()
    
    # Apply parameter effects
    conf_bonus = (conf_high - 0.75) * 10  # Higher confidence = slight bonus
    pyramid_bonus = (0.7 - pyramid_r2) * 5  # Lower pyramid mult = safer
    risk_penalty = (streak_base - 2.0) * 3  # Higher base risk = penalty
    
    return base_score + conf_bonus + pyramid_bonus - abs(risk_penalty)
