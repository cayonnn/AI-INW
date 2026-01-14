# src/retrain/bayesian_space.py
"""
Regime-aware Bayesian Search Space
===================================

Different parameter ranges for each market regime:
- TREND: Aggressive, maximize alpha
- CHOP: Conservative, survive first
- HIGH_VOL: Balanced, stay in game
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("BAYESIAN_SPACE")


# Regime-specific parameter ranges
REGIME_SPACES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "TREND": {
        "risk": (1.5, 3.5),
        "pyramid_levels": (2, 4),
        "trail_start_r": (1.5, 3.0),
        "conf_high": (0.70, 0.85),
        "atr_tp_mult": (2.0, 4.0),
    },
    "CHOP": {
        "risk": (0.5, 1.5),
        "pyramid_levels": (0, 1),
        "trail_start_r": (0.8, 1.5),
        "conf_high": (0.80, 0.90),
        "atr_tp_mult": (1.0, 2.0),
    },
    "HIGH_VOL": {
        "risk": (0.8, 2.0),
        "pyramid_levels": (1, 2),
        "trail_start_r": (1.2, 2.0),
        "conf_high": (0.75, 0.88),
        "atr_tp_mult": (1.5, 3.0),
    },
}


# Regime-specific penalty weights
REGIME_PENALTIES: Dict[str, Dict[str, float]] = {
    "TREND": {
        "dd_weight": 1.0,       # Normal DD penalty
        "vol_weight": 0.5,     # Low vol penalty
        "trade_weight": 0.3,   # Want more trades
    },
    "CHOP": {
        "dd_weight": 1.5,       # Higher DD penalty
        "vol_weight": 1.0,     # Normal vol penalty
        "trade_weight": 0.1,   # OK with fewer trades
    },
    "HIGH_VOL": {
        "dd_weight": 2.0,       # Highest DD penalty
        "vol_weight": 2.0,     # High vol penalty
        "trade_weight": 0.2,   # Careful with trades
    },
}


def get_space_for_regime(regime: str) -> Dict[str, Tuple[float, float]]:
    """Get parameter space for a regime."""
    return REGIME_SPACES.get(regime, REGIME_SPACES["CHOP"])


def get_penalty_weights(regime: str) -> Dict[str, float]:
    """Get penalty weights for a regime."""
    return REGIME_PENALTIES.get(regime, REGIME_PENALTIES["CHOP"])


def regime_objective(metrics: Dict, regime: str) -> float:
    """
    Regime-aware objective function.
    
    Score - Regime-specific penalties.
    """
    score = metrics.get("score", 0)
    weights = get_penalty_weights(regime)
    
    penalty = 0.0
    
    # DD penalty (scaled by regime)
    max_dd = metrics.get("max_dd", 0)
    if max_dd > 0.04:
        penalty += (max_dd - 0.04) * 100 * weights["dd_weight"]
    
    # Volatility penalty
    volatility = metrics.get("volatility", 0)
    penalty += volatility * weights["vol_weight"]
    
    # Trade count penalty
    trades = metrics.get("trades", 30)
    if trades < 20:
        penalty += (20 - trades) * weights["trade_weight"]
    
    return score - penalty


def build_skopt_space(regime: str):
    """Build scikit-optimize space for a regime."""
    try:
        from skopt.space import Real, Integer
    except ImportError:
        return None
    
    space_config = get_space_for_regime(regime)
    
    space = [
        Real(space_config["risk"][0], space_config["risk"][1], name="risk"),
        Integer(int(space_config["pyramid_levels"][0]), int(space_config["pyramid_levels"][1]), name="pyramid"),
        Real(space_config["trail_start_r"][0], space_config["trail_start_r"][1], name="trail"),
        Real(space_config["conf_high"][0], space_config["conf_high"][1], name="conf_high"),
        Real(space_config["atr_tp_mult"][0], space_config["atr_tp_mult"][1], name="atr_tp"),
    ]
    
    return space


# Default balanced space
DEFAULT_SPACE = {
    "risk": (1.0, 2.5),
    "pyramid_levels": (1, 3),
    "trail_start_r": (1.0, 2.5),
    "conf_high": (0.75, 0.88),
    "atr_tp_mult": (1.5, 3.0),
}
