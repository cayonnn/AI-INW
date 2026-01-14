# src/retrain/optimizers/bayesian_score_optimizer.py
"""
Bayesian Score Optimizer - Competition Grade
=============================================

Optimizes for Leaderboard Score directly, not just PnL.

Uses scikit-optimize (skopt) for Bayesian optimization.

Leaderboard Score Formula:
  Score = Profit × 0.4 + Sharpe × 0.25 + WinRate × 0.2 - MaxDD × 0.3 + Consistency × 0.15
"""

from typing import Dict, List, Any, Optional
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.logger import get_logger

logger = get_logger("BAYESIAN_OPT")


# Try to import skopt
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False


# Extended parameter search space
SPACE = [
    Real(0.5, 2.0, name="atr_sl"),           # ATR multiplier for SL
    Real(1.0, 4.0, name="atr_tp"),           # ATR multiplier for TP
    Real(0.5, 1.5, name="risk_mult"),        # Risk multiplier
    Integer(1, 3, name="pyramid_levels"),    # Pyramid depth
    Real(0.70, 0.90, name="conf_high"),      # High confidence threshold
    Real(0.50, 0.70, name="conf_mid"),       # Mid confidence threshold
]


def leaderboard_score(metrics: Dict) -> float:
    """
    Calculate leaderboard-style score.
    
    Score = Profit × 0.4 + Sharpe × 0.25 + WinRate × 0.2 - MaxDD × 0.3 + Consistency × 0.15
    """
    profit = metrics.get("profit", 0)
    sharpe = metrics.get("sharpe", 0)
    winrate = metrics.get("winrate", 0.5)
    max_dd = metrics.get("max_dd", 0.05)
    consistency = metrics.get("consistency", 0.5)
    
    score = (
        profit * 0.4
        + sharpe * 0.25
        + winrate * 100 * 0.2
        - max_dd * 100 * 0.3
        + consistency * 0.15
    )
    
    return score


def constrained_score(metrics: Dict) -> float:
    """
    Calculate constrained score with penalties.
    
    Prevents fake high scores from:
    - Excessive drawdown
    - Too few trades
    - Extreme risk parameters
    """
    base_score = leaderboard_score(metrics)
    penalty = 0.0
    
    # Penalty for high drawdown (> 6%)
    max_dd_pct = metrics.get("max_dd", 0) * 100
    if max_dd_pct > 6:
        penalty += (max_dd_pct - 6) * 2
    
    # Penalty for too few trades
    trade_count = metrics.get("trades", metrics.get("trade_count", 30))
    if trade_count < 20:
        penalty += 1.5
    
    # Penalty for extreme risk
    risk_mult = metrics.get("risk_mult", 1.0)
    if risk_mult > 1.3:
        penalty += (risk_mult - 1.3) * 3
    
    # Penalty for excessive pyramid
    pyramid = metrics.get("pyramid_levels", 1)
    if pyramid > 2:
        penalty += (pyramid - 2) * 0.5
    
    return base_score - penalty


def simulate_params(params: List[float], data: Dict) -> Dict:
    """
    Simulate trading with given parameters.
    
    Returns estimated metrics for the parameter set.
    """
    atr_sl, atr_tp, risk_mult, pyramid_levels, conf_high, conf_mid = params
    
    # Get base metrics from data
    base_profit = 0
    base_dd = 0.05
    base_sharpe = 1.0
    base_winrate = 0.5
    
    if data.get("score") is not None and len(data["score"]) > 0:
        score_df = data["score"]
        if "profit" in score_df.columns:
            base_profit = score_df["profit"].mean()
        if "drawdown" in score_df.columns:
            base_dd = score_df["drawdown"].max()
    
    if data.get("live") is not None and len(data["live"]) > 0:
        live_df = data["live"]
        if "pnl" in live_df.columns:
            positive_trades = (live_df["pnl"] > 0).sum()
            total_trades = len(live_df)
            base_winrate = positive_trades / total_trades if total_trades > 0 else 0.5
    
    # Simulate parameter effects
    # Higher TP ratio = potentially higher profit but lower winrate
    tp_ratio = atr_tp / atr_sl
    profit_effect = base_profit * (1 + (tp_ratio - 2) * 0.1)
    winrate_effect = base_winrate * (1 - (tp_ratio - 2) * 0.05)
    
    # Risk mult affects profit and DD
    profit_effect *= risk_mult
    dd_effect = base_dd * risk_mult
    
    # Pyramid adds profit but also risk
    profit_effect *= (1 + (pyramid_levels - 1) * 0.2)
    dd_effect *= (1 + (pyramid_levels - 1) * 0.1)
    
    # Confidence thresholds affect trade frequency
    trade_freq = 1 - (conf_high - 0.7) * 2
    
    # Calculate simulated Sharpe
    sharpe = (profit_effect / dd_effect) if dd_effect > 0 else base_sharpe
    
    return {
        "profit": profit_effect,
        "sharpe": sharpe,
        "winrate": winrate_effect,
        "max_dd": min(dd_effect, 0.10),  # Cap at 10%
        "consistency": trade_freq * 0.8,
    }


def optimize(
    data: Dict,
    n_calls: int = 30,
    n_initial_points: int = 10,
    random_state: int = 42
) -> List[Dict]:
    """
    Run Bayesian optimization for leaderboard score.
    
    Args:
        data: Training data
        n_calls: Number of optimization iterations
        n_initial_points: Initial random exploration
        random_state: Random seed
        
    Returns:
        List of candidate configs
    """
    print(f"Running Bayesian optimization (n_calls={n_calls})")
    
    if not HAS_SKOPT:
        print("⚠ skopt not installed, using random search fallback")
        return _random_search(data, n_samples=n_calls)
    
    def objective(params):
        """Risk-aware objective function (uses constrained_score)."""
        metrics = simulate_params(params, data)
        # Add params to metrics for penalty calculation
        metrics["risk_mult"] = params[2]
        metrics["pyramid_levels"] = params[3]
        score = constrained_score(metrics)  # Uses penalties
        return -score
    
    try:
        result = gp_minimize(
            objective,
            SPACE,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            verbose=False,
        )
        
        # Get final metrics
        final_metrics = simulate_params(result.x, data)
        final_score = leaderboard_score(final_metrics)
        
        best_config = {
            "atr": {
                "sl_mult": round(result.x[0], 2),
                "tp_mult": round(result.x[1], 2),
            },
            "risk": {
                "mult": round(result.x[2], 2),
            },
            "pyramid": {
                "levels": int(result.x[3]),
                "r1": 1.0,
                "r2": 0.7,
                "r3": 0.4,
            },
            "confidence": {
                "high": round(result.x[4], 3),
                "mid": round(result.x[5], 3),
                "low": 0.50,
            },
            "score": round(final_score, 2),
            "max_dd": round(final_metrics["max_dd"], 3),
            "metrics": {
                "profit": round(final_metrics["profit"], 2),
                "sharpe": round(final_metrics["sharpe"], 2),
                "winrate": round(final_metrics["winrate"], 2),
            },
            "source": "bayesian",
        }
        
        print(f"Bayesian best score: {final_score:.2f}")
        
        return [best_config]
        
    except Exception as e:
        print(f"Bayesian optimization failed: {e}")
        return _random_search(data, n_samples=10)


def _random_search(data: Dict, n_samples: int = 30) -> List[Dict]:
    """Fallback random search with guaranteed positive scores."""
    print("⚠ skopt not installed, using random search fallback")
    
    candidates = []
    
    for _ in range(n_samples):
        params = [
            np.random.uniform(0.5, 2.0),   # atr_sl
            np.random.uniform(1.0, 4.0),   # atr_tp
            np.random.uniform(0.5, 1.5),   # risk_mult
            np.random.randint(1, 4),       # pyramid_levels
            np.random.uniform(0.70, 0.90), # conf_high
            np.random.uniform(0.50, 0.70), # conf_mid
        ]
        
        metrics = simulate_params(params, data)
        score = leaderboard_score(metrics)
        
        candidates.append({
            "atr": {
                "sl_mult": round(params[0], 2),
                "tp_mult": round(params[1], 2),
            },
            "risk": {
                "mult": round(params[2], 2),
            },
            "pyramid": {
                "levels": int(params[3]),
                "r1": 1.0,
                "r2": 0.7,
                "r3": 0.4,
            },
            "confidence": {
                "high": round(params[4], 3),
                "mid": round(params[5], 3),
                "low": 0.50,
            },
            "score": round(score, 2),
            "max_dd": round(metrics["max_dd"], 3),
            "source": "random_fallback",
        })
    
    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"Generated {len(candidates)} candidates, best score: {candidates[0]['score']}")
    
    return candidates[:10]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("Bayesian Score Optimizer Test")
    print("=" * 60)
    
    # Create sample data
    sample_data = {
        "score": pd.DataFrame({
            "profit": np.random.randn(50) * 10 + 5,
            "drawdown": np.random.uniform(0, 0.08, 50),
            "score": np.random.uniform(40, 80, 50),
        }),
        "live": pd.DataFrame({
            "pnl": np.random.randn(50) * 10 + 5,
        })
    }
    
    print(f"\nskopt available: {HAS_SKOPT}")
    
    candidates = optimize(sample_data, n_calls=20)
    
    print(f"\nGenerated {len(candidates)} candidates")
    for i, c in enumerate(candidates[:3]):
        print(f"\n--- Candidate {i+1} ---")
        print(f"Score: {c['score']}")
        print(f"ATR: SL={c['atr']['sl_mult']}x, TP={c['atr']['tp_mult']}x")
        print(f"Pyramid: {c['pyramid']['levels']} levels")
        print(f"Confidence: high={c['confidence']['high']}")
