# src/retrain/__init__.py
"""
Auto Retrain Module - Competition Grade
=======================================

Daily retraining of meta-parameters after market close.

Components:
- DataCollector: Collects trading data
- MetaParameterOptimizer: Optimizes meta-parameters
- DailyRetrainJob: Orchestrates daily retraining

Usage:
    from src.retrain import run_daily_retrain
    result = run_daily_retrain(force=True)
"""

from src.retrain.data_collector import (
    RetrainDataCollector,
    get_data_collector,
    DailyData,
    TradeRecord
)

from src.retrain.meta_optimizer import (
    MetaParameterOptimizer,
    MetaConfig,
    OptimizationResult,
    get_meta_optimizer
)

from src.retrain.daily_retrain_job import (
    DailyRetrainJob,
    RetrainResult,
    get_daily_retrain_job,
    run_daily_retrain
)


__all__ = [
    # Data Collector
    "RetrainDataCollector",
    "get_data_collector",
    "DailyData",
    "TradeRecord",
    
    # Meta Optimizer
    "MetaParameterOptimizer",
    "MetaConfig",
    "OptimizationResult",
    "get_meta_optimizer",
    
    # Daily Job
    "DailyRetrainJob",
    "RetrainResult",
    "get_daily_retrain_job",
    "run_daily_retrain",
]
