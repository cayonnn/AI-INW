# src/registry/__init__.py
"""
Model Registry Package - Fund-Grade
=====================================

Versioned model management with:
- Model Registry
- Promotion Gate  
- Hot Reload
- Auto Rollback
"""

from .model_registry import ModelRegistry
from .promotion_gate import PromotionGate, allow_promotion
from .daily_retrain import DailyRetrainPipeline, run_daily_retrain
from .hot_reload import HotReloader, load_latest_model, get_model_version

__all__ = [
    "ModelRegistry",
    "PromotionGate",
    "allow_promotion",
    "DailyRetrainPipeline",
    "run_daily_retrain",
    "HotReloader",
    "load_latest_model",
    "get_model_version",
]
