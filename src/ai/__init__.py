"""
AI Module for Fund-Grade Trading System
========================================

Mental Model:
- Rule-Based (Teacher) - provides training signals
- AI Model (Student) - learns to imitate teacher
- Risk Guard (Principal) - supervises all decisions

Components:
- dataset_recorder: Records decisions for AI training
- imitation_trainer: Trains XGBoost/LightGBM models
- ai_signal: AI prediction and hybrid modes
"""

from .dataset_recorder import (
    ImitationDatasetRecorder,
    get_dataset_recorder,
    DatasetRecord,
)

from .imitation_trainer import (
    ImitationTrainer,
    train_imitation_model,
    TrainingResult,
)

from .ai_signal import (
    AISignal,
    AISignalResult,
    HybridSignal,
    get_ai_signal,
    get_hybrid_signal,
)

__all__ = [
    # Dataset
    "ImitationDatasetRecorder",
    "get_dataset_recorder",
    "DatasetRecord",
    
    # Training
    "ImitationTrainer", 
    "train_imitation_model",
    "TrainingResult",
    
    # AI Signal
    "AISignal",
    "AISignalResult",
    "HybridSignal",
    "get_ai_signal",
    "get_hybrid_signal",
]
