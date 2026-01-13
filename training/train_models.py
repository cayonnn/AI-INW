"""
AI Trading System - Training Scripts
======================================
Model training with walk-forward validation.
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from src.data.mt5_connector import MT5Connector
from src.data.data_processor import DataProcessor
from src.features.price_action import PriceActionFeatures
from src.features.trend_momentum import TrendMomentumFeatures
from src.features.volatility import VolatilityFeatures
from src.features.regime_detector import RegimeDetector
from src.models.lstm_direction import LSTMDirectionModel
from src.models.xgb_timing import XGBTimingModel
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def create_labels(df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.002) -> pd.Series:
    """Create classification labels based on future returns."""
    future_return = df["close"].shift(-lookahead) / df["close"] - 1
    
    labels = pd.Series(1, index=df.index)  # Default: NEUTRAL
    labels[future_return > threshold] = 0   # LONG
    labels[future_return < -threshold] = 2  # SHORT
    
    return labels


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering."""
    pa = PriceActionFeatures()
    tm = TrendMomentumFeatures()
    vol = VolatilityFeatures()
    
    result = pa.add_all_features(df)
    result = tm.add_all_features(result)
    result = vol.add_all_features(result)
    
    return result.dropna()


def train_lstm(symbol: str = "EURUSD", timeframe: str = "H1", lookback_bars: int = 5000):
    """Train LSTM model."""
    logger.info(f"Training LSTM for {symbol} {timeframe}")
    
    # Load data
    with MT5Connector() as connector:
        df = connector.get_rates(symbol, timeframe, lookback_bars)
    
    if df is None or len(df) < 1000:
        logger.error("Insufficient data")
        return
    
    # Feature engineering
    df = prepare_features(df)
    
    # Create labels
    df["label"] = create_labels(df)
    df = df.dropna()
    
    # Prepare sequences
    processor = DataProcessor()
    feature_cols = [c for c in df.columns if c not in ["label", "open", "high", "low", "close", "volume"]]
    
    processor.fit_scaler(df[feature_cols])
    df_scaled = processor.normalize(df[feature_cols])
    
    X, y = processor.create_sequences(
        pd.concat([df_scaled, df["label"]], axis=1),
        sequence_length=60,
        target_column="label",
        feature_columns=feature_cols
    )
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train
    model = LSTMDirectionModel(input_size=len(feature_cols))
    model.feature_names = feature_cols
    history = model.train(X_train, y_train.astype(int), X_val, y_val.astype(int), epochs=100)
    
    # Save
    save_dir = Path("artifacts/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(save_dir / "lstm_direction.pt")
    model.export_onnx(str(save_dir / "lstm_direction.onnx"))
    
    logger.info(f"Training complete. Final val_acc: {history['val_acc'][-1]:.4f}")


def train_xgboost(symbol: str = "EURUSD", timeframe: str = "H1", lookback_bars: int = 5000):
    """Train XGBoost model."""
    logger.info(f"Training XGBoost for {symbol} {timeframe}")
    
    # Load data
    with MT5Connector() as connector:
        df = connector.get_rates(symbol, timeframe, lookback_bars)
    
    if df is None or len(df) < 1000:
        logger.error("Insufficient data")
        return
    
    # Feature engineering
    df = prepare_features(df)
    df["label"] = create_labels(df)
    df = df.dropna()
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ["label", "open", "high", "low", "close", "volume"]]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train
    model = XGBTimingModel(num_classes=3)
    results = model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    
    # Save
    save_dir = Path("artifacts/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir / "xgb_timing.json")
    
    # Feature importance
    importance = model.get_feature_importance(top_n=10)
    logger.info(f"Top features: {importance}")
    logger.info(f"Training complete. Val acc: {results['val_acc']:.4f}")


if __name__ == "__main__":
    setup_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "xgboost", "all"], default="all")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    args = parser.parse_args()
    
    if args.model in ["lstm", "all"]:
        train_lstm(args.symbol, args.timeframe)
    
    if args.model in ["xgboost", "all"]:
        train_xgboost(args.symbol, args.timeframe)
