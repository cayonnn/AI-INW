# src/train/xgb_sl_tp_train.py
"""
XGBoost SL/TP Training Pipeline (Regression)
=============================================

Train XGBoost regressors for dynamic SL/TP prediction:
- SL Model: Predicts MAE (Max Adverse Excursion)
- TP Model: Predicts MFE (Max Favorable Excursion)

Usage:
    python src/train/xgb_sl_tp_train.py
    python src/train/xgb_sl_tp_train.py --path data/custom.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib


# =========================
# CONFIG
# =========================

DEFAULT_DATASET = "data/imitation_full_dataset.csv"
SL_MODEL_OUTPUT = "models/xgb_sl.pkl"
TP_MODEL_OUTPUT = "models/xgb_tp.pkl"

# Features for SL/TP prediction
FEATURE_COLS = [
    "ema_spread",     # EMA spread (trend strength)
    "atr14",          # ATR (volatility)
    "rsi14"           # RSI (momentum)
]

# Extended features (if available)
EXTENDED_FEATURES = [
    "ema20", "ema50", "hour", "day_of_week",
    "close", "high", "low"
]


# =========================
# DATA LOADING
# =========================

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"📊 Loaded {len(df)} samples from {filepath}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features for training.
    
    Returns:
        (X, feature_names)
    """
    # Find available features
    available = []
    
    # Primary features
    for col in FEATURE_COLS:
        if col in df.columns:
            available.append(col)
    
    # Extended features
    for col in EXTENDED_FEATURES:
        if col in df.columns and col not in available:
            available.append(col)
    
    if len(available) == 0:
        raise ValueError("No valid features found in dataset")
    
    print(f"   Features: {available}")
    
    X = df[available].fillna(0)
    
    return X, available


def prepare_targets(df: pd.DataFrame, atr_fallback: float = None) -> tuple:
    """
    Prepare target variables for SL/TP.
    
    Returns:
        (y_sl, y_tp)
    """
    # Check for MAE column
    if "mae" in df.columns:
        y_sl = df["mae"].abs()
    elif "sl_distance" in df.columns:
        y_sl = df["sl_distance"].abs()
    elif "atr14" in df.columns:
        # Fallback: use ATR * 1.5 as SL
        y_sl = df["atr14"].abs() * 1.5
        print("⚠️  Using ATR × 1.5 as SL target (no MAE column)")
    else:
        raise ValueError("No SL target column found (mae, sl_distance, atr14)")
    
    # Check for MFE column
    if "mfe" in df.columns:
        y_tp = df["mfe"].abs()
    elif "tp_distance" in df.columns:
        y_tp = df["tp_distance"].abs()
    elif "atr14" in df.columns:
        # Fallback: use ATR * 3.0 as TP
        y_tp = df["atr14"].abs() * 3.0
        print("⚠️  Using ATR × 3.0 as TP target (no MFE column)")
    else:
        raise ValueError("No TP target column found (mfe, tp_distance, atr14)")
    
    # Basic statistics
    print(f"\n📈 Target Statistics:")
    print(f"   SL: mean={y_sl.mean():.5f}, std={y_sl.std():.5f}")
    print(f"   TP: mean={y_tp.mean():.5f}, std={y_tp.std():.5f}")
    
    return y_sl, y_tp


# =========================
# TRAINING
# =========================

def train_regressor(X_train, y_train, n_estimators=200, max_depth=4, lr=0.1):
    """Train XGBoost regressor."""
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_regressor(model, X_test, y_test, name: str, feature_names):
    """Evaluate regression model."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 {name} Model Performance:")
    print(f"   MAE:  {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   R²:   {r2:.3f}")
    
    # Feature importance
    print(f"\n🔥 {name} Feature Importance:")
    importance = sorted(zip(feature_names, model.feature_importances_),
                       key=lambda x: x[1], reverse=True)
    for feat, imp in importance[:5]:
        print(f"   {feat}: {imp:.3f}")
    
    return mae, rmse, r2


def save_model(model, filepath: str):
    """Save model to file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Saved: {filepath}")


# =========================
# MAIN PIPELINE
# =========================

def main(dataset_path: str = DEFAULT_DATASET):
    """Full training pipeline for SL/TP models."""
    print("=" * 60)
    print("🎯 XGBoost SL/TP Regression Training (Fund-Grade)")
    print("=" * 60)
    
    # Load data
    df = load_data(dataset_path)
    
    # Prepare features
    X, features = prepare_features(df)
    
    # Prepare targets
    y_sl, y_tp = prepare_targets(df)
    
    # Split data
    print(f"\n📊 Train/Test Split (80/20)...")
    X_train, X_test, y_sl_train, y_sl_test, y_tp_train, y_tp_test = train_test_split(
        X, y_sl, y_tp, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train SL Model
    print("\n🚀 Training SL Model (MAE Target)...")
    sl_model = train_regressor(X_train, y_sl_train)
    sl_mae, sl_rmse, sl_r2 = evaluate_regressor(
        sl_model, X_test, y_sl_test, "SL", features
    )
    
    # Train TP Model
    print("\n🚀 Training TP Model (MFE Target)...")
    tp_model = train_regressor(X_train, y_tp_train)
    tp_mae, tp_rmse, tp_r2 = evaluate_regressor(
        tp_model, X_test, y_tp_test, "TP", features
    )
    
    # Save models
    print("\n💾 Saving Models...")
    save_model(sl_model, SL_MODEL_OUTPUT)
    save_model(tp_model, TP_MODEL_OUTPUT)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TRAINING SUMMARY")
    print("=" * 60)
    print(f"   SL Model: MAE={sl_mae:.6f}, R²={sl_r2:.3f}")
    print(f"   TP Model: MAE={tp_mae:.6f}, R²={tp_r2:.3f}")
    
    if sl_r2 > 0.3 and tp_r2 > 0.3:
        print("\n✅ GOOD: Models ready for production")
    elif sl_r2 > 0.1 and tp_r2 > 0.1:
        print("\n📊 FAIR: Models usable, more data recommended")
    else:
        print("\n⚠️  POOR: Consider more data or feature engineering")
    
    print("=" * 60)
    
    return sl_model, tp_model


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost SL/TP Models")
    parser.add_argument("--path", type=str, default=DEFAULT_DATASET,
                       help="Path to dataset CSV")
    parser.add_argument("--estimators", type=int, default=200,
                       help="Number of trees")
    parser.add_argument("--depth", type=int, default=4,
                       help="Max tree depth")
    args = parser.parse_args()
    
    main(args.path)
