# src/train/xgb_train.py
"""
XGBoost Training Pipeline
=========================
Train XGBoost from Imitation Dataset.

Usage:
    python src/train/xgb_train.py
    python src/train/xgb_train.py --path data/custom.csv
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb


# =========================
# CONFIG
# =========================

DEFAULT_DATASET = "data/imitation_full_dataset.csv"
MODEL_OUTPUT = "models/xgb_imitation.json"

FEATURE_COLS = [
    "ema20", "ema50", "ema_spread",
    "atr14", "rsi14",
    "hour", "day_of_week"
]


# =========================
# TRAINING PIPELINE
# =========================

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"📊 Loaded {len(df)} samples from {filepath}")
    return df


def prepare_data(df: pd.DataFrame):
    """Prepare features and labels."""
    # Find available features
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"   Features: {available}")
    
    X = df[available].fillna(0)
    
    # Get labels
    if "label" in df.columns:
        y = df["label"]
    elif "signal" in df.columns:
        label_map = {"SELL": 0, "HOLD": 1, "BUY": 2}
        y = df["signal"].map(label_map)
    else:
        raise ValueError("No label column found")
    
    # Distribution
    print(f"\n📈 Class Distribution:")
    print(f"   SELL (0): {(y == 0).sum()}")
    print(f"   HOLD (1): {(y == 1).sum()}")
    print(f"   BUY  (2): {(y == 2).sum()}")
    
    return X, y, available


def train_xgboost(X_train, y_train, n_estimators=200, max_depth=4, lr=0.1):
    """Train XGBoost classifier."""
    print(f"\n🚀 Training XGBoost...")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth}")
    print(f"   learning_rate: {lr}")
    
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
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


def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model performance."""
    print("\n📈 Evaluating...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.1%}")
    
    # Classification report
    try:
        print("\n📋 Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=["SELL", "HOLD", "BUY"],
            zero_division=0
        ))
    except:
        pass
    
    # Confusion matrix
    try:
        print("📉 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    except:
        pass
    
    # Feature importance
    print("\n🔥 Feature Importance:")
    importance = sorted(zip(feature_names, model.feature_importances_),
                       key=lambda x: x[1], reverse=True)
    for name, imp in importance[:5]:
        print(f"   {name}: {imp:.3f}")
    
    return accuracy


def save_model(model, filepath: str):
    """Save model to file."""
    import joblib
    
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    # Use joblib for compatibility
    joblib.dump(model, filepath.replace(".json", ".pkl"))
    print(f"\n✅ Model saved: {filepath.replace('.json', '.pkl')}")


def main(dataset_path: str = DEFAULT_DATASET, output_path: str = MODEL_OUTPUT):
    """Full training pipeline."""
    print("=" * 50)
    print("🤖 XGBoost Imitation Learning")
    print("=" * 50)
    
    # Load data
    df = load_data(dataset_path)
    
    # Prepare
    X, y, features = prepare_data(df)
    
    # Split
    print(f"\n📊 Train/Test Split (80/20)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train
    model = train_xgboost(X_train, y_train)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test, features)
    
    # Save
    save_model(model, output_path)
    
    # Summary
    print("\n" + "=" * 50)
    if accuracy > 0.7:
        print("✅ GOOD: Model ready for inference")
    elif accuracy > 0.5:
        print("📊 FAIR: Needs more data")
    else:
        print("❌ POOR: Check data quality")
    print("=" * 50)
    
    return model


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Imitation Model")
    parser.add_argument("--path", type=str, default=DEFAULT_DATASET,
                       help="Path to dataset CSV")
    parser.add_argument("--output", type=str, default=MODEL_OUTPUT,
                       help="Output model path")
    parser.add_argument("--estimators", type=int, default=200,
                       help="Number of trees")
    parser.add_argument("--depth", type=int, default=4,
                       help="Max tree depth")
    args = parser.parse_args()
    
    main(args.path, args.output)
