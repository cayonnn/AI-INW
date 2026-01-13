"""
train_ai.py
===========
Train AI Model (XGBoost / LightGBM) from prepared dataset

Features:
- Load dataset from various sources
- Train XGBoost multi-class classifier
- Evaluate with metrics (Accuracy, Classification Report, Confusion Matrix)
- Save model for inference

Usage:
    python src/ai/train_ai.py
    python src/ai/train_ai.py --source mt5 --bars 1000
    python src/ai/train_ai.py --source backtest
    python src/ai/train_ai.py --source csv --path data/dataset.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =========================
# CONFIG
# =========================

DEFAULT_DATASET = "data/XAUUSD_dataset.csv"
MODEL_OUTPUT = "models/xgb_signal_model.pkl"

FEATURE_COLS = [
    # Trend
    "ema20", "ema50", "ema_spread", "ema_spread_pct", "ema20_slope",
    # Volatility
    "atr14", "atr_ratio", "spread", "spread_ratio",
    # Momentum
    "rsi14", "macd", "macd_signal", "macd_hist",
    # Price
    "return_1", "return_5", "return_10", "bb_position",
    # Time
    "hour", "day_of_week",
]

LABEL_MAP = {"BUY": 2, "HOLD": 1, "SELL": 0}
REVERSE_LABEL = {v: k for k, v in LABEL_MAP.items()}


# =========================
# DATA LOADING
# =========================

def load_dataset(source: str = "csv", path: str = None, **kwargs) -> pd.DataFrame:
    """
    Load dataset from various sources.
    
    Args:
        source: "csv", "mt5", "backtest", "live"
        path: Path for csv source
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with features and labels
    """
    print(f"📥 Loading dataset from: {source}")
    
    if source == "csv":
        path = path or DEFAULT_DATASET
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        print(f"   Loaded {len(df)} rows from {path}")
        
    elif source == "mt5":
        from src.ai.dataset_preparation import prepare_dataset_from_mt5
        bars = kwargs.get("bars", 1000)
        symbol = kwargs.get("symbol", "XAUUSD")
        prep = prepare_dataset_from_mt5(symbol=symbol, bars=bars)
        df = prep.df
        
    elif source == "backtest":
        from src.ai.dataset_preparation import prepare_dataset_from_backtest
        db_path = kwargs.get("db_path", "data/backtest.db")
        df = prepare_dataset_from_backtest(db_path=db_path)
        
    elif source == "live":
        from src.ai.dataset_preparation import prepare_dataset_from_live_log
        csv_path = kwargs.get("csv_path", "data/imitation_dataset.csv")
        df = prepare_dataset_from_live_log(csv_path=csv_path)
        
    else:
        raise ValueError(f"Unknown source: {source}")
    
    return df


def prepare_features_labels(df: pd.DataFrame) -> tuple:
    """
    Extract features and labels from DataFrame.
    
    Returns:
        (X, y, available_features)
    """
    # Find available feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"   Available features: {len(available)}/{len(FEATURE_COLS)}")
    
    if len(available) == 0:
        # Fallback to basic columns
        basic_cols = ["ema_fast", "ema_slow", "atr", "atr_ratio", "hour", "day_of_week"]
        available = [c for c in basic_cols if c in df.columns]
        
        # Backtest-specific fallback
        if len(available) == 0:
            backtest_cols = ["entry_price", "exit_price", "mae", "mfe", "pnl", "duration",
                            "mae_ratio", "mfe_ratio", "pnl_ratio"]
            available = [c for c in backtest_cols if c in df.columns]
        
        print(f"   Using fallback features: {available}")
    
    if len(available) == 0:
        raise ValueError("No valid features found in dataset")
    
    X = df[available].fillna(0)
    
    # Handle different label column names
    if "signal_code" in df.columns:
        y = df["signal_code"]
    elif "signal" in df.columns:
        y = df["signal"].map(LABEL_MAP)
    elif "action_code" in df.columns:
        y = df["action_code"]
    elif "action" in df.columns:
        y = df["action"].map(LABEL_MAP)
    else:
        raise ValueError("No label column found (signal, signal_code, action, action_code)")
    
    # Show class distribution
    print(f"\n📊 Class Distribution:")
    for code, name in REVERSE_LABEL.items():
        count = (y == code).sum()
        print(f"   {name}: {count} ({count/len(y):.1%})")
    
    return X, y, available


# =========================
# TRAINING
# =========================

def train_xgboost(
    X_train, y_train, X_test, y_test,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier.
    """
    print(f"\n🚀 Training XGBoost...")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth}")
    print(f"   learning_rate: {learning_rate}")
    
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test, feature_names: list):
    """
    Evaluate trained model.
    """
    print("\n📈 Evaluating Model...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.1%}")
    
    # Classification report
    try:
        report = classification_report(
            y_test, y_pred,
            target_names=["SELL", "HOLD", "BUY"],
            zero_division=0
        )
        print(f"\n📋 Classification Report:\n{report}")
    except:
        print("   (Classification report skipped)")
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        print(f"📉 Confusion Matrix:")
        print(cm)
    except:
        pass
    
    # Feature importance
    print("\n🔥 Feature Importance:")
    importance = sorted(zip(feature_names, model.feature_importances_), 
                       key=lambda x: x[1], reverse=True)
    for name, imp in importance[:5]:
        print(f"   {name}: {imp:.3f}")
    
    return accuracy


def save_model(model, path: str = MODEL_OUTPUT, feature_names: list = None):
    """
    Save trained model.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    model_data = {
        "model": model,
        "feature_names": feature_names,
        "label_map": LABEL_MAP,
        "timestamp": datetime.now().isoformat()
    }
    
    joblib.dump(model_data, path)
    print(f"\n✅ Model saved: {path}")


# =========================
# MAIN TRAINING PIPELINE
# =========================

def train_pipeline(
    source: str = "csv",
    path: str = None,
    test_size: float = 0.2,
    **kwargs
):
    """
    Full training pipeline.
    """
    print("=" * 50)
    print("🤖 AI Training Pipeline")
    print("=" * 50)
    
    # 1. Load data
    df = load_dataset(source, path, **kwargs)
    
    if df.empty or len(df) < 5:
        print(f"❌ Not enough data: {len(df)} samples (need at least 5)")
        return None
    
    if len(df) < 50:
        print(f"⚠️ Warning: Only {len(df)} samples. Recommend 50+ for reliable model.")
    
    # 2. Prepare features
    X, y, feature_names = prepare_features_labels(df)
    
    # 3. Train/test split
    print(f"\n📊 Splitting data ({1-test_size:.0%} train, {test_size:.0%} test)...")
    
    # Handle stratification with small datasets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 4. Train model (extract only training kwargs)
    train_kwargs = {
        "n_estimators": kwargs.get("n_estimators", 200),
        "max_depth": kwargs.get("max_depth", 4),
        "learning_rate": kwargs.get("learning_rate", 0.1),
    }
    model = train_xgboost(X_train, y_train, X_test, y_test, **train_kwargs)
    
    # 5. Evaluate
    accuracy = evaluate_model(model, X_test, y_test, feature_names)
    
    # 6. Save model
    save_model(model, feature_names=feature_names)
    
    # 7. Quality check
    print("\n" + "=" * 50)
    if accuracy > 0.95:
        print("⚠️ CAUTION: Accuracy > 95% may indicate overfitting!")
    elif accuracy > 0.70:
        print("✅ GOOD: Model accuracy is in healthy range")
    elif accuracy > 0.50:
        print("📊 FAIR: Model needs more training data")
    else:
        print("❌ POOR: Model performance is below random chance")
    print("=" * 50)
    
    return model


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Signal Model")
    parser.add_argument("--source", type=str, default="csv",
                       choices=["csv", "mt5", "backtest", "live"],
                       help="Data source")
    parser.add_argument("--path", type=str, default=None,
                       help="Path to CSV file (for csv source)")
    parser.add_argument("--bars", type=int, default=1000,
                       help="Number of bars (for mt5 source)")
    parser.add_argument("--symbol", type=str, default="XAUUSD",
                       help="Symbol (for mt5 source)")
    parser.add_argument("--n_estimators", type=int, default=200,
                       help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=4,
                       help="Max tree depth")
    args = parser.parse_args()
    
    train_pipeline(
        source=args.source,
        path=args.path,
        bars=args.bars,
        symbol=args.symbol,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
