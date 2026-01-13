"""
imitation_trainer.py
====================
AI Imitation Learning Trainer

Train models to imitate SignalEngine V2 decisions:
- XGBoost (recommended starter)
- LightGBM
- LSTM (future)

Usage:
    trainer = ImitationTrainer()
    trainer.load_dataset("data/imitation_dataset.csv")
    trainer.train()
    trainer.save_model("models/imitation_xgb.pkl")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
import json

# ML Libraries
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import xgboost as xgb
    HAS_ML = True
except ImportError:
    HAS_ML = False


# =========================
# Feature Engineering
# =========================

FEATURE_COLUMNS = [
    "ema_fast",
    "ema_slow",
    "ema_spread",
    "ema_slope",
    "atr",
    "atr_ratio",
    "atr_threshold",
    "htf_trend",
    "volatility_ok",
    "hour",
    "day_of_week",
]

TARGET_COLUMN = "action_code"  # 1=BUY, -1=SELL, 0=HOLD


@dataclass
class TrainingResult:
    """Training result summary."""
    accuracy: float
    train_accuracy: float
    test_accuracy: float
    cv_scores: List[float]
    classification_report: str
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]


# =========================
# Imitation Trainer
# =========================

class ImitationTrainer:
    """
    Train AI to imitate SignalEngine V2 decisions.
    
    Supports:
    - XGBoost (default, best for tabular)
    - LightGBM
    - Custom models
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        random_state: int = 42
    ):
        """
        Args:
            model_type: "xgboost" or "lightgbm"
            random_state: Random seed for reproducibility
        """
        if not HAS_ML:
            raise ImportError("ML libraries not installed. Run: pip install xgboost scikit-learn")
        
        self.model_type = model_type
        self.random_state = random_state
        
        self.df: Optional[pd.DataFrame] = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_dataset(self, csv_path: str, min_samples: int = 100) -> Dict:
        """
        Load and prepare dataset.
        
        Args:
            csv_path: Path to imitation_dataset.csv
            min_samples: Minimum samples required
            
        Returns:
            Dataset statistics
        """
        self.df = pd.read_csv(csv_path)
        
        if len(self.df) < min_samples:
            raise ValueError(f"Need at least {min_samples} samples, got {len(self.df)}")
        
        # Stats
        stats = {
            "total_samples": len(self.df),
            "buy_samples": len(self.df[self.df["action"] == "BUY"]),
            "sell_samples": len(self.df[self.df["action"] == "SELL"]),
            "hold_samples": len(self.df[self.df["action"] == "HOLD"]),
        }
        
        print(f"📊 Dataset loaded: {stats['total_samples']} samples")
        print(f"   BUY: {stats['buy_samples']} | SELL: {stats['sell_samples']} | HOLD: {stats['hold_samples']}")
        
        return stats
    
    def prepare_features(self, test_size: float = 0.2) -> Tuple:
        """
        Prepare features and split data.
        
        Args:
            test_size: Fraction for test set
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            raise ValueError("Load dataset first")
        
        # Extract features
        X = self.df[FEATURE_COLUMNS].copy()
        y = self.df[TARGET_COLUMN].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels (convert -1,0,1 to 0,1,2)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"✅ Features prepared: {X_scaled.shape[1]} features")
        print(f"   Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        cv_folds: int = 5
    ) -> TrainingResult:
        """
        Train the model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Max tree depth
            learning_rate: XGBoost learning rate
            cv_folds: Cross-validation folds
            
        Returns:
            TrainingResult with metrics
        """
        if self.X_train is None:
            self.prepare_features()
        
        print(f"\n🚀 Training {self.model_type}...")
        
        # Create model
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss',
                objective='multi:softmax',
                num_class=3
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Adaptive CV folds (at least 2, max cv_folds, based on sample size)
        actual_cv = min(cv_folds, len(self.X_train))
        if actual_cv < 2:
            actual_cv = 2
            print(f"   ⚠️ Too few samples for proper CV, using {actual_cv} folds")
        
        # Cross-validation (skip if not enough samples)
        if len(self.X_train) >= actual_cv:
            cv_scores = cross_val_score(
                self.model, self.X_train, self.y_train,
                cv=actual_cv, scoring='accuracy'
            )
            print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        else:
            cv_scores = np.array([0.0])
            print(f"   ⚠️ Skipping CV (not enough samples)")
        
        # Train on full training set
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        print(f"   Train Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy: {test_acc:.3f}")
        
        # Classification report
        target_names = ["SELL", "HOLD", "BUY"]  # Based on label encoding order
        report = classification_report(self.y_test, test_pred, target_names=target_names)
        cm = confusion_matrix(self.y_test, test_pred)
        
        print(f"\n📋 Classification Report:\n{report}")
        
        # Feature importance
        feature_imp = dict(zip(
            FEATURE_COLUMNS,
            self.model.feature_importances_
        ))
        feature_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
        
        print("\n🔥 Top Features:")
        for feat, imp in list(feature_imp.items())[:5]:
            print(f"   {feat}: {imp:.3f}")
        
        return TrainingResult(
            accuracy=test_acc,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            cv_scores=cv_scores.tolist(),
            classification_report=report,
            confusion_matrix=cm,
            feature_importance=feature_imp
        )
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """
        Predict action for given features.
        
        Args:
            features: Dict with feature values
            
        Returns:
            (action: str, confidence: float)
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        # Prepare features
        X = pd.DataFrame([features])[FEATURE_COLUMNS].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        # Decode
        action_code = self.label_encoder.inverse_transform([pred])[0]
        confidence = float(proba.max())
        
        # Convert to action string
        action_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        action = action_map.get(action_code, "HOLD")
        
        return action, confidence
    
    def save_model(self, path: str = "models/imitation_xgb.pkl"):
        """Save trained model."""
        if self.model is None:
            raise ValueError("Train model first")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_columns": FEATURE_COLUMNS,
            "model_type": self.model_type,
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved: {path}")
    
    def load_model(self, path: str = "models/imitation_xgb.pkl"):
        """Load trained model."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.model_type = model_data["model_type"]
        
        print(f"✅ Model loaded: {path}")


# =========================
# Quick Train Function
# =========================

def train_imitation_model(
    dataset_path: str = "data/imitation_dataset.csv",
    model_path: str = "models/imitation_xgb.pkl",
    min_samples: int = 3  # Very low for testing, recommend 100+ for production
) -> TrainingResult:
    """
    Quick function to train imitation model.
    
    Args:
        dataset_path: Path to dataset CSV
        model_path: Path to save model
        min_samples: Minimum samples required (default 10 for testing)
        
    Returns:
        TrainingResult
    """
    trainer = ImitationTrainer()
    trainer.load_dataset(dataset_path, min_samples=min_samples)
    
    if len(trainer.df) < 100:
        print(f"⚠️ Warning: Only {len(trainer.df)} samples. Recommend 100+ for reliable AI.")
    
    trainer.prepare_features()
    result = trainer.train()
    trainer.save_model(model_path)
    
    return result


if __name__ == "__main__":
    # Example usage
    result = train_imitation_model()
    print(f"\n🏆 Final Test Accuracy: {result.test_accuracy:.1%}")
