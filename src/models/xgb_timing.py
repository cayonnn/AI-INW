"""
AI Trading System - XGBoost Timing Model
==========================================
XGBoost model for entry timing and confirmation.
Outputs probability of good entry, NOT raw price.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class XGBTimingModel:
    """
    XGBoost model for entry timing prediction.
    
    Predicts: Good Entry (1) vs Bad Entry (0)
    Can also do 3-class: LONG_ENTRY, NO_ENTRY, SHORT_ENTRY
    """
    
    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.01, num_classes: int = 3):
        config = get_config()
        xgb_config = config.models.xgboost
        
        self.n_estimators = n_estimators or xgb_config.n_estimators
        self.max_depth = max_depth or xgb_config.max_depth
        self.learning_rate = learning_rate or xgb_config.learning_rate
        self.num_classes = num_classes
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.version = "1.0.0"
        self.class_names = ["LONG_ENTRY", "NO_ENTRY", "SHORT_ENTRY"] if num_classes == 3 else ["BAD_ENTRY", "GOOD_ENTRY"]
    
    def build_model(self) -> None:
        """Initialize XGBoost model."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multi:softprob" if self.num_classes > 2 else "binary:logistic",
            num_class=self.num_classes if self.num_classes > 2 else None,
            tree_method="hist",
            eval_metric="mlogloss" if self.num_classes > 2 else "logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        logger.info("XGBoost model initialized")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 50, feature_names: Optional[List[str]] = None) -> Dict:
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features (samples, features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            early_stopping_rounds: Early stopping patience
            feature_names: List of feature names
        
        Returns:
            Training results dict
        """
        if self.model is None:
            self.build_model()
        
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        
        # Handle class imbalance
        class_counts = np.bincount(y_train.astype(int), minlength=self.num_classes)
        total = len(y_train)
        scale_pos_weight = total / (self.num_classes * class_counts + 1)
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate
        results = {"train_acc": accuracy_score(y_train, self.model.predict(X_train))}
        
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            results["val_acc"] = accuracy_score(y_val, val_preds)
            logger.info(f"Training complete - Train Acc: {results['train_acc']:.4f}, Val Acc: {results['val_acc']:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Input features (samples, features)
        
        Returns:
            Tuple of (class_predictions, probability_distributions)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        return preds.astype(int), probs
    
    def predict_single(self, X: np.ndarray) -> Dict:
        """Predict for single sample with detailed output."""
        X_2d = X.reshape(1, -1) if X.ndim == 1 else X
        preds, probs = self.predict(X_2d)
        
        pred_class = int(preds[0])
        prob_dist = probs[0]
        
        return {
            "prediction": self.class_names[pred_class],
            "probability": float(prob_dist[pred_class]),
            "confidence": float(prob_dist.max() - np.sort(prob_dist)[-2]) if len(prob_dist) > 1 else prob_dist[pred_class],
            "probabilities": {self.class_names[i]: float(p) for i, p in enumerate(prob_dist)},
            "is_good_entry": pred_class != 1 if self.num_classes == 3 else pred_class == 1,
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top important features."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        return {self.feature_names[i]: float(importance[i]) for i in indices}
    
    def save(self, path: str) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(path))
        
        # Save metadata
        meta_path = path.with_suffix(".meta.json")
        import json
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "class_names": self.class_names,
                "num_classes": self.num_classes,
                "version": self.version,
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model."""
        self.build_model()
        self.model.load_model(path)
        
        # Load metadata
        meta_path = Path(path).with_suffix(".meta.json")
        if meta_path.exists():
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.class_names = meta.get("class_names", self.class_names)
            self.num_classes = meta.get("num_classes", self.num_classes)
            self.version = meta.get("version", "1.0.0")
        
        logger.info(f"Model loaded from {path}")
