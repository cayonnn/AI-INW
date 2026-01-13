"""
AI Trading System - LSTM Direction Model
==========================================
LSTM model for predicting directional bias (LONG/SHORT/NEUTRAL).
Training and inference are separated. Outputs probability, NOT raw price.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM neural network architecture."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        logits = self.fc(context)
        probs = torch.softmax(logits, dim=1)
        return logits, probs


class LSTMDirectionModel:
    """
    LSTM model for directional prediction.
    
    Predicts: LONG (0), NEUTRAL (1), SHORT (2)
    Outputs: probability distribution, NOT raw prices
    """
    
    CLASS_NAMES = ["LONG", "NEUTRAL", "SHORT"]
    
    def __init__(self, input_size: int = 50, sequence_length: int = 60,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        config = get_config()
        lstm_config = config.models.lstm
        
        self.input_size = input_size
        self.sequence_length = sequence_length or lstm_config.sequence_length
        self.hidden_size = hidden_size or lstm_config.hidden_size
        self.num_layers = num_layers or lstm_config.num_layers
        self.dropout = dropout or lstm_config.dropout
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMNetwork] = None
        self.version = "1.0.0"
        self.feature_names: List[str] = []
    
    def build_model(self) -> None:
        """Initialize the LSTM model."""
        self.model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        logger.info(f"LSTM model built: {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, lr: float = 0.001,
              early_stopping_patience: int = 10) -> Dict:
        """
        Train the LSTM model with early stopping.
        
        Args:
            X_train: Training sequences (samples, seq_len, features)
            y_train: Training labels (0=LONG, 1=NEUTRAL, 2=SHORT)
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Max training epochs
            batch_size: Batch size
            lr: Learning rate
            early_stopping_patience: Epochs without improvement before stopping
        
        Returns:
            Training history dict
        """
        if self.model is None:
            self.input_size = X_train.shape[2]
            self.build_model()
        
        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.LongTensor(y_train).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Class weights for imbalanced data
        class_counts = np.bincount(y_train, minlength=3)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum()
        weights = torch.FloatTensor(class_weights).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits, _ = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation
            if X_val is not None:
                val_loss, val_acc = self._evaluate(X_val, y_val, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        return history
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray, criterion) -> Tuple[float, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            logits, probs = self.model(X_tensor)
            loss = criterion(logits, y_tensor).item()
            preds = probs.argmax(dim=1)
            acc = (preds == y_tensor).float().mean().item()
        return loss, acc
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Input sequences (samples, seq_len, features)
        
        Returns:
            Tuple of (class_predictions, probability_distributions)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, probs = self.model(X_tensor)
            probs = probs.cpu().numpy()
            preds = probs.argmax(axis=1)
        
        return preds, probs
    
    def predict_single(self, X: np.ndarray) -> Dict:
        """
        Predict for single sequence with detailed output.
        
        Returns:
            Dict with direction, probability, and confidence
        """
        preds, probs = self.predict(X.reshape(1, *X.shape) if X.ndim == 2 else X)
        
        pred_class = int(preds[0])
        prob_dist = probs[0]
        
        return {
            "direction": self.CLASS_NAMES[pred_class],
            "probability": float(prob_dist[pred_class]),
            "confidence": float(prob_dist.max() - prob_dist.min()),
            "probabilities": {self.CLASS_NAMES[i]: float(p) for i, p in enumerate(prob_dist)}
        }
    
    def save(self, path: str) -> None:
        """Save model and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "config": {
                "input_size": self.input_size,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            "version": self.version,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model and config."""
        checkpoint = torch.load(path, map_location=self.device)
        
        config = checkpoint["config"]
        self.input_size = config["input_size"]
        self.sequence_length = config["sequence_length"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.version = checkpoint.get("version", "1.0.0")
        self.feature_names = checkpoint.get("feature_names", [])
        
        self.build_model()
        if checkpoint["model_state"]:
            self.model.load_state_dict(checkpoint["model_state"])
        
        logger.info(f"Model loaded from {path}")
    
    def export_onnx(self, path: str) -> None:
        """Export model to ONNX format for MT5."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        dummy_input = torch.randn(1, self.sequence_length, self.input_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            input_names=["input"],
            output_names=["logits", "probabilities"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}, "probabilities": {0: "batch_size"}}
        )
        logger.info(f"Model exported to ONNX: {path}")
