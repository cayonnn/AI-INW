"""
AI Trading System - Model Hot-Swap Manager
============================================
Enables model updates without restarting the system.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import threading
import time
from src.models.lstm_direction import LSTMDirectionModel
from src.models.xgb_timing import XGBTimingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Model metadata."""
    name: str
    path: str
    version: str
    loaded_at: datetime
    file_hash: str
    is_active: bool = True


class ModelHotSwap:
    """
    Manages model hot-swapping without system restart.
    
    Features:
    - Automatic model file monitoring
    - Graceful model switching
    - Version tracking
    - Fallback on load failure
    """
    
    def __init__(self, model_dir: str = "artifacts/models", check_interval: int = 60):
        self.model_dir = Path(model_dir)
        self.check_interval = check_interval
        
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.file_hashes: Dict[str, str] = {}
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def _get_file_hash(self, path: Path) -> str:
        """Calculate MD5 hash of file."""
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_model(self, name: str, model_class: type, path: str) -> bool:
        """
        Load or reload a model.
        
        Args:
            name: Model identifier
            model_class: Model class (LSTMDirectionModel, XGBTimingModel)
            path: Path to model file
        
        Returns:
            True if successful
        """
        full_path = self.model_dir / path
        
        if not full_path.exists():
            logger.error(f"Model file not found: {full_path}")
            return False
        
        try:
            with self._lock:
                # Create and load model
                model = model_class()
                model.load(str(full_path))
                
                # Store model and info
                file_hash = self._get_file_hash(full_path)
                
                self.models[name] = model
                self.model_info[name] = ModelInfo(
                    name=name,
                    path=str(full_path),
                    version=getattr(model, "version", "1.0.0"),
                    loaded_at=datetime.now(),
                    file_hash=file_hash,
                    is_active=True
                )
                self.file_hashes[str(full_path)] = file_hash
                
                logger.info(f"Model loaded: {name} v{self.model_info[name].version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            return False
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get loaded model by name."""
        with self._lock:
            return self.models.get(name)
    
    def reload_model(self, name: str) -> bool:
        """Force reload a model from disk."""
        if name not in self.model_info:
            logger.warning(f"Model {name} not registered")
            return False
        
        info = self.model_info[name]
        
        # Determine model class from current model
        current_model = self.models.get(name)
        if isinstance(current_model, LSTMDirectionModel):
            model_class = LSTMDirectionModel
        elif isinstance(current_model, XGBTimingModel):
            model_class = XGBTimingModel
        else:
            logger.error(f"Unknown model class for {name}")
            return False
        
        return self.load_model(name, model_class, Path(info.path).name)
    
    def check_for_updates(self) -> List[str]:
        """Check if any model files have been updated."""
        updated = []
        
        for name, info in self.model_info.items():
            path = Path(info.path)
            if path.exists():
                current_hash = self._get_file_hash(path)
                if current_hash != info.file_hash:
                    updated.append(name)
                    logger.info(f"Model file changed: {name}")
        
        return updated
    
    def auto_reload_updated(self) -> int:
        """Check for updates and reload changed models."""
        updated = self.check_for_updates()
        reloaded = 0
        
        for name in updated:
            if self.reload_model(name):
                reloaded += 1
        
        return reloaded
    
    def start_monitoring(self):
        """Start background model file monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Model monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Model monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                reloaded = self.auto_reload_updated()
                if reloaded > 0:
                    logger.info(f"Auto-reloaded {reloaded} models")
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(self.check_interval)
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all models."""
        status = {}
        for name, info in self.model_info.items():
            status[name] = {
                "version": info.version,
                "loaded_at": info.loaded_at.isoformat(),
                "is_active": info.is_active,
                "path": info.path,
            }
        return status


# Global instance
_model_manager: Optional[ModelHotSwap] = None


def get_model_manager() -> ModelHotSwap:
    """Get global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelHotSwap()
    return _model_manager
