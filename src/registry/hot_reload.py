# src/registry/hot_reload.py
"""
Hot Reload - Live Model Switching
==================================

Live ไม่ต้อง restart
Model ใหม่จะถูกโหลดอัตโนมัติเมื่อมีการ promote

Features:
- Detect file changes
- Load new model without restart
- Log model version
- Thread-safe
"""

import logging
import threading
import joblib
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger("HOT_RELOAD")


class HotReloader:
    """
    Hot-reload model from registry.
    
    Usage:
        reloader = HotReloader("xgb_signal")
        model = reloader.get_model()  # Auto-reloads if updated
    """

    def __init__(
        self,
        model_name: str,
        registry_dir: str = "models/registry"
    ):
        self.model_name = model_name
        self.registry_dir = Path(registry_dir)
        
        self._model: Optional[Any] = None
        self._last_mtime: Optional[float] = None
        self._current_version: Optional[str] = None
        self._lock = threading.Lock()
        
        # Model paths
        self._model_path = self.registry_dir / model_name / "latest_model.pkl"
        self._alt_path = self.registry_dir / model_name / "latest" / "model.pkl"
        
        logger.info(f"HotReloader initialized: {model_name}")

    # =========================================================
    # MAIN INTERFACE
    # =========================================================
    
    def get_model(self) -> Optional[Any]:
        """
        Get the current model, auto-reloading if updated.
        
        Returns:
            The loaded model, or None if not available
        """
        with self._lock:
            if self._should_reload():
                self._reload()
            return self._model

    def _should_reload(self) -> bool:
        """Check if model file has been updated."""
        path = self._get_model_path()
        
        if path is None:
            return False
        
        try:
            mtime = path.stat().st_mtime
            
            if self._last_mtime is None:
                return True
            
            return mtime != self._last_mtime
            
        except Exception:
            return False

    def _reload(self):
        """Reload the model from disk."""
        path = self._get_model_path()
        
        if path is None:
            logger.warning(f"No model found for: {self.model_name}")
            return
        
        try:
            self._model = joblib.load(path)
            self._last_mtime = path.stat().st_mtime
            
            # Try to get version
            self._current_version = self._get_version()
            
            logger.info(
                f"🔁 Model hot-reloaded: {self.model_name} "
                f"v{self._current_version or 'unknown'}"
            )
            
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")

    def _get_model_path(self) -> Optional[Path]:
        """Get the path to the model file."""
        if self._model_path.exists():
            return self._model_path
        if self._alt_path.exists():
            return self._alt_path
        return None

    def _get_version(self) -> Optional[str]:
        """Get version from meta.json."""
        import json
        
        meta_paths = [
            self.registry_dir / self.model_name / "latest" / "meta.json",
        ]
        
        for meta_path in meta_paths:
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        return json.load(f).get("version")
                except Exception:
                    pass
        
        return None

    # =========================================================
    # STATUS
    # =========================================================
    
    def get_version(self) -> Optional[str]:
        """Get current model version."""
        return self._current_version

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def get_status(self) -> dict:
        """Get reloader status."""
        return {
            "model_name": self.model_name,
            "loaded": self.is_loaded(),
            "version": self._current_version,
            "last_reload": datetime.fromtimestamp(self._last_mtime).isoformat() if self._last_mtime else None
        }


# =========================================================
# GLOBAL RELOADERS
# =========================================================

_reloaders: dict = {}

def get_reloader(model_name: str) -> HotReloader:
    """Get or create a reloader for a model."""
    global _reloaders
    
    if model_name not in _reloaders:
        _reloaders[model_name] = HotReloader(model_name)
    
    return _reloaders[model_name]


def load_latest_model(model_name: str = "xgb_signal") -> Optional[Any]:
    """
    Convenience function to load latest model.
    
    Usage in live_loop_v3.py:
        from src.registry.hot_reload import load_latest_model
        
        model = load_latest_model()
        if model:
            signal = model.predict(X)
    """
    reloader = get_reloader(model_name)
    return reloader.get_model()


def get_model_version(model_name: str = "xgb_signal") -> Optional[str]:
    """Get current model version."""
    reloader = get_reloader(model_name)
    return reloader.get_version()
