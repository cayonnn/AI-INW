# src/registry/model_registry.py
"""
Model Registry - Fund-Grade Version Control
=============================================

Architecture:
models/
 ├─ registry/
 │   ├─ xgb_signal/
 │   │   ├─ v001/
 │   │   │   ├─ model.pkl
 │   │   │   ├─ metrics.json
 │   │   │   └─ meta.json
 │   │   ├─ v002/
 │   │   └─ latest → v002

Features:
- Versioned model storage
- Metrics tracking
- Metadata storage
- Latest pointer
- Rollback support
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

logger = logging.getLogger("MODEL_REGISTRY")


class ModelRegistry:
    """
    Fund-Grade Model Registry.
    
    Manages versioned models with:
    - Semantic versioning (v001, v002, ...)
    - Metrics tracking
    - Latest pointer for hot-reload
    - Rollback capability
    """

    def __init__(self, base_dir: str = "models/registry"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelRegistry initialized: {self.base}")

    # =========================================================
    # REGISTRATION
    # =========================================================
    
    def register(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict,
        meta: Optional[Dict] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model (e.g., "xgb_signal")
            model_path: Path to the trained model file
            metrics: Model performance metrics
            meta: Additional metadata
            
        Returns:
            Path to the registered version directory
        """
        model_dir = self.base / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Get next version
        version = self._next_version(model_dir)
        vdir = model_dir / version
        vdir.mkdir()

        # Copy model file
        shutil.copy(model_path, vdir / "model.pkl")

        # Save metrics
        metrics["registered_at"] = datetime.utcnow().isoformat()
        with open(vdir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save metadata
        meta = meta or {}
        meta["trained_at"] = datetime.utcnow().isoformat()
        meta["version"] = version
        meta["model_name"] = model_name
        with open(vdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Registered {model_name} {version}")
        return str(vdir)

    def promote(self, model_name: str, version: str) -> bool:
        """
        Promote a version to 'latest'.
        
        Args:
            model_name: Model name
            version: Version to promote (e.g., "v002")
            
        Returns:
            True if promoted successfully
        """
        model_dir = self.base / model_name
        vdir = model_dir / version
        
        if not vdir.exists():
            logger.error(f"Version not found: {version}")
            return False

        # Update latest pointer (Windows-compatible)
        latest = model_dir / "latest"
        latest_model = model_dir / "latest_model.pkl"
        
        # Remove old latest
        if latest.exists():
            if latest.is_symlink():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        
        if latest_model.exists():
            latest_model.unlink()
        
        # Copy latest model (Windows doesn't always support symlinks)
        try:
            latest.symlink_to(vdir.name)
        except OSError:
            # Fallback: copy instead of symlink
            shutil.copytree(vdir, latest)
        
        # Also copy model directly for easy access
        shutil.copy(vdir / "model.pkl", latest_model)
        
        logger.info(f"Promoted {model_name} to {version}")
        return True

    def _next_version(self, model_dir: Path) -> str:
        """Get next version number."""
        versions = [
            d.name for d in model_dir.iterdir() 
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ]
        if not versions:
            return "v001"
        max_v = max(int(v[1:]) for v in versions)
        return f"v{max_v + 1:03d}"

    # =========================================================
    # QUERIES
    # =========================================================
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest promoted version."""
        model_dir = self.base / model_name
        latest = model_dir / "latest"
        
        if latest.exists():
            if latest.is_symlink():
                return latest.resolve().name
            # Check meta.json for version
            meta_path = latest / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    return json.load(f).get("version")
        
        return None

    def get_latest_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to latest model file."""
        model_dir = self.base / model_name
        
        # Try direct latest_model.pkl first
        latest_model = model_dir / "latest_model.pkl"
        if latest_model.exists():
            return latest_model
        
        # Try latest/model.pkl
        latest = model_dir / "latest" / "model.pkl"
        if latest.exists():
            return latest
        
        return None

    def get_metrics(self, model_name: str, version: str) -> Optional[Dict]:
        """Get metrics for a specific version."""
        metrics_path = self.base / model_name / version / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return None

    def get_meta(self, model_name: str, version: str) -> Optional[Dict]:
        """Get metadata for a specific version."""
        meta_path = self.base / model_name / version / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model."""
        model_dir = self.base / model_name
        if not model_dir.exists():
            return []
        
        versions = [
            d.name for d in model_dir.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ]
        return sorted(versions)

    # =========================================================
    # ROLLBACK
    # =========================================================
    
    def rollback(self, model_name: str) -> Optional[str]:
        """
        Rollback to previous version.
        
        Returns:
            Previous version string, or None if not possible
        """
        versions = self.list_versions(model_name)
        current = self.get_latest_version(model_name)
        
        if not versions or len(versions) < 2:
            logger.warning(f"Cannot rollback {model_name}: not enough versions")
            return None
        
        if current:
            try:
                current_idx = versions.index(current)
                if current_idx > 0:
                    prev_version = versions[current_idx - 1]
                    self.promote(model_name, prev_version)
                    logger.warning(f"Rolled back {model_name} to {prev_version}")
                    return prev_version
            except ValueError:
                pass
        
        # Fallback: promote second-to-last
        prev_version = versions[-2]
        self.promote(model_name, prev_version)
        return prev_version

    # =========================================================
    # STATUS
    # =========================================================
    
    def get_status(self, model_name: str) -> Dict:
        """Get status of a model."""
        versions = self.list_versions(model_name)
        latest = self.get_latest_version(model_name)
        latest_metrics = None
        
        if latest:
            latest_metrics = self.get_metrics(model_name, latest)
        
        return {
            "model_name": model_name,
            "versions": versions,
            "total_versions": len(versions),
            "latest": latest,
            "latest_metrics": latest_metrics
        }
