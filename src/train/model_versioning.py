# src/train/model_versioning.py
"""
Model Versioning - Fund-Grade
==============================

Semantic Versioning สำหรับ models:
- เก็บ training data hash
- เก็บ hyperparameters
- เก็บ performance metrics
- Promotion gate + auto-rollback
"""

import os
import json
import hashlib
import shutil
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List

logger = logging.getLogger("MODEL_VERSION")


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    accuracy: float
    profit_factor: float = 0.0
    total_trades: int = 0


@dataclass
class ModelVersion:
    """Model version information."""
    version: str           # e.g., "v2.3.1"
    major: int
    minor: int
    patch: int
    model_type: str        # e.g., "xgb_imitation"
    created_at: str
    data_hash: str
    hyperparameters: Dict
    metrics: ModelMetrics
    is_active: bool = False


class ModelVersionManager:
    """
    Fund-Grade Model Version Manager.
    
    Manages:
    - Version history
    - Promotion gates
    - Rollback capability
    """

    # Promotion thresholds
    PROMOTION_THRESHOLDS = {
        "win_rate": {"op": ">=", "prev": True},      # Must be >= previous
        "max_drawdown": {"op": "<=", "prev": True},  # Must be <= previous
        "sharpe_ratio": {"value": 1.2}               # Must be >= 1.2
    }

    def __init__(self, models_dir: str = "models", versions_dir: str = "models/versions"):
        self.models_dir = Path(models_dir)
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_path = self.versions_dir / "registry.json"
        self.registry: Dict[str, List[Dict]] = self._load_registry()
        
        logger.info(f"ModelVersionManager initialized: {self.versions_dir}")

    # =========================================================
    # REGISTRY MANAGEMENT
    # =========================================================
    
    def _load_registry(self) -> Dict:
        """Load version registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}}

    def _save_registry(self):
        """Save version registry."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    # =========================================================
    # VERSION CREATION
    # =========================================================
    
    def create_version(
        self,
        model_type: str,
        model_path: str,
        training_data_path: str,
        hyperparameters: Dict,
        metrics: ModelMetrics
    ) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            model_type: Type of model (e.g., "xgb_imitation")
            model_path: Path to trained model file
            training_data_path: Path to training data
            hyperparameters: Model hyperparameters
            metrics: Training metrics
            
        Returns:
            Created ModelVersion
        """
        # Get next version number
        current = self.get_active_version(model_type)
        if current:
            major, minor, patch = current.major, current.minor, current.patch + 1
        else:
            major, minor, patch = 1, 0, 0
        
        version_str = f"v{major}.{minor}.{patch}"
        
        # Calculate data hash
        data_hash = self._hash_file(training_data_path) if os.path.exists(training_data_path) else "N/A"
        
        # Create version object
        version = ModelVersion(
            version=version_str,
            major=major,
            minor=minor,
            patch=patch,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            data_hash=data_hash,
            hyperparameters=hyperparameters,
            metrics=metrics,
            is_active=False
        )
        
        # Save model file with version
        version_model_path = self.versions_dir / f"{model_type}_{version_str}.pkl"
        shutil.copy(model_path, version_model_path)
        
        # Register version
        if model_type not in self.registry["models"]:
            self.registry["models"][model_type] = []
        
        self.registry["models"][model_type].append(asdict(version))
        self._save_registry()
        
        logger.info(f"Created version {version_str} for {model_type}")
        
        return version

    # =========================================================
    # PROMOTION GATE
    # =========================================================
    
    def check_promotion(
        self,
        model_type: str,
        new_metrics: ModelMetrics
    ) -> tuple:
        """
        Check if new model passes promotion gate.
        
        Returns:
            (passed: bool, reason: str)
        """
        current = self.get_active_version(model_type)
        prev_metrics = current.metrics if current else None
        
        failures = []
        
        for metric, rule in self.PROMOTION_THRESHOLDS.items():
            new_val = getattr(new_metrics, metric, 0)
            
            if "prev" in rule and prev_metrics:
                prev_val = getattr(prev_metrics, metric, 0)
                op = rule["op"]
                
                if op == ">=" and new_val < prev_val:
                    failures.append(f"{metric}: {new_val:.2f} < previous {prev_val:.2f}")
                elif op == "<=" and new_val > prev_val:
                    failures.append(f"{metric}: {new_val:.2f} > previous {prev_val:.2f}")
            
            elif "value" in rule:
                threshold = rule["value"]
                if new_val < threshold:
                    failures.append(f"{metric}: {new_val:.2f} < threshold {threshold}")
        
        if failures:
            reason = "Promotion gate failed: " + ", ".join(failures)
            logger.warning(reason)
            return False, reason
        
        logger.info(f"Promotion gate passed for {model_type}")
        return True, "All checks passed"

    def promote(self, model_type: str, version_str: str) -> bool:
        """
        Promote a version to active.
        
        Args:
            model_type: Model type
            version_str: Version to promote
            
        Returns:
            True if promoted successfully
        """
        versions = self.registry["models"].get(model_type, [])
        
        for v in versions:
            if v["version"] == version_str:
                # Deactivate all others
                for other in versions:
                    other["is_active"] = False
                
                # Activate this one
                v["is_active"] = True
                
                # Copy to main models dir
                src = self.versions_dir / f"{model_type}_{version_str}.pkl"
                dst = self.models_dir / f"{model_type}.pkl"
                if src.exists():
                    shutil.copy(src, dst)
                
                self._save_registry()
                logger.info(f"Promoted {model_type} to {version_str}")
                return True
        
        return False

    # =========================================================
    # ROLLBACK
    # =========================================================
    
    def rollback(self, model_type: str) -> Optional[str]:
        """
        Rollback to previous version.
        
        Returns:
            Version string rolled back to, or None
        """
        versions = self.registry["models"].get(model_type, [])
        
        if len(versions) < 2:
            logger.warning(f"Cannot rollback {model_type}: not enough versions")
            return None
        
        # Find current active and previous
        active_idx = None
        for i, v in enumerate(versions):
            if v["is_active"]:
                active_idx = i
                break
        
        if active_idx is None or active_idx == 0:
            logger.warning(f"Cannot rollback {model_type}: no previous version")
            return None
        
        # Rollback to previous
        prev_version = versions[active_idx - 1]["version"]
        self.promote(model_type, prev_version)
        
        logger.warning(f"Rolled back {model_type} to {prev_version}")
        return prev_version

    # =========================================================
    # QUERIES
    # =========================================================
    
    def get_active_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get currently active version."""
        versions = self.registry["models"].get(model_type, [])
        
        for v in versions:
            if v.get("is_active"):
                metrics = ModelMetrics(**v["metrics"]) if isinstance(v["metrics"], dict) else v["metrics"]
                return ModelVersion(
                    version=v["version"],
                    major=v["major"],
                    minor=v["minor"],
                    patch=v["patch"],
                    model_type=v["model_type"],
                    created_at=v["created_at"],
                    data_hash=v["data_hash"],
                    hyperparameters=v["hyperparameters"],
                    metrics=metrics,
                    is_active=True
                )
        
        return None

    def get_version_history(self, model_type: str) -> List[Dict]:
        """Get version history for a model type."""
        return self.registry["models"].get(model_type, [])

    # =========================================================
    # HELPERS
    # =========================================================
    
    def _hash_file(self, filepath: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]
