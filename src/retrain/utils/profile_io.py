# src/retrain/utils/profile_io.py
"""
Profile I/O - YAML Profile Management
=====================================

Read/write YAML profiles for trading configurations.
Supports versioning and active profile tracking.
"""

import os
import json
import yaml
from typing import Dict, Optional, Any
from glob import glob

from src.utils.logger import get_logger

logger = get_logger("PROFILE_IO")

PROFILES_DIR = "profiles"


def save_profile(
    config: Dict,
    name: str = "alpha",
    version: int = None,
    active: bool = False
) -> str:
    """
    Save profile to YAML file.
    
    Args:
        config: Profile configuration
        name: Profile name
        version: Version number (auto-increment if None)
        active: Set as active profile
        
    Returns:
        File path
    """
    os.makedirs(PROFILES_DIR, exist_ok=True)
    
    if version is None:
        version = get_next_version(name)
    
    # Build profile structure
    profile = {
        "profile": {
            "name": name.title(),
            "version": version,
            "mode": "optimized",
        },
        "risk": {
            "base": config.get("streak", {}).get("base", 0.02),
            "max_daily_dd": config.get("max_dd", 0.06),
        },
        "confidence_engine": {
            "thresholds": {
                "high": config.get("confidence", {}).get("high", 0.82),
                "mid": config.get("confidence", {}).get("mid", 0.66),
                "low": config.get("confidence", {}).get("low", 0.50),
            },
            "multipliers": {
                "high": 1.4,
                "mid": 1.2,
                "base": 1.0,
                "low": 0.6,
            }
        },
        "pyramid": {
            "enabled": True,
            "layers": [
                {"r": 0.0, "mult": config.get("pyramid", {}).get("r1", 1.0)},
                {"r": 1.0, "mult": config.get("pyramid", {}).get("r2", 0.7)},
                {"r": 2.0, "mult": config.get("pyramid", {}).get("r3", 0.4)},
            ]
        },
        "mode_switch": {
            "trend": {"score_bias": 1.1},
            "chop": {"score_bias": 0.8},
        },
        "_meta": {
            "score": config.get("score", 0),
            "created_by": "retrain_optimizer",
        }
    }
    
    # Save YAML
    filename = f"{name}_v{version}.yaml"
    filepath = os.path.join(PROFILES_DIR, filename)
    
    with open(filepath, 'w') as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved profile: {filepath}")
    
    if active:
        set_active_profile(name, version)
    
    return filepath


def load_profile(name: str, version: int = None) -> Optional[Dict]:
    """
    Load profile from YAML file.
    
    Args:
        name: Profile name
        version: Version (latest if None)
        
    Returns:
        Profile configuration
    """
    if version is None:
        version = get_latest_version(name)
    
    if version is None:
        logger.warning(f"No profile found for {name}")
        return None
    
    filename = f"{name}_v{version}.yaml"
    filepath = os.path.join(PROFILES_DIR, filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"Profile not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        profile = yaml.safe_load(f)
    
    return profile


def get_next_version(name: str) -> int:
    """Get next version number for a profile."""
    latest = get_latest_version(name)
    return (latest or 0) + 1


def get_latest_version(name: str) -> Optional[int]:
    """Get latest version number for a profile."""
    pattern = os.path.join(PROFILES_DIR, f"{name}_v*.yaml")
    files = glob(pattern)
    
    if not files:
        return None
    
    versions = []
    for f in files:
        try:
            v = int(os.path.basename(f).split("_v")[1].split(".")[0])
            versions.append(v)
        except:
            pass
    
    return max(versions) if versions else None


def set_active_profile(name: str, version: int) -> None:
    """Set the active profile."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    
    active_file = os.path.join(PROFILES_DIR, "active.json")
    
    with open(active_file, 'w') as f:
        json.dump({
            "name": name,
            "version": version,
            "file": f"{name}_v{version}.yaml",
        }, f, indent=2)
    
    logger.info(f"Active profile set: {name} v{version}")


def get_active_profile() -> Optional[Dict]:
    """Get the currently active profile."""
    active_file = os.path.join(PROFILES_DIR, "active.json")
    
    if not os.path.exists(active_file):
        return None
    
    with open(active_file, 'r') as f:
        active = json.load(f)
    
    return load_profile(active["name"], active["version"])


def list_profiles(name: str = None) -> list:
    """List all profile versions."""
    if name:
        pattern = os.path.join(PROFILES_DIR, f"{name}_v*.yaml")
    else:
        pattern = os.path.join(PROFILES_DIR, "*_v*.yaml")
    
    files = glob(pattern)
    
    profiles = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace(".yaml", "").split("_v")
        if len(parts) == 2:
            profiles.append({
                "name": parts[0],
                "version": int(parts[1]),
                "file": basename,
            })
    
    profiles.sort(key=lambda x: (x["name"], x["version"]))
    return profiles
