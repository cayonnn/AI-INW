# src/retrain/deployer.py
"""
Config Deployer - Safe Deployment
=================================

Safely deploys new configs with:
- Version control
- Activation scheduling
- Rollback support
"""

import os
from datetime import datetime
from typing import Dict, Optional

from src.retrain.utils.profile_io import save_profile, get_next_version
from src.utils.logger import get_logger

logger = get_logger("DEPLOYER")


def deploy_candidate(
    profile: Dict,
    active: bool = False,
    profile_name: str = "alpha"
) -> bool:
    """
    Deploy a candidate profile.
    
    Args:
        profile: Profile config to deploy
        active: Whether to activate immediately
        profile_name: Base profile name
        
    Returns:
        True if deployed successfully
    """
    try:
        # Get next version
        version = get_next_version(profile_name)
        
        # Add metadata
        profile["_meta"] = {
            "name": profile_name,
            "version": version,
            "deployed_at": datetime.now().isoformat(),
            "active": active,
        }
        
        # Save profile
        filepath = save_profile(profile, profile_name, version, active=active)
        
        logger.info(
            f"Deployed {profile_name} v{version} "
            f"(active={active}): {filepath}"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False


def activate_profile(profile_name: str, version: int) -> bool:
    """Activate a specific profile version."""
    from src.retrain.utils.profile_io import set_active_profile
    
    try:
        set_active_profile(profile_name, version)
        logger.info(f"Activated {profile_name} v{version}")
        return True
    except Exception as e:
        logger.error(f"Activation failed: {e}")
        return False


def rollback_to_version(profile_name: str, version: int) -> bool:
    """Rollback to a previous version."""
    logger.warning(f"Rolling back {profile_name} to v{version}")
    return activate_profile(profile_name, version)


def schedule_activation(
    profile_name: str,
    version: int,
    activate_at: datetime
) -> bool:
    """
    Schedule profile activation for future time.
    
    Note: Requires external scheduler (cron/task scheduler)
    """
    schedule_file = f"profiles/scheduled_{profile_name}_v{version}.json"
    
    import json
    schedule_data = {
        "profile_name": profile_name,
        "version": version,
        "activate_at": activate_at.isoformat(),
        "created_at": datetime.now().isoformat(),
    }
    
    with open(schedule_file, 'w') as f:
        json.dump(schedule_data, f, indent=2)
    
    logger.info(
        f"Scheduled {profile_name} v{version} "
        f"for activation at {activate_at}"
    )
    
    return True
