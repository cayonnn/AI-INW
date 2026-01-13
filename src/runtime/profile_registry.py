# src/runtime/profile_registry.py
"""
Runtime Profile Registry - Diff Detector
==========================================

🔥 Production-Grade Runtime Safety:
- Central registry for all module profile checksums
- Detect config drift at runtime
- Halt system on mismatch

Usage:
    # Each module registers on init
    ProfileRegistry.register("TrailingManager", checksum, profile.name)
    
    # Guard in live loop
    if ProfileRegistry.detect_drift():
        raise RuntimeError("PROFILE DRIFT DETECTED")
"""

from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ModuleRegistration:
    """Registration record for a module."""
    module_name: str
    checksum: str
    profile_name: str
    registered_at: datetime


class ProfileRegistry:
    """
    Central Registry for Profile Consistency.
    
    All modules register their profile checksum here.
    Detects drift when checksums don't match.
    """
    
    _modules: Dict[str, ModuleRegistration] = {}
    _master_checksum: Optional[str] = None
    _master_profile: Optional[str] = None
    
    @classmethod
    def set_master(cls, checksum: str, profile_name: str):
        """Set the master checksum from orchestrator."""
        cls._master_checksum = checksum
        cls._master_profile = profile_name
    
    @classmethod
    def register(cls, module_name: str, checksum: str, profile_name: str):
        """
        Register a module with its profile checksum.
        
        Args:
            module_name: Name of the module (e.g., "TrailingManager")
            checksum: Profile checksum used by this module
            profile_name: Name of the profile used
        """
        cls._modules[module_name] = ModuleRegistration(
            module_name=module_name,
            checksum=checksum,
            profile_name=profile_name,
            registered_at=datetime.now()
        )
    
    @classmethod
    def detect_drift(cls) -> bool:
        """
        Detect if any module has a different checksum.
        
        Returns:
            True if drift detected (checksums don't match)
        """
        if not cls._modules:
            return False
        
        checksums = {m.checksum for m in cls._modules.values()}
        
        # All modules should have same checksum
        if len(checksums) > 1:
            return True
        
        # If master is set, compare against it
        if cls._master_checksum:
            return checksums.pop() != cls._master_checksum
        
        return False
    
    @classmethod
    def get_drift_report(cls) -> Dict:
        """
        Get detailed drift report.
        
        Returns:
            Dict with drift analysis
        """
        report = {
            "has_drift": cls.detect_drift(),
            "master_checksum": cls._master_checksum,
            "master_profile": cls._master_profile,
            "modules": {},
            "unique_checksums": [],
        }
        
        checksums = set()
        for name, reg in cls._modules.items():
            match_master = reg.checksum == cls._master_checksum if cls._master_checksum else None
            report["modules"][name] = {
                "checksum": reg.checksum,
                "profile": reg.profile_name,
                "registered_at": reg.registered_at.isoformat(),
                "matches_master": match_master,
            }
            checksums.add(reg.checksum)
        
        report["unique_checksums"] = list(checksums)
        
        return report
    
    @classmethod
    def get_drifted_modules(cls) -> List[str]:
        """Get list of modules that don't match master."""
        if not cls._master_checksum:
            return []
        
        return [
            name for name, reg in cls._modules.items()
            if reg.checksum != cls._master_checksum
        ]
    
    @classmethod
    def validate_or_halt(cls, logger=None):
        """
        Validate all modules match master, halt if drift detected.
        
        Raises:
            RuntimeError if drift detected
        """
        if cls.detect_drift():
            report = cls.get_drift_report()
            drifted = cls.get_drifted_modules()
            
            msg = (
                f"❌ PROFILE DRIFT DETECTED - SYSTEM HALTED!\n"
                f"   Master: {cls._master_checksum} ({cls._master_profile})\n"
                f"   Drifted modules: {drifted}\n"
                f"   Report: {report}"
            )
            
            if logger:
                logger.critical(msg)
            
            raise RuntimeError(msg)
        
        if logger:
            logger.info(
                f"✅ Profile Registry validated: {len(cls._modules)} modules, "
                f"all match checksum {cls._master_checksum}"
            )
    
    @classmethod
    def reset(cls):
        """Reset registry (for testing)."""
        cls._modules.clear()
        cls._master_checksum = None
        cls._master_profile = None
    
    @classmethod
    def export_status(cls) -> Dict:
        """Export registry status for dashboard."""
        return {
            "module_count": len(cls._modules),
            "has_drift": cls.detect_drift(),
            "master_checksum": cls._master_checksum,
            "master_profile": cls._master_profile,
            "modules": list(cls._modules.keys()),
        }


# =============================================================================
# RUNTIME GUARD FUNCTION
# =============================================================================

def guard_profile_drift(logger=None):
    """
    Guard function to call in live loop.
    
    Usage:
        # In live_loop_v3.py main loop
        guard_profile_drift(logger)
    """
    ProfileRegistry.validate_or_halt(logger)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔐 PROFILE REGISTRY TEST")
    print("=" * 60)
    
    # Reset for test
    ProfileRegistry.reset()
    
    # Set master
    ProfileRegistry.set_master("abc123def456", "🔥 Competition")
    
    # Register modules with same checksum
    ProfileRegistry.register("RiskManager", "abc123def456", "🔥 Competition")
    ProfileRegistry.register("TrailingManager", "abc123def456", "🔥 Competition")
    ProfileRegistry.register("PositionManager", "abc123def456", "🔥 Competition")
    
    print("\n✅ All modules matching:")
    print(f"   Drift detected: {ProfileRegistry.detect_drift()}")
    
    # Simulate drift
    ProfileRegistry.register("BadModule", "wrong_checksum", "Conservative")
    
    print("\n❌ After drift:")
    print(f"   Drift detected: {ProfileRegistry.detect_drift()}")
    print(f"   Drifted modules: {ProfileRegistry.get_drifted_modules()}")
    
    print("\n📊 Full Report:")
    import json
    print(json.dumps(ProfileRegistry.get_drift_report(), indent=2, default=str))
    
    print("=" * 60)
