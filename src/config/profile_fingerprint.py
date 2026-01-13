# src/config/profile_fingerprint.py
"""
Profile Fingerprint - Checksum Validation
==========================================

🔐 Production-Grade Profile Integrity:
- Generate unique hash for each profile config
- Detect silent misconfiguration
- Fail fast on config drift

Usage:
    profile = get_active_profile()
    checksum = profile_checksum(profile)
    
    # Pass to all modules
    trailing = TrailingManager(profile, checksum)
"""

import json
import hashlib
from typing import Any


def profile_checksum(profile) -> str:
    """
    Generate unique checksum for a trading profile.
    
    Creates SHA256 hash of all profile settings to detect any drift.
    
    Args:
        profile: TradingProfile object
        
    Returns:
        12-character hex checksum
    """
    payload = {
        "name": profile.name,
        "mode": profile.mode.value if hasattr(profile.mode, 'value') else str(profile.mode),
    }
    
    # Risk config
    if hasattr(profile, 'risk') and profile.risk:
        payload["risk"] = {
            k: v for k, v in profile.risk.__dict__.items()
            if not k.startswith('_')
        }
    
    # Entry config
    if hasattr(profile, 'entry') and profile.entry:
        payload["entry"] = {
            k: v for k, v in profile.entry.__dict__.items()
            if not k.startswith('_')
        }
    
    # Trailing config
    if hasattr(profile, 'trailing') and profile.trailing:
        payload["trailing"] = {
            k: v for k, v in profile.trailing.__dict__.items()
            if not k.startswith('_')
        }
    
    # SLTP config
    if hasattr(profile, 'sltp') and profile.sltp:
        payload["sltp"] = {
            k: v for k, v in profile.sltp.__dict__.items()
            if not k.startswith('_')
        }
    
    # Model config
    if hasattr(profile, 'model') and profile.model:
        payload["model"] = {
            k: v for k, v in profile.model.__dict__.items()
            if not k.startswith('_')
        }
    
    # Create deterministic JSON and hash
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def validate_checksum(expected: str, actual: str, module_name: str = "Unknown"):
    """
    Validate checksum matches expected value.
    
    Raises:
        ValueError if checksum mismatch (config drift detected)
    """
    if expected != actual:
        raise ValueError(
            f"❌ PROFILE CHECKSUM MISMATCH in {module_name}!\n"
            f"   Expected: {expected}\n"
            f"   Actual: {actual}\n"
            f"   This indicates config drift - SYSTEM HALTED"
        )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from src.config.trading_profiles import get_active_profile, AGGRESSIVE_PROFILE, CONSERVATIVE_PROFILE
    
    print("=" * 60)
    print("🔐 PROFILE FINGERPRINT TEST")
    print("=" * 60)
    
    agg = AGGRESSIVE_PROFILE
    con = CONSERVATIVE_PROFILE
    
    agg_checksum = profile_checksum(agg)
    con_checksum = profile_checksum(con)
    
    print(f"\n🔥 Aggressive Profile: {agg.name}")
    print(f"   Checksum: {agg_checksum}")
    
    print(f"\n🛡️ Conservative Profile: {con.name}")
    print(f"   Checksum: {con_checksum}")
    
    print(f"\n✅ Checksums are different: {agg_checksum != con_checksum}")
    
    # Verify same profile gives same checksum
    agg_checksum2 = profile_checksum(agg)
    print(f"✅ Same profile same checksum: {agg_checksum == agg_checksum2}")
    
    print("=" * 60)
