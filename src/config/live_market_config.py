# src/config/live_market_config.py
"""
Live Market Configuration
==========================

Production-grade config for REAL market trading.
Separated from Competition Mode for safety.

Key Differences from Competition:
    - Lower risk per trade
    - Stricter DD limits
    - More conservative Guardian
    - PPO disabled by default

WARNING: Competition preset is TOO AGGRESSIVE for real money!
"""

from dataclasses import dataclass, field
from typing import Dict, Any

# =============================================================================
# LIVE MARKET PROFILE (PRODUCTION)
# =============================================================================

LIVE_MARKET = {
    "name": "LIVE_MARKET_SAFE",
    
    # --- Risk & Capital ---
    "risk_per_trade": 0.5,        # % per trade (conservative)
    "max_positions": 3,
    "max_daily_loss": 4.0,        # % hard stop per day
    "max_total_dd": 10.0,         # % absolute account DD
    
    # --- Margin / Safety ---
    "min_free_margin_pct": 25,    # > 25% only
    "margin_buffer": 150,         # conservative
    
    # --- Guardian ---
    "guardian": {
        "daily_dd_limit": 6.0,        # Guardian latch
        "soft_dd_warning": 4.0,       # Start tightening
        "freeze_cooldown_sec": 900,   # 15 minutes
        "max_blocks_before_freeze": 2,
        "strict_mode": True
    },
    
    # --- PPO Usage ---
    "ppo": {
        "enabled": False,             # START WITH OFF!
        "confidence_threshold": 0.85,
        "max_control_ratio": 0.2      # 20% of trades max
    },
    
    # --- Auto Protection ---
    "loss_streak_protection": {
        "max_losses": 3,
        "risk_reduce_factor": 0.5
    },
    
    # --- Capital Lockdown ---
    "lockdown": {
        "equity_drop_pct": 3.0,       # If equity drops 3% from ATH
        "risk_reduce_factor": 0.5,    # Reduce risk by 50%
        "first_day_max_dd": 4.0       # First day of week limit
    }
}

# =============================================================================
# COMPETITION PROFILE (AGGRESSIVE - FOR COMPETITION ONLY)
# =============================================================================

COMPETITION_MODE = {
    "name": "COMPETITION_AGGRESSIVE",
    
    # --- Risk & Capital ---
    "risk_per_trade": 2.0,        # % per trade
    "max_positions": 5,
    "max_daily_loss": 14.0,       # % - HIGH!
    "max_total_dd": 20.0,
    
    # --- Margin / Safety ---
    "min_free_margin_pct": 15,
    "margin_buffer": 120,
    
    # --- Guardian ---
    "guardian": {
        "daily_dd_limit": 14.0,
        "soft_dd_warning": 10.0,
        "freeze_cooldown_sec": 300,
        "max_blocks_before_freeze": 3,
        "strict_mode": False
    },
    
    # --- PPO Usage ---
    "ppo": {
        "enabled": True,
        "confidence_threshold": 0.70,
        "max_control_ratio": 0.5
    },
    
    # --- Auto Protection ---
    "loss_streak_protection": {
        "max_losses": 5,
        "risk_reduce_factor": 0.7
    }
}

# =============================================================================
# PROFILES REGISTRY
# =============================================================================

PROFILES = {
    "LIVE_MARKET": LIVE_MARKET,
    "COMPETITION": COMPETITION_MODE,
}


@dataclass
class LiveMarketConfig:
    """Live market configuration loader."""
    profile: Dict[str, Any] = field(default_factory=lambda: LIVE_MARKET)
    
    @classmethod
    def load(cls, name: str = "LIVE_MARKET") -> "LiveMarketConfig":
        """Load a named profile."""
        if name not in PROFILES:
            raise ValueError(f"Unknown profile: {name}. Available: {list(PROFILES.keys())}")
        return cls(profile=PROFILES[name])
    
    @property
    def name(self) -> str:
        return self.profile.get("name", "unknown")
    
    @property
    def risk_per_trade(self) -> float:
        return self.profile.get("risk_per_trade", 0.5)
    
    @property
    def max_positions(self) -> int:
        return self.profile.get("max_positions", 3)
    
    @property
    def guardian_config(self) -> Dict:
        return self.profile.get("guardian", {})
    
    @property
    def ppo_config(self) -> Dict:
        return self.profile.get("ppo", {})
    
    @property
    def lockdown_config(self) -> Dict:
        return self.profile.get("lockdown", {})
    
    def summary(self) -> str:
        return (
            f"Profile: {self.name}\n"
            f"  Risk/Trade: {self.risk_per_trade}%\n"
            f"  Max Positions: {self.max_positions}\n"
            f"  Guardian DD: {self.guardian_config.get('daily_dd_limit', 'N/A')}%\n"
            f"  PPO Enabled: {self.ppo_config.get('enabled', False)}"
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Live Market Configurations")
    print("=" * 60)
    
    for name in PROFILES:
        config = LiveMarketConfig.load(name)
        print(f"\n{config.summary()}")
    
    print("\n" + "=" * 60)
    print("⚠️  WARNING: Use LIVE_MARKET for real money!")
    print("    Competition mode is TOO AGGRESSIVE!")
    print("=" * 60)
