# src/config/demo_live_config.py
"""
Demo Live Configuration
========================

Safe preset for Phase A: Soft Live trading.

Risk Level: VERY LOW
Purpose: Prove Alpha PPO doesn't destroy account

Phases:
    A: Soft Live (0.25% risk)
    B: Controlled Aggressive (0.5-1%)
    C: Full Demo Live (Competition-grade)
"""

from dataclasses import dataclass
from typing import Dict, Any

# =============================================================================
# PHASE A: SOFT LIVE (Demo Safe)
# =============================================================================

PHASE_A_SOFT_LIVE = {
    "name": "PHASE_A_SOFT_LIVE",
    "description": "Alpha PPO Live with minimal risk",
    
    # --- Risk ---
    "risk_per_trade": 0.25,       # 0.25% per trade
    "max_positions": 1,
    "max_daily_dd": 5.0,          # 5% max daily DD
    "margin_threshold": 150,      # 150% minimum margin
    
    # --- Alpha PPO ---
    "alpha": {
        "enabled": True,          # Alpha PPO makes decisions
        "confidence_threshold": 0.55,
        "max_consecutive_losses": 3,
        "dd_disable_threshold": 0.03,  # 3% → disable Alpha
    },
    
    # --- Guardian ---
    "guardian": {
        "enabled": True,
        "rule_based": True,
        "ppo_advisor": True,
        "hard_override": True,    # Can always block Alpha
    },
    
    # --- Kill Switch ---
    "kill_switch": {
        "enabled": True,
        "alpha_dd_limit": 0.03,   # 3% Alpha-specific DD
        "error_count_limit": 3,
        "chaos_force_hold": True,
    }
}

# =============================================================================
# PHASE B: CONTROLLED AGGRESSIVE
# =============================================================================

PHASE_B_CONTROLLED = {
    "name": "PHASE_B_CONTROLLED",
    "description": "Increased risk after Phase A validation",
    
    # --- Risk ---
    "risk_per_trade": 0.75,
    "max_positions": 2,
    "max_daily_dd": 6.0,
    "margin_threshold": 130,
    
    # --- Alpha PPO ---
    "alpha": {
        "enabled": True,
        "confidence_threshold": 0.60,
        "max_consecutive_losses": 4,
        "dd_disable_threshold": 0.04,
    },
    
    # --- Guardian ---
    "guardian": {
        "enabled": True,
        "rule_based": True,
        "ppo_advisor": True,
        "hard_override": True,  # Only CRITICAL
    },
    
    # --- Kill Switch ---
    "kill_switch": {
        "enabled": True,
        "alpha_dd_limit": 0.04,
        "error_count_limit": 3,
        "chaos_force_hold": True,
    }
}

# =============================================================================
# PHASE C: FULL DEMO LIVE (Competition-grade)
# =============================================================================

PHASE_C_FULL_LIVE = {
    "name": "PHASE_C_FULL_LIVE",
    "description": "Competition-grade full live trading",
    
    # --- Risk ---
    "risk_per_trade": 1.5,
    "max_positions": 3,
    "max_daily_dd": 8.0,
    "margin_threshold": 120,
    
    # --- Alpha PPO ---
    "alpha": {
        "enabled": True,
        "confidence_threshold": 0.55,
        "max_consecutive_losses": 5,
        "dd_disable_threshold": 0.06,
    },
    
    # --- Guardian ---
    "guardian": {
        "enabled": True,
        "rule_based": False,  # Rule as fallback only
        "ppo_advisor": True,  # Co-agent
        "hard_override": True,
    },
    
    # --- Kill Switch ---
    "kill_switch": {
        "enabled": True,
        "alpha_dd_limit": 0.06,
        "error_count_limit": 3,
        "chaos_force_hold": True,
    }
}

# =============================================================================
# PHASE REGISTRY
# =============================================================================

PHASES = {
    "A": PHASE_A_SOFT_LIVE,
    "B": PHASE_B_CONTROLLED,
    "C": PHASE_C_FULL_LIVE,
}


@dataclass
class DemoLiveConfig:
    """Demo Live configuration loader."""
    phase: str = "A"
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = PHASES.get(self.phase, PHASE_A_SOFT_LIVE)
    
    @classmethod
    def load(cls, phase: str = "A") -> "DemoLiveConfig":
        return cls(phase=phase, config=PHASES.get(phase, PHASE_A_SOFT_LIVE))
    
    @property
    def name(self) -> str:
        return self.config.get("name", "unknown")
    
    @property
    def risk_per_trade(self) -> float:
        return self.config.get("risk_per_trade", 0.25)
    
    @property
    def alpha_enabled(self) -> bool:
        return self.config.get("alpha", {}).get("enabled", False)
    
    @property
    def alpha_confidence(self) -> float:
        return self.config.get("alpha", {}).get("confidence_threshold", 0.55)
    
    def summary(self) -> str:
        return (
            f"Phase {self.phase}: {self.config.get('description', '')}\n"
            f"  Risk/Trade: {self.risk_per_trade}%\n"
            f"  Alpha PPO: {'ENABLED' if self.alpha_enabled else 'DISABLED'}\n"
            f"  Confidence: {self.alpha_confidence:.0%}\n"
            f"  Max Daily DD: {self.config.get('max_daily_dd', 5)}%"
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo Live Configurations")
    print("=" * 60)
    
    for phase in ["A", "B", "C"]:
        config = DemoLiveConfig.load(phase)
        print(f"\n{config.summary()}")
    
    print("\n" + "=" * 60)
