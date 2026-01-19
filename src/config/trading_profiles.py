# src/config/trading_profiles.py
"""
Trading Profiles Configuration
==============================

Three modes for different trading styles:
- CONSERVATIVE: Fund / Prop Firm (capital preservation)
- BALANCED: Semi-Prop (balanced approach)
- AGGRESSIVE: Competition / Leaderboard (max ROI)

Usage:
    from src.config.trading_profiles import get_profile, TradingMode
    
    profile = get_profile(TradingMode.AGGRESSIVE)
    print(profile.risk_per_trade)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


class TradingMode(Enum):
    """Trading mode selector."""
    CONSERVATIVE = "conservative"  # Fund / Prop
    BALANCED = "balanced"          # Semi-Prop
    AGGRESSIVE = "aggressive"      # Competition / Leaderboard


@dataclass
class RiskConfig:
    """Risk & Exposure settings."""
    risk_per_trade: float           # % of equity per trade
    max_open_positions: int          # Max concurrent positions
    max_daily_loss: float           # Daily loss limit %
    max_weekly_loss: float          # Weekly loss limit %
    allow_correlation: bool         # Allow correlated positions
    hard_equity_dd_kill: float      # Hard kill switch DD %


@dataclass
class EntryConfig:
    """Entry policy settings."""
    allow_reentry_same_direction: bool
    cooldown_bars: int              # Bars between entries
    duplicate_block: bool           # Block duplicate signals
    pyramid_allowed: bool           # Allow adding to winners
    max_pyramid_levels: int         # Max pyramid entries


@dataclass
class SLTPConfig:
    """Stop Loss / Take Profit settings."""
    min_rr: float                   # Minimum R:R ratio
    default_rr: float               # Default R:R ratio
    allow_dynamic_tp: bool          # Extend TP with volatility
    early_be_exit: bool             # Move to BE early
    atr_multiplier_sl: float        # ATR × for SL


@dataclass
class TrailingConfig:
    """Trailing stop settings."""
    be_trigger_r: float             # Move to BE at this R
    trail_start_r: float            # Start trailing at this R
    trail_atr_multiplier: float     # ATR × for trailing
    partial_close_1_r: float        # First partial close at R
    partial_close_1_pct: float      # % to close at first partial
    partial_close_2_r: float        # Second partial close at R
    partial_close_2_pct: float      # % to close at second partial


@dataclass
class ModelConfig:
    """AI Model settings."""
    auto_retrain: bool
    retrain_interval: str           # daily, weekly, on_trigger
    retrain_dd_trigger: float       # DD % to trigger retrain
    retrain_winrate_trigger: float  # Winrate below to trigger
    model_error_kill: float         # Model error rate to kill


@dataclass
class CompetitionConfig:
    """Competition specific envelope (Fairness constraints)."""
    enabled: bool
    max_daily_loss_limit: float     # Absolute hard cap (16%)
    min_margin_level: float         # Minimum margin level (1.5%)
    max_risk_flex: float            # Max multiplier for risk (e.g. 1.2x)


@dataclass
class GovernanceEnvelope:
    """Hard governance bounds for Auto-Tuner (Constitution)."""
    max_daily_loss_range: tuple[float, float]   # (Min, Max) %
    margin_buffer_range: tuple[float, float]    # (Min, Max) %
    risk_per_trade_range: tuple[float, float]   # (Min, Max) %


@dataclass
class TradingProfile:
    """Complete trading profile."""
    mode: TradingMode
    name: str
    description: str
    
    risk: RiskConfig
    entry: EntryConfig
    sltp: SLTPConfig
    trailing: TrailingConfig
    model: ModelConfig
    competition: CompetitionConfig = field(default_factory=lambda: CompetitionConfig(False, 0.0, 0.0, 1.0))
    envelope: GovernanceEnvelope = field(default_factory=lambda: GovernanceEnvelope((0, 100), (0, 100), (0, 100)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "mode": self.mode.value,
            "name": self.name,
            "risk": {
                "risk_per_trade": self.risk.risk_per_trade,
                "max_open_positions": self.risk.max_open_positions,
                "max_daily_loss": self.risk.max_daily_loss,
                "allow_correlation": self.risk.allow_correlation,
            },
            "competition": {
                "enabled": self.competition.enabled,
                "max_daily_loss_limit": self.competition.max_daily_loss_limit,
            },
            "entry": {
                "cooldown_bars": self.entry.cooldown_bars,
                "pyramid_allowed": self.entry.pyramid_allowed,
            },
            "trailing": {
                "be_trigger_r": self.trailing.be_trigger_r,
                "trail_start_r": self.trailing.trail_start_r,
            }
        }


# =============================================================================
# PROFILE DEFINITIONS
# =============================================================================

CONSERVATIVE_PROFILE = TradingProfile(
    mode=TradingMode.CONSERVATIVE,
    name="Fund-Grade Conservative",
    description="Capital preservation, Prop Firm compliant, low DD",
    
    risk=RiskConfig(
        risk_per_trade=0.5,
        max_open_positions=3,
        max_daily_loss=3.0,
        max_weekly_loss=6.0,
        allow_correlation=False,
        hard_equity_dd_kill=8.0,
    ),
    
    entry=EntryConfig(
        allow_reentry_same_direction=False,
        cooldown_bars=5,
        duplicate_block=True,
        pyramid_allowed=False,
        max_pyramid_levels=1,
    ),
    
    sltp=SLTPConfig(
        min_rr=2.0,
        default_rr=2.5,
        allow_dynamic_tp=False,
        early_be_exit=True,
        atr_multiplier_sl=1.5,
    ),
    
    trailing=TrailingConfig(
        be_trigger_r=1.0,
        trail_start_r=2.0,
        trail_atr_multiplier=1.2,
        partial_close_1_r=2.0,
        partial_close_1_pct=0.33,
        partial_close_2_r=3.0,
        partial_close_2_pct=0.50,
    ),
    
    model=ModelConfig(
        auto_retrain=True,
        retrain_interval="weekly",
        retrain_dd_trigger=4.0,
        retrain_winrate_trigger=40.0,
        model_error_kill=3.0,
    ),
)


BALANCED_PROFILE = TradingProfile(
    mode=TradingMode.BALANCED,
    name="Balanced Semi-Prop",
    description="Balance between growth and preservation",
    
    risk=RiskConfig(
        risk_per_trade=1.0,
        max_open_positions=4,
        max_daily_loss=5.0,
        max_weekly_loss=10.0,
        allow_correlation=True,
        hard_equity_dd_kill=12.0,
    ),
    
    entry=EntryConfig(
        allow_reentry_same_direction=True,
        cooldown_bars=3,
        duplicate_block=True,
        pyramid_allowed=True,
        max_pyramid_levels=2,
    ),
    
    sltp=SLTPConfig(
        min_rr=1.5,
        default_rr=2.0,
        allow_dynamic_tp=True,
        early_be_exit=True,
        atr_multiplier_sl=1.5,
    ),
    
    trailing=TrailingConfig(
        be_trigger_r=1.0,
        trail_start_r=2.0,
        trail_atr_multiplier=1.5,
        partial_close_1_r=1.5,
        partial_close_1_pct=0.25,
        partial_close_2_r=2.5,
        partial_close_2_pct=0.50,
    ),
    
    model=ModelConfig(
        auto_retrain=True,
        retrain_interval="daily",
        retrain_dd_trigger=5.0,
        retrain_winrate_trigger=42.0,
        model_error_kill=4.0,
    ),
)


AGGRESSIVE_PROFILE = TradingProfile(
    mode=TradingMode.AGGRESSIVE,
    name="🔥 Competition / Leaderboard",
    description="Max ROI, accept high DD, fast iteration",
    
    risk=RiskConfig(
        risk_per_trade=2.0,          # 2.0% Aggressive
        max_open_positions=6,         # 5 - 8
        max_daily_loss=14.0,          # 14% (Base Limit)
        max_weekly_loss=25.0,
        allow_correlation=True,       # ✅ Allow correlated
        hard_equity_dd_kill=10.0,     # Kill switch at 10% (Live Patch Synced)
    ),
    
    entry=EntryConfig(
        allow_reentry_same_direction=True,   # ✅ Follow-through
        cooldown_bars=2,                      # 1-2 bars only
        duplicate_block=False,                # ❌ No block
        pyramid_allowed=True,                 # ✅ Pyramid OK
        max_pyramid_levels=3,
    ),
    
    sltp=SLTPConfig(
        min_rr=1.5,                   # Lower min R:R
        default_rr=2.0,
        allow_dynamic_tp=True,        # ✅ TP extends with volatility
        early_be_exit=False,          # ❌ Don't rush to BE
        atr_multiplier_sl=1.8,        # Wider SL
    ),
    
    trailing=TrailingConfig(
        be_trigger_r=1.5,             # Later BE at 1.5R
        trail_start_r=2.5,            # Late trailing at 2.5R
        trail_atr_multiplier=2.0,     # 1.8 - 2.2 (give space)
        partial_close_1_r=2.0,        # First partial at 2R
        partial_close_1_pct=0.25,     # 25%
        partial_close_2_r=3.5,        # Second partial at 3.5R
        partial_close_2_pct=0.40,     # 40%
    ),
    
    model=ModelConfig(
        auto_retrain=True,
        retrain_interval="daily",     # Fast iteration
        retrain_dd_trigger=6.0,       # Retrain on 6% DD
        retrain_winrate_trigger=45.0, # Retrain if WR < 45%
        model_error_kill=5.0,         # Kill at 5% error rate
    ),
    
    competition=CompetitionConfig(
        enabled=True,
        max_daily_loss_limit=16.0,    # Envelope Cap (Fairness)
        min_margin_level=1.5,         # Aggressive margin floor
        max_risk_flex=1.2             # Max 20% deviation from limit
    ),
    
    envelope=GovernanceEnvelope(
        max_daily_loss_range=(12.0, 16.0),    # Hard bounds: 12% - 16%
        margin_buffer_range=(1.5, 2.5),       # Keep 1.5% - 2.5% free
        risk_per_trade_range=(1.5, 2.5)       # Flex risk between 1.5% and 2.5%
    )
)


# =============================================================================
# PROFILE REGISTRY
# =============================================================================

PROFILES: Dict[TradingMode, TradingProfile] = {
    TradingMode.CONSERVATIVE: CONSERVATIVE_PROFILE,
    TradingMode.BALANCED: BALANCED_PROFILE,
    TradingMode.AGGRESSIVE: AGGRESSIVE_PROFILE,
}


def get_profile(mode: TradingMode = TradingMode.AGGRESSIVE) -> TradingProfile:
    """Get trading profile by mode."""
    return PROFILES.get(mode, AGGRESSIVE_PROFILE)


def get_profile_by_name(name: str) -> TradingProfile:
    """Get profile by name string."""
    mode_map = {
        "conservative": TradingMode.CONSERVATIVE,
        "balanced": TradingMode.BALANCED,
        "aggressive": TradingMode.AGGRESSIVE,
        "comp": TradingMode.AGGRESSIVE,
        "leaderboard": TradingMode.AGGRESSIVE,
    }
    mode = mode_map.get(name.lower(), TradingMode.AGGRESSIVE)
    return get_profile(mode)


# =============================================================================
# CURRENT ACTIVE PROFILE
# =============================================================================

# 🔥 DEFAULT TO AGGRESSIVE FOR COMPETITION
ACTIVE_MODE = TradingMode.AGGRESSIVE
ACTIVE_PROFILE = get_profile(ACTIVE_MODE)


def set_active_profile(mode: TradingMode):
    """Set the active trading profile."""
    global ACTIVE_MODE, ACTIVE_PROFILE
    ACTIVE_MODE = mode
    ACTIVE_PROFILE = get_profile(mode)


def get_active_profile() -> TradingProfile:
    """Get currently active profile."""
    return ACTIVE_PROFILE


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 TRADING PROFILES")
    print("=" * 60)
    
    for mode, profile in PROFILES.items():
        print(f"\n{'='*40}")
        print(f"📊 {profile.name}")
        print(f"{'='*40}")
        print(f"Mode: {mode.value}")
        print(f"Description: {profile.description}")
        print(f"\n🎲 Risk:")
        print(f"   Risk/Trade: {profile.risk.risk_per_trade}%")
        print(f"   Max Positions: {profile.risk.max_open_positions}")
        print(f"   Max Daily Loss: {profile.risk.max_daily_loss}%")
        print(f"   Kill Switch: {profile.risk.hard_equity_dd_kill}% DD")
        print(f"\n📈 Entry:")
        print(f"   Cooldown: {profile.entry.cooldown_bars} bars")
        print(f"   Pyramid: {'✅' if profile.entry.pyramid_allowed else '❌'}")
        print(f"\n🎯 Trailing:")
        print(f"   BE at: {profile.trailing.be_trigger_r}R")
        print(f"   Trail at: {profile.trailing.trail_start_r}R")
        print(f"   ATR×: {profile.trailing.trail_atr_multiplier}")
    
    print("\n" + "=" * 60)
    print(f"🔥 ACTIVE PROFILE: {ACTIVE_PROFILE.name}")
    print("=" * 60)
