# src/dashboard/effective_risk.py
"""
Effective Risk Stack - Dashboard Explainability
=================================================

🎯 Shows ACTUAL risk being used, not just config.

Components:
- Base risk from profile
- Win-streak booster multiplier
- Capital allocation adjustments
- Final effective risk %

Usage:
    stack = EffectiveRiskStack(orchestrator)
    data = stack.export()  # JSON for dashboard
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class RiskStackData:
    """Complete risk stack for dashboard."""
    
    # Profile Info
    profile_name: str
    profile_checksum: str
    
    # Base Risk
    base_risk_pct: float
    
    # Win-Streak Booster
    booster_enabled: bool
    booster_streak: int
    booster_multiplier: float
    
    # Capital Allocation
    allocation_multiplier: float
    
    # Final Effective
    effective_risk_pct: float
    
    # Position Info
    max_positions: int
    current_positions: int
    
    # Trailing Config
    be_trigger_r: float
    trail_start_r: float
    atr_multiplier: float
    
    # AI Models
    sl_model: str
    tp_model: str
    
    # Timestamps
    updated_at: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EffectiveRiskStack:
    """
    Aggregates actual effective risk from all components.
    
    Shows the REAL risk being used, accounting for:
    - Base profile risk
    - Win-streak booster
    - Capital allocation engine
    - Position limits
    """
    
    def __init__(
        self,
        profile = None,
        checksum: str = "",
        booster = None,
        allocator = None,
        position_manager = None,
        trailing_manager = None,
    ):
        self.profile = profile
        self.checksum = checksum
        self.booster = booster
        self.allocator = allocator
        self.position_manager = position_manager
        self.trailing_manager = trailing_manager
    
    def calculate_effective_risk(self) -> float:
        """Calculate final effective risk after all multipliers."""
        if not self.profile:
            return 0.0
        
        base = self.profile.risk.risk_per_trade
        
        # Win-streak booster multiplier
        booster_mult = 1.0
        if self.booster:
            booster_mult = self.booster.current_risk() / self.booster.base_risk
        
        # Capital allocation multiplier (if available)
        alloc_mult = 1.0
        if self.allocator and hasattr(self.allocator, 'get_multiplier'):
            alloc_mult = self.allocator.get_multiplier()
        
        return base * booster_mult * alloc_mult
    
    def export(self) -> Dict:
        """Export complete risk stack as JSON-serializable dict."""
        if not self.profile:
            return {"error": "No profile loaded"}
        
        # Get booster info
        booster_enabled = self.booster is not None
        booster_streak = self.booster.win_streak if self.booster else 0
        booster_mult = 1.0
        if self.booster:
            booster_mult = self.booster.current_risk() / self.booster.base_risk
        
        # Get allocation multiplier
        alloc_mult = 1.0
        if self.allocator and hasattr(self.allocator, 'get_multiplier'):
            alloc_mult = self.allocator.get_multiplier()
        
        # Get position info
        max_pos = self.profile.risk.max_open_positions
        current_pos = 0
        if self.position_manager:
            current_pos = len(getattr(self.position_manager, '_positions', {}))
        
        # Get trailing config
        be_r = self.profile.trailing.be_trigger_r
        trail_r = self.profile.trailing.trail_start_r
        atr_mult = self.profile.trailing.trail_atr_multiplier
        
        if self.trailing_manager:
            be_r = self.trailing_manager.be_rr
            trail_r = self.trailing_manager.trail_rr
            atr_mult = self.trailing_manager.atr_multiplier
        
        data = RiskStackData(
            profile_name=self.profile.name,
            profile_checksum=self.checksum,
            base_risk_pct=self.profile.risk.risk_per_trade,
            booster_enabled=booster_enabled,
            booster_streak=booster_streak,
            booster_multiplier=booster_mult,
            allocation_multiplier=alloc_mult,
            effective_risk_pct=self.calculate_effective_risk(),
            max_positions=max_pos,
            current_positions=current_pos,
            be_trigger_r=be_r,
            trail_start_r=trail_r,
            atr_multiplier=atr_mult,
            sl_model="xgb_sl_v3",
            tp_model="xgb_tp_v3",
            updated_at=datetime.now().isoformat(),
        )
        
        return data.to_dict()
    
    def format_display(self) -> str:
        """Format for terminal/log display."""
        data = self.export()
        if "error" in data:
            return str(data)
        
        lines = [
            "=" * 50,
            "🔥 EFFECTIVE RISK STACK",
            "=" * 50,
            f"Profile: {data['profile_name']}",
            f"Checksum: {data['profile_checksum']}",
            "-" * 50,
            f"Base Risk: {data['base_risk_pct']:.1f}%",
        ]
        
        if data['booster_enabled']:
            lines.append(f"Win-Streak Booster: x{data['booster_multiplier']:.2f} (streak={data['booster_streak']})")
        
        if data['allocation_multiplier'] != 1.0:
            lines.append(f"Capital Allocation: x{data['allocation_multiplier']:.2f}")
        
        lines.extend([
            "-" * 50,
            f">>> EFFECTIVE RISK: {data['effective_risk_pct']:.2f}% <<<",
            "-" * 50,
            f"Positions: {data['current_positions']}/{data['max_positions']}",
            f"Trailing: BE@{data['be_trigger_r']}R, Trail@{data['trail_start_r']}R",
            "=" * 50,
        ])
        
        return "\n".join(lines)


# =============================================================================
# FASTAPI/FLASK ENDPOINT EXAMPLE
# =============================================================================

def create_risk_stack_endpoint(orchestrator):
    """
    Create endpoint for dashboard.
    
    Usage with FastAPI:
        @app.get("/debug/effective-risk")
        def effective_risk():
            return create_risk_stack_endpoint(orchestrator)
    """
    from src.config.profile_fingerprint import profile_checksum
    
    stack = EffectiveRiskStack(
        profile=orchestrator.trading_profile,
        checksum=profile_checksum(orchestrator.trading_profile),
        booster=getattr(orchestrator, 'streak_booster', None),
        allocator=getattr(orchestrator, 'capital_allocator', None),
        position_manager=getattr(orchestrator, 'position_manager', None),
        trailing_manager=getattr(orchestrator, 'trailing_manager', None),
    )
    
    return stack.export()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 EFFECTIVE RISK STACK TEST")
    print("=" * 60)
    
    from src.config.trading_profiles import get_active_profile
    from src.config.profile_fingerprint import profile_checksum
    
    profile = get_active_profile()
    checksum = profile_checksum(profile)
    
    # Mock booster
    class MockBooster:
        base_risk = 2.0
        win_streak = 3
        def current_risk(self):
            return 3.0  # 1.5x multiplier
    
    stack = EffectiveRiskStack(
        profile=profile,
        checksum=checksum,
        booster=MockBooster(),
    )
    
    print(stack.format_display())
    
    print("\n📊 JSON Export:")
    import json
    print(json.dumps(stack.export(), indent=2))
    
    print("=" * 60)
