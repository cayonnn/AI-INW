# src/risk/win_streak_booster.py
"""
Win-Streak Risk Booster - Leaderboard Grade
============================================

🔥 Convex Payoff Strategy:
- Win streak → Ramp up risk stepwise
- Any loss → Reset immediately
- Respects Crisis Mode & Max DD

Competition-Optimized:
- Stepped exponential equity curve
- Controlled max DD
- Sharpe / Profit Factor boost

Safety:
- Does NOT touch SL/RR/execution
- Respects all safety guards
- Auto-reset on any red flag
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("WIN_STREAK_BOOSTER")


class TradeResult(Enum):
    """Trade outcome classification."""
    WIN = "win"           # TP hit or profitable close
    LOSS = "loss"         # SL hit or loss close
    BREAKEVEN = "be"      # Closed at entry
    PARTIAL_WIN = "partial_win"   # Partial TP
    PARTIAL_LOSS = "partial_loss" # Partial SL


@dataclass
class BoosterState:
    """Current booster state for logging/dashboard."""
    win_streak: int
    loss_streak: int
    current_risk_pct: float
    risk_level: int  # 0-5 (base to max)
    is_boosted: bool
    blocked: bool
    block_reason: str = ""


class WinStreakRiskBooster:
    """
    Win-Streak Based Risk Ramping.
    
    Risk Ramp Table (Competition Mode):
    ┌────────────┬────────────┬─────────┐
    │ Win Streak │ Risk %     │ Level   │
    ├────────────┼────────────┼─────────┤
    │ 0-1        │ 0.5% base  │ 0       │
    │ 2          │ 0.75%      │ 1       │
    │ 3          │ 1.0%       │ 2       │
    │ 4          │ 1.25%      │ 3       │
    │ ≥5         │ 1.5% cap   │ 4       │
    └────────────┴────────────┴─────────┘
    
    Reset Triggers:
    - Any loss (SL hit)
    - Breakeven close
    - Partial loss
    - Crisis mode active
    - Max DD exceeded
    """
    
    # Risk ramp table (streak -> risk %)
    RISK_TABLE = {
        0: 0.5,    # Base
        1: 0.5,    # Still base
        2: 0.75,   # First ramp
        3: 1.0,    # Second ramp
        4: 1.25,   # Third ramp
        5: 1.5,    # Cap (max)
    }
    
    def __init__(
        self,
        base_risk: float = 0.5,
        max_risk: float = 1.5,
        ramp_mode: str = "standard",  # "standard" or "aggressive"
        profile = None
    ):
        """
        Initialize booster.
        
        Args:
            base_risk: Starting risk % (default 0.5%)
            max_risk: Maximum risk % cap (default 1.5%)
            ramp_mode: "standard" or "aggressive"
            profile: TradingProfile for integration
        """
        # Get from profile if available
        if profile is not None:
            base_risk = profile.risk.risk_per_trade
            # Aggressive profile might have higher base
            if hasattr(profile, 'mode') and profile.mode.value == 'aggressive':
                max_risk = min(3.0, base_risk * 3)  # 3x max for aggressive
        
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.ramp_mode = ramp_mode
        
        # State
        self.win_streak = 0
        self.loss_streak = 0
        self._blocked = False
        self._block_reason = ""
        
        # Statistics
        self.stats = {
            "total_trades": 0,
            "total_wins": 0,
            "total_losses": 0,
            "max_streak": 0,
            "times_at_max": 0,
            "resets": 0,
        }
        
        # Build custom risk table if aggressive
        self._build_risk_table()
        
        logger.info(
            f"🔥 WinStreakBooster initialized: "
            f"base={base_risk}%, max={max_risk}%, mode={ramp_mode}"
        )
    
    def _build_risk_table(self):
        """Build risk table based on mode and base/max."""
        if self.ramp_mode == "aggressive":
            # More aggressive ramp
            self._risk_table = {
                0: self.base_risk,
                1: self.base_risk,
                2: self.base_risk * 1.5,
                3: self.base_risk * 2.0,
                4: self.base_risk * 2.5,
                5: min(self.max_risk, self.base_risk * 3.0),
            }
        else:
            # Standard ramp (linear steps)
            step = (self.max_risk - self.base_risk) / 4
            self._risk_table = {
                0: self.base_risk,
                1: self.base_risk,
                2: self.base_risk + step,
                3: self.base_risk + step * 2,
                4: self.base_risk + step * 3,
                5: self.max_risk,
            }
    
    # =========================================================================
    # CORE METHODS
    # =========================================================================
    
    def on_trade_result(self, result: TradeResult | str):
        """
        Update state based on trade result.
        
        Args:
            result: TradeResult enum or string ("WIN", "LOSS", "BE", etc.)
        """
        # Convert string to enum if needed
        if isinstance(result, str):
            result_map = {
                "WIN": TradeResult.WIN,
                "LOSS": TradeResult.LOSS,
                "BE": TradeResult.BREAKEVEN,
                "BREAKEVEN": TradeResult.BREAKEVEN,
                "PARTIAL_WIN": TradeResult.PARTIAL_WIN,
                "PARTIAL_LOSS": TradeResult.PARTIAL_LOSS,
            }
            result = result_map.get(result.upper(), TradeResult.LOSS)
        
        self.stats["total_trades"] += 1
        
        if result == TradeResult.WIN:
            self._on_win()
        elif result == TradeResult.PARTIAL_WIN:
            # Partial win = 0.5 streak increment (round down)
            self._on_partial_win()
        else:
            # LOSS, BE, PARTIAL_LOSS = all reset
            self._on_reset(result.value)
    
    def _on_win(self):
        """Handle win trade."""
        self.win_streak += 1
        self.loss_streak = 0
        self.stats["total_wins"] += 1
        
        # Track max streak
        if self.win_streak > self.stats["max_streak"]:
            self.stats["max_streak"] = self.win_streak
        
        # Track times at max
        if self.win_streak >= 5:
            self.stats["times_at_max"] += 1
        
        new_risk = self.current_risk()
        logger.info(
            f"🔥 WIN STREAK: {self.win_streak} | "
            f"📈 Risk ramped → {new_risk:.2f}%"
        )
    
    def _on_partial_win(self):
        """Handle partial win - half increment."""
        # Don't increment streak, but don't reset either
        self.stats["total_wins"] += 1
        logger.info(f"📊 Partial win - streak maintained at {self.win_streak}")
    
    def _on_reset(self, reason: str):
        """Reset streak on loss/BE/partial loss."""
        old_streak = self.win_streak
        self.win_streak = 0
        self.loss_streak += 1
        self.stats["total_losses"] += 1
        self.stats["resets"] += 1
        
        if old_streak > 0:
            logger.info(
                f"🔻 Streak reset: {old_streak} → 0 | "
                f"Reason: {reason} | Risk → {self.base_risk:.2f}%"
            )
    
    def current_risk(self) -> float:
        """
        Get current risk % based on streak.
        
        Returns:
            Risk percentage (0.5 - 1.5 typically)
        """
        if self._blocked:
            return self.base_risk
        
        # Lookup in risk table (cap at 5)
        level = min(self.win_streak, 5)
        return self._risk_table.get(level, self.base_risk)
    
    def get_risk_level(self) -> int:
        """Get current risk level (0-5)."""
        return min(self.win_streak, 5)
    
    def is_boosted(self) -> bool:
        """Check if currently boosted (streak >= 2)."""
        return self.win_streak >= 2 and not self._blocked
    
    # =========================================================================
    # SAFETY GUARDS
    # =========================================================================
    
    def check_safety(
        self,
        crisis_mode: bool = False,
        daily_dd: float = 0,
        max_dd: float = 10.0,
        spread_ok: bool = True,
        latency_ok: bool = True
    ) -> bool:
        """
        Check if boosting is safe.
        
        Args:
            crisis_mode: Is crisis mode active?
            daily_dd: Current daily drawdown %
            max_dd: Max allowed daily DD %
            spread_ok: Is spread within limits?
            latency_ok: Is latency acceptable?
            
        Returns:
            True if safe to boost, False if blocked
        """
        reasons = []
        
        if crisis_mode:
            reasons.append("crisis_mode")
        if daily_dd > max_dd * 0.8:  # Block at 80% of max DD
            reasons.append(f"dd_warning({daily_dd:.1f}%)")
        if not spread_ok:
            reasons.append("spread_high")
        if not latency_ok:
            reasons.append("latency_spike")
        
        if reasons:
            self._blocked = True
            self._block_reason = ", ".join(reasons)
            logger.warning(f"⚠️ Booster BLOCKED: {self._block_reason}")
            return False
        
        self._blocked = False
        self._block_reason = ""
        return True
    
    def force_reset(self, reason: str = "manual"):
        """Force reset streak (called by external safety systems)."""
        old_streak = self.win_streak
        self.win_streak = 0
        self._blocked = True
        self._block_reason = reason
        self.stats["resets"] += 1
        
        logger.warning(f"🛑 Booster FORCE RESET: {reason} (was {old_streak})")
    
    def unblock(self):
        """Unblock booster after safety condition resolved."""
        self._blocked = False
        self._block_reason = ""
        logger.info("✅ Booster unblocked")
    
    # =========================================================================
    # STATE & STATS
    # =========================================================================
    
    def get_state(self) -> BoosterState:
        """Get current booster state."""
        return BoosterState(
            win_streak=self.win_streak,
            loss_streak=self.loss_streak,
            current_risk_pct=self.current_risk(),
            risk_level=self.get_risk_level(),
            is_boosted=self.is_boosted(),
            blocked=self._blocked,
            block_reason=self._block_reason,
        )
    
    def get_stats(self) -> dict:
        """Get booster statistics."""
        return {
            **self.stats,
            "current_streak": self.win_streak,
            "current_risk": self.current_risk(),
            "is_boosted": self.is_boosted(),
        }
    
    def reset_stats(self):
        """Reset statistics (for new session)."""
        self.stats = {k: 0 for k in self.stats}
        self.win_streak = 0
        self.loss_streak = 0


# =============================================================================
# R-MULTIPLE BASED BOOSTER (Advanced)
# =============================================================================

class RMultipleBooster(WinStreakRiskBooster):
    """
    🔥 Ultra-Aggressive: Ramp based on R-multiple, not just win count.
    
    Rules:
    - Big win (R >= 2.0) = Jump 2 levels
    - Normal win (R >= 1.0) = Jump 1 level
    - Small win (R < 1.0) = No increment
    - BE = No change
    - Loss = Full reset
    """
    
    def on_trade_result_with_r(self, result: str, r_multiple: float):
        """
        Update based on R-multiple.
        
        Args:
            result: "WIN", "LOSS", "BE"
            r_multiple: Actual R achieved (e.g., 2.5 = 2.5R profit)
        """
        self.stats["total_trades"] += 1
        
        if result == "WIN":
            if r_multiple >= 2.0:
                # Big win = double increment
                self.win_streak += 2
                self.stats["total_wins"] += 1
                logger.info(f"🔥🔥 BIG WIN ({r_multiple:.1f}R) | Streak +2 → {self.win_streak}")
            elif r_multiple >= 1.0:
                # Normal win
                self.win_streak += 1
                self.stats["total_wins"] += 1
                logger.info(f"🔥 WIN ({r_multiple:.1f}R) | Streak +1 → {self.win_streak}")
            else:
                # Small win - no increment
                self.stats["total_wins"] += 1
                logger.info(f"📊 Small win ({r_multiple:.1f}R) | Streak unchanged")
        elif result == "BE":
            # Breakeven - no change
            logger.info(f"📊 BE | Streak unchanged at {self.win_streak}")
        else:
            # Loss - full reset
            self._on_reset("loss")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_booster: Optional[WinStreakRiskBooster] = None

def get_win_streak_booster(profile=None) -> WinStreakRiskBooster:
    """Get global booster instance."""
    global _booster
    if _booster is None:
        _booster = WinStreakRiskBooster(profile=profile)
    return _booster


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔥 WIN-STREAK RISK BOOSTER TEST")
    print("=" * 60)
    
    booster = WinStreakRiskBooster(base_risk=0.5, max_risk=1.5)
    
    # Simulate trades
    results = ["WIN", "WIN", "WIN", "WIN", "WIN", "LOSS", "WIN", "WIN", "BE"]
    
    print("\n📊 Simulating trade sequence:")
    for i, result in enumerate(results):
        booster.on_trade_result(result)
        state = booster.get_state()
        print(f"   Trade {i+1}: {result:4} → Streak={state.win_streak}, Risk={state.current_risk_pct:.2f}%")
    
    print("\n📈 Final Stats:")
    stats = booster.get_stats()
    print(f"   Max Streak: {stats['max_streak']}")
    print(f"   Times at Max: {stats['times_at_max']}")
    print(f"   Resets: {stats['resets']}")
    
    print("\n" + "=" * 60)
    print("🔥 R-MULTIPLE BOOSTER TEST")
    print("=" * 60)
    
    r_booster = RMultipleBooster(base_risk=0.5, max_risk=1.5)
    
    # Simulate with R-multiples
    trades = [
        ("WIN", 1.5),
        ("WIN", 2.5),  # Big win
        ("WIN", 0.8),  # Small win - no increment
        ("BE", 0),
        ("LOSS", -1.0),
    ]
    
    print("\n📊 Simulating R-multiple trades:")
    for i, (result, r) in enumerate(trades):
        r_booster.on_trade_result_with_r(result, r)
        state = r_booster.get_state()
        print(f"   Trade {i+1}: {result:4} ({r:+.1f}R) → Streak={state.win_streak}, Risk={state.current_risk_pct:.2f}%")
    
    print("=" * 60)
