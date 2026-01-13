# src/risk/position_manager.py
"""
PositionManager Module - Fund-Grade Active Position Management
==============================================================

🎯 Purpose: Active position management with AI intelligence

ไม่ใช่ HOLD = นิ่ง แต่เป็น HOLD = กำลังบริหาร

Components:
- R-Multiple Tracking
- Position State Machine
- AI Exit Intelligence
- Adaptive Actions (BE, Trail, Partial Close, Exit)

Usage:
    from src.risk.position_manager import PositionManager, PositionAction
    
    pm = PositionManager()
    action = pm.manage_position(position, market_data, ai_prediction)
    
    if action == PositionAction.MOVE_BE:
        # Move SL to breakeven
        ...
"""

import MetaTrader5 as mt5
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd

from src.utils.logger import get_logger
from src.config.trading_profiles import get_active_profile, TradingProfile

logger = get_logger("POSITION_MANAGER")


# =============================================================================
# ENUMS
# =============================================================================

class PositionAction(Enum):
    """Actions that PositionManager can take."""
    HOLD = "hold"                       # Continue monitoring (active)
    MOVE_BE = "move_be"                 # Move SL to breakeven
    TRAIL_CONSERVATIVE = "trail_cons"   # Trail with wide ATR
    TRAIL_AGGRESSIVE = "trail_aggr"     # Trail with tight ATR
    TIGHTEN_SL = "tighten_sl"           # Reduce SL distance
    PARTIAL_CLOSE = "partial_close"     # Close portion of position
    FULL_EXIT = "full_exit"             # Exit entire position
    SCALE_IN = "scale_in"               # Add to winning position


class PositionState(Enum):
    """Position lifecycle states."""
    OPENED = "opened"                   # Just opened
    MONITORING = "monitoring"           # Active monitoring
    AT_BE = "at_be"                     # SL at breakeven
    TRAILING = "trailing"               # Trailing active
    PARTIALLY_CLOSED = "partial"        # Partial close done
    EXITED = "exited"                   # Position closed


class TrailingMode(Enum):
    """Trailing stop modes."""
    OFF = "off"
    CONSERVATIVE = "conservative"   # ATR × 2.0
    STANDARD = "standard"           # ATR × 1.5
    AGGRESSIVE = "aggressive"       # ATR × 1.0


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class PositionStatus:
    """Current position status."""
    ticket: int
    symbol: str
    direction: str                  # "LONG" or "SHORT"
    entry_price: float
    current_price: float
    current_sl: float
    current_tp: float
    volume: float
    
    # Calculated
    profit_pips: float
    profit_r: float                 # Profit in R-multiples
    state: PositionState
    
    # Tracking
    be_applied: bool = False
    partial_closes: int = 0
    scale_ins: int = 0


@dataclass
class ManagementDecision:
    """Result from position management decision."""
    action: PositionAction
    reason: str
    
    # For modifications
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    
    # For partial close
    close_volume: Optional[float] = None
    close_pct: Optional[float] = None
    
    # For scale in
    add_volume: Optional[float] = None
    
    # Meta
    r_multiple: float = 0.0
    trend_strength: float = 0.0
    ai_exit_signal: bool = False


# =============================================================================
# POSITION MANAGER
# =============================================================================

class PositionManager:
    """
    Fund-Grade Position Management Layer
    
    Decision Matrix (Aggressive Mode):
    ┌─────────┬────────────────────────────────────────────┐
    │ R-Value │ Actions                                    │
    ├─────────┼────────────────────────────────────────────┤
    │ < -0.5  │ Check invalidation, consider early exit    │
    │ -0.5-0  │ Monitor, hold position                     │
    │ 0.0-1.0 │ Monitor, prepare for BE                    │
    │ 1.0-1.5 │ Hold (aggressive waits for 1.5R BE)        │
    │ 1.5-2.0 │ Move to BE, lock profit                    │
    │ 2.0-2.5 │ Partial close 25%, start conservative trail│
    │ 2.5-3.5 │ Trail aggressive                           │
    │ > 3.5   │ Partial close 40%, full trailing           │
    └─────────┴────────────────────────────────────────────┘
    """
    
    def __init__(self, profile: TradingProfile = None):
        """Initialize with trading profile."""
        self.profile = profile or get_active_profile()
        
        # State tracking per position (ticket -> PositionStatus)
        self._position_states: Dict[int, PositionStatus] = {}
        
        # Statistics
        self.stats = {
            "decisions": 0,
            "be_moves": 0,
            "partial_closes": 0,
            "full_exits": 0,
            "scale_ins": 0,
        }
        
        logger.info(
            f"PositionManager initialized | Profile: {self.profile.name} | "
            f"BE@{self.profile.trailing.be_trigger_r}R, "
            f"Trail@{self.profile.trailing.trail_start_r}R"
        )
    
    # =========================================================================
    # MAIN MANAGEMENT FUNCTION
    # =========================================================================
    
    def manage_position(
        self,
        position,
        market_data: pd.DataFrame = None,
        ai_prediction: Dict = None,
        atr: float = 0.0
    ) -> ManagementDecision:
        """
        Active position management decision.
        
        Args:
            position: MT5 position object or PositionStatus
            market_data: DataFrame with recent price data
            ai_prediction: AI model prediction dict
            atr: Current ATR value
            
        Returns:
            ManagementDecision with action and parameters
        """
        self.stats["decisions"] += 1
        
        # Get or create position status
        status = self._get_position_status(position, atr)
        if status is None:
            return ManagementDecision(
                action=PositionAction.HOLD,
                reason="Invalid position"
            )
        
        # Calculate key metrics
        r = status.profit_r
        trend_strength = self._assess_trend(market_data) if market_data is not None else 0.5
        ai_exit = self._check_ai_exit(ai_prediction, status)
        
        # Log current state
        logger.debug(
            f"[{status.symbol}] #{status.ticket} | "
            f"R={r:.2f} | State={status.state.value} | "
            f"Trend={trend_strength:.2f} | AI_Exit={ai_exit}"
        )
        
        # Decision logic based on profile
        decision = self._make_decision(status, r, trend_strength, ai_exit, atr)
        
        # Update stats
        self._update_stats(decision.action)
        
        # Log decision
        if decision.action != PositionAction.HOLD:
            logger.info(
                f"[{status.symbol}] #{status.ticket} | "
                f"🎯 {decision.action.value.upper()} | "
                f"R={r:.2f} | {decision.reason}"
            )
        
        return decision
    
    def manage_all_positions(
        self,
        market_data: Dict[str, pd.DataFrame] = None,
        ai_predictions: Dict[str, Dict] = None,
        atr_dict: Dict[str, float] = None
    ) -> List[ManagementDecision]:
        """
        Manage all open positions.
        
        Args:
            market_data: Dict of symbol -> DataFrame
            ai_predictions: Dict of symbol -> prediction
            atr_dict: Dict of symbol -> ATR value
            
        Returns:
            List of decisions for each position
        """
        decisions = []
        
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return decisions
            
            for pos in positions:
                symbol = pos.symbol
                data = market_data.get(symbol) if market_data else None
                pred = ai_predictions.get(symbol) if ai_predictions else None
                atr = atr_dict.get(symbol, 0) if atr_dict else 0
                
                decision = self.manage_position(pos, data, pred, atr)
                decisions.append(decision)
                
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
        
        return decisions
    
    # =========================================================================
    # DECISION LOGIC
    # =========================================================================
    
    def _make_decision(
        self,
        status: PositionStatus,
        r: float,
        trend_strength: float,
        ai_exit: bool,
        atr: float
    ) -> ManagementDecision:
        """Make management decision based on current state and profile."""
        
        cfg = self.profile.trailing
        
        # === NEGATIVE R: Potential early exit ===
        if r < -0.5:
            # AI says exit and losing badly
            if ai_exit and r < -0.8:
                return ManagementDecision(
                    action=PositionAction.FULL_EXIT,
                    reason=f"AI exit signal + losing {abs(r):.1f}R",
                    r_multiple=r,
                    trend_strength=trend_strength,
                    ai_exit_signal=ai_exit
                )
            # Trend reversed strongly
            if trend_strength < 0.2:
                return ManagementDecision(
                    action=PositionAction.TIGHTEN_SL,
                    reason=f"Trend weak ({trend_strength:.0%}), tighten SL",
                    new_sl=self._calculate_tightened_sl(status, atr),
                    r_multiple=r,
                    trend_strength=trend_strength
                )
        
        # === BREAKEVEN ZONE (1.5R for aggressive) ===
        if r >= cfg.be_trigger_r and not status.be_applied:
            new_sl = status.entry_price
            return ManagementDecision(
                action=PositionAction.MOVE_BE,
                reason=f"R={r:.2f} >= {cfg.be_trigger_r}R, move to BE",
                new_sl=new_sl,
                r_multiple=r,
                trend_strength=trend_strength
            )
        
        # === FIRST PARTIAL CLOSE ===
        if r >= cfg.partial_close_1_r and status.partial_closes == 0:
            close_vol = status.volume * cfg.partial_close_1_pct
            return ManagementDecision(
                action=PositionAction.PARTIAL_CLOSE,
                reason=f"R={r:.2f} >= {cfg.partial_close_1_r}R, close {cfg.partial_close_1_pct:.0%}",
                close_volume=close_vol,
                close_pct=cfg.partial_close_1_pct,
                r_multiple=r,
                trend_strength=trend_strength
            )
        
        # === TRAILING ZONE ===
        if r >= cfg.trail_start_r:
            if trend_strength > 0.6:
                # Strong trend = aggressive trail
                new_sl = self._calculate_trailing_sl(status, atr, TrailingMode.AGGRESSIVE)
                return ManagementDecision(
                    action=PositionAction.TRAIL_AGGRESSIVE,
                    reason=f"R={r:.2f}, strong trend ({trend_strength:.0%}), aggressive trail",
                    new_sl=new_sl,
                    r_multiple=r,
                    trend_strength=trend_strength
                )
            else:
                # Weak trend = conservative trail
                new_sl = self._calculate_trailing_sl(status, atr, TrailingMode.CONSERVATIVE)
                return ManagementDecision(
                    action=PositionAction.TRAIL_CONSERVATIVE,
                    reason=f"R={r:.2f}, moderate trend, conservative trail",
                    new_sl=new_sl,
                    r_multiple=r,
                    trend_strength=trend_strength
                )
        
        # === SECOND PARTIAL CLOSE ===
        if r >= cfg.partial_close_2_r and status.partial_closes == 1:
            close_vol = status.volume * cfg.partial_close_2_pct
            return ManagementDecision(
                action=PositionAction.PARTIAL_CLOSE,
                reason=f"R={r:.2f} >= {cfg.partial_close_2_r}R, close {cfg.partial_close_2_pct:.0%}",
                close_volume=close_vol,
                close_pct=cfg.partial_close_2_pct,
                r_multiple=r,
                trend_strength=trend_strength
            )
        
        # === AI EXIT SIGNAL ===
        if ai_exit:
            if r > 1.0:
                # In profit, respect AI
                return ManagementDecision(
                    action=PositionAction.PARTIAL_CLOSE,
                    reason=f"AI exit signal at {r:.2f}R, partial close",
                    close_volume=status.volume * 0.5,
                    close_pct=0.5,
                    r_multiple=r,
                    trend_strength=trend_strength,
                    ai_exit_signal=True
                )
            elif r < 0 and trend_strength < 0.3:
                # Losing + AI exit + weak trend = cut
                return ManagementDecision(
                    action=PositionAction.FULL_EXIT,
                    reason=f"AI exit + weak trend + losing, cut position",
                    r_multiple=r,
                    trend_strength=trend_strength,
                    ai_exit_signal=True
                )
        
        # === SCALE IN (Aggressive only) ===
        if (self.profile.entry.pyramid_allowed and 
            r > 0.5 and 
            trend_strength > 0.7 and
            status.scale_ins < self.profile.entry.max_pyramid_levels):
            
            # Check if we should scale in
            # (This is just a signal, execution happens elsewhere)
            if self._should_scale_in(status, trend_strength):
                return ManagementDecision(
                    action=PositionAction.SCALE_IN,
                    reason=f"Strong trend ({trend_strength:.0%}), pyramid entry",
                    add_volume=status.volume * 0.5,  # Add 50% of original
                    r_multiple=r,
                    trend_strength=trend_strength
                )
        
        # === DEFAULT: ACTIVE HOLD ===
        return ManagementDecision(
            action=PositionAction.HOLD,
            reason=f"Monitoring | R={r:.2f} | Trend={trend_strength:.0%}",
            r_multiple=r,
            trend_strength=trend_strength,
            ai_exit_signal=ai_exit
        )
    
    # =========================================================================
    # CALCULATIONS
    # =========================================================================
    
    def _get_position_status(self, position, atr: float = 0) -> Optional[PositionStatus]:
        """Get or create position status."""
        try:
            # Handle both MT5 position object and dict
            if hasattr(position, 'ticket'):
                ticket = position.ticket
                symbol = position.symbol
                entry = position.price_open
                current_sl = position.sl
                current_tp = position.tp
                volume = position.volume
                pos_type = position.type
            else:
                ticket = position.get('ticket', 0)
                symbol = position.get('symbol', 'UNKNOWN')
                entry = position.get('price_open', 0)
                current_sl = position.get('sl', 0)
                current_tp = position.get('tp', 0)
                volume = position.get('volume', 0)
                pos_type = position.get('type', 0)
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                current_price = tick.bid if pos_type == 0 else tick.ask
            else:
                current_price = entry  # Fallback
            
            # Direction
            direction = "LONG" if pos_type == 0 else "SHORT"
            
            # Calculate profit in pips
            if direction == "LONG":
                profit_pips = (current_price - entry) / 0.01 if symbol.endswith("JPY") else (current_price - entry) / 0.0001
            else:
                profit_pips = (entry - current_price) / 0.01 if symbol.endswith("JPY") else (entry - current_price) / 0.0001
            
            # Calculate R-multiple
            if current_sl != 0:
                risk = abs(entry - current_sl)
            elif atr > 0:
                risk = atr * self.profile.sltp.atr_multiplier_sl
            else:
                risk = abs(entry * 0.005)  # Fallback 0.5%
            
            if direction == "LONG":
                profit_r = (current_price - entry) / risk if risk > 0 else 0
            else:
                profit_r = (entry - current_price) / risk if risk > 0 else 0
            
            # Get existing state or create new
            existing = self._position_states.get(ticket)
            if existing:
                state = existing.state
                be_applied = existing.be_applied
                partial_closes = existing.partial_closes
                scale_ins = existing.scale_ins
            else:
                state = PositionState.MONITORING
                be_applied = current_sl == entry  # Check if already at BE
                partial_closes = 0
                scale_ins = 0
            
            status = PositionStatus(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                current_price=current_price,
                current_sl=current_sl,
                current_tp=current_tp,
                volume=volume,
                profit_pips=profit_pips,
                profit_r=profit_r,
                state=state,
                be_applied=be_applied,
                partial_closes=partial_closes,
                scale_ins=scale_ins
            )
            
            # Store for tracking
            self._position_states[ticket] = status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting position status: {e}")
            return None
    
    def _calculate_r_multiple(self, entry: float, current: float, sl: float, direction: str) -> float:
        """Calculate current R-multiple."""
        risk = abs(entry - sl) if sl != 0 else abs(entry * 0.005)
        if risk == 0:
            return 0
            
        if direction == "LONG":
            return (current - entry) / risk
        else:
            return (entry - current) / risk
    
    def _calculate_trailing_sl(
        self,
        status: PositionStatus,
        atr: float,
        mode: TrailingMode
    ) -> float:
        """Calculate new trailing SL."""
        if atr <= 0:
            return status.current_sl
        
        # ATR multiplier based on mode
        multipliers = {
            TrailingMode.CONSERVATIVE: 2.0,
            TrailingMode.STANDARD: 1.5,
            TrailingMode.AGGRESSIVE: 1.0,
        }
        mult = multipliers.get(mode, self.profile.trailing.trail_atr_multiplier)
        distance = atr * mult
        
        if status.direction == "LONG":
            new_sl = status.current_price - distance
            # Never move SL backwards
            return max(new_sl, status.current_sl) if status.current_sl > 0 else new_sl
        else:
            new_sl = status.current_price + distance
            # Never move SL backwards (for SHORT, lower is better)
            return min(new_sl, status.current_sl) if status.current_sl > 0 else new_sl
    
    def _calculate_tightened_sl(self, status: PositionStatus, atr: float) -> float:
        """Calculate tightened SL for weak trend."""
        if atr <= 0 or status.current_sl == 0:
            return status.current_sl
        
        current_distance = abs(status.current_price - status.current_sl)
        new_distance = current_distance * 0.7  # Reduce by 30%
        
        if status.direction == "LONG":
            return status.current_price - new_distance
        else:
            return status.current_price + new_distance
    
    def _assess_trend(self, df: pd.DataFrame) -> float:
        """
        Assess trend strength from market data.
        
        Returns:
            0.0 - 1.0: trend strength (0 = no trend, 1 = strong trend)
        """
        if df is None or df.empty or len(df) < 20:
            return 0.5  # Neutral
        
        try:
            close = df['close'] if 'close' in df.columns else df.iloc[:, 3]
            
            # EMA momentum
            ema20 = close.ewm(span=20).mean()
            ema50 = close.ewm(span=50).mean()
            
            # Trend direction
            bullish = ema20.iloc[-1] > ema50.iloc[-1]
            
            # Momentum strength (EMA separation)
            separation = abs(ema20.iloc[-1] - ema50.iloc[-1]) / close.iloc[-1]
            
            # Price momentum (last 5 bars)
            momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            
            # Combine
            strength = min(1.0, abs(separation) * 100 + abs(momentum) * 10)
            
            return strength
            
        except Exception as e:
            logger.warning(f"Trend assessment failed: {e}")
            return 0.5
    
    def _check_ai_exit(self, ai_prediction: Dict, status: PositionStatus) -> bool:
        """
        Check if AI recommends exit.
        
        Returns:
            True if AI signals exit
        """
        if ai_prediction is None:
            return False
        
        try:
            ai_action = ai_prediction.get('action', ai_prediction.get('signal', ''))
            ai_confidence = ai_prediction.get('confidence', 0)
            
            # Threshold from profile (use 70% for aggressive)
            exit_threshold = 0.70
            
            # AI says opposite direction with high confidence
            if status.direction == "LONG":
                if ai_action in ['SELL', 'SHORT'] and ai_confidence > exit_threshold:
                    return True
            else:
                if ai_action in ['BUY', 'LONG'] and ai_confidence > exit_threshold:
                    return True
            
            # Trend strength from prediction
            trend_strength = ai_prediction.get('trend_strength', 0.5)
            if trend_strength < 0.3:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"AI exit check failed: {e}")
            return False
    
    def _should_scale_in(self, status: PositionStatus, trend_strength: float) -> bool:
        """Check if conditions are right for scale-in."""
        if not self.profile.entry.pyramid_allowed:
            return False
        if status.scale_ins >= self.profile.entry.max_pyramid_levels:
            return False
        if status.profit_r < 0.5:
            return False
        if trend_strength < 0.7:
            return False
        return True
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def update_position_state(self, ticket: int, **updates):
        """Update position state after action executed."""
        if ticket in self._position_states:
            status = self._position_states[ticket]
            
            if 'be_applied' in updates:
                status.be_applied = updates['be_applied']
            if 'partial_closes' in updates:
                status.partial_closes = updates['partial_closes']
            if 'scale_ins' in updates:
                status.scale_ins = updates['scale_ins']
            if 'state' in updates:
                status.state = updates['state']
    
    def remove_position(self, ticket: int):
        """Remove closed position from tracking."""
        if ticket in self._position_states:
            del self._position_states[ticket]
    
    def _update_stats(self, action: PositionAction):
        """Update statistics."""
        if action == PositionAction.MOVE_BE:
            self.stats["be_moves"] += 1
        elif action == PositionAction.PARTIAL_CLOSE:
            self.stats["partial_closes"] += 1
        elif action == PositionAction.FULL_EXIT:
            self.stats["full_exits"] += 1
        elif action == PositionAction.SCALE_IN:
            self.stats["scale_ins"] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {k: 0 for k in self.stats}


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_manager: Optional[PositionManager] = None

def get_position_manager() -> PositionManager:
    """Get global PositionManager instance."""
    global _manager
    if _manager is None:
        _manager = PositionManager()
    return _manager


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 POSITION MANAGER TEST")
    print("=" * 60)
    
    from src.config.trading_profiles import AGGRESSIVE_PROFILE
    
    pm = PositionManager(AGGRESSIVE_PROFILE)
    
    # Mock position data
    mock_positions = [
        {"ticket": 1001, "symbol": "XAUUSD", "type": 0, "price_open": 2650.0, 
         "sl": 2640.0, "tp": 2680.0, "volume": 0.1},
        {"ticket": 1002, "symbol": "XAUUSD", "type": 1, "price_open": 2650.0,
         "sl": 2660.0, "tp": 2620.0, "volume": 0.1},
    ]
    
    # Mock different R scenarios
    scenarios = [
        (2655.0, "R = 0.5"),
        (2660.0, "R = 1.0"),
        (2665.0, "R = 1.5 (BE trigger)"),
        (2670.0, "R = 2.0 (Partial 1)"),
        (2680.0, "R = 3.0 (Trail)"),
        (2645.0, "R = -0.5"),
    ]
    
    print("\n📊 Testing decision matrix with mock positions:")
    print("-" * 60)
    
    for price, desc in scenarios:
        print(f"\n🔹 Scenario: {desc}")
        
        # Simulate tick update
        class MockTick:
            bid = price
            ask = price
        
        # Patch mt5.symbol_info_tick
        original_tick = mt5.symbol_info_tick
        mt5.symbol_info_tick = lambda s: MockTick()
        
        decision = pm.manage_position(mock_positions[0], atr=5.0)
        
        print(f"   Action: {decision.action.value}")
        print(f"   Reason: {decision.reason}")
        if decision.new_sl:
            print(f"   New SL: {decision.new_sl:.2f}")
        
        # Restore
        mt5.symbol_info_tick = original_tick
    
    print("\n" + "=" * 60)
    print("📈 Stats:", pm.get_stats())
    print("=" * 60)
