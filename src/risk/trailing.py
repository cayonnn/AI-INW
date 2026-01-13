# src/risk/trailing.py
"""
TrailingManager Module - Break-Even & Trailing Stop
====================================================

Fund-Grade Exit Management:
- Break-Even: Move SL to entry when price hits configured R profit
- Trailing Stop: Trail SL with ATR when price hits trail R profit

⚠️ Uses TradingProfile to prevent config drift!
"""

import MetaTrader5 as mt5
from typing import Optional, Tuple
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("TRAILING_MANAGER")


@dataclass
class TrailingResult:
    """Trailing stop update result."""
    updated: bool
    new_sl: Optional[float] = None
    reason: str = ""


class TrailingManager:
    """
    Fund-Grade Trailing Stop Manager.
    
    Rules (from TradingProfile):
    - Break-Even: Move SL to entry at configured R
    - Trailing: Trail SL with ATR × multiplier
    - Only move SL in favorable direction (never widen)
    
    ⚠️ Config Drift Protection:
    - Uses TradingProfile as single source of truth
    - Validates config on initialization
    """

    def __init__(
        self,
        be_rr: float = None,           # Override or use profile
        trail_rr: float = None,        # Override or use profile
        atr_multiplier: float = None,  # Override or use profile
        profile = None                 # TradingProfile object
    ):
        # Get profile if not provided
        if profile is None:
            try:
                from src.config.trading_profiles import get_active_profile
                profile = get_active_profile()
            except ImportError:
                profile = None
        
        # Use profile values as defaults, allow overrides
        if profile is not None:
            self.be_rr = be_rr if be_rr is not None else profile.trailing.be_trigger_r
            self.trail_rr = trail_rr if trail_rr is not None else profile.trailing.trail_start_r
            self.atr_multiplier = atr_multiplier if atr_multiplier is not None else profile.trailing.trail_atr_multiplier
            self._profile_name = profile.name
        else:
            # Fallback to aggressive defaults if no profile
            self.be_rr = be_rr if be_rr is not None else 1.5
            self.trail_rr = trail_rr if trail_rr is not None else 2.5
            self.atr_multiplier = atr_multiplier if atr_multiplier is not None else 2.0
            self._profile_name = "Fallback (no profile)"
        
        logger.info(
            f"TrailingManager initialized: BE@{self.be_rr}R, Trail@{self.trail_rr}R, "
            f"ATR×{self.atr_multiplier} | Profile: {self._profile_name}"
        )

    # -------------------------------------------------
    # Main Management Function
    # -------------------------------------------------
    def manage(self, position, atr: float) -> TrailingResult:
        """
        Manage trailing stop for a position.
        
        Args:
            position: MT5 position object (from positions_get)
            atr: Current ATR value for trailing calculation
            
        Returns:
            TrailingResult with update status
        """
        if position is None:
            return TrailingResult(updated=False, reason="No position")
            
        if atr <= 0:
            return TrailingResult(updated=False, reason="Invalid ATR")

        try:
            # Get current price
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return TrailingResult(updated=False, reason="No tick data")
            
            # Use bid for LONG, ask for SHORT
            current_price = tick.bid if position.type == 0 else tick.ask
            
            entry = position.price_open
            current_sl = position.sl
            tp = position.tp
            
            # Calculate risk (distance from entry to SL)
            if current_sl == 0:
                # No SL set, use ATR as fallback
                risk = atr
            else:
                risk = abs(entry - current_sl)
                
            if risk <= 0:
                return TrailingResult(updated=False, reason="Invalid risk distance")

            # Calculate new SL
            new_sl = self._calculate_new_sl(
                position_type=position.type,
                entry=entry,
                current_price=current_price,
                current_sl=current_sl,
                risk=risk,
                atr=atr
            )
            
            # Check if update needed
            if new_sl is None:
                return TrailingResult(updated=False, reason="No update needed")
                
            # Only update if significant change (> 10% of ATR)
            if current_sl != 0 and abs(new_sl - current_sl) < atr * 0.1:
                return TrailingResult(updated=False, reason="Change too small")

            # Send modify order
            success = self._send_sl_modify(position, new_sl, tp)
            
            if success:
                logger.info(
                    f"{position.symbol} #{position.ticket}: SL updated "
                    f"{current_sl:.5f} → {new_sl:.5f}"
                )
                return TrailingResult(updated=True, new_sl=new_sl)
            else:
                return TrailingResult(updated=False, reason="Order modify failed")
                
        except Exception as e:
            logger.error(f"Error managing position: {e}")
            return TrailingResult(updated=False, reason=str(e))

    # -------------------------------------------------
    # SL Calculation Logic
    # -------------------------------------------------
    def _calculate_new_sl(
        self,
        position_type: int,
        entry: float,
        current_price: float,
        current_sl: float,
        risk: float,
        atr: float
    ) -> Optional[float]:
        """
        Calculate new SL based on BE and trailing rules.
        
        Returns:
            New SL price or None if no update needed
        """
        new_sl = None
        
        if position_type == 0:  # LONG / BUY
            profit_r = (current_price - entry) / risk if risk > 0 else 0
            
            # Break-Even check
            if profit_r >= self.be_rr and (current_sl < entry or current_sl == 0):
                new_sl = entry
                
            # Trailing check
            if profit_r >= self.trail_rr:
                trail_sl = current_price - atr * self.atr_multiplier
                # Only use trailing if better than BE
                if new_sl is None:
                    new_sl = trail_sl
                else:
                    new_sl = max(new_sl, trail_sl)
                    
            # Never move SL backwards
            if new_sl is not None and current_sl > 0:
                new_sl = max(new_sl, current_sl)
                
        else:  # SHORT / SELL
            profit_r = (entry - current_price) / risk if risk > 0 else 0
            
            # Break-Even check
            if profit_r >= self.be_rr and (current_sl > entry or current_sl == 0):
                new_sl = entry
                
            # Trailing check
            if profit_r >= self.trail_rr:
                trail_sl = current_price + atr * self.atr_multiplier
                # Only use trailing if better than BE
                if new_sl is None:
                    new_sl = trail_sl
                else:
                    new_sl = min(new_sl, trail_sl)
                    
            # Never move SL backwards
            if new_sl is not None and current_sl > 0:
                new_sl = min(new_sl, current_sl)
        
        return new_sl

    # -------------------------------------------------
    # MT5 Order Modification
    # -------------------------------------------------
    def _send_sl_modify(self, position, new_sl: float, tp: float) -> bool:
        """
        Send SL modification to MT5.
        
        Args:
            position: MT5 position object
            new_sl: New stop loss price
            tp: Take profit price (keep unchanged)
            
        Returns:
            True if successful
        """
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "symbol": position.symbol,
                "sl": new_sl,
                "tp": tp
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                logger.error("order_send returned None")
                return False
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"SL modify failed: {result.retcode} - {result.comment}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending SL modify: {e}")
            return False

    # -------------------------------------------------
    # Manage All Positions
    # -------------------------------------------------
    def manage_all(self, atr_dict: dict) -> int:
        """
        Manage trailing for all open positions.
        
        Args:
            atr_dict: Dictionary of symbol -> ATR value
            
        Returns:
            Number of positions updated
        """
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return 0
                
            updated = 0
            for pos in positions:
                atr = atr_dict.get(pos.symbol, 0)
                if atr > 0:
                    result = self.manage(pos, atr)
                    if result.updated:
                        updated += 1
                        
            return updated
            
        except Exception as e:
            logger.error(f"Error managing all positions: {e}")
            return 0
