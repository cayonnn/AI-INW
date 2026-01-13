# src/risk/risk_manager.py
"""
RiskManager Module - Fund-Grade Position Sizing & Risk Control
===============================================================

MT5-Integrated Risk Management:
- Risk-based position sizing (% of equity)
- Max positions per symbol control
- Max daily loss guard

This is the PRINCIPAL that controls all trading.
"""

import MetaTrader5 as mt5
from typing import Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("RISK_MANAGER")


@dataclass
class RiskCheckResult:
    """Risk check result."""
    can_trade: bool
    lot_size: float
    rejection_reason: str = ""


class RiskManager:
    """
    Fund-Grade Risk Manager.
    
    Responsibilities:
    - Calculate lot size based on risk per trade
    - Control max positions per symbol
    - Guard daily loss limit
    """

    def __init__(
        self,
        risk_per_trade: float = 0.005,      # 0.5% risk per trade
        max_positions_per_symbol: int = 3,   # Max 3 positions per symbol
        max_daily_loss_pct: float = 0.03     # 3% max daily loss
    ):
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions_per_symbol
        self.max_daily_loss_pct = max_daily_loss_pct
        
        # Track daily P/L
        self._daily_pnl = 0.0
        self._start_equity = None
        
        logger.info(
            f"RiskManager initialized: risk={risk_per_trade:.1%}, "
            f"max_pos={max_positions_per_symbol}, max_loss={max_daily_loss_pct:.1%}"
        )

    # -------------------------------------------------
    # Position Sizing
    # -------------------------------------------------
    def calc_lot(self, sl_points: float, symbol: str) -> float:
        """
        Calculate lot size based on risk percentage of equity.
        
        Formula: lot = (Equity × Risk%) / (SL points / tick_size × tick_value)
        
        Args:
            sl_points: Stop loss distance in points (price difference)
            symbol: Trading symbol
            
        Returns:
            Calculated lot size (min 0.01)
        """
        if sl_points <= 0:
            logger.warning(f"Invalid SL points: {sl_points}")
            return 0.01
            
        try:
            # Get account info
            acc = mt5.account_info()
            if acc is None:
                logger.error("Failed to get account info")
                return 0.01
                
            equity = acc.equity
            
            # Initialize start equity for daily tracking
            if self._start_equity is None:
                self._start_equity = equity
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info: {symbol}")
                return 0.01
                
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            
            if tick_size <= 0 or tick_value <= 0:
                logger.error(f"Invalid tick info: size={tick_size}, value={tick_value}")
                return 0.01
            
            # Calculate risk amount
            risk_amount = equity * self.risk_per_trade
            
            # Cost per lot for SL distance
            ticks_in_sl = sl_points / tick_size
            cost_per_lot = ticks_in_sl * tick_value
            
            if cost_per_lot <= 0:
                logger.error(f"Invalid cost per lot: {cost_per_lot}")
                return 0.01
            
            # Calculate lot size
            lot = risk_amount / cost_per_lot
            
            # Clamp to symbol limits
            lot = max(symbol_info.volume_min, lot)
            lot = min(symbol_info.volume_max, lot)
            
            # Round to step
            step = symbol_info.volume_step
            if step > 0:
                lot = round(lot / step) * step
            
            lot = round(lot, 2)
            
            logger.info(
                f"{symbol}: lot={lot:.2f}, equity={equity:.2f}, "
                f"risk={risk_amount:.2f}, sl_pts={sl_points:.5f}"
            )
            
            return lot
            
        except Exception as e:
            logger.error(f"Error calculating lot: {e}")
            return 0.01

    # -------------------------------------------------
    # Trade Permission Checks
    # -------------------------------------------------
    def can_trade(self, symbol: str) -> bool:
        """
        Check if trading is allowed for this symbol.
        
        Checks:
        1. Max positions per symbol
        2. Daily loss limit
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if trading allowed
        """
        # Check max positions
        if not self._check_max_positions(symbol):
            return False
            
        # Check daily loss
        if not self._check_daily_loss():
            return False
            
        return True

    def _check_max_positions(self, symbol: str) -> bool:
        """Check if max positions for symbol reached."""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is not None and len(positions) >= self.max_positions:
                logger.warning(
                    f"{symbol}: Max positions reached ({len(positions)}/{self.max_positions})"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return True  # Allow on error

    def _check_daily_loss(self) -> bool:
        """Check if daily loss limit reached."""
        try:
            if self._start_equity is None:
                return True
                
            acc = mt5.account_info()
            if acc is None:
                return True
                
            current_equity = acc.equity
            daily_loss = (self._start_equity - current_equity) / self._start_equity
            
            if daily_loss >= self.max_daily_loss_pct:
                logger.warning(
                    f"Daily loss limit reached: {daily_loss:.2%} >= {self.max_daily_loss_pct:.2%}"
                )
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking daily loss: {e}")
            return True  # Allow on error

    # -------------------------------------------------
    # Full Risk Check
    # -------------------------------------------------
    def check_and_size(self, symbol: str, sl_points: float) -> RiskCheckResult:
        """
        Combined check: can trade + calculate lot size.
        
        Args:
            symbol: Trading symbol
            sl_points: Stop loss distance in points
            
        Returns:
            RiskCheckResult with permission and lot size
        """
        if not self.can_trade(symbol):
            return RiskCheckResult(
                can_trade=False,
                lot_size=0.0,
                rejection_reason="Trading blocked by risk manager"
            )
            
        lot = self.calc_lot(sl_points, symbol)
        
        if lot <= 0:
            return RiskCheckResult(
                can_trade=False,
                lot_size=0.0,
                rejection_reason="Invalid lot size calculation"
            )
            
        return RiskCheckResult(
            can_trade=True,
            lot_size=lot
        )

    # -------------------------------------------------
    # Daily Reset
    # -------------------------------------------------
    def reset_daily(self):
        """Reset daily tracking (call at start of new trading day)."""
        try:
            acc = mt5.account_info()
            if acc is not None:
                self._start_equity = acc.equity
                self._daily_pnl = 0.0
                logger.info(f"Daily reset: start_equity={self._start_equity:.2f}")
        except Exception as e:
            logger.error(f"Error resetting daily: {e}")

    def update_pnl(self, pnl: float):
        """Update daily P/L tracking."""
        self._daily_pnl += pnl
