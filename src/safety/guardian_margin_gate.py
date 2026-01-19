# src/safety/guardian_margin_gate.py
"""
Guardian Margin Gate - Hard Margin Protection Layer
=====================================================

Purpose:
- Prevent OrderSend retcode=10019 (No money)
- Daily drawdown limit with latch
- Margin sufficiency check (order_calc_margin)
- Act as HARD GATE before live trading

This is a HARD GATE - cannot be bypassed.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime, date
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("MARGIN_GATE")

# Try to import MT5
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    logger.warning("MetaTrader5 not available, using mock mode")


class GuardianDecision(Enum):
    """Guardian decision types."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    CLAMP = "CLAMP"


@dataclass
class GuardianResult:
    """Result from Guardian evaluation."""
    decision: GuardianDecision
    allowed_lot: float
    reason: str
    margin_required: float
    margin_free: float
    original_lot: float = 0.0


class GuardianMarginGate:
    """
    Hard Margin Protection Guardian.
    
    Prevents 'No money' errors by:
    1. Daily drawdown tracking with latch
    2. Margin check via order_calc_margin before order
    3. Clamping lot size if needed
    4. Blocking order if margin insufficient
    
    This is a HARD GATE - cannot be bypassed.
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        min_lot: float = 0.01,
        lot_step: float = 0.01,
        safety_buffer: float = 1.10,
        daily_loss_limit: float = 0.10,  # 10% max daily loss
        magic: int = 900005,
        verbose: bool = True,
    ):
        """
        Initialize Guardian Margin Gate.
        
        Args:
            symbol: Trading symbol
            min_lot: Minimum allowed lot
            lot_step: Lot size step
            safety_buffer: Margin safety multiplier (1.1 = 10% extra)
            daily_loss_limit: Max daily loss before latch (0.10 = 10%)
            magic: Magic number for orders
            verbose: Log decisions
        """
        self.symbol = symbol
        self.min_lot = min_lot
        self.lot_step = lot_step
        self.safety_buffer = safety_buffer
        self.daily_loss_limit = daily_loss_limit
        self.magic = magic
        self.verbose = verbose
        
        # Daily tracking
        self.start_balance: Optional[float] = None
        self.current_day = date.today()
        
        # DD LATCH STATE
        self.dd_latched = False
        self.dd_latch_reason: Optional[str] = None
        self.latch_time: Optional[datetime] = None
        self._latch_warned = False  # Prevent log spam
        
        self._mt5_initialized = False
        self._init_mt5()
        
        logger.info(
            f"GuardianMarginGate initialized: {symbol}, "
            f"buffer={safety_buffer:.0%}, DD_limit={daily_loss_limit:.0%}"
        )
    
    def _init_mt5(self) -> bool:
        """Initialize MT5 connection."""
        if not HAS_MT5:
            return False
        
        try:
            if not mt5.initialize():
                logger.warning("MT5 initialization failed")
                return False
            
            self.symbol_info = mt5.symbol_info(self.symbol)
            if self.symbol_info is None:
                logger.warning(f"Symbol not found: {self.symbol}")
                return False
            
            self._mt5_initialized = True
            return True
        except Exception as e:
            logger.error(f"MT5 init error: {e}")
            return False

    # --------------------------------------------------
    # Daily reset
    # --------------------------------------------------
    def daily_reset_if_needed(self, log=None):
        """Check and perform daily reset if new day."""
        today = date.today()
        if today != self.current_day:
            self.current_day = today
            self.start_balance = None
            self.dd_latched = False
            self.dd_latch_reason = None
            self.latch_time = None
            self._latch_warned = False
            
            if log:
                log.info("🧹 GuardianMarginGate DAILY RESET")
            else:
                logger.info("🧹 GuardianMarginGate DAILY RESET")

    def reset_daily(self) -> None:
        """Manual reset latch for new trading day."""
        if self.dd_latched:
            logger.info("🔓 Guardian Latch RESET (New Day)")
        
        self.start_balance = None
        self.dd_latched = False
        self.dd_latch_reason = None
        self.latch_time = None
        self._latch_warned = False
        self.current_day = date.today()

    def reset_new_day(self) -> None:
        """Alias for reset_daily."""
        self.reset_daily()
    
    # --------------------------------------------------
    # Core DD tracking
    # --------------------------------------------------
    def _update_start_balance(self, account):
        """Capture start of day balance."""
        if self.start_balance is None:
            self.start_balance = account.balance
            logger.info(f"📊 Start balance captured: ${self.start_balance:.2f}")
    
    def _daily_dd(self, account) -> float:
        """Calculate daily drawdown percentage."""
        if self.start_balance is None or self.start_balance <= 0:
            return 0.0
        return max(0.0, (self.start_balance - account.equity) / self.start_balance)
    
    def update_dd(self, daily_dd_percent: float) -> None:
        """
        Update daily drawdown state from external source.
        Triggers latch if limit exceeded.
        
        Args:
            daily_dd_percent: DD as percentage (e.g., 10.5 for 10.5%)
        """
        if self.dd_latched:
            return

        # Convert to same scale as limit
        limit_val = self.daily_loss_limit * 100  # 0.10 -> 10.0
        
        if daily_dd_percent >= limit_val:
            self.dd_latched = True
            self.dd_latch_reason = f"Daily DD limit exceeded: {daily_dd_percent:.2f}% >= {limit_val:.2f}%"
            self.latch_time = datetime.utcnow()
            self._trigger_latch()

    def _trigger_latch(self):
        """Trigger hard latch."""
        logger.critical("=" * 50)
        logger.critical("🔒 GUARDIAN HARD LATCH TRIGGERED")
        logger.critical(f"   Reason: {self.dd_latch_reason}")
        logger.critical("   Action: ALL TRADING BLOCKED UNTIL NEXT DAY")
        logger.critical("=" * 50)

    def allow_trade(self) -> bool:
        """Check if trading is allowed (not latched)."""
        if self.dd_latched and not self._latch_warned:
            logger.critical(f"🛑 TRADING BLOCKED BY GUARDIAN: {self.dd_latch_reason}")
            self._latch_warned = True
        return not self.dd_latched

    def can_trade_ex(self, action: str, volume: float, price: float) -> tuple:
        """
        Extended can_trade check with reason.
        
        Returns:
            (bool, reason): (True, None) if OK, (False, "REASON") if blocked
        """
        # Daily reset check
        self.daily_reset_if_needed()
        
        # Check latch
        if self.dd_latched:
            return False, "LATCHED"
        
        if not HAS_MT5 or not self._mt5_initialized:
            return True, None  # Mock mode allows
        
        try:
            account = mt5.account_info()
            if account is None:
                return False, "NO_ACCOUNT"
            
            # Update start balance
            self._update_start_balance(account)
            
            # Check daily DD
            daily_dd = self._daily_dd(account)
            if daily_dd >= self.daily_loss_limit:
                self.dd_latched = True
                self.dd_latch_reason = f"Daily DD: {daily_dd*100:.2f}%"
                self._trigger_latch()
                return False, "DAILY_DD"
            
            # Check margin
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return False, "NO_TICK"
            
            mt5_order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if action == "BUY" else tick.bid
            
            margin_required = self._calc_margin(mt5_order_type, volume, price)
            if margin_required is None:
                return False, "MARGIN_CALC_FAIL"
            
            if account.margin_free < margin_required * self.safety_buffer:
                # Try to find minimum viable lot
                min_margin = self._calc_margin(mt5_order_type, self.min_lot, price)
                if min_margin and account.margin_free >= min_margin * self.safety_buffer:
                    return True, "CLAMP_NEEDED"
                return False, "MARGIN"
            
            return True, None
            
        except Exception as e:
            return False, f"ERROR:{e}"


    def is_latched(self) -> bool:
        """Check if latch is active."""
        return self.dd_latched

    def status(self) -> dict:
        """Get current status."""
        return {
            "is_latched": self.dd_latched,
            "reason": self.dd_latch_reason,
            "daily_limit": self.daily_loss_limit,
            "start_balance": self.start_balance,
            "current_day": str(self.current_day),
        }

    def allow_retrain(self) -> bool:
        """Block retraining if we are in a dangerous state."""
        return not self.dd_latched

    def on_trade_result(self, result):
        """Update state based on trade result (loss tracking)."""
        pass
    
    # --------------------------------------------------
    # Margin evaluation
    # --------------------------------------------------
    def evaluate(
        self,
        desired_lot: float,
        order_type: str = "BUY"
    ) -> GuardianResult:
        """
        Evaluate if trade is allowed based on margin AND latch state.
        
        Args:
            desired_lot: Requested lot size
            order_type: "BUY" or "SELL"
        
        Returns:
            GuardianResult with decision and allowed lot
        """
        # 0. DAILY RESET CHECK
        self.daily_reset_if_needed()
        
        # 1. CHECK LATCH FIRST
        if not self.allow_trade():
            return self._block(
                reason=f"LATCHED: {self.dd_latch_reason}",
                margin_required=0,
                margin_free=0,
                original_lot=desired_lot
            )

        if not HAS_MT5 or not self._mt5_initialized:
            # Mock mode - allow with warning
            return GuardianResult(
                decision=GuardianDecision.ALLOW,
                allowed_lot=desired_lot,
                reason="MOCK_MODE_NO_MT5",
                margin_required=0,
                margin_free=0,
                original_lot=desired_lot
            )
        
        try:
            account = mt5.account_info()
            tick = mt5.symbol_info_tick(self.symbol)
            
            if account is None or tick is None:
                return self._block(
                    reason="ACCOUNT_OR_TICK_UNAVAILABLE",
                    margin_required=0,
                    margin_free=0,
                    original_lot=desired_lot
                )
            
            # Update start balance on first call of the day
            self._update_start_balance(account)
            
            # Check daily DD
            daily_dd = self._daily_dd(account)
            if daily_dd >= self.daily_loss_limit:
                self.dd_latched = True
                self.dd_latch_reason = f"Daily DD: {daily_dd*100:.2f}% >= {self.daily_loss_limit*100:.2f}%"
                self._trigger_latch()
                return self._block(
                    reason=self.dd_latch_reason,
                    margin_required=0,
                    margin_free=account.margin_free,
                    original_lot=desired_lot
                )
            
            margin_free = account.margin_free
            mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if order_type == "BUY" else tick.bid
            
            # Step 1: Check desired lot
            margin_required = self._calc_margin(mt5_order_type, desired_lot, price)
            
            if margin_required is None:
                return self._block(
                    reason="MARGIN_CALC_FAILED",
                    margin_required=0,
                    margin_free=margin_free,
                    original_lot=desired_lot
                )
            
            # Check if margin is sufficient
            if margin_free > margin_required * self.safety_buffer:
                return self._allow(
                    desired_lot,
                    reason="MARGIN_OK",
                    margin_required=margin_required,
                    margin_free=margin_free,
                    original_lot=desired_lot
                )
            
            # Step 2: Try to clamp lot
            clamped_lot, clamped_margin = self._clamp_lot(
                mt5_order_type, desired_lot, price, margin_free
            )
            
            if clamped_lot >= self.min_lot:
                return self._clamp(
                    clamped_lot,
                    reason="LOT_CLAMPED_BY_MARGIN",
                    margin_required=clamped_margin,
                    margin_free=margin_free,
                    original_lot=desired_lot
                )
            
            # Step 3: Hard block
            return self._block(
                reason="INSUFFICIENT_MARGIN_HARD_BLOCK",
                margin_required=margin_required,
                margin_free=margin_free,
                original_lot=desired_lot
            )
            
        except Exception as e:
            logger.error(f"Margin evaluation error: {e}")
            return self._block(
                reason=f"ERROR: {e}",
                margin_required=0,
                margin_free=0,
                original_lot=desired_lot
            )
    
    def _calc_margin(
        self,
        order_type: int,
        lot: float,
        price: float
    ) -> Optional[float]:
        """Calculate margin required."""
        try:
            return mt5.order_calc_margin(
                order_type,
                self.symbol,
                lot,
                price
            )
        except:
            return None
    
    def _clamp_lot(
        self,
        order_type: int,
        desired_lot: float,
        price: float,
        margin_free: float
    ) -> tuple:
        """Clamp lot to fit within margin."""
        lot = round(desired_lot, 2)
        
        while lot >= self.min_lot:
            margin = self._calc_margin(order_type, lot, price)
            if margin and margin_free > margin * self.safety_buffer:
                return lot, margin
            lot = round(lot - self.lot_step, 2)
        
        return 0.0, 0.0
    
    def _allow(
        self,
        lot: float,
        reason: str,
        margin_required: float,
        margin_free: float,
        original_lot: float
    ) -> GuardianResult:
        """Allow trade."""
        if self.verbose:
            logger.info(
                f"✅ MARGIN ALLOW | lot={lot:.2f} | "
                f"req={margin_required:.0f} | free={margin_free:.0f}"
            )
        
        return GuardianResult(
            decision=GuardianDecision.ALLOW,
            allowed_lot=lot,
            reason=reason,
            margin_required=margin_required,
            margin_free=margin_free,
            original_lot=original_lot
        )
    
    def _clamp(
        self,
        lot: float,
        reason: str,
        margin_required: float,
        margin_free: float,
        original_lot: float
    ) -> GuardianResult:
        """Allow with clamped lot."""
        if self.verbose:
            logger.warning(
                f"⚠️ MARGIN CLAMP | {original_lot:.2f}→{lot:.2f} | "
                f"req={margin_required:.0f} | free={margin_free:.0f}"
            )
        
        return GuardianResult(
            decision=GuardianDecision.CLAMP,
            allowed_lot=lot,
            reason=reason,
            margin_required=margin_required,
            margin_free=margin_free,
            original_lot=original_lot
        )
    
    def _block(
        self,
        reason: str,
        margin_required: float,
        margin_free: float,
        original_lot: float
    ) -> GuardianResult:
        """Block trade."""
        if self.verbose:
            logger.warning(
                f"🚫 MARGIN BLOCK | lot={original_lot:.2f} | "
                f"req={margin_required:.0f} | free={margin_free:.0f} | {reason}"
            )
        
        return GuardianResult(
            decision=GuardianDecision.BLOCK,
            allowed_lot=0.0,
            reason=reason,
            margin_required=margin_required,
            margin_free=margin_free,
            original_lot=original_lot
        )
    
    def get_max_lot(self, order_type: str = "BUY") -> float:
        """Get maximum lot size for current margin."""
        if not HAS_MT5 or not self._mt5_initialized:
            return 0.01
        
        try:
            account = mt5.account_info()
            tick = mt5.symbol_info_tick(self.symbol)
            
            if not account or not tick:
                return 0.01
            
            margin_free = account.margin_free
            mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if order_type == "BUY" else tick.bid
            
            # Binary search for max lot
            max_lot = 0.01
            for lot in [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01]:
                margin = self._calc_margin(mt5_order_type, lot, price)
                if margin and margin_free > margin * self.safety_buffer:
                    max_lot = lot
                    break
            
            return max_lot
        except:
            return 0.01


# Singleton
_gate: Optional[GuardianMarginGate] = None


def get_margin_gate(symbol: str = "XAUUSD") -> GuardianMarginGate:
    """Get singleton margin gate."""
    global _gate
    if _gate is None:
        _gate = GuardianMarginGate(symbol)
    return _gate


def check_margin(desired_lot: float, order_type: str = "BUY") -> GuardianResult:
    """Convenience function to check margin."""
    gate = get_margin_gate()
    return gate.evaluate(desired_lot, order_type)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Guardian Margin Gate Test")
    print("=" * 50)
    
    guardian = GuardianMarginGate(
        symbol="XAUUSD",
        safety_buffer=1.15,
        verbose=True
    )
    
    test_lots = [0.20, 0.15, 0.10, 0.05, 0.01]
    
    for lot in test_lots:
        print(f"\nTesting desired lot: {lot}")
        result = guardian.evaluate(lot)
        
        print(f"  Decision: {result.decision.value}")
        print(f"  Allowed lot: {result.allowed_lot}")
        print(f"  Reason: {result.reason}")
    
    print(f"\nMax safe lot: {guardian.get_max_lot()}")
