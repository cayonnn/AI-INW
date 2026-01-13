# src/core/capital_allocation_engine.py
"""
Capital Allocation Engine (CAE) - Fund-Grade
=============================================

จัดสรรทุน อัตโนมัติ + ปลอดภัย + คุมความเสี่ยงระดับพอร์ต

Core Responsibilities:
- กำหนด lot size ต่อ symbol / strategy
- คุม max exposure ต่อวัน / ต่อคู่เงิน
- ปรับทุนตาม model confidence + drawdown

Formula:
    risk_per_trade = equity × base_risk × confidence_score
    lot = risk_per_trade / (SL_pips × pip_value)

Safety Rules (บังคับ):
- ❌ ห้ามเปิดออเดอร์ถ้า SL/TP ไม่ครบ
- ❌ ห้ามเกิน Max Daily Loss
- ❌ ห้าม stacking direction เดียวเกิน N order
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime, date

logger = logging.getLogger("CAE")


@dataclass
class AllocationResult:
    """Capital allocation result."""
    approved: bool
    symbol: str
    lot: float
    sl: float
    tp: float
    risk_pct: float
    rejection_reason: str = ""


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str  # TREND, RANGE, HIGH_VOL, LOW_VOL
    strength: float  # 0-1
    

class CapitalAllocationEngine:
    """
    Fund-Grade Capital Allocation Engine.
    
    Allocates capital based on:
    - Account equity & free margin
    - Model signal strength (confidence)
    - Historical win rate & drawdown
    - Market regime
    """

    def __init__(
        self,
        base_risk: float = 0.005,           # 0.5% base risk per trade
        max_risk: float = 0.02,             # 2% max risk per trade
        max_daily_loss: float = 0.03,       # 3% max daily loss
        max_exposure_per_symbol: float = 0.1,  # 10% max exposure per symbol
        max_orders_same_direction: int = 3,  # Max orders same direction
        min_lot: float = 0.01,
        max_lot: float = 10.0
    ):
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.max_daily_loss = max_daily_loss
        self.max_exposure_per_symbol = max_exposure_per_symbol
        self.max_orders_same_direction = max_orders_same_direction
        self.min_lot = min_lot
        self.max_lot = max_lot
        
        # Daily tracking
        self._daily_pnl = 0.0
        self._start_equity = None
        self._trade_date = date.today()
        
        # Order tracking
        self._open_orders: Dict[str, List[Dict]] = {}  # symbol -> [orders]
        
        logger.info(
            f"CAE initialized: base_risk={base_risk:.1%}, max_daily_loss={max_daily_loss:.1%}"
        )

    # =========================================================
    # MAIN ALLOCATION
    # =========================================================
    
    def allocate(
        self,
        symbol: str,
        direction: str,
        equity: float,
        confidence: float,
        sl_pips: float,
        tp_pips: float,
        pip_value: float = 1.0,
        market_regime: Optional[MarketRegime] = None
    ) -> AllocationResult:
        """
        Allocate capital for a trade.
        
        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            equity: Account equity
            confidence: Model confidence (0-1)
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
            pip_value: Value per pip per lot
            market_regime: Current market regime
            
        Returns:
            AllocationResult with lot size and parameters
        """
        # Initialize daily tracking
        self._check_daily_reset(equity)
        
        # =========================================================
        # SAFETY RULES (บังคับ)
        # =========================================================
        
        # Rule 1: SL/TP ต้องครบ
        if sl_pips <= 0 or tp_pips <= 0:
            return self._reject(symbol, "❌ SL/TP ไม่ครบ - ต้องระบุทั้งคู่")
        
        # Rule 2: Max Daily Loss
        if not self._check_daily_loss(equity):
            return self._reject(symbol, f"❌ เกิน Max Daily Loss ({self.max_daily_loss:.1%})")
        
        # Rule 3: Max orders same direction
        if not self._check_direction_stacking(symbol, direction):
            return self._reject(
                symbol, 
                f"❌ Stacking เกิน {self.max_orders_same_direction} orders ใน direction เดียว"
            )
        
        # Rule 4: Max exposure per symbol
        if not self._check_symbol_exposure(symbol, equity):
            return self._reject(symbol, f"❌ Exposure เกิน {self.max_exposure_per_symbol:.1%}")
        
        # =========================================================
        # ALLOCATION FORMULA
        # =========================================================
        
        # Adjust risk based on confidence
        adjusted_risk = self._calculate_adjusted_risk(confidence, market_regime)
        
        # Calculate risk amount
        risk_amount = equity * adjusted_risk
        
        # Calculate lot size
        sl_cost_per_lot = sl_pips * pip_value
        if sl_cost_per_lot <= 0:
            return self._reject(symbol, "❌ Invalid SL cost calculation")
        
        lot = risk_amount / sl_cost_per_lot
        
        # Clamp lot size
        lot = max(self.min_lot, min(lot, self.max_lot))
        lot = round(lot, 2)
        
        # Calculate SL/TP prices (placeholder - actual prices calculated elsewhere)
        sl_price = sl_pips  # Will be calculated by caller
        tp_price = tp_pips
        
        logger.info(
            f"CAE: {symbol} {direction} | lot={lot:.2f}, risk={adjusted_risk:.2%}, "
            f"conf={confidence:.1%}"
        )
        
        return AllocationResult(
            approved=True,
            symbol=symbol,
            lot=lot,
            sl=sl_price,
            tp=tp_price,
            risk_pct=adjusted_risk * 100
        )

    # =========================================================
    # RISK ADJUSTMENT
    # =========================================================
    
    def _calculate_adjusted_risk(
        self,
        confidence: float,
        regime: Optional[MarketRegime]
    ) -> float:
        """
        Calculate adjusted risk based on confidence and regime.
        
        Formula: adjusted_risk = base_risk × confidence × regime_multiplier
        """
        # Start with base risk
        risk = self.base_risk
        
        # Scale by confidence (0.5 to 1.5 multiplier)
        confidence_mult = 0.5 + confidence  # 0.5 at 0%, 1.5 at 100%
        risk *= confidence_mult
        
        # Scale by regime
        if regime:
            if regime.regime == "HIGH_VOL":
                risk *= 0.5  # Cut risk in high volatility
            elif regime.regime == "TREND" and regime.strength > 0.7:
                risk *= 1.2  # Increase in strong trend
            elif regime.regime == "RANGE":
                risk *= 0.8  # Reduce in ranging market
        
        # Clamp to max risk
        risk = min(risk, self.max_risk)
        
        return risk

    # =========================================================
    # SAFETY CHECKS
    # =========================================================
    
    def _check_daily_reset(self, equity: float):
        """Reset daily tracking if new day."""
        today = date.today()
        if today != self._trade_date:
            self._daily_pnl = 0.0
            self._start_equity = equity
            self._trade_date = today
            logger.info(f"CAE: Daily reset, start_equity={equity:.2f}")
        
        if self._start_equity is None:
            self._start_equity = equity

    def _check_daily_loss(self, current_equity: float) -> bool:
        """Check if daily loss limit reached."""
        if self._start_equity is None or self._start_equity <= 0:
            return True
        
        daily_loss = (self._start_equity - current_equity) / self._start_equity
        
        if daily_loss >= self.max_daily_loss:
            logger.warning(f"CAE: Daily loss limit reached: {daily_loss:.2%}")
            return False
        
        return True

    def _check_direction_stacking(self, symbol: str, direction: str) -> bool:
        """Check if too many orders in same direction."""
        orders = self._open_orders.get(symbol, [])
        same_direction = sum(1 for o in orders if o.get("direction") == direction)
        
        return same_direction < self.max_orders_same_direction

    def _check_symbol_exposure(self, symbol: str, equity: float) -> bool:
        """Check if exposure per symbol is within limits."""
        orders = self._open_orders.get(symbol, [])
        
        if not orders:
            return True
        
        # Calculate total exposure
        total_exposure = sum(o.get("lot", 0) * o.get("margin", 0) for o in orders)
        exposure_pct = total_exposure / equity if equity > 0 else 0
        
        return exposure_pct < self.max_exposure_per_symbol

    # =========================================================
    # ORDER TRACKING
    # =========================================================
    
    def register_order(self, symbol: str, direction: str, lot: float, margin: float = 0):
        """Register an open order for tracking."""
        if symbol not in self._open_orders:
            self._open_orders[symbol] = []
        
        self._open_orders[symbol].append({
            "direction": direction,
            "lot": lot,
            "margin": margin,
            "time": datetime.now()
        })

    def close_order(self, symbol: str, direction: str):
        """Remove a closed order from tracking."""
        if symbol in self._open_orders:
            orders = self._open_orders[symbol]
            for i, o in enumerate(orders):
                if o.get("direction") == direction:
                    del orders[i]
                    break

    def update_pnl(self, pnl: float):
        """Update daily P/L tracking."""
        self._daily_pnl += pnl

    # =========================================================
    # HELPERS
    # =========================================================
    
    def _reject(self, symbol: str, reason: str) -> AllocationResult:
        """Create rejection result."""
        logger.warning(f"CAE: {symbol} REJECTED - {reason}")
        return AllocationResult(
            approved=False,
            symbol=symbol,
            lot=0.0,
            sl=0.0,
            tp=0.0,
            risk_pct=0.0,
            rejection_reason=reason
        )

    def get_status(self) -> Dict:
        """Get current CAE status."""
        return {
            "daily_pnl": self._daily_pnl,
            "start_equity": self._start_equity,
            "trade_date": str(self._trade_date),
            "open_orders": {k: len(v) for k, v in self._open_orders.items()}
        }
