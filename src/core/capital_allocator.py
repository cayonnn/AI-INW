# src/core/capital_allocator.py
"""
Capital Allocator - PRODUCTION READY
=====================================

คำนวณ lot จาก equity + risk + confidence
ป้องกัน martingale / overtrade

Hard Rules:
- BASE_RISK = 1% per trade
- MAX_DAILY_RISK = 4%
- MAX_SYMBOL_EXPOSURE = 2%
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import date

from src.core.execution_contract import ExecutionBlocked

logger = logging.getLogger("CAE")


@dataclass
class AllocationResult:
    """Capital allocation result."""
    lot: float
    risk_amount: float
    risk_pct: float
    approved: bool
    rejection: str = ""


class CapitalAllocator:
    """
    PRODUCTION-READY Capital Allocator.
    
    ป้องกัน:
    - Martingale
    - Overtrade
    - Symbol concentration
    """

    # HARD LIMITS - DO NOT CHANGE WITHOUT APPROVAL
    BASE_RISK = 0.01           # 1% per trade
    MAX_RISK = 0.02            # 2% max per trade
    MAX_DAILY_RISK = 0.04      # 4% max daily
    MAX_SYMBOL_EXPOSURE = 0.02  # 2% max per symbol
    MIN_LOT = 0.01
    MAX_LOT = 5.0

    def __init__(self, equity: float, pip_value: float = 1.0):
        self.equity = equity
        self.pip_value = pip_value
        
        # Daily tracking
        self._daily_risk_used = 0.0
        self._symbol_exposure: Dict[str, float] = {}
        self._trade_date = date.today()
        
        logger.info(
            f"CapitalAllocator: equity={equity:.2f}, "
            f"base_risk={self.BASE_RISK:.1%}, max_daily={self.MAX_DAILY_RISK:.1%}"
        )

    # =========================================================
    # ALLOCATION
    # =========================================================
    
    def allocate(
        self,
        symbol: str,
        sl_pips: float,
        confidence: float,
        direction: str = "BUY"
    ) -> AllocationResult:
        """
        Allocate capital for a trade.
        
        Formula: lot = (equity × risk × confidence) / (sl_pips × pip_value)
        
        Args:
            symbol: Trading symbol
            sl_pips: Stop loss in pips
            confidence: Model confidence (0-1)
            direction: BUY or SELL
            
        Returns:
            AllocationResult with lot size
            
        Raises:
            ExecutionBlocked: If allocation denied
        """
        self._check_daily_reset()
        
        # Check symbol exposure
        current_exposure = self._symbol_exposure.get(symbol, 0.0)
        if current_exposure >= self.MAX_SYMBOL_EXPOSURE:
            raise ExecutionBlocked(
                f"Symbol exposure exceeded: {symbol} = {current_exposure:.1%} >= {self.MAX_SYMBOL_EXPOSURE:.1%}"
            )
        
        # Check daily risk
        if self._daily_risk_used >= self.MAX_DAILY_RISK:
            raise ExecutionBlocked(
                f"Daily risk exceeded: {self._daily_risk_used:.1%} >= {self.MAX_DAILY_RISK:.1%}"
            )
        
        # Calculate risk adjusted by confidence
        adjusted_risk = min(self.BASE_RISK * confidence, self.MAX_RISK)
        
        # Calculate risk amount
        risk_amount = self.equity * adjusted_risk
        
        # Calculate lot
        if sl_pips <= 0 or self.pip_value <= 0:
            raise ExecutionBlocked(f"Invalid SL pips ({sl_pips}) or pip value ({self.pip_value})")
        
        lot = risk_amount / (sl_pips * self.pip_value)
        
        # Clamp lot
        lot = max(self.MIN_LOT, min(lot, self.MAX_LOT))
        lot = round(lot, 2)
        
        logger.info(
            f"CAE: {symbol} lot={lot:.2f}, risk={adjusted_risk:.2%}, conf={confidence:.1%}"
        )
        
        return AllocationResult(
            lot=lot,
            risk_amount=risk_amount,
            risk_pct=adjusted_risk * 100,
            approved=True
        )

    def allocate_safe(
        self,
        symbol: str,
        sl_pips: float,
        confidence: float,
        direction: str = "BUY"
    ) -> AllocationResult:
        """
        Safe allocation that returns rejection instead of raising.
        
        Returns:
            AllocationResult (approved=False if denied)
        """
        try:
            return self.allocate(symbol, sl_pips, confidence, direction)
        except ExecutionBlocked as e:
            return AllocationResult(
                lot=0.0,
                risk_amount=0.0,
                risk_pct=0.0,
                approved=False,
                rejection=str(e)
            )

    # =========================================================
    # TRACKING
    # =========================================================
    
    def register_trade(self, symbol: str, risk_pct: float):
        """Register a trade for exposure tracking."""
        self._daily_risk_used += risk_pct / 100
        self._symbol_exposure[symbol] = self._symbol_exposure.get(symbol, 0) + risk_pct / 100
        
        logger.debug(
            f"Registered: {symbol} risk={risk_pct:.1f}%, "
            f"daily_total={self._daily_risk_used:.1%}"
        )

    def close_trade(self, symbol: str, risk_pct: float):
        """Remove closed trade from exposure tracking."""
        if symbol in self._symbol_exposure:
            self._symbol_exposure[symbol] = max(0, self._symbol_exposure[symbol] - risk_pct / 100)

    def update_equity(self, new_equity: float):
        """Update account equity."""
        self.equity = new_equity

    def _check_daily_reset(self):
        """Reset daily tracking if new day."""
        today = date.today()
        if today != self._trade_date:
            self._daily_risk_used = 0.0
            self._trade_date = today
            logger.info("CapitalAllocator: Daily reset")

    # =========================================================
    # STATUS
    # =========================================================
    
    def get_status(self) -> Dict:
        """Get allocator status."""
        return {
            "equity": self.equity,
            "daily_risk_used": self._daily_risk_used,
            "daily_risk_remaining": self.MAX_DAILY_RISK - self._daily_risk_used,
            "symbol_exposure": dict(self._symbol_exposure),
            "trade_date": str(self._trade_date)
        }

    def can_trade(self, symbol: str) -> bool:
        """Check if trading is allowed."""
        return (
            self._daily_risk_used < self.MAX_DAILY_RISK and
            self._symbol_exposure.get(symbol, 0) < self.MAX_SYMBOL_EXPOSURE
        )
