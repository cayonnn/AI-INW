"""
position_sizer.py
=================
Capital allocation engine using:
- Fractional Kelly Criterion
- Volatility-adjusted exposure
- Signal confidence scaling

This is where "AI ฉลาดแค่ไหนก็แพ้ได้" if done wrong.
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("POSITION_SIZER")


@dataclass
class PositionResult:
    """Position sizing result."""
    lot_size: float
    risk_pct: float
    kelly_fraction: float
    vol_adjustment: float
    confidence: float
    is_valid: bool
    rejection_reason: str = ""


class PositionSizer:
    """
    Hedge-fund grade position sizing.
    
    Formula: Fractional Kelly × Volatility Scaling × Signal Confidence
    
    - ไม่ใช้ fixed lot
    - ไม่ใช้ martingale  
    - ไม่ all-in Kelly
    """

    def __init__(self, risk_cfg: dict = None):
        risk_cfg = risk_cfg or {}
        
        self.max_risk = risk_cfg.get("max_risk_per_trade", 0.01)     # 1%
        self.min_risk = risk_cfg.get("min_risk_per_trade", 0.001)    # 0.1%
        self.kelly_fraction = risk_cfg.get("kelly_fraction", 0.3)    # 30% of Kelly
        self.min_position = risk_cfg.get("min_position", 0.01)       # 0.01 lot
        self.max_position = risk_cfg.get("max_position", 5.0)        # 5 lots
        self.vol_target = risk_cfg.get("target_volatility", 0.01)    # 1%
        
        # Cache for volatility values
        self._vol_cache = {}

    # -------------------------------------------------
    def calculate(self, balance: float, sl_pips: float, pip_value: float,
                  symbol: str, confidence: float, rr_ratio: float = 1.5,
                  current_atr: float = None) -> PositionResult:
        """
        Calculate position size using Fractional Kelly + Volatility Scaling.
        
        Args:
            balance: Account balance
            sl_pips: Stop loss in pips
            pip_value: Value per pip per lot
            symbol: Trading symbol
            confidence: Signal confidence (0-1)
            rr_ratio: Expected risk:reward ratio
            current_atr: Current ATR for volatility adjustment
            
        Returns:
            PositionResult with lot size and details
        """
        # Input validation
        if balance <= 0:
            return self._reject("Invalid balance")
        if sl_pips <= 0:
            return self._reject("Invalid SL pips")
        if pip_value <= 0:
            return self._reject("Invalid pip value")
        if confidence <= 0 or confidence > 1:
            return self._reject("Invalid confidence")

        # 1️⃣ Kelly Criterion (Base)
        f_kelly = self._kelly_fraction_calc(confidence, rr_ratio)
        
        if f_kelly <= 0:
            logger.info(f"{symbol}: Kelly suggests no trade (f={f_kelly:.4f})")
            return self._reject(f"Kelly negative: {f_kelly:.4f}")

        # 2️⃣ Fractional Kelly (ลดความผันผวน)
        f_fractional = f_kelly * self.kelly_fraction

        # 3️⃣ Volatility Scaling
        vol_adj = self._volatility_adjustment(symbol, current_atr)

        # 4️⃣ Confidence Scaling
        conf_adj = confidence

        # Combined exposure
        exposure = f_fractional * vol_adj * conf_adj
        
        # Clamp to risk limits
        risk_pct = min(max(exposure, self.min_risk), self.max_risk)
        
        # Convert to lot size
        risk_amount = balance * risk_pct
        sl_cost_per_lot = sl_pips * pip_value
        
        if sl_cost_per_lot <= 0:
            return self._reject("Invalid SL cost")
            
        lot_size = risk_amount / sl_cost_per_lot
        
        # Clamp lot size
        lot_size = self._clamp_position(lot_size)
        
        if lot_size < self.min_position:
            return self._reject(f"Position too small: {lot_size:.4f}")

        logger.info(
            f"{symbol}: size={lot_size:.3f} lots, "
            f"kelly={f_kelly:.3f}, vol_adj={vol_adj:.2f}, conf={confidence:.2f}"
        )

        return PositionResult(
            lot_size=lot_size,
            risk_pct=risk_pct,
            kelly_fraction=f_kelly,
            vol_adjustment=vol_adj,
            confidence=confidence,
            is_valid=True,
        )

    # -------------------------------------------------
    # Kelly Criterion
    # -------------------------------------------------
    def _kelly_fraction_calc(self, p: float, rr: float) -> float:
        """
        Kelly formula: f = ((RR × p) - (1 - p)) / RR
        
        Args:
            p: Win probability (signal confidence)
            rr: Risk:Reward ratio
            
        Returns:
            Kelly fraction (can be negative)
        """
        if rr <= 0:
            return 0.0
        
        f = ((rr * p) - (1 - p)) / rr
        return f

    # -------------------------------------------------
    # Volatility Adjustment
    # -------------------------------------------------
    def _volatility_adjustment(self, symbol: str, current_atr: float = None) -> float:
        """
        Volatility scaling:
        - ตลาดผันผวนสูง → ลด size
        - ตลาดนิ่ง → เพิ่ม size
        
        Returns multiplier 0.5 to 2.0
        """
        if current_atr is None:
            realized_vol = self._get_realized_vol(symbol)
        else:
            realized_vol = current_atr

        if realized_vol <= 0:
            return 1.0

        # Inverse relationship: higher vol = smaller size
        adjustment = self.vol_target / realized_vol
        
        # Clamp to reasonable range
        return max(0.5, min(2.0, adjustment))

    def _get_realized_vol(self, symbol: str) -> float:
        """Get cached realized volatility."""
        return self._vol_cache.get(symbol, self.vol_target)

    def set_volatility(self, symbol: str, vol: float):
        """Update volatility cache."""
        self._vol_cache[symbol] = vol

    # -------------------------------------------------
    # Position Sizing Helpers
    # -------------------------------------------------
    def _clamp_position(self, size: float) -> float:
        """Clamp to min/max lot size."""
        if size < self.min_position:
            return 0.0
        return max(self.min_position, min(size, self.max_position))

    def _reject(self, reason: str) -> PositionResult:
        """Create rejection result."""
        return PositionResult(
            lot_size=0.0,
            risk_pct=0.0,
            kelly_fraction=0.0,
            vol_adjustment=0.0,
            confidence=0.0,
            is_valid=False,
            rejection_reason=reason,
        )

    # -------------------------------------------------
    # Alternative: Fixed Fractional (Simpler)
    # -------------------------------------------------
    def calculate_fixed_fractional(self, balance: float, sl_pips: float, 
                                   pip_value: float, risk_pct: float = 0.01) -> float:
        """
        Simple fixed fractional position sizing.
        
        Args:
            balance: Account balance
            sl_pips: Stop loss in pips
            pip_value: Value per pip per lot
            risk_pct: Fixed risk percentage (default 1%)
            
        Returns:
            Lot size
        """
        risk_amount = balance * risk_pct
        sl_cost_per_lot = sl_pips * pip_value
        
        if sl_cost_per_lot <= 0:
            return 0.0
            
        lot_size = risk_amount / sl_cost_per_lot
        return self._clamp_position(lot_size)
