"""
stop_loss.py
============
Structure + Volatility based Stop Loss Engine

❗ ไม่ใช่ SL แบบ fixed pips
❗ ไม่ใช่ ATR อย่างเดียว
✅ ใช้ "Market Structure ก่อน → Volatility เสริม"

Components:
- Market Structure: หา swing high / low
- Volatility (ATR): buffer กัน stop hunt
- Confidence: signal ดี → ให้พื้นที่มากขึ้น
- RR Control: TP = SL × RR
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("STOP_LOSS")


@dataclass
class SLTPResult:
    """Stop Loss / Take Profit calculation result."""
    stop_loss: float
    take_profit: float
    sl_pips: float
    tp_pips: float
    rr_ratio: float
    method: str           # "structure", "atr", "combined"
    is_valid: bool
    rejection_reason: str = ""


class StopLossEngine:
    """
    Hedge-fund grade Stop Loss Engine.
    
    Principles:
    - Stop ต้องอยู่ "นอก noise"
    - Stop ต้อง invalidate idea
    - Stop ต้องสัมพันธ์กับ position sizing
    - Take Profit derive จาก SL (RR control)
    """

    def __init__(self, risk_cfg: dict = None):
        risk_cfg = risk_cfg or {}
        
        self.atr_multiplier = risk_cfg.get("atr_multiplier", 1.5)
        self.min_sl_pips = risk_cfg.get("min_stop_pips", 10)
        self.max_sl_pips = risk_cfg.get("max_stop_pips", 200)
        self.default_rr = risk_cfg.get("default_rr", 2.0)
        self.min_rr = risk_cfg.get("min_rr", 1.5)
        self.structure_buffer_pips = risk_cfg.get("structure_buffer_pips", 3)

    # -------------------------------------------------
    def calculate(self, symbol: str, decision) -> SLTPResult:
        """
        Calculate SL/TP using structure + volatility.
        
        Decision object must have:
        - entry_price
        - structure_high
        - structure_low
        - atr
        - confidence
        - rr_ratio
        - action
        
        Returns:
            SLTPResult with SL, TP, and validation
        """
        entry = decision.entry_price
        action = decision.action

        # Get structure-based SL
        structure_sl = self._structure_stop(decision)
        
        # Get ATR buffer
        atr_buffer = self._atr_buffer(symbol, decision)

        # Merge: push SL beyond structure by ATR buffer
        raw_sl = self._merge_sl(structure_sl, atr_buffer, action)
        
        # Clamp to limits
        final_sl = self._clamp_sl(entry, raw_sl, action)

        # Calculate TP from RR
        rr = decision.rr_ratio if decision.rr_ratio > 0 else self.default_rr
        tp = self._take_profit(entry, final_sl, action, rr)

        # Calculate pips
        sl_pips = self._price_to_pips(abs(entry - final_sl))
        tp_pips = self._price_to_pips(abs(tp - entry))
        
        # Validate minimum R:R
        actual_rr = tp_pips / sl_pips if sl_pips > 0 else 0
        
        if actual_rr < self.min_rr:
            return self._reject(f"R:R {actual_rr:.2f} < min {self.min_rr}")

        logger.info(f"{symbol}: SL={final_sl:.5f} ({sl_pips:.1f} pips), TP={tp:.5f} ({tp_pips:.1f} pips), RR={actual_rr:.2f}")

        return SLTPResult(
            stop_loss=final_sl,
            take_profit=tp,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            rr_ratio=actual_rr,
            method="combined",
            is_valid=True,
        )

    # -------------------------------------------------
    def calculate_atr_based(self, entry_price: float, atr: float, direction: str,
                           rr_ratio: float = 2.0, confidence: float = 0.6) -> SLTPResult:
        """
        Simple ATR-based SL/TP calculation.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: "LONG" or "SHORT"
            rr_ratio: Risk:Reward ratio
            confidence: Signal confidence for buffer adjustment
        """
        # Confidence adjustment: higher confidence = tighter stop allowed
        conf_adj = max(0.8, min(1.2, 2 - confidence))
        sl_distance = atr * self.atr_multiplier * conf_adj
        
        # Calculate SL and TP
        if direction in ["LONG", "BUY"]:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * rr_ratio)
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - (sl_distance * rr_ratio)
        
        # Calculate pips
        sl_pips = self._price_to_pips(sl_distance)
        tp_pips = self._price_to_pips(sl_distance * rr_ratio)
        
        # Clamp SL pips
        if sl_pips < self.min_sl_pips:
            logger.warning(f"SL too tight ({sl_pips:.1f} pips), adjusting to min")
            sl_pips = self.min_sl_pips
            sl_distance = self._pips_to_price(entry_price, sl_pips)
            tp_pips = sl_pips * rr_ratio
            
            if direction in ["LONG", "BUY"]:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + (sl_distance * rr_ratio)
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - (sl_distance * rr_ratio)
        
        if sl_pips > self.max_sl_pips:
            logger.warning(f"SL too wide ({sl_pips:.1f} pips), rejecting")
            return self._reject(f"SL too wide: {sl_pips:.1f} pips")
        
        return SLTPResult(
            stop_loss=stop_loss,
            take_profit=take_profit,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            rr_ratio=rr_ratio,
            method="atr",
            is_valid=True,
        )

    # -------------------------------------------------
    def calculate_structure_based(self, entry_price: float, swing_high: float, 
                                  swing_low: float, direction: str,
                                  rr_ratio: float = 2.0) -> SLTPResult:
        """
        Structure-based SL using swing points.
        
        Args:
            entry_price: Entry price
            swing_high: Recent swing high
            swing_low: Recent swing low
            direction: "LONG" or "SHORT"
            rr_ratio: Risk:Reward ratio
        """
        buffer = self._pips_to_price(entry_price, self.structure_buffer_pips)
        
        if direction in ["LONG", "BUY"]:
            stop_loss = swing_low - buffer
            sl_distance = entry_price - stop_loss
            take_profit = entry_price + (sl_distance * rr_ratio)
        else:
            stop_loss = swing_high + buffer
            sl_distance = stop_loss - entry_price
            take_profit = entry_price - (sl_distance * rr_ratio)
        
        sl_pips = self._price_to_pips(sl_distance)
        tp_pips = sl_pips * rr_ratio
        
        # Validate
        if sl_pips < self.min_sl_pips:
            return self._reject(f"Structure SL too tight: {sl_pips:.1f} pips")
        if sl_pips > self.max_sl_pips:
            return self._reject(f"Structure SL too wide: {sl_pips:.1f} pips")
        
        return SLTPResult(
            stop_loss=stop_loss,
            take_profit=take_profit,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            rr_ratio=rr_ratio,
            method="structure",
            is_valid=True,
        )

    # -------------------------------------------------
    def calculate_combined(self, entry_price: float, atr: float,
                          swing_high: float, swing_low: float,
                          direction: str, rr_ratio: float = 2.0,
                          confidence: float = 0.6) -> SLTPResult:
        """
        Combined: Structure + ATR buffer.
        
        Uses swing point as base, adds ATR buffer for stop hunt protection.
        """
        # ATR buffer
        conf_adj = max(0.8, min(1.2, 2 - confidence))
        atr_buffer = atr * self.atr_multiplier * conf_adj * 0.5  # Half ATR as buffer
        
        if direction in ["LONG", "BUY"]:
            # SL below swing low with buffer
            stop_loss = swing_low - atr_buffer
            sl_distance = entry_price - stop_loss
            take_profit = entry_price + (sl_distance * rr_ratio)
        else:
            # SL above swing high with buffer
            stop_loss = swing_high + atr_buffer
            sl_distance = stop_loss - entry_price
            take_profit = entry_price - (sl_distance * rr_ratio)
        
        sl_pips = self._price_to_pips(sl_distance)
        tp_pips = sl_pips * rr_ratio
        
        # Clamp
        if sl_pips < self.min_sl_pips:
            return self._reject(f"Combined SL too tight: {sl_pips:.1f} pips")
        if sl_pips > self.max_sl_pips:
            return self._reject(f"Combined SL too wide: {sl_pips:.1f} pips")
        
        if rr_ratio < self.min_rr:
            return self._reject(f"R:R {rr_ratio:.2f} < min {self.min_rr}")
        
        return SLTPResult(
            stop_loss=stop_loss,
            take_profit=take_profit,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            rr_ratio=rr_ratio,
            method="combined",
            is_valid=True,
        )

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------
    def _structure_stop(self, decision) -> float:
        """Uses recent swing high / low as invalidation point."""
        if decision.action in ["BUY", "LONG"]:
            return decision.structure_low
        else:
            return decision.structure_high

    def _atr_buffer(self, symbol: str, decision) -> float:
        """Adds volatility buffer using ATR."""
        atr = decision.atr
        confidence_adj = max(0.8, decision.confidence)
        return atr * self.atr_multiplier * confidence_adj

    def _merge_sl(self, structure_sl: float, atr_buffer: float, action: str) -> float:
        """Push SL beyond structure by ATR buffer."""
        if action in ["BUY", "LONG"]:
            return structure_sl - atr_buffer
        else:
            return structure_sl + atr_buffer

    def _clamp_sl(self, entry: float, sl: float, action: str) -> float:
        """Prevents SL being too tight or too wide."""
        dist = abs(entry - sl)
        
        min_dist = self._pips_to_price(entry, self.min_sl_pips)
        max_dist = self._pips_to_price(entry, self.max_sl_pips)

        if dist < min_dist:
            sl = entry - min_dist if action in ["BUY", "LONG"] else entry + min_dist
        if dist > max_dist:
            sl = entry - max_dist if action in ["BUY", "LONG"] else entry + max_dist

        return sl

    def _take_profit(self, entry: float, sl: float, action: str, rr: float) -> float:
        """RR-based take profit."""
        rr = rr if rr > 0 else self.default_rr
        risk = abs(entry - sl)

        if action in ["BUY", "LONG"]:
            return entry + (risk * rr)
        else:
            return entry - (risk * rr)

    def _pips_to_price(self, price: float, pips: float) -> float:
        """Convert pips to price distance."""
        # Assuming 4/5 digit broker
        if price > 10:  # JPY pairs
            return pips * 0.01
        return pips * 0.0001

    def _price_to_pips(self, distance: float) -> float:
        """Convert price distance to pips."""
        # Simplified: assume standard pairs
        return distance / 0.0001

    def _reject(self, reason: str) -> SLTPResult:
        """Create rejection result."""
        logger.warning(f"SL/TP rejected: {reason}")
        return SLTPResult(
            stop_loss=0.0,
            take_profit=0.0,
            sl_pips=0.0,
            tp_pips=0.0,
            rr_ratio=0.0,
            method="rejected",
            is_valid=False,
            rejection_reason=reason,
        )


# Alias for backward compatibility
StopLossCalculator = StopLossEngine
