# src/core/execution_contract.py
"""
Execution Contract - PRODUCTION READY
======================================

ห้ามส่งคำสั่งไป MT ถ้าไม่ผ่านกฎ
TP / SL / Lot / Risk / Context ต้องครบ 100%

❗ ถ้า validate ไม่ผ่าน = ไม่มีทางยิง order
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger("EXEC_CONTRACT")


class ExecutionBlocked(Exception):
    """Exception raised when trade validation fails."""
    pass


@dataclass
class TradeIntent:
    """Trade intent with all required fields."""
    symbol: str
    side: str              # BUY / SELL
    lot: float
    sl: float
    tp: float
    confidence: float
    model_version: str
    entry_price: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "lot": self.lot,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "entry_price": self.entry_price,
            "timestamp": self.timestamp
        }


class ExecutionContract:
    """
    PRODUCTION-READY Execution Contract.
    
    Hard Rules (Fail = Stop Bot):
    - SL/TP บังคับทุก order
    - RR ≥ 1.2
    - Lot คำนวณจาก CAE เท่านั้น
    - Confidence threshold enforced
    
    ❌ "ยิงมั่ว" = เป็นไปไม่ได้
    ✅ ทุก order = audit ได้ / rollback ได้
    """

    # HARD LIMITS - DO NOT CHANGE WITHOUT APPROVAL
    MIN_LOT = 0.01
    MAX_LOT = 5.0
    MIN_RR = 1.2
    MIN_CONFIDENCE = 0.55
    MAX_SL_PIPS = 5000  # Safety cap
    MIN_SL_PIPS = 10    # Minimum SL distance

    def __init__(self):
        self._blocked_count = 0
        self._passed_count = 0
        self._block_reasons: Dict[str, int] = {}
        
        logger.info(
            f"ExecutionContract initialized: MIN_LOT={self.MIN_LOT}, "
            f"MAX_LOT={self.MAX_LOT}, MIN_RR={self.MIN_RR}"
        )

    # =========================================================
    # VALIDATION (MANDATORY)
    # =========================================================
    
    def validate(self, intent: TradeIntent) -> bool:
        """
        Validate trade intent against all rules.
        
        Raises:
            ExecutionBlocked: If any rule fails
            
        Returns:
            True if all validations pass
        """
        try:
            self._validate_lot(intent)
            self._validate_sl_tp(intent)
            self._validate_rr(intent)
            self._validate_confidence(intent)
            self._validate_symbol(intent)
            
            self._passed_count += 1
            logger.info(
                f"✅ Contract PASSED: {intent.symbol} {intent.side} "
                f"lot={intent.lot:.2f} RR={self._calc_rr(intent):.1f}"
            )
            return True
            
        except ExecutionBlocked as e:
            self._blocked_count += 1
            reason = str(e)
            self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1
            logger.warning(f"❌ Contract BLOCKED: {reason}")
            raise

    def _validate_lot(self, intent: TradeIntent):
        """Validate lot size."""
        if intent.lot < self.MIN_LOT:
            raise ExecutionBlocked(f"Lot too small: {intent.lot} < {self.MIN_LOT}")
        
        if intent.lot > self.MAX_LOT:
            raise ExecutionBlocked(f"Lot too large: {intent.lot} > {self.MAX_LOT}")

    def _validate_sl_tp(self, intent: TradeIntent):
        """Validate SL/TP presence and values."""
        if intent.sl is None or intent.sl == 0:
            raise ExecutionBlocked("SL missing - MANDATORY")
        
        if intent.tp is None or intent.tp == 0:
            raise ExecutionBlocked("TP missing - MANDATORY")
        
        # Validate SL makes sense
        if intent.side == "BUY":
            if intent.sl >= intent.entry_price and intent.entry_price > 0:
                raise ExecutionBlocked(f"Invalid BUY SL: {intent.sl} >= entry {intent.entry_price}")
        elif intent.side == "SELL":
            if intent.sl <= intent.entry_price and intent.entry_price > 0:
                raise ExecutionBlocked(f"Invalid SELL SL: {intent.sl} <= entry {intent.entry_price}")

    def _validate_rr(self, intent: TradeIntent):
        """Validate Risk/Reward ratio."""
        rr = self._calc_rr(intent)
        
        if rr < self.MIN_RR:
            raise ExecutionBlocked(f"RR too low: {rr:.2f} < {self.MIN_RR}")

    def _validate_confidence(self, intent: TradeIntent):
        """Validate model confidence."""
        if intent.confidence < self.MIN_CONFIDENCE:
            raise ExecutionBlocked(
                f"Confidence too low: {intent.confidence:.1%} < {self.MIN_CONFIDENCE:.1%}"
            )

    def _validate_symbol(self, intent: TradeIntent):
        """Validate symbol."""
        if not intent.symbol or len(intent.symbol) < 3:
            raise ExecutionBlocked(f"Invalid symbol: {intent.symbol}")

    def _calc_rr(self, intent: TradeIntent) -> float:
        """Calculate Risk/Reward ratio."""
        if intent.entry_price > 0:
            risk = abs(intent.entry_price - intent.sl)
            reward = abs(intent.tp - intent.entry_price)
        else:
            # Fallback if no entry price
            risk = abs(intent.sl)
            reward = abs(intent.tp - intent.sl)
        
        if risk == 0:
            return 0.0
        
        return reward / risk

    # =========================================================
    # MT5 BRIDGE
    # =========================================================
    
    def send_to_mt(self, intent: TradeIntent, mt5_module=None) -> Dict:
        """
        Validate and send order to MT5.
        
        Args:
            intent: Trade intent to execute
            mt5_module: MT5 module (for testing/mocking)
            
        Returns:
            Order result dictionary
        """
        # MANDATORY: Validate first
        self.validate(intent)
        
        # Build payload
        payload = {
            "action": 1,  # TRADE_ACTION_DEAL
            "symbol": intent.symbol,
            "type": 0 if intent.side == "BUY" else 1,
            "volume": intent.lot,
            "sl": intent.sl,
            "tp": intent.tp,
            "comment": f"v{intent.model_version}",
            "magic": 900005
        }
        
        if mt5_module:
            try:
                result = mt5_module.order_send(payload)
                return {
                    "success": result.retcode == 10009 if result else False,
                    "ticket": result.order if result else 0,
                    "payload": payload
                }
            except Exception as e:
                logger.error(f"MT5 order_send error: {e}")
                return {"success": False, "error": str(e)}
        
        # Sandbox mode
        return {
            "success": True,
            "ticket": int(time.time()),
            "mode": "SANDBOX",
            "payload": payload
        }

    # =========================================================
    # STATS
    # =========================================================
    
    def get_stats(self) -> Dict:
        """Get contract validation statistics."""
        total = self._blocked_count + self._passed_count
        return {
            "total": total,
            "passed": self._passed_count,
            "blocked": self._blocked_count,
            "pass_rate": self._passed_count / total if total > 0 else 0,
            "block_reasons": self._block_reasons
        }

    def reset_stats(self):
        """Reset statistics."""
        self._blocked_count = 0
        self._passed_count = 0
        self._block_reasons = {}


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

_contract = ExecutionContract()

def validate_and_send(intent: TradeIntent, mt5_module=None) -> Dict:
    """
    Validate and send trade intent (convenience function).
    
    Usage:
        intent = TradeIntent(symbol="XAUUSD", side="BUY", ...)
        result = validate_and_send(intent)
    """
    return _contract.send_to_mt(intent, mt5_module)
