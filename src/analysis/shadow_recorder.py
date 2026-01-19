import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
import logging

logger = logging.getLogger("SHADOW_RECORDER")

@dataclass
class ShadowOrder:
    ticket: str
    symbol: str
    order_type: str  # 'BUY' or 'SELL'
    open_price: float
    sl: float
    tp: float
    volume: float
    open_time: datetime
    status: str = "OPEN"  # OPEN, CLOSED_TP, CLOSED_SL, CLOSED_MANUAL
    close_price: float = 0.0
    close_time: Optional[datetime] = None
    pnl: float = 0.0
    highest_floating_pnl: float = 0.0
    lowest_floating_pnl: float = 0.0
    reason: str = "UNKNOWN"

class ShadowRecorder:
    """
    Shadow Recorder
    ===============
    Simulates execution of trades blocked by Guardian.
    Tracks 'DD Avoided' (Losses that didn't happen) and 'Missed Profit'.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.active_orders: List[ShadowOrder] = []
        self.closed_orders: List[ShadowOrder] = []
        self.log_file = os.path.join(log_dir, "shadow_trades.csv")
        self.contract_size = 100.0 # XAUUSD Standard assumption
        
        self._init_csv()
        
    def _init_csv(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ticket", "open_time", "type", "volume", 
                    "open_price", "sl", "tp", 
                    "status", "close_time", "close_price", "pnl", 
                    "max_favorable", "max_adverse", "reason"
                ])

    def place_trade(self, signal: str, price: float, sl: float, tp: float, volume: float, reason: str = "UNKNOWN", symbol: str = "XAUUSD") -> str:
        """Record a virtual trade."""
        ticket = f"SHADOW-{int(datetime.now().timestamp())}"
        order = ShadowOrder(
            ticket=ticket,
            symbol=symbol,
            order_type=signal,
            open_price=price,
            sl=sl,
            tp=tp,
            volume=volume,
            open_time=datetime.now(),
            reason=reason
        )
        self.active_orders.append(order)
        logger.info(f"👻 Shadow Trade Placed: {signal} @ {price} (SL={sl}, TP={tp}, Reason={reason})")
        return ticket

    def update(self, current_bid: float, current_ask: float):
        """Update active shadow trades with current market price."""
        for order in self.active_orders[:]:
            current_price = current_bid if order.order_type == "BUY" else current_ask
            
            # Calculate Floating PnL
            if order.order_type == "BUY":
                diff = current_price - order.open_price
            else:
                diff = order.open_price - current_price
                
            floating_pnl = diff * order.volume * self.contract_size
            
            # Update Watermarks
            if floating_pnl > order.highest_floating_pnl:
                order.highest_floating_pnl = floating_pnl
            if floating_pnl < order.lowest_floating_pnl:
                order.lowest_floating_pnl = floating_pnl
            
            # Check Exit Conditions
            closed = False
            exit_reason = ""
            
            if order.order_type == "BUY":
                if current_price >= order.tp:
                    order.status = "CLOSED_TP"
                    order.close_price = order.tp # Slippage ignored for shadow
                    closed = True
                elif current_price <= order.sl:
                    order.status = "CLOSED_SL"
                    order.close_price = order.sl
                    closed = True
            elif order.order_type == "SELL":
                if current_price <= order.tp:
                    order.status = "CLOSED_TP"
                    order.close_price = order.tp
                    closed = True
                elif current_price >= order.sl:
                    order.status = "CLOSED_SL"
                    order.close_price = order.sl
                    closed = True
            
            if closed:
                # Calculate Final PnL
                if order.order_type == "BUY":
                    final_diff = order.close_price - order.open_price
                else:
                    final_diff = order.open_price - order.close_price
                
                order.pnl = final_diff * order.volume * self.contract_size
                order.close_time = datetime.now()
                
                self.active_orders.remove(order)
                self.closed_orders.append(order)
                self._log_trade(order)
                
                logger.info(f"👻 Shadow Trade Closed: {order.status} | PnL: ${order.pnl:.2f} | Reason: {order.reason}")

    def _log_trade(self, order: ShadowOrder):
        """Append closed trade to CSV."""
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    order.ticket, order.open_time, order.order_type, 
                    order.volume, order.open_price, order.sl, order.tp, 
                    order.status, order.close_time, order.close_price, 
                    f"{order.pnl:.2f}", 
                    f"{order.highest_floating_pnl:.2f}", 
                    f"{order.lowest_floating_pnl:.2f}",
                    order.reason
                ])
        except Exception as e:
            logger.error(f"Failed to log shadow trade: {e}")

    def get_stats(self) -> Dict[str, float]:
        """Get cumulative stats."""
        total_pnl = sum(o.pnl for o in self.closed_orders)
        
        # Avoided DD = Sum of NEGATIVE PnL (money we would have lost)
        # Missed Profit = Sum of POSITIVE PnL (money we could have made)
        avoided_dd = sum(abs(o.pnl) for o in self.closed_orders if o.pnl < 0)
        missed_profit = sum(o.pnl for o in self.closed_orders if o.pnl > 0)
        
        return {
            "pnl": total_pnl,
            "avoided_dd": avoided_dd,
            "missed_profit": missed_profit,
            "active_count": len(self.active_orders)
        }
