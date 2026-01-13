# src/execution/position_executor.py
"""
PositionExecutor - Execute Position Management Decisions
=========================================================

Takes decisions from PositionManager and executes them via MT5.

Actions:
- MODIFY: Change SL/TP
- PARTIAL_CLOSE: Close portion of position
- FULL_EXIT: Close entire position
- SCALE_IN: Add to position (new order)
"""

import MetaTrader5 as mt5
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.risk.position_manager import (
    PositionManager,
    ManagementDecision,
    PositionAction,
    get_position_manager,
)
from src.execution.mt5_command_writer import get_command_writer
from src.utils.logger import get_logger

logger = get_logger("POSITION_EXECUTOR")


@dataclass
class ExecutionResult:
    """Result of position action execution."""
    success: bool
    action: PositionAction
    ticket: int
    message: str
    details: Dict[str, Any] = None


class PositionExecutor:
    """
    Executes position management decisions.
    
    Works with both:
    - Direct MT5 API
    - MT5 Command Writer (for EA execution)
    """
    
    def __init__(self, use_command_writer: bool = True):
        """
        Initialize executor.
        
        Args:
            use_command_writer: If True, use JSON command writer for EA.
                               If False, use direct MT5 API.
        """
        self.use_command_writer = use_command_writer
        self.command_writer = get_command_writer() if use_command_writer else None
        
        self.stats = {
            "executions": 0,
            "success": 0,
            "failed": 0,
        }
        
        logger.info(f"PositionExecutor initialized | Mode: {'CommandWriter' if use_command_writer else 'DirectMT5'}")
    
    def execute(self, decision: ManagementDecision, position) -> ExecutionResult:
        """
        Execute a position management decision.
        
        Args:
            decision: ManagementDecision from PositionManager
            position: MT5 position object
            
        Returns:
            ExecutionResult
        """
        self.stats["executions"] += 1
        
        ticket = position.ticket if hasattr(position, 'ticket') else position.get('ticket', 0)
        symbol = position.symbol if hasattr(position, 'symbol') else position.get('symbol', '')
        
        try:
            if decision.action == PositionAction.HOLD:
                return ExecutionResult(
                    success=True,
                    action=decision.action,
                    ticket=ticket,
                    message="No action needed"
                )
            
            elif decision.action == PositionAction.MOVE_BE:
                return self._execute_modify_sl(position, decision.new_sl)
            
            elif decision.action in [PositionAction.TRAIL_CONSERVATIVE, 
                                     PositionAction.TRAIL_AGGRESSIVE,
                                     PositionAction.TIGHTEN_SL]:
                return self._execute_modify_sl(position, decision.new_sl)
            
            elif decision.action == PositionAction.PARTIAL_CLOSE:
                return self._execute_partial_close(position, decision.close_volume)
            
            elif decision.action == PositionAction.FULL_EXIT:
                return self._execute_full_close(position)
            
            elif decision.action == PositionAction.SCALE_IN:
                return self._execute_scale_in(position, decision.add_volume)
            
            else:
                return ExecutionResult(
                    success=False,
                    action=decision.action,
                    ticket=ticket,
                    message=f"Unknown action: {decision.action}"
                )
                
        except Exception as e:
            self.stats["failed"] += 1
            logger.error(f"Execution error for #{ticket}: {e}")
            return ExecutionResult(
                success=False,
                action=decision.action,
                ticket=ticket,
                message=str(e)
            )
    
    def _execute_modify_sl(self, position, new_sl: float) -> ExecutionResult:
        """Modify position SL."""
        ticket = position.ticket if hasattr(position, 'ticket') else position.get('ticket')
        symbol = position.symbol if hasattr(position, 'symbol') else position.get('symbol')
        current_tp = position.tp if hasattr(position, 'tp') else position.get('tp', 0)
        
        if new_sl is None:
            return ExecutionResult(
                success=False,
                action=PositionAction.MOVE_BE,
                ticket=ticket,
                message="No new SL provided"
            )
        
        if self.use_command_writer and self.command_writer:
            # Use command writer
            success = self.command_writer.send_modify(ticket, new_sl, current_tp)
        else:
            # Direct MT5
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": new_sl,
                "tp": current_tp
            }
            result = mt5.order_send(request)
            success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        
        if success:
            self.stats["success"] += 1
            logger.info(f"✅ SL modified #{ticket}: {new_sl:.5f}")
            
            # Update position manager state
            pm = get_position_manager()
            current_sl = position.sl if hasattr(position, 'sl') else position.get('sl', 0)
            entry = position.price_open if hasattr(position, 'price_open') else position.get('price_open', 0)
            
            if abs(new_sl - entry) < 0.0001:  # At BE
                pm.update_position_state(ticket, be_applied=True)
            
            return ExecutionResult(
                success=True,
                action=PositionAction.MOVE_BE,
                ticket=ticket,
                message=f"SL modified to {new_sl:.5f}",
                details={"new_sl": new_sl}
            )
        else:
            self.stats["failed"] += 1
            return ExecutionResult(
                success=False,
                action=PositionAction.MOVE_BE,
                ticket=ticket,
                message="Failed to modify SL"
            )
    
    def _execute_partial_close(self, position, close_volume: float) -> ExecutionResult:
        """Close partial position."""
        ticket = position.ticket if hasattr(position, 'ticket') else position.get('ticket')
        symbol = position.symbol if hasattr(position, 'symbol') else position.get('symbol')
        pos_type = position.type if hasattr(position, 'type') else position.get('type', 0)
        
        if close_volume is None or close_volume <= 0:
            return ExecutionResult(
                success=False,
                action=PositionAction.PARTIAL_CLOSE,
                ticket=ticket,
                message="Invalid close volume"
            )
        
        # Round to valid lot size
        close_volume = round(close_volume, 2)
        
        if self.use_command_writer and self.command_writer:
            # Command writer doesn't support partial close directly
            # Need to send a CLOSE command with volume
            command = {
                "action": "PARTIAL_CLOSE",
                "ticket": ticket,
                "symbol": symbol,
                "volume": close_volume,
            }
            success = self.command_writer._write_command(command)
        else:
            # Direct MT5
            order_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": close_volume,
                "type": order_type,
                "magic": 900001,
                "comment": "partial_close",
            }
            result = mt5.order_send(request)
            success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        
        if success:
            self.stats["success"] += 1
            logger.info(f"✅ Partial close #{ticket}: {close_volume} lots")
            
            # Update state
            pm = get_position_manager()
            current_partials = pm._position_states.get(ticket)
            if current_partials:
                pm.update_position_state(
                    ticket, 
                    partial_closes=current_partials.partial_closes + 1
                )
            
            return ExecutionResult(
                success=True,
                action=PositionAction.PARTIAL_CLOSE,
                ticket=ticket,
                message=f"Closed {close_volume} lots",
                details={"volume": close_volume}
            )
        else:
            self.stats["failed"] += 1
            return ExecutionResult(
                success=False,
                action=PositionAction.PARTIAL_CLOSE,
                ticket=ticket,
                message="Failed to partial close"
            )
    
    def _execute_full_close(self, position) -> ExecutionResult:
        """Close entire position."""
        ticket = position.ticket if hasattr(position, 'ticket') else position.get('ticket')
        symbol = position.symbol if hasattr(position, 'symbol') else position.get('symbol')
        volume = position.volume if hasattr(position, 'volume') else position.get('volume', 0)
        pos_type = position.type if hasattr(position, 'type') else position.get('type', 0)
        
        if self.use_command_writer and self.command_writer:
            success = self.command_writer.send_close(ticket=ticket)
        else:
            order_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "magic": 900001,
                "comment": "full_exit",
            }
            result = mt5.order_send(request)
            success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        
        if success:
            self.stats["success"] += 1
            logger.info(f"✅ Full exit #{ticket}")
            
            # Remove from tracking
            pm = get_position_manager()
            pm.remove_position(ticket)
            
            return ExecutionResult(
                success=True,
                action=PositionAction.FULL_EXIT,
                ticket=ticket,
                message="Position closed"
            )
        else:
            self.stats["failed"] += 1
            return ExecutionResult(
                success=False,
                action=PositionAction.FULL_EXIT,
                ticket=ticket,
                message="Failed to close position"
            )
    
    def _execute_scale_in(self, position, add_volume: float) -> ExecutionResult:
        """Add to position (pyramid entry)."""
        symbol = position.symbol if hasattr(position, 'symbol') else position.get('symbol')
        pos_type = position.type if hasattr(position, 'type') else position.get('type', 0)
        current_sl = position.sl if hasattr(position, 'sl') else position.get('sl', 0)
        current_tp = position.tp if hasattr(position, 'tp') else position.get('tp', 0)
        ticket = position.ticket if hasattr(position, 'ticket') else position.get('ticket')
        
        if add_volume is None or add_volume <= 0:
            return ExecutionResult(
                success=False,
                action=PositionAction.SCALE_IN,
                ticket=ticket,
                message="Invalid add volume"
            )
        
        add_volume = round(add_volume, 2)
        direction = "BUY" if pos_type == 0 else "SELL"
        
        if self.use_command_writer and self.command_writer:
            success = self.command_writer.send_open(
                symbol=symbol,
                direction=direction,
                volume=add_volume,
                sl=current_sl,
                tp=current_tp,
                magic=900001,
                comment="scale_in"
            )
        else:
            order_type = mt5.ORDER_TYPE_BUY if pos_type == 0 else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": add_volume,
                "type": order_type,
                "sl": current_sl,
                "tp": current_tp,
                "magic": 900001,
                "comment": "scale_in",
            }
            result = mt5.order_send(request)
            success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        
        if success:
            self.stats["success"] += 1
            logger.info(f"✅ Scale-in {symbol}: +{add_volume} lots")
            
            # Update state
            pm = get_position_manager()
            current_state = pm._position_states.get(ticket)
            if current_state:
                pm.update_position_state(
                    ticket,
                    scale_ins=current_state.scale_ins + 1
                )
            
            return ExecutionResult(
                success=True,
                action=PositionAction.SCALE_IN,
                ticket=ticket,
                message=f"Added {add_volume} lots",
                details={"volume": add_volume}
            )
        else:
            self.stats["failed"] += 1
            return ExecutionResult(
                success=False,
                action=PositionAction.SCALE_IN,
                ticket=ticket,
                message="Failed to add position"
            )
    
    def get_stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return self.stats.copy()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_executor: Optional[PositionExecutor] = None

def get_position_executor(use_command_writer: bool = True) -> PositionExecutor:
    """Get global PositionExecutor instance."""
    global _executor
    if _executor is None:
        _executor = PositionExecutor(use_command_writer)
    return _executor


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 POSITION EXECUTOR TEST")
    print("=" * 60)
    
    executor = PositionExecutor(use_command_writer=True)
    
    # Mock decision
    decision = ManagementDecision(
        action=PositionAction.MOVE_BE,
        reason="Test BE move",
        new_sl=2650.0,
        r_multiple=1.5
    )
    
    mock_position = {
        "ticket": 12345,
        "symbol": "XAUUSD",
        "type": 0,
        "price_open": 2650.0,
        "sl": 2640.0,
        "tp": 2680.0,
        "volume": 0.1
    }
    
    print(f"\n📊 Mock Decision: {decision.action.value}")
    print(f"   Reason: {decision.reason}")
    print(f"   New SL: {decision.new_sl}")
    
    # Note: Actual execution would fail without MT5 connection
    print("\n⚠️ Actual execution requires MT5 connection")
    print("=" * 60)
