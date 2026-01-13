"""
mt5_command_writer.py
======================
MT5 Command Writer - Python to EA Communication

เขียนคำสั่ง JSON ให้ Zero-Logic EA อ่านและ execute
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional
from src.utils.logger import get_logger

logger = get_logger("MT5_COMMAND")


class MT5CommandWriter:
    """
    Writes trade commands for MT5 EA to execute.
    
    Communication: Python (Brain) → JSON File → MQL5 EA (Executor)
    """

    def __init__(self, 
                 command_file: str = "ai_command.json",
                 response_file: str = "ai_response.json"):
        # MT5 Common folder path
        self.mt5_common = self._get_mt5_common_path()
        self.command_path = os.path.join(self.mt5_common, command_file)
        self.response_path = os.path.join(self.mt5_common, response_file)

    def _get_mt5_common_path(self) -> str:
        """Get MT5 common data folder."""
        # Default paths
        paths = [
            os.path.expanduser("~/AppData/Roaming/MetaQuotes/Terminal/Common/Files"),
            "C:/Users/Public/Documents/MetaQuotes/Terminal/Common/Files",
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        # Fallback
        return os.path.expanduser("~/MT5_Commands")

    # =========================================================
    # WRITE COMMANDS
    # =========================================================
    def send_open(self, symbol: str, direction: str, volume: float,
                  sl: float, tp: float, magic: int = 900001,
                  comment: str = "") -> bool:
        """Send OPEN command."""
        command = {
            "action": "OPEN",
            "symbol": symbol,
            "direction": direction.upper(),
            "volume": round(volume, 2),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "magic": magic,
            "comment": comment or f"AI_{datetime.now().strftime('%H%M%S')}",
        }
        return self._write_command(command)

    def send_close(self, ticket: int = 0, magic: int = 0) -> bool:
        """Send CLOSE command."""
        command = {
            "action": "CLOSE",
            "ticket": ticket,
            "magic": magic,
        }
        return self._write_command(command)

    def send_close_all(self, magic: int = 900001) -> bool:
        """Send CLOSE_ALL command."""
        command = {
            "action": "CLOSE_ALL",
            "magic": magic,
        }
        return self._write_command(command)

    def send_modify(self, ticket: int, sl: float, tp: float) -> bool:
        """Send MODIFY command."""
        command = {
            "action": "MODIFY",
            "ticket": ticket,
            "sl": round(sl, 5),
            "tp": round(tp, 5),
        }
        return self._write_command(command)

    def send_trade(self, trade: Dict) -> bool:
        """Send trade dict from orchestrator."""
        return self.send_open(
            symbol=trade["symbol"],
            direction=trade["direction"],
            volume=trade["volume"],
            sl=trade["sl"],
            tp=trade["tp"],
            magic=trade.get("magic", 900001),
            comment=trade.get("comment", ""),
        )

    # =========================================================
    # INTERNAL
    # =========================================================
    def _write_command(self, command: Dict) -> bool:
        """Write command to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.command_path), exist_ok=True)
            
            # Clear old response file first
            if os.path.exists(self.response_path):
                try:
                    os.remove(self.response_path)
                except:
                    pass
            
            # Write ASCII-only JSON for MQL5 compatibility
            with open(self.command_path, "w", encoding="ascii") as f:
                json.dump(command, f, separators=(',', ':'), ensure_ascii=True)
            
            logger.info(f"Command sent: {command['action']} {command.get('symbol', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write command: {e}")
            return False

    def read_response(self) -> Optional[Dict]:
        """Read response from EA."""
        try:
            if not os.path.exists(self.response_path):
                return None
            
            # Check file size - skip if empty
            if os.path.getsize(self.response_path) < 5:
                return None
            
            # Try different encodings (MQL5 may write UTF-16)
            content = None
            for encoding in ["utf-8", "utf-16", "utf-16-le", "latin-1"]:
                try:
                    with open(self.response_path, "r", encoding=encoding) as f:
                        content = f.read().strip()
                    if content and "{" in content:
                        break
                except:
                    continue
            
            if not content or not "{" in content:
                return None
            
            # Extract JSON from content (skip any BOM or garbage)
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
            
            response = json.loads(content)
            
            # Delete after successful reading
            os.remove(self.response_path)
            
            return response
            
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.error(f"Failed to read response: {e}")
            return None

    def wait_for_response(self, timeout_sec: float = 5.0) -> Optional[Dict]:
        """Wait for EA response."""
        import time
        start = time.time()
        
        while time.time() - start < timeout_sec:
            response = self.read_response()
            if response:
                return response
            time.sleep(0.1)
        
        logger.warning("Response timeout")
        return None


# =========================================================
# GLOBAL
# =========================================================
_writer: Optional[MT5CommandWriter] = None


def get_command_writer() -> MT5CommandWriter:
    global _writer
    if _writer is None:
        _writer = MT5CommandWriter()
    return _writer
