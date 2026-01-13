"""
AI Trading System - Professional Logging Module
================================================
Structured logging with rotation, multiple outputs, and trade-specific formatting.

Usage:
    from src.utils.logger import get_logger, setup_logging
    
    setup_logging()  # Call once at startup
    logger = get_logger(__name__)
    logger.info("Trade executed", extra={"symbol": "EURUSD", "action": "BUY"})
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from dataclasses import dataclass, asdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

LOG_DIR = Path(__file__).parent.parent.parent.parent / "logs"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
JSON_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeLogEntry:
    """Structured trade log entry for analytics."""
    
    timestamp: str
    event_type: str  # SIGNAL | DECISION | EXECUTION | CLOSE | ERROR
    symbol: str
    direction: Optional[str] = None
    probability: Optional[float] = None
    confidence: Optional[float] = None
    regime: Optional[str] = None
    action: Optional[str] = None  # TRADE | NO_TRADE | REJECTED
    lot_size: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    ticket: Optional[int] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().strftime(JSON_DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "message", "thread",
                "threadName", "taskName"
            ):
                log_data[key] = value
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class TradeFormatter(logging.Formatter):
    """CSV formatter for trade logs."""
    
    COLUMNS = [
        "timestamp", "event_type", "symbol", "direction", "probability",
        "confidence", "regime", "action", "lot_size", "entry_price",
        "stop_loss", "take_profit", "ticket", "pnl", "reason"
    ]
    
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "trade_entry") and isinstance(record.trade_entry, TradeLogEntry):
            entry = record.trade_entry.to_dict()
            values = [str(entry.get(col, "")) for col in self.COLUMNS]
            return ",".join(values)
        return record.getMessage()


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

class TradeLogHandler(logging.Handler):
    """Handler specifically for trade events - writes to CSV."""
    
    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write header if file doesn't exist
        if not self.filepath.exists():
            with open(self.filepath, "w") as f:
                f.write(",".join(TradeFormatter.COLUMNS) + "\n")
    
    def emit(self, record: logging.LogRecord):
        try:
            if hasattr(record, "trade_entry"):
                msg = self.format(record)
                with open(self.filepath, "a") as f:
                    f.write(msg + "\n")
        except Exception:
            self.handleError(record)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGER SETUP
# ═══════════════════════════════════════════════════════════════════════════════

_loggers: Dict[str, logging.Logger] = {}
_is_initialized = False


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: Optional[Path] = None,
    json_format: bool = True,
    colored_console: bool = True,
    trade_log_enabled: bool = True,
    max_file_size_mb: int = 100,
    backup_count: int = 10,
) -> None:
    """
    Initialize the logging system.
    
    Args:
        console_level: Log level for console output
        file_level: Log level for file output
        log_dir: Directory for log files
        json_format: Use JSON format for file logs
        colored_console: Use colored output in console
        trade_log_enabled: Enable separate trade log file
        max_file_size_mb: Max size per log file
        backup_count: Number of backup files to keep
    """
    global _is_initialized
    
    if _is_initialized:
        return
    
    log_path = log_dir or LOG_DIR
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    
    if colored_console and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(DEFAULT_LOG_FORMAT))
    else:
        console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    
    root_logger.addHandler(console_handler)
    
    # File handler - General logs
    general_log_file = log_path / "trading.log"
    file_handler = RotatingFileHandler(
        general_log_file,
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, file_level.upper()))
    
    if json_format:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    
    root_logger.addHandler(file_handler)
    
    # Trade log handler
    if trade_log_enabled:
        trade_log_file = log_path / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
        trade_handler = TradeLogHandler(trade_log_file)
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(TradeFormatter())
        
        # Only attach to trade-specific logger
        trade_logger = logging.getLogger("trades")
        trade_logger.addHandler(trade_handler)
        trade_logger.propagate = True
    
    _is_initialized = True
    
    # Log initialization
    root_logger.info(
        "Logging system initialized",
        extra={
            "log_dir": str(log_path),
            "console_level": console_level,
            "file_level": file_level,
            "json_format": json_format,
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


def get_trade_logger() -> logging.Logger:
    """Get the dedicated trade logger."""
    return logging.getLogger("trades")


def log_trade_event(entry: TradeLogEntry) -> None:
    """
    Log a trade event to the trade log.
    
    Args:
        entry: TradeLogEntry with trade details
    """
    logger = get_trade_logger()
    logger.info(
        f"Trade event: {entry.event_type}",
        extra={"trade_entry": entry}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def log_signal(
    logger: logging.Logger,
    symbol: str,
    direction: str,
    probability: float,
    confidence: float,
    regime: str,
    **kwargs,
) -> None:
    """Helper to log AI signal generation."""
    entry = TradeLogEntry(
        timestamp=datetime.utcnow().strftime(JSON_DATE_FORMAT),
        event_type="SIGNAL",
        symbol=symbol,
        direction=direction,
        probability=probability,
        confidence=confidence,
        regime=regime,
        metadata=kwargs if kwargs else None,
    )
    log_trade_event(entry)
    logger.info(
        f"Signal generated: {symbol} {direction} (prob={probability:.2%}, conf={confidence:.2%})",
        extra=entry.to_dict(),
    )


def log_decision(
    logger: logging.Logger,
    symbol: str,
    action: str,
    reason: str,
    **kwargs,
) -> None:
    """Helper to log trade decision."""
    entry = TradeLogEntry(
        timestamp=datetime.utcnow().strftime(JSON_DATE_FORMAT),
        event_type="DECISION",
        symbol=symbol,
        action=action,
        reason=reason,
        metadata=kwargs if kwargs else None,
    )
    log_trade_event(entry)
    
    level = logging.INFO if action == "TRADE" else logging.DEBUG
    logger.log(level, f"Decision: {symbol} -> {action} | {reason}", extra=entry.to_dict())


def log_execution(
    logger: logging.Logger,
    symbol: str,
    direction: str,
    lot_size: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    ticket: int,
    **kwargs,
) -> None:
    """Helper to log trade execution."""
    entry = TradeLogEntry(
        timestamp=datetime.utcnow().strftime(JSON_DATE_FORMAT),
        event_type="EXECUTION",
        symbol=symbol,
        direction=direction,
        lot_size=lot_size,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        ticket=ticket,
        metadata=kwargs if kwargs else None,
    )
    log_trade_event(entry)
    logger.info(
        f"Trade executed: {symbol} {direction} {lot_size} lots @ {entry_price} "
        f"[SL: {stop_loss}, TP: {take_profit}] Ticket: {ticket}",
        extra=entry.to_dict(),
    )


def log_close(
    logger: logging.Logger,
    symbol: str,
    ticket: int,
    pnl: float,
    reason: str,
    **kwargs,
) -> None:
    """Helper to log position close."""
    entry = TradeLogEntry(
        timestamp=datetime.utcnow().strftime(JSON_DATE_FORMAT),
        event_type="CLOSE",
        symbol=symbol,
        ticket=ticket,
        pnl=pnl,
        reason=reason,
        metadata=kwargs if kwargs else None,
    )
    log_trade_event(entry)
    
    level = logging.INFO if pnl >= 0 else logging.WARNING
    logger.log(level, f"Position closed: {symbol} #{ticket} P/L: {pnl:.2f} | {reason}", extra=entry.to_dict())
