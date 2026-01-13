# src/data/database.py
"""
Database Layer - Fund-Grade Storage
=====================================

SQLite for minimal production-ready storage.

Tables:
- trades: All executed trades
- signals: Signal history
- models: Model versions and metrics
- daily_stats: Daily performance stats

Can be upgraded to PostgreSQL by changing connection string.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger("DATABASE")

DB_PATH = Path("data/trading.db")


@dataclass
class Trade:
    """Trade record."""
    id: Optional[int] = None
    symbol: str = ""
    side: str = ""
    lot: float = 0.0
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    confidence: float = 0.0
    model_version: str = ""
    ticket: int = 0
    profit: float = 0.0
    opened_at: str = ""
    closed_at: Optional[str] = None
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED


@dataclass
class Signal:
    """Signal record."""
    id: Optional[int] = None
    symbol: str = ""
    action: str = ""
    confidence: float = 0.0
    model_version: str = ""
    features: str = ""  # JSON
    reason: str = ""
    created_at: str = ""


@dataclass
class ModelRecord:
    """Model record."""
    id: Optional[int] = None
    model_name: str = ""
    version: str = ""
    accuracy: float = 0.0
    winrate: float = 0.0
    drawdown: float = 0.0
    sharpe: float = 0.0
    samples: int = 0
    is_active: bool = False
    trained_at: str = ""


@dataclass
class DailyStat:
    """Daily statistics."""
    id: Optional[int] = None
    date: str = ""
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    profit: float = 0.0
    drawdown: float = 0.0
    max_dd: float = 0.0


class TradingDatabase:
    """
    Fund-Grade SQLite Database.
    
    Production-ready storage for:
    - Trade history
    - Signal logs
    - Model versions
    - Daily stats
    """

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_tables()
        logger.info(f"Database initialized: {self.db_path}")

    # =========================================================
    # CONNECTION
    # =========================================================
    
    @contextmanager
    def _connect(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_tables(self):
        """Initialize database tables."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    lot REAL NOT NULL,
                    entry_price REAL,
                    sl REAL,
                    tp REAL,
                    confidence REAL,
                    model_version TEXT,
                    ticket INTEGER,
                    profit REAL DEFAULT 0,
                    opened_at TEXT,
                    closed_at TEXT,
                    status TEXT DEFAULT 'OPEN'
                )
            """)
            
            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL,
                    model_version TEXT,
                    features TEXT,
                    reason TEXT,
                    created_at TEXT
                )
            """)
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    accuracy REAL,
                    winrate REAL,
                    drawdown REAL,
                    sharpe REAL,
                    samples INTEGER,
                    is_active INTEGER DEFAULT 0,
                    trained_at TEXT,
                    UNIQUE(model_name, version)
                )
            """)
            
            # Daily stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    trades_count INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    profit REAL DEFAULT 0,
                    drawdown REAL DEFAULT 0,
                    max_dd REAL DEFAULT 0
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active)")

    # =========================================================
    # TRADES
    # =========================================================
    
    def insert_trade(self, trade: Trade) -> int:
        """Insert a new trade."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (symbol, side, lot, entry_price, sl, tp, 
                    confidence, model_version, ticket, profit, opened_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.side, trade.lot, trade.entry_price,
                trade.sl, trade.tp, trade.confidence, trade.model_version,
                trade.ticket, trade.profit, 
                trade.opened_at or datetime.now().isoformat(),
                trade.status
            ))
            return cursor.lastrowid

    def close_trade(self, ticket: int, profit: float):
        """Close a trade."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades 
                SET profit = ?, closed_at = ?, status = 'CLOSED'
                WHERE ticket = ? AND status = 'OPEN'
            """, (profit, datetime.now().isoformat(), ticket))

    def get_open_trades(self, symbol: str = None) -> List[Dict]:
        """Get all open trades."""
        with self._connect() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM trades WHERE status = 'OPEN' AND symbol = ?",
                    (symbol,)
                )
            else:
                cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            return [dict(row) for row in cursor.fetchall()]

    def get_trades(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get recent trades."""
        with self._connect() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================
    # SIGNALS
    # =========================================================
    
    def insert_signal(self, signal: Signal) -> int:
        """Insert a signal."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals (symbol, action, confidence, model_version, 
                    features, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol, signal.action, signal.confidence,
                signal.model_version, signal.features, signal.reason,
                signal.created_at or datetime.now().isoformat()
            ))
            return cursor.lastrowid

    def get_signals(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get recent signals."""
        with self._connect() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM signals WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================
    # MODELS
    # =========================================================
    
    def insert_model(self, model: ModelRecord) -> int:
        """Insert a model record."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO models (model_name, version, accuracy, 
                    winrate, drawdown, sharpe, samples, is_active, trained_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.model_name, model.version, model.accuracy,
                model.winrate, model.drawdown, model.sharpe,
                model.samples, 1 if model.is_active else 0,
                model.trained_at or datetime.now().isoformat()
            ))
            return cursor.lastrowid

    def set_active_model(self, model_name: str, version: str):
        """Set a model as active."""
        with self._connect() as conn:
            cursor = conn.cursor()
            # Deactivate all
            cursor.execute(
                "UPDATE models SET is_active = 0 WHERE model_name = ?",
                (model_name,)
            )
            # Activate specified
            cursor.execute(
                "UPDATE models SET is_active = 1 WHERE model_name = ? AND version = ?",
                (model_name, version)
            )

    def get_active_model(self, model_name: str) -> Optional[Dict]:
        """Get active model."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM models WHERE model_name = ? AND is_active = 1",
                (model_name,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_model_history(self, model_name: str) -> List[Dict]:
        """Get model version history."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM models WHERE model_name = ? ORDER BY trained_at DESC",
                (model_name,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================
    # DAILY STATS
    # =========================================================
    
    def update_daily_stat(self, stat: DailyStat):
        """Update or insert daily stats."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO daily_stats 
                    (date, trades_count, wins, losses, profit, drawdown, max_dd)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                stat.date or date.today().isoformat(),
                stat.trades_count, stat.wins, stat.losses,
                stat.profit, stat.drawdown, stat.max_dd
            ))

    def get_daily_stats(self, days: int = 30) -> List[Dict]:
        """Get recent daily stats."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?",
                (days,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stat_by_date(self, date_str: str) -> Optional[Dict]:
        """Get stats for a specific date."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM daily_stats WHERE date = ?",
                (date_str,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================
    # UTILITIES
    # =========================================================
    
    def get_summary(self) -> Dict:
        """Get database summary."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM trades")
            trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'")
            open_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM signals")
            signals = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM models")
            models = cursor.fetchone()[0]
            
            return {
                "total_trades": trades,
                "open_trades": open_trades,
                "total_signals": signals,
                "total_models": models,
                "db_path": str(self.db_path)
            }


# =========================================================
# GLOBAL INSTANCE
# =========================================================

_db: Optional[TradingDatabase] = None

def get_database() -> TradingDatabase:
    """Get global database instance."""
    global _db
    if _db is None:
        _db = TradingDatabase()
    return _db
