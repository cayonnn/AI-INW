"""
AI Trading System - MetaTrader 5 Data Connector
=================================================
Professional MT5 connection with rate limiting, error handling, and data caching.

Usage:
    from src.data.mt5_connector import MT5Connector
    
    connector = MT5Connector()
    if connector.connect():
        df = connector.get_rates("EURUSD", "H1", 1000)
        connector.disconnect()
"""

import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from src.utils.logger import get_logger
from src.utils.config_loader import get_config


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

logger = get_logger(__name__)


class Timeframe(Enum):
    """MT5 Timeframe mappings."""
    M1 = auto()
    M5 = auto()
    M15 = auto()
    M30 = auto()
    H1 = auto()
    H4 = auto()
    D1 = auto()
    W1 = auto()
    MN1 = auto()


# MT5 timeframe constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
    "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
    "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 16385,
    "H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 16388,
    "D1": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 16408,
    "W1": mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 32769,
    "MN1": mt5.TIMEFRAME_MN1 if MT5_AVAILABLE else 49153,
}


@dataclass
class SymbolInfo:
    """Symbol information from MT5."""
    
    name: str
    description: str
    path: str
    point: float
    digits: int
    spread: int
    tick_value: float
    tick_size: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_mode: int
    swap_long: float
    swap_short: float
    margin_initial: float


@dataclass  
class AccountInfo:
    """Account information from MT5."""
    
    login: int
    server: str
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    profit: float
    currency: str
    leverage: int
    trade_mode: int


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 CONNECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MT5Connector:
    """
    MetaTrader 5 data connector with robust error handling.
    
    Features:
    - Automatic retry on connection failure
    - Rate limiting to prevent API abuse
    - Data caching for efficiency
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        terminal_path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        timeout_ms: int = 60000,
    ):
        """
        Initialize MT5 connector.
        
        Args:
            terminal_path: Path to MT5 terminal (optional, uses config)
            login: Account login (optional, uses config/env)
            password: Account password (optional, uses config/env)
            server: Broker server (optional, uses config/env)
            timeout_ms: Connection timeout in milliseconds
        """
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 package not installed. Install with: pip install MetaTrader5")
        
        config = get_config()
        mt5_config = config.mt5
        
        self.terminal_path = terminal_path or mt5_config.terminal_path
        self.login = login or mt5_config.login
        self.password = password or mt5_config.password
        self.server = server or mt5_config.server
        self.timeout_ms = timeout_ms or mt5_config.timeout_ms
        self.retry_attempts = mt5_config.retry_attempts
        self.retry_delay_ms = mt5_config.retry_delay_ms
        
        self._connected = False
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms rate limit
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._cache_ttl = config.data.cache_ttl_seconds
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal.
        
        Returns:
            True if connection successful, False otherwise
        """
        for attempt in range(self.retry_attempts):
            try:
                # Initialize MT5
                init_kwargs = {"path": self.terminal_path} if self.terminal_path else {}
                
                if not mt5.initialize(**init_kwargs):
                    error = mt5.last_error()
                    logger.warning(f"MT5 init failed (attempt {attempt + 1}): {error}")
                    time.sleep(self.retry_delay_ms / 1000)
                    continue
                
                # Login if credentials provided
                if self.login and self.password and self.server:
                    if not mt5.login(
                        login=self.login,
                        password=self.password,
                        server=self.server,
                        timeout=self.timeout_ms,
                    ):
                        error = mt5.last_error()
                        logger.warning(f"MT5 login failed (attempt {attempt + 1}): {error}")
                        mt5.shutdown()
                        time.sleep(self.retry_delay_ms / 1000)
                        continue
                
                self._connected = True
                account = self.get_account_info()
                logger.info(
                    f"MT5 connected successfully",
                    extra={
                        "login": account.login if account else "N/A",
                        "server": account.server if account else "N/A",
                        "balance": account.balance if account else 0,
                    }
                )
                return True
                
            except Exception as e:
                logger.error(f"MT5 connection error (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay_ms / 1000)
        
        logger.error("Failed to connect to MT5 after all retries")
        return False
    
    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            self._cache.clear()
            logger.info("MT5 disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        if not self._connected:
            return False
        
        # Verify connection is still alive
        info = mt5.terminal_info()
        if info is None:
            self._connected = False
            return False
        return True
    
    def ensure_connected(self) -> bool:
        """Ensure connection is established, reconnect if needed."""
        if self.is_connected():
            return True
        return self.connect()
    
    # ─────────────────────────────────────────────────────────────────────────
    # RATE LIMITING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    # ─────────────────────────────────────────────────────────────────────────
    # CACHING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _get_cache_key(self, symbol: str, timeframe: str, count: int) -> str:
        """Generate cache key for data request."""
        key_str = f"{symbol}_{timeframe}_{count}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if valid."""
        if key in self._cache:
            df, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return df.copy()
            else:
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, df: pd.DataFrame) -> None:
        """Store data in cache."""
        self._cache[key] = (df.copy(), time.time())
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA RETRIEVAL
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_rates(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        start_pos: int = 0,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV rates from MT5.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "H1", "M15")
            count: Number of bars to retrieve
            start_pos: Starting position (0 = current bar)
            use_cache: Whether to use cache
        
        Returns:
            DataFrame with columns [time, open, high, low, close, tick_volume, spread, real_volume]
            or None if error
        """
        if not self.ensure_connected():
            logger.error("Not connected to MT5")
            return None
        
        # Check cache
        cache_key = self._get_cache_key(symbol, timeframe, count)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {symbol} {timeframe}")
                return cached
        
        # Rate limit
        self._rate_limit()
        
        # Get MT5 timeframe constant
        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        try:
            # Fetch rates
            rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
            
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.error(f"Failed to get rates for {symbol}: {error}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.columns = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            
            # Rename tick_volume to volume for convenience
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            
            # Cache result
            if use_cache:
                self._set_cache(cache_key, df)
            
            logger.debug(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting rates for {symbol}: {e}")
            return None
    
    def get_rates_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV rates for a specific date range.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            date_from: Start datetime
            date_to: End datetime
        
        Returns:
            DataFrame or None
        """
        if not self.ensure_connected():
            return None
        
        self._rate_limit()
        
        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        try:
            rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
            
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.error(f"Failed to get rates range: {error}")
                return None
            
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.columns = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting rates range: {e}")
            return None
    
    def get_current_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current tick data for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dict with bid, ask, last, time, etc. or None
        """
        if not self.ensure_connected():
            return None
        
        self._rate_limit()
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "time": datetime.fromtimestamp(tick.time),
                "flags": tick.flags,
            }
        except Exception as e:
            logger.error(f"Error getting tick for {symbol}: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # SYMBOL INFORMATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get detailed symbol information.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            SymbolInfo dataclass or None
        """
        if not self.ensure_connected():
            return None
        
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            
            return SymbolInfo(
                name=info.name,
                description=info.description,
                path=info.path,
                point=info.point,
                digits=info.digits,
                spread=info.spread,
                tick_value=info.trade_tick_value,
                tick_size=info.trade_tick_size,
                contract_size=info.trade_contract_size,
                volume_min=info.volume_min,
                volume_max=info.volume_max,
                volume_step=info.volume_step,
                trade_mode=info.trade_mode,
                swap_long=info.swap_long,
                swap_short=info.swap_short,
                margin_initial=info.margin_initial,
            )
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_symbols(
        self,
        group: Optional[str] = None,
        trade_mode: Optional[int] = None,
    ) -> List[str]:
        """
        Get list of available symbols.
        
        Args:
            group: Symbol group filter (e.g., "*USD*")
            trade_mode: Trade mode filter
        
        Returns:
            List of symbol names
        """
        if not self.ensure_connected():
            return []
        
        try:
            if group:
                symbols = mt5.symbols_get(group=group)
            else:
                symbols = mt5.symbols_get()
            
            if symbols is None:
                return []
            
            names = [s.name for s in symbols]
            
            # Filter by trade mode if specified
            if trade_mode is not None:
                names = [s.name for s in symbols if s.trade_mode == trade_mode]
            
            return names
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []
    
    def select_symbol(self, symbol: str) -> bool:
        """
        Enable symbol in Market Watch.
        
        Args:
            symbol: Symbol to enable
        
        Returns:
            True if successful
        """
        if not self.ensure_connected():
            return False
        
        try:
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Failed to select symbol: {symbol}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error selecting symbol {symbol}: {e}")
            return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # ACCOUNT INFORMATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get current account information.
        
        Returns:
            AccountInfo dataclass or None
        """
        if not self.ensure_connected():
            return None
        
        try:
            info = mt5.account_info()
            if info is None:
                return None
            
            return AccountInfo(
                login=info.login,
                server=info.server,
                balance=info.balance,
                equity=info.equity,
                margin=info.margin,
                margin_free=info.margin_free,
                margin_level=info.margin_level if info.margin_level else 0.0,
                profit=info.profit,
                currency=info.currency,
                leverage=info.leverage,
                trade_mode=info.trade_mode,
            )
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-SYMBOL DATA
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_multi_symbol_rates(
        self,
        symbols: List[str],
        timeframe: str,
        count: int,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get rates for multiple symbols efficiently.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe string
            count: Number of bars
        
        Returns:
            Dict mapping symbol to DataFrame
        """
        result = {}
        
        for symbol in symbols:
            df = self.get_rates(symbol, timeframe, count)
            if df is not None:
                result[symbol] = df
            else:
                logger.warning(f"Failed to get data for {symbol}")
        
        return result
    
    def get_multi_timeframe_rates(
        self,
        symbol: str,
        timeframes: List[str],
        count: int,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get rates for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframe strings
            count: Number of bars
        
        Returns:
            Dict mapping timeframe to DataFrame
        """
        result = {}
        
        for tf in timeframes:
            df = self.get_rates(symbol, tf, count)
            if df is not None:
                result[tf] = df
            else:
                logger.warning(f"Failed to get {tf} data for {symbol}")
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONTEXT MANAGER
    # ─────────────────────────────────────────────────────────────────────────
    
    def __enter__(self) -> "MT5Connector":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
