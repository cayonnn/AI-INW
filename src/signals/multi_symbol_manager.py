"""
AI Trading System - Multi-Symbol Manager
==========================================
Manages trading across multiple symbols with correlation checks.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from src.models.signal_schema import AISignal, SignalDirection
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


@dataclass
class SymbolState:
    """State for individual symbol."""
    symbol: str
    enabled: bool = True
    has_position: bool = False
    last_signal: Optional[AISignal] = None
    exposure_pct: float = 0.0
    correlation_blocked: Set[str] = field(default_factory=set)


class MultiSymbolManager:
    """
    Manages trading operations across multiple symbols.
    
    Features:
    - Correlation-based position blocking
    - Per-symbol exposure tracking
    - Unified signal collection
    """
    
    def __init__(self, symbols: Optional[List[str]] = None, max_correlation: float = 0.7,
                 max_total_exposure: float = 15.0, max_per_symbol_exposure: float = 5.0):
        config = get_config()
        
        self.symbols = symbols or config.get_enabled_symbols()
        self.max_correlation = max_correlation or config.risk.max_correlation
        self.max_total_exposure = max_total_exposure
        self.max_per_symbol_exposure = max_per_symbol_exposure
        
        self.states: Dict[str, SymbolState] = {s: SymbolState(symbol=s) for s in self.symbols}
        self.correlation_matrix: Optional[pd.DataFrame] = None
    
    def update_correlation_matrix(self, price_data: Dict[str, pd.DataFrame], lookback: int = 100):
        """Calculate correlation matrix from recent price data."""
        returns = {}
        for symbol, df in price_data.items():
            if len(df) >= lookback:
                returns[symbol] = df["close"].pct_change().tail(lookback)
        
        if len(returns) >= 2:
            returns_df = pd.DataFrame(returns)
            self.correlation_matrix = returns_df.corr()
            logger.debug(f"Updated correlation matrix for {len(returns)} symbols")
    
    def check_correlation_allowed(self, symbol: str, direction: SignalDirection) -> bool:
        """Check if opening position would violate correlation limits."""
        if self.correlation_matrix is None:
            return True
        
        for other_symbol, state in self.states.items():
            if other_symbol == symbol or not state.has_position:
                continue
            
            if symbol in self.correlation_matrix.columns and other_symbol in self.correlation_matrix.columns:
                corr = abs(self.correlation_matrix.loc[symbol, other_symbol])
                
                if corr > self.max_correlation:
                    # Same direction on correlated pairs = blocked
                    if state.last_signal and state.last_signal.direction == direction:
                        logger.info(f"Blocked {symbol}: correlated with {other_symbol} (r={corr:.2f})")
                        return False
        
        return True
    
    def check_exposure_allowed(self, symbol: str, new_exposure_pct: float) -> bool:
        """Check if adding exposure would exceed limits."""
        # Per-symbol check
        current_exposure = self.states[symbol].exposure_pct
        if current_exposure + new_exposure_pct > self.max_per_symbol_exposure:
            logger.info(f"Blocked {symbol}: would exceed per-symbol limit")
            return False
        
        # Total exposure check
        total_exposure = sum(s.exposure_pct for s in self.states.values())
        if total_exposure + new_exposure_pct > self.max_total_exposure:
            logger.info(f"Blocked {symbol}: would exceed total exposure limit")
            return False
        
        return True
    
    def update_position(self, symbol: str, has_position: bool, exposure_pct: float = 0.0,
                       signal: Optional[AISignal] = None):
        """Update symbol position state."""
        if symbol in self.states:
            self.states[symbol].has_position = has_position
            self.states[symbol].exposure_pct = exposure_pct if has_position else 0.0
            self.states[symbol].last_signal = signal
    
    def get_active_symbols(self) -> List[str]:
        """Get list of symbols with open positions."""
        return [s for s, state in self.states.items() if state.has_position]
    
    def get_available_symbols(self) -> List[str]:
        """Get symbols available for new trades."""
        return [s for s, state in self.states.items() if state.enabled and not state.has_position]
    
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure percentage."""
        return sum(s.exposure_pct for s in self.states.values())
    
    def can_trade(self, symbol: str, direction: SignalDirection, exposure_pct: float) -> tuple:
        """Check if trade is allowed, return (allowed, reason)."""
        if symbol not in self.states:
            return False, "Symbol not configured"
        
        if not self.states[symbol].enabled:
            return False, "Symbol disabled"
        
        if self.states[symbol].has_position:
            return False, "Already has position"
        
        if not self.check_correlation_allowed(symbol, direction):
            return False, "Correlation limit violated"
        
        if not self.check_exposure_allowed(symbol, exposure_pct):
            return False, "Exposure limit violated"
        
        return True, "Allowed"
