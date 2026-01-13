"""
AI Trading System - Configuration Loader
=========================================
Type-safe configuration loading with validation and environment variable support.

Usage:
    from src.utils.config_loader import get_config, ConfigLoader
    
    config = get_config()
    risk_config = config.risk
    symbol_config = config.get_symbol("EURUSD")
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from functools import lru_cache
import yaml

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# __file__ = src/utils/config_loader.py -> parent.parent.parent = project root
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
SYMBOLS_FILE = CONFIG_DIR / "symbols.yaml"
RISK_PARAMS_FILE = CONFIG_DIR / "risk_params.yaml"



# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class MT5Config(BaseModel):
    """MetaTrader 5 connection configuration."""
    
    terminal_path: str = "C:/Program Files/MetaTrader 5/terminal64.exe"
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    timeout_ms: int = 60000
    retry_attempts: int = 3
    retry_delay_ms: int = 1000
    
    @field_validator("login", "password", "server", mode="before")
    @classmethod
    def load_from_env(cls, v, info):
        """Load sensitive values from environment variables."""
        if v is None:
            env_key = f"MT5_{info.field_name.upper()}"
            return os.getenv(env_key)
        return v


class DataConfig(BaseModel):
    """Data layer configuration."""
    
    primary_timeframe: str = "H1"
    secondary_timeframes: List[str] = ["M15", "H4", "D1"]
    lookback_bars: int = 1000
    min_bars_required: int = 500
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    max_gap_seconds: int = 3600
    max_missing_bars_pct: float = 5.0


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    
    swing_lookback: int = 20
    support_resistance_sensitivity: float = 0.001
    ema_periods: List[int] = [8, 21, 55, 200]
    rsi_period: int = 14
    macd: Dict[str, int] = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
    atr_period: int = 14
    bollinger: Dict[str, Union[int, float]] = {"period": 20, "std_dev": 2.0}
    volatility_percentile_lookback: int = 100
    regime_lookback: int = 50
    trend_threshold: float = 0.6
    volatility_threshold_high: int = 75
    volatility_threshold_low: int = 25


class ModelConfig(BaseModel):
    """Individual model configuration."""
    
    enabled: bool = True
    model_path: str = ""
    version: str = "1.0.0"


class LSTMConfig(ModelConfig):
    """LSTM model specific configuration."""
    
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


class TreeModelConfig(ModelConfig):
    """Tree-based model configuration (XGBoost/LightGBM)."""
    
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.01


class ModelsConfig(BaseModel):
    """All AI models configuration."""
    
    lstm: LSTMConfig = LSTMConfig()
    xgboost: TreeModelConfig = TreeModelConfig()
    lightgbm: TreeModelConfig = TreeModelConfig()
    ppo: ModelConfig = ModelConfig(enabled=False)


class SignalConfig(BaseModel):
    """Signal fusion configuration."""
    
    weights: Dict[str, float] = {"lstm": 0.4, "xgboost": 0.35, "lightgbm": 0.25}
    min_probability: float = 0.55
    min_confidence: float = 0.60
    min_model_agreement: int = 2
    allow_trending: bool = True
    allow_ranging: bool = True
    allow_volatile: bool = False
    allow_quiet: bool = True


class RiskConfig(BaseModel):
    """Risk management configuration."""
    
    risk_per_trade_pct: float = 1.0
    max_risk_per_trade_pct: float = 2.0
    sl_atr_multiplier: float = 2.0
    min_sl_pips: int = 20
    max_sl_pips: int = 100
    min_risk_reward: float = 1.5
    default_risk_reward: float = 2.0
    max_exposure_per_symbol_pct: float = 5.0
    max_total_exposure_pct: float = 15.0
    max_positions_per_symbol: int = 1
    max_total_positions: int = 5
    max_correlation: float = 0.7
    max_daily_drawdown_pct: float = 3.0
    max_weekly_drawdown_pct: float = 5.0
    max_total_drawdown_pct: float = 10.0


class ExecutionConfig(BaseModel):
    """Order execution configuration."""
    
    slippage_points: int = 10
    max_spread_points: int = 30
    magic_number: int = 123456
    order_comment: str = "AI_Trading_v1"
    fill_policy: str = "FOK"
    emergency_stop_enabled: bool = True
    max_consecutive_losses: int = 5
    pause_after_loss_minutes: int = 60


class SymbolConfig(BaseModel):
    """Individual symbol configuration."""
    
    enabled: bool = True
    description: str = ""
    category: str = "forex_major"
    pip_value: float = 0.0001
    contract_size: float = 100000
    min_lot: float = 0.01
    max_lot: float = 100.0
    margin_rate: float = 0.01
    sl_atr_multiplier: Optional[float] = None
    max_spread_points: Optional[int] = None
    risk_per_trade_pct: Optional[float] = None
    correlations: Dict[str, float] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONFIGURATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SystemConfig(BaseModel):
    """Complete system configuration."""
    
    name: str = "AI-Trading-System"
    version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"


class TradingConfig(BaseModel):
    """Master configuration combining all settings."""
    
    system: SystemConfig = SystemConfig()
    mt5: MT5Config = MT5Config()
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    models: ModelsConfig = ModelsConfig()
    signals: SignalConfig = SignalConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    symbols: Dict[str, SymbolConfig] = {}
    symbol_groups: Dict[str, List[str]] = {}
    
    def get_symbol(self, symbol: str) -> Optional[SymbolConfig]:
        """Get configuration for a specific symbol."""
        return self.symbols.get(symbol)
    
    def get_enabled_symbols(self) -> List[str]:
        """Get list of enabled trading symbols."""
        return [s for s, cfg in self.symbols.items() if cfg.enabled]
    
    def get_symbol_param(
        self, 
        symbol: str, 
        param: str, 
        default: Any = None
    ) -> Any:
        """
        Get symbol-specific parameter with fallback to global defaults.
        
        Args:
            symbol: Symbol name
            param: Parameter name
            default: Default value if not found
        
        Returns:
            Parameter value or default
        """
        sym_config = self.symbols.get(symbol)
        if sym_config:
            value = getattr(sym_config, param, None)
            if value is not None:
                return value
        
        # Fallback to risk config
        return getattr(self.risk, param, default)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigLoader:
    """
    Configuration loader with YAML parsing and validation.
    
    Loads and merges configurations from multiple YAML files.
    """
    
    def __init__(
        self,
        settings_path: Optional[Path] = None,
        symbols_path: Optional[Path] = None,
        risk_params_path: Optional[Path] = None,
    ):
        self.settings_path = settings_path or SETTINGS_FILE
        self.symbols_path = symbols_path or SYMBOLS_FILE
        self.risk_params_path = risk_params_path or RISK_PARAMS_FILE
        self._config: Optional[TradingConfig] = None
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def _process_symbols(self, symbols_data: Dict[str, Any]) -> Dict[str, SymbolConfig]:
        """Process symbols configuration with defaults."""
        defaults = symbols_data.get("defaults", {})
        symbols = {}
        
        for symbol, config in symbols_data.get("symbols", {}).items():
            # Merge with defaults
            merged = {**defaults, **config}
            symbols[symbol] = SymbolConfig(**merged)
        
        return symbols
    
    def load(self) -> TradingConfig:
        """
        Load and validate all configuration files.
        
        Returns:
            Validated TradingConfig instance
        """
        # Load YAML files
        settings = self._load_yaml(self.settings_path)
        symbols_data = self._load_yaml(self.symbols_path)
        risk_params = self._load_yaml(self.risk_params_path)
        
        # Process symbols
        symbols = self._process_symbols(symbols_data)
        symbol_groups = symbols_data.get("groups", {})
        
        # Build configuration
        self._config = TradingConfig(
            system=SystemConfig(**settings.get("system", {})),
            mt5=MT5Config(**settings.get("mt5", {})),
            data=DataConfig(**settings.get("data", {})),
            features=FeatureConfig(**settings.get("features", {})),
            models=ModelsConfig(**settings.get("models", {})),
            signals=SignalConfig(**settings.get("signals", {})),
            risk=RiskConfig(**settings.get("risk", {})),
            execution=ExecutionConfig(**settings.get("execution", {})),
            symbols=symbols,
            symbol_groups=symbol_groups,
        )
        
        return self._config
    
    @property
    def config(self) -> TradingConfig:
        """Get loaded configuration (loads if not already loaded)."""
        if self._config is None:
            self.load()
        return self._config


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_config_instance: Optional[TradingConfig] = None


@lru_cache(maxsize=1)
def get_config() -> TradingConfig:
    """
    Get the global configuration instance (singleton).
    
    Returns:
        Validated TradingConfig instance
    """
    global _config_instance
    if _config_instance is None:
        loader = ConfigLoader()
        _config_instance = loader.load()
    return _config_instance


def reload_config() -> TradingConfig:
    """
    Force reload configuration from files.
    
    Returns:
        Fresh TradingConfig instance
    """
    global _config_instance
    get_config.cache_clear()
    _config_instance = None
    return get_config()
