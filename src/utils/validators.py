"""
AI Trading System - Input Validators
=====================================
Validation functions for signals, trade parameters, and data quality.

Usage:
    from src.utils.validators import validate_signal, validate_trade_params
    
    is_valid, errors = validate_signal(signal_dict)
    is_valid, errors = validate_trade_params(params_dict)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS FOR VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(str, Enum):
    """Valid trade directions."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Regime(str, Enum):
    """Valid market regimes."""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"


class VolatilityState(str, Enum):
    """Valid volatility states."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of validation operation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    @classmethod
    def success(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create successful validation result."""
        return cls(is_valid=True, errors=[], warnings=warnings or [])
    
    @classmethod
    def failure(cls, errors: List[str], warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create failed validation result."""
        return cls(is_valid=False, errors=errors, warnings=warnings or [])


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_signal(signal: Dict[str, Any]) -> ValidationResult:
    """
    Validate an AI signal dictionary.
    
    Expected signal structure:
    {
        "direction": "LONG" | "SHORT" | "NEUTRAL",
        "probability": float (0.0 - 1.0),
        "confidence": float (0.0 - 1.0),
        "expected_rr": float (> 0),
        "volatility_state": "LOW" | "NORMAL" | "HIGH",
        "regime": "TRENDING" | "RANGING" | "VOLATILE" | "QUIET",
        "timestamp": datetime or ISO string,
        "model_version": str
    }
    
    Args:
        signal: Signal dictionary to validate
    
    Returns:
        ValidationResult with is_valid flag and any errors
    """
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ["direction", "probability", "confidence", "regime"]
    for field in required_fields:
        if field not in signal:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return ValidationResult.failure(errors)
    
    # Direction validation
    if signal["direction"] not in [d.value for d in Direction]:
        errors.append(f"Invalid direction: {signal['direction']}. Must be LONG, SHORT, or NEUTRAL")
    
    # Probability validation
    prob = signal.get("probability")
    if not isinstance(prob, (int, float)):
        errors.append(f"Probability must be a number, got {type(prob).__name__}")
    elif not 0.0 <= prob <= 1.0:
        errors.append(f"Probability must be between 0 and 1, got {prob}")
    elif prob < 0.51 and signal.get("direction") != "NEUTRAL":
        warnings.append(f"Probability {prob:.2%} is close to random chance")
    
    # Confidence validation
    conf = signal.get("confidence")
    if not isinstance(conf, (int, float)):
        errors.append(f"Confidence must be a number, got {type(conf).__name__}")
    elif not 0.0 <= conf <= 1.0:
        errors.append(f"Confidence must be between 0 and 1, got {conf}")
    
    # Regime validation
    if signal.get("regime") not in [r.value for r in Regime]:
        errors.append(f"Invalid regime: {signal.get('regime')}")
    
    # Optional field validations
    if "expected_rr" in signal:
        rr = signal["expected_rr"]
        if not isinstance(rr, (int, float)) or rr <= 0:
            errors.append(f"Expected R:R must be a positive number, got {rr}")
    
    if "volatility_state" in signal:
        if signal["volatility_state"] not in [v.value for v in VolatilityState]:
            errors.append(f"Invalid volatility_state: {signal['volatility_state']}")
    
    if "timestamp" in signal:
        ts = signal["timestamp"]
        if isinstance(ts, str):
            try:
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"Invalid timestamp format: {ts}")
        elif not isinstance(ts, datetime):
            errors.append(f"Timestamp must be datetime or ISO string, got {type(ts).__name__}")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE PARAMETERS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_trade_params(
    params: Dict[str, Any],
    min_sl_pips: float = 20,
    max_sl_pips: float = 100,
    min_rr: float = 1.0,
    max_lot: float = 100.0,
) -> ValidationResult:
    """
    Validate trade execution parameters.
    
    Expected params structure:
    {
        "symbol": str,
        "direction": "LONG" | "SHORT",
        "lot_size": float,
        "entry_price": float,
        "stop_loss": float,
        "take_profit": float,
        "magic_number": int (optional),
        "comment": str (optional)
    }
    
    Args:
        params: Trade parameters dictionary
        min_sl_pips: Minimum allowed stop loss in pips
        max_sl_pips: Maximum allowed stop loss in pips
        min_rr: Minimum required risk:reward ratio
        max_lot: Maximum allowed lot size
    
    Returns:
        ValidationResult with is_valid flag and any errors
    """
    errors = []
    warnings = []
    
    # Required fields
    required = ["symbol", "direction", "lot_size", "entry_price", "stop_loss", "take_profit"]
    for field in required:
        if field not in params:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return ValidationResult.failure(errors)
    
    # Direction
    if params["direction"] not in ["LONG", "SHORT"]:
        errors.append(f"Direction must be LONG or SHORT, got {params['direction']}")
    
    # Lot size
    lot = params["lot_size"]
    if not isinstance(lot, (int, float)) or lot <= 0:
        errors.append(f"Lot size must be positive, got {lot}")
    elif lot > max_lot:
        errors.append(f"Lot size {lot} exceeds maximum {max_lot}")
    elif lot < 0.01:
        errors.append(f"Lot size {lot} is below minimum 0.01")
    
    # Price validations
    entry = params["entry_price"]
    sl = params["stop_loss"]
    tp = params["take_profit"]
    
    for name, value in [("entry_price", entry), ("stop_loss", sl), ("take_profit", tp)]:
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"{name} must be a positive number, got {value}")
    
    if not errors:
        direction = params["direction"]
        
        # SL direction check
        if direction == "LONG" and sl >= entry:
            errors.append(f"LONG trade: stop_loss ({sl}) must be below entry ({entry})")
        elif direction == "SHORT" and sl <= entry:
            errors.append(f"SHORT trade: stop_loss ({sl}) must be above entry ({entry})")
        
        # TP direction check
        if direction == "LONG" and tp <= entry:
            errors.append(f"LONG trade: take_profit ({tp}) must be above entry ({entry})")
        elif direction == "SHORT" and tp >= entry:
            errors.append(f"SHORT trade: take_profit ({tp}) must be below entry ({entry})")
        
        # Risk:Reward calculation
        if not errors:
            sl_distance = abs(entry - sl)
            tp_distance = abs(tp - entry)
            
            if sl_distance > 0:
                actual_rr = tp_distance / sl_distance
                if actual_rr < min_rr:
                    errors.append(f"Risk:Reward ratio {actual_rr:.2f} is below minimum {min_rr}")
            else:
                errors.append("Stop loss distance is zero")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_ohlcv_data(
    df: pd.DataFrame,
    min_bars: int = 100,
    max_gap_seconds: int = 3600,
    max_missing_pct: float = 5.0,
) -> ValidationResult:
    """
    Validate OHLCV DataFrame for data quality.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume] 
            and DatetimeIndex
        min_bars: Minimum required number of bars
        max_gap_seconds: Maximum allowed gap between bars
        max_missing_pct: Maximum allowed percentage of missing values
    
    Returns:
        ValidationResult with quality assessment
    """
    errors = []
    warnings = []
    
    # Check DataFrame structure
    required_columns = ["open", "high", "low", "close"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
        return ValidationResult.failure(errors)
    
    # Check minimum bars
    if len(df) < min_bars:
        errors.append(f"Insufficient data: {len(df)} bars, minimum {min_bars} required")
    
    # Check for missing values
    if df[required_columns].isnull().any().any():
        missing_pct = df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns)) * 100
        if missing_pct > max_missing_pct:
            errors.append(f"Too many missing values: {missing_pct:.1f}% (max {max_missing_pct}%)")
        else:
            warnings.append(f"Missing values detected: {missing_pct:.1f}%")
    
    # Check OHLC consistency
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    if invalid_ohlc.any():
        count = invalid_ohlc.sum()
        errors.append(f"Invalid OHLC values in {count} bars (high < low or similar)")
    
    # Check for zero/negative prices
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append("Zero or negative prices detected")
    
    # Check time gaps (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        time_diffs = df.index.to_series().diff()
        if time_diffs.notna().any():
            max_gap = time_diffs.max()
            if pd.notna(max_gap) and max_gap.total_seconds() > max_gap_seconds:
                warnings.append(f"Large time gap detected: {max_gap}")
    
    # Check for duplicate timestamps
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        errors.append(f"Duplicate timestamps: {dup_count}")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_features(
    features: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    check_nan: bool = True,
    check_inf: bool = True,
) -> ValidationResult:
    """
    Validate feature array for model input.
    
    Args:
        features: Feature array
        expected_shape: Expected shape (ignores None dimensions)
        check_nan: Check for NaN values
        check_inf: Check for infinite values
    
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    if not isinstance(features, np.ndarray):
        errors.append(f"Features must be numpy array, got {type(features).__name__}")
        return ValidationResult.failure(errors)
    
    # Shape check
    if expected_shape is not None:
        for i, (actual, expected) in enumerate(zip(features.shape, expected_shape)):
            if expected is not None and actual != expected:
                errors.append(f"Shape mismatch at dim {i}: expected {expected}, got {actual}")
    
    # NaN check
    if check_nan and np.isnan(features).any():
        nan_count = np.isnan(features).sum()
        nan_pct = nan_count / features.size * 100
        errors.append(f"Features contain NaN values: {nan_count} ({nan_pct:.1f}%)")
    
    # Infinity check
    if check_inf and np.isinf(features).any():
        inf_count = np.isinf(features).sum()
        errors.append(f"Features contain infinite values: {inf_count}")
    
    # Range check (warn on extreme values)
    if features.size > 0:
        std = np.std(features)
        mean = np.mean(features)
        extreme = np.abs(features - mean) > 10 * std
        if extreme.any():
            warnings.append(f"Extreme values detected: {extreme.sum()} beyond 10 std")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# RISK PARAMETER VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_risk_params(
    balance: float,
    risk_pct: float,
    sl_pips: float,
    pip_value: float,
    lot_size: float,
) -> ValidationResult:
    """
    Validate risk parameters are consistent.
    
    Args:
        balance: Account balance
        risk_pct: Risk percentage per trade
        sl_pips: Stop loss in pips
        pip_value: Value per pip per lot
        lot_size: Calculated lot size
    
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    # Calculate expected lot size
    risk_amount = balance * (risk_pct / 100)
    expected_lot = risk_amount / (sl_pips * pip_value)
    
    # Check if lot size matches expected
    tolerance = 0.02  # 2% tolerance
    if abs(lot_size - expected_lot) / expected_lot > tolerance:
        errors.append(
            f"Lot size mismatch: calculated {lot_size:.4f}, "
            f"expected {expected_lot:.4f} for {risk_pct}% risk"
        )
    
    # Check risk limits
    actual_risk = (lot_size * sl_pips * pip_value) / balance * 100
    if actual_risk > 2.5:
        warnings.append(f"Risk per trade {actual_risk:.2f}% exceeds recommended 2%")
    if actual_risk > 5.0:
        errors.append(f"Risk per trade {actual_risk:.2f}% exceeds maximum 5%")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)
