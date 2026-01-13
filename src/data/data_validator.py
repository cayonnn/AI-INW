"""
AI Trading System - Data Validator
===================================
Data quality validation and integrity checks.

Usage:
    from src.data.data_validator import DataValidator
    
    validator = DataValidator()
    result = validator.validate(df)
    if not result.is_valid:
        print(result.errors)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.validators import ValidationResult


logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    
    is_valid: bool
    total_rows: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Quality metrics
    missing_pct: float = 0.0
    duplicate_pct: float = 0.0
    outlier_pct: float = 0.0
    gap_count: int = 0
    max_gap_hours: float = 0.0
    
    # OHLC validity
    invalid_ohlc_count: int = 0
    negative_price_count: int = 0
    zero_volume_pct: float = 0.0
    
    # Statistical summary
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": {
                "missing_pct": self.missing_pct,
                "duplicate_pct": self.duplicate_pct,
                "outlier_pct": self.outlier_pct,
                "gap_count": self.gap_count,
                "max_gap_hours": self.max_gap_hours,
                "invalid_ohlc_count": self.invalid_ohlc_count,
                "negative_price_count": self.negative_price_count,
                "zero_volume_pct": self.zero_volume_pct,
            },
            "summary": self.summary,
        }


class DataValidator:
    """
    Comprehensive data quality validator for OHLCV data.
    
    Performs checks on:
    - Data completeness
    - OHLC consistency
    - Time series gaps
    - Outliers
    - Data range validity
    """
    
    def __init__(
        self,
        min_rows: int = 100,
        max_missing_pct: float = 5.0,
        max_gap_hours: float = 24.0,
        outlier_threshold: float = 4.0,
    ):
        """
        Initialize validator.
        
        Args:
            min_rows: Minimum required rows
            max_missing_pct: Maximum allowed missing percentage
            max_gap_hours: Maximum allowed gap in hours
            outlier_threshold: Z-score threshold for outliers
        """
        self.min_rows = min_rows
        self.max_missing_pct = max_missing_pct
        self.max_gap_hours = max_gap_hours
        self.outlier_threshold = outlier_threshold
    
    def validate(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        strict: bool = False,
    ) -> DataQualityReport:
        """
        Perform comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol name for logging
            strict: If True, any warning becomes an error
        
        Returns:
            DataQualityReport with validation results
        """
        report = DataQualityReport(
            is_valid=True,
            total_rows=len(df),
        )
        
        symbol_str = f"{symbol}: " if symbol else ""
        
        # Check minimum rows
        if len(df) < self.min_rows:
            report.errors.append(
                f"{symbol_str}Insufficient data: {len(df)} rows (min: {self.min_rows})"
            )
            report.is_valid = False
        
        # Check required columns
        required = ["open", "high", "low", "close"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            report.errors.append(f"{symbol_str}Missing columns: {missing_cols}")
            report.is_valid = False
            return report
        
        # Check for missing values
        self._check_missing(df, report, symbol_str)
        
        # Check for duplicates
        self._check_duplicates(df, report, symbol_str)
        
        # Check OHLC consistency
        self._check_ohlc_consistency(df, report, symbol_str)
        
        # Check for time gaps
        self._check_time_gaps(df, report, symbol_str)
        
        # Check for outliers
        self._check_outliers(df, report, symbol_str)
        
        # Check for negative/zero prices
        self._check_price_validity(df, report, symbol_str)
        
        # Check volume
        self._check_volume(df, report, symbol_str)
        
        # Generate summary statistics
        report.summary = self._compute_summary(df)
        
        # In strict mode, warnings become errors
        if strict and report.warnings:
            report.errors.extend(report.warnings)
            report.is_valid = False
        
        if report.is_valid:
            logger.debug(f"{symbol_str}Data validation passed ({len(df)} rows)")
        else:
            logger.warning(f"{symbol_str}Data validation failed: {report.errors}")
        
        return report
    
    def _check_missing(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        prefix: str,
    ) -> None:
        """Check for missing values."""
        price_cols = ["open", "high", "low", "close"]
        missing = df[price_cols].isnull().sum().sum()
        total = len(df) * len(price_cols)
        
        report.missing_pct = (missing / total) * 100 if total > 0 else 0
        
        if report.missing_pct > self.max_missing_pct:
            report.errors.append(
                f"{prefix}Missing values: {report.missing_pct:.2f}% (max: {self.max_missing_pct}%)"
            )
            report.is_valid = False
        elif report.missing_pct > 0:
            report.warnings.append(f"{prefix}Missing values: {report.missing_pct:.2f}%")
    
    def _check_duplicates(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        prefix: str,
    ) -> None:
        """Check for duplicate timestamps."""
        if isinstance(df.index, pd.DatetimeIndex):
            dups = df.index.duplicated().sum()
            report.duplicate_pct = (dups / len(df)) * 100 if len(df) > 0 else 0
            
            if dups > 0:
                report.warnings.append(f"{prefix}Duplicate timestamps: {dups}")
    
    def _check_ohlc_consistency(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        prefix: str,
    ) -> None:
        """Check OHLC data consistency."""
        invalid_mask = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        
        report.invalid_ohlc_count = invalid_mask.sum()
        
        if report.invalid_ohlc_count > 0:
            pct = (report.invalid_ohlc_count / len(df)) * 100
            if pct > 1:
                report.errors.append(
                    f"{prefix}Invalid OHLC: {report.invalid_ohlc_count} bars ({pct:.2f}%)"
                )
                report.is_valid = False
            else:
                report.warnings.append(
                    f"{prefix}Invalid OHLC: {report.invalid_ohlc_count} bars ({pct:.2f}%)"
                )
    
    def _check_time_gaps(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        prefix: str,
    ) -> None:
        """Check for time series gaps."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        
        time_diff = df.index.to_series().diff()
        
        if len(time_diff) > 1:
            # Calculate expected gap (median)
            expected_gap = time_diff.median()
            
            if pd.notna(expected_gap):
                # Find gaps larger than 2x expected
                large_gaps = time_diff > (expected_gap * 2)
                report.gap_count = large_gaps.sum()
                
                # Max gap in hours
                max_gap = time_diff.max()
                if pd.notna(max_gap):
                    report.max_gap_hours = max_gap.total_seconds() / 3600
                    
                    if report.max_gap_hours > self.max_gap_hours:
                        report.warnings.append(
                            f"{prefix}Large gap detected: {report.max_gap_hours:.1f} hours"
                        )
    
    def _check_outliers(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        prefix: str,
    ) -> None:
        """Check for outliers using returns."""
        if len(df) < 2:
            return
        
        returns = df["close"].pct_change().dropna()
        
        if len(returns) > 0:
            mean = returns.mean()
            std = returns.std()
            
            if std > 0:
                z_scores = np.abs((returns - mean) / std)
                outliers = (z_scores > self.outlier_threshold).sum()
                report.outlier_pct = (outliers / len(returns)) * 100
                
                if report.outlier_pct > 5:
                    report.warnings.append(
                        f"{prefix}High outlier count: {report.outlier_pct:.2f}%"
                    )
    
    def _check_price_validity(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        prefix: str,
    ) -> None:
        """Check for negative or zero prices."""
        price_cols = ["open", "high", "low", "close"]
        
        negative_mask = (df[price_cols] <= 0).any(axis=1)
        report.negative_price_count = negative_mask.sum()
        
        if report.negative_price_count > 0:
            report.errors.append(
                f"{prefix}Invalid prices (<=0): {report.negative_price_count} bars"
            )
            report.is_valid = False
    
    def _check_volume(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        prefix: str,
    ) -> None:
        """Check volume data."""
        if "volume" in df.columns:
            zero_vol = (df["volume"] == 0).sum()
            report.zero_volume_pct = (zero_vol / len(df)) * 100 if len(df) > 0 else 0
            
            if report.zero_volume_pct > 50:
                report.warnings.append(
                    f"{prefix}High zero-volume bars: {report.zero_volume_pct:.1f}%"
                )
    
    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics."""
        summary = {}
        
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                summary[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "median": df[col].median(),
                }
        
        if "close" in df.columns and len(df) > 1:
            returns = df["close"].pct_change().dropna()
            summary["returns"] = {
                "mean": returns.mean() * 252,  # Annualized
                "std": returns.std() * np.sqrt(252),  # Annualized
                "min": returns.min(),
                "max": returns.max(),
                "skew": returns.skew(),
                "kurtosis": returns.kurtosis(),
            }
        
        return summary
    
    # ─────────────────────────────────────────────────────────────────────────
    # SPECIFIC VALIDATORS
    # ─────────────────────────────────────────────────────────────────────────
    
    def validate_for_training(
        self,
        df: pd.DataFrame,
        min_rows: int = 1000,
    ) -> ValidationResult:
        """
        Validate data specifically for model training.
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum rows for training
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        if len(df) < min_rows:
            errors.append(f"Insufficient data for training: {len(df)} (min: {min_rows})")
        
        # Check for lookahead bias potential
        if isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                errors.append("Data is not sorted chronologically")
        
        # Check for constant columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() < 2:
                warnings.append(f"Column '{col}' has constant values")
        
        # Check for high correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.99:
                        high_corr.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j])
                        )
            if high_corr:
                warnings.append(f"Highly correlated features: {high_corr[:3]}")
        
        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult.success(warnings)
    
    def validate_for_inference(
        self,
        df: pd.DataFrame,
        expected_features: List[str],
    ) -> ValidationResult:
        """
        Validate data for model inference.
        
        Args:
            df: DataFrame to validate
            expected_features: Expected feature columns
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check all required features present
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            errors.append(f"Missing features for inference: {missing}")
        
        # Check for NaN in features
        if not errors:
            nan_cols = df[expected_features].isnull().any()
            if nan_cols.any():
                nan_features = nan_cols[nan_cols].index.tolist()
                errors.append(f"NaN values in features: {nan_features}")
        
        # Check for infinite values
        if not errors:
            inf_mask = np.isinf(df[expected_features].values)
            if inf_mask.any():
                errors.append("Infinite values in features")
        
        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult.success(warnings)
