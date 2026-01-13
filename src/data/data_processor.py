"""
AI Trading System - Data Processor
===================================
Data cleaning, normalization, and transformation for ML pipeline.

Usage:
    from src.data.data_processor import DataProcessor
    
    processor = DataProcessor()
    clean_df = processor.process(raw_df)
    normalized = processor.normalize(clean_df)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils.logger import get_logger


logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class DataProcessor:
    """
    Data processing pipeline for trading data.
    
    Handles:
    - Missing value imputation
    - Outlier detection and handling
    - Data normalization/standardization
    - Sequence preparation for LSTM
    - Train/test splitting with proper handling
    """
    
    def __init__(
        self,
        fill_method: str = "ffill",
        outlier_method: str = "clip",
        outlier_threshold: float = 3.0,
        scaler_type: str = "robust",
    ):
        """
        Initialize data processor.
        
        Args:
            fill_method: Method for filling missing values ("ffill", "bfill", "interpolate")
            outlier_method: Outlier handling method ("clip", "remove", "winsorize")
            outlier_threshold: Number of standard deviations for outlier detection
            scaler_type: Normalization method ("standard", "minmax", "robust")
        """
        self.fill_method = fill_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.scaler_type = scaler_type
        
        self._scaler = None
        self._feature_stats: Dict[str, Dict[str, float]] = {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # MISSING VALUE HANDLING
    # ─────────────────────────────────────────────────────────────────────────
    
    def fill_missing(
        self, 
        df: pd.DataFrame,
        method: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fill missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            method: Override fill method
        
        Returns:
            DataFrame with filled values
        """
        method = method or self.fill_method
        result = df.copy()
        
        before_count = result.isnull().sum().sum()
        
        if method == "ffill":
            result = result.ffill()
        elif method == "bfill":
            result = result.bfill()
        elif method == "interpolate":
            result = result.interpolate(method="linear")
        elif method == "mean":
            result = result.fillna(result.mean())
        elif method == "median":
            result = result.fillna(result.median())
        else:
            raise ValueError(f"Unknown fill method: {method}")
        
        # Handle any remaining NaN at edges
        result = result.bfill().ffill()
        
        after_count = result.isnull().sum().sum()
        
        if before_count > 0:
            logger.debug(f"Filled {before_count - after_count} missing values using {method}")
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # OUTLIER HANDLING
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Detect outliers using z-score method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check (None = all numeric)
            threshold: Z-score threshold
        
        Returns:
            Boolean DataFrame indicating outliers
        """
        threshold = threshold or self.outlier_threshold
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
        outliers = z_scores > threshold
        
        return outliers
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Handle outliers in data.
        
        Args:
            df: Input DataFrame
            columns: Columns to process
            method: Override outlier method
            threshold: Z-score threshold
        
        Returns:
            DataFrame with handled outliers
        """
        method = method or self.outlier_method
        threshold = threshold or self.outlier_threshold
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            col_data = result[col]
            mean = col_data.mean()
            std = col_data.std()
            
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            if method == "clip":
                result[col] = col_data.clip(lower_bound, upper_bound)
            elif method == "remove":
                mask = (col_data >= lower_bound) & (col_data <= upper_bound)
                result = result[mask]
            elif method == "winsorize":
                percentile_low = col_data.quantile(0.01)
                percentile_high = col_data.quantile(0.99)
                result[col] = col_data.clip(percentile_low, percentile_high)
            elif method == "median":
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                result.loc[outlier_mask, col] = col_data.median()
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def fit_scaler(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        scaler_type: Optional[str] = None,
    ) -> None:
        """
        Fit scaler on training data.
        
        Args:
            df: Training DataFrame
            columns: Columns to scale
            scaler_type: Override scaler type
        """
        scaler_type = scaler_type or self.scaler_type
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create scaler
        if scaler_type == "standard":
            self._scaler = StandardScaler()
        elif scaler_type == "minmax":
            self._scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self._scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit on specified columns
        self._scaler.fit(df[columns])
        self._scaler_columns = columns
        
        # Store feature statistics
        for i, col in enumerate(columns):
            self._feature_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
            }
        
        logger.debug(f"Fitted {scaler_type} scaler on {len(columns)} columns")
    
    def normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Normalize data using fitted scaler.
        
        Args:
            df: DataFrame to normalize
            columns: Columns to normalize (must match fitted columns)
        
        Returns:
            Normalized DataFrame
        """
        if self._scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        columns = columns or self._scaler_columns
        
        result = df.copy()
        scaled_values = self._scaler.transform(df[columns])
        result[columns] = scaled_values
        
        return result
    
    def inverse_normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Inverse normalization to original scale.
        
        Args:
            df: Normalized DataFrame
            columns: Columns to inverse
        
        Returns:
            Original-scale DataFrame
        """
        if self._scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        columns = columns or self._scaler_columns
        
        result = df.copy()
        original_values = self._scaler.inverse_transform(df[columns])
        result[columns] = original_values
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # SEQUENCE CREATION (FOR LSTM)
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        step: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM training.
        
        Args:
            df: Input DataFrame
            sequence_length: Length of each sequence
            target_column: Column to predict (None = no target)
            feature_columns: Feature columns (None = all numeric)
            step: Step size between sequences
        
        Returns:
            Tuple of (X sequences, y targets) - y is None if no target
        """
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column:
                feature_columns = [c for c in feature_columns if c != target_column]
        
        data = df[feature_columns].values
        n_samples = (len(data) - sequence_length) // step + 1
        
        # Pre-allocate arrays
        X = np.empty((n_samples, sequence_length, len(feature_columns)), dtype=np.float32)
        
        for i in range(n_samples):
            start_idx = i * step
            end_idx = start_idx + sequence_length
            X[i] = data[start_idx:end_idx]
        
        # Handle target
        y = None
        if target_column:
            target_data = df[target_column].values
            y = np.empty(n_samples, dtype=np.float32)
            for i in range(n_samples):
                target_idx = i * step + sequence_length
                if target_idx < len(target_data):
                    y[i] = target_data[target_idx]
                else:
                    y[i] = target_data[-1]
        
        logger.debug(f"Created {n_samples} sequences of length {sequence_length}")
        return X, y
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAIN/TEST SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    
    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split data into train/validation/test sets chronologically.
        
        Args:
            df: Input DataFrame
            test_size: Fraction for test set
            validation_size: Fraction for validation set (0 = no validation)
        
        Returns:
            Tuple of (train_df, test_df, validation_df)
        """
        n = len(df)
        
        test_start = int(n * (1 - test_size))
        
        if validation_size > 0:
            val_start = int(n * (1 - test_size - validation_size))
            train_df = df.iloc[:val_start]
            val_df = df.iloc[val_start:test_start]
            test_df = df.iloc[test_start:]
            
            logger.debug(
                f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )
            return train_df, test_df, val_df
        else:
            train_df = df.iloc[:test_start]
            test_df = df.iloc[test_start:]
            
            logger.debug(f"Split: train={len(train_df)}, test={len(test_df)}")
            return train_df, test_df, None
    
    # ─────────────────────────────────────────────────────────────────────────
    # FULL PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    
    def process(
        self,
        df: pd.DataFrame,
        fill_missing: bool = True,
        handle_outliers: bool = True,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """
        Run full processing pipeline.
        
        Args:
            df: Input DataFrame
            fill_missing: Whether to fill missing values
            handle_outliers: Whether to handle outliers
            normalize: Whether to normalize (requires prior fit_scaler)
        
        Returns:
            Processed DataFrame
        """
        result = df.copy()
        
        if fill_missing:
            result = self.fill_missing(result)
        
        if handle_outliers:
            result = self.handle_outliers(result)
        
        if normalize:
            if self._scaler is None:
                self.fit_scaler(result)
            result = self.normalize(result)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stored feature statistics."""
        return self._feature_stats.copy()
    
    def compute_returns(
        self,
        df: pd.DataFrame,
        price_column: str = "close",
        periods: List[int] = [1, 5, 10],
    ) -> pd.DataFrame:
        """
        Compute returns for various periods.
        
        Args:
            df: DataFrame with price data
            price_column: Column containing prices
            periods: List of periods for return calculation
        
        Returns:
            DataFrame with return columns added
        """
        result = df.copy()
        
        for period in periods:
            result[f"return_{period}"] = result[price_column].pct_change(period)
        
        # Log returns
        result[f"log_return_1"] = np.log(result[price_column] / result[price_column].shift(1))
        
        return result
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features from DatetimeIndex.
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            DataFrame with time features added
        """
        result = df.copy()
        
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping time features")
            return result
        
        result["hour"] = result.index.hour
        result["day_of_week"] = result.index.dayofweek
        result["day_of_month"] = result.index.day
        result["month"] = result.index.month
        result["is_month_start"] = result.index.is_month_start.astype(int)
        result["is_month_end"] = result.index.is_month_end.astype(int)
        
        # Cyclical encoding for hour
        result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
        result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
        
        # Cyclical encoding for day of week
        result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
        result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
        
        return result
