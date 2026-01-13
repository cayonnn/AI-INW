"""
dataset_preparation.py
======================
Prepare Dataset for AI Training (Imitation Learning)

Features:
- EMA12 / EMA26 → trend
- ATR14 → volatility  
- RSI14 → momentum
- Spread → liquidity / noise

Labels:
- Rule-Based Signal (BUY, SELL, HOLD) → Teacher

Usage:
    prep = DatasetPreparation(df)
    X, y = prep.create_features_labels()
    prep.save_csv("data/XAUUSD_dataset.csv")
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


# =========================
# INDICATOR FUNCTIONS
# =========================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 0.001)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """MACD and Signal line."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (upper, middle, lower)."""
    middle = sma(series, period)
    std_dev = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


# =========================
# DATASET PREPARATION
# =========================

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    ema_fast: int = 12
    ema_slow: int = 26
    ema_trend_fast: int = 20
    ema_trend_slow: int = 50
    atr_period: int = 14
    rsi_period: int = 14
    lookback: int = 50


class DatasetPreparation:
    """
    Prepare dataset for AI training.
    
    Features: Technical indicators
    Labels: Rule-Based signals (Teacher)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: DatasetConfig = None
    ):
        """
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            config: Optional configuration
        """
        self.df = df.copy()
        self.config = config or DatasetConfig()
        self.features: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
    
    def compute_indicators(self):
        """Compute all technical indicators."""
        cfg = self.config
        close = self.df['close']
        
        # === Trend Indicators ===
        self.df["ema12"] = ema(close, cfg.ema_fast)
        self.df["ema26"] = ema(close, cfg.ema_slow)
        self.df["ema20"] = ema(close, cfg.ema_trend_fast)
        self.df["ema50"] = ema(close, cfg.ema_trend_slow)
        
        # EMA Spreads
        self.df["ema_spread"] = self.df["ema20"] - self.df["ema50"]
        self.df["ema_spread_pct"] = self.df["ema_spread"] / close * 100
        
        # EMA Slope (momentum)
        self.df["ema20_slope"] = self.df["ema20"].diff(5) / 5
        
        # === Volatility ===
        self.df["atr14"] = atr(self.df, cfg.atr_period)
        self.df["atr_ratio"] = self.df["atr14"] / close
        self.df["spread"] = self.df["high"] - self.df["low"]
        self.df["spread_ratio"] = self.df["spread"] / close
        
        # === Momentum ===
        self.df["rsi14"] = rsi(close, cfg.rsi_period)
        
        # MACD
        macd_line, macd_signal = macd(close)
        self.df["macd"] = macd_line
        self.df["macd_signal"] = macd_signal
        self.df["macd_hist"] = macd_line - macd_signal
        
        # === Price Action ===
        self.df["return_1"] = close.pct_change(1)
        self.df["return_5"] = close.pct_change(5)
        self.df["return_10"] = close.pct_change(10)
        
        # Bollinger Bands position
        bb_upper, bb_middle, bb_lower = bollinger_bands(close)
        self.df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # === Time Features ===
        if 'time' in self.df.columns:
            self.df['time'] = pd.to_datetime(self.df['time'])
            self.df["hour"] = self.df["time"].dt.hour
            self.df["day_of_week"] = self.df["time"].dt.dayofweek
        else:
            self.df["hour"] = 0
            self.df["day_of_week"] = 0
        
        print(f"✅ Computed {len([c for c in self.df.columns if c not in ['open','high','low','close','volume','time']])} indicators")
    
    def generate_labels(self):
        """Generate labels using Rule-Based Signal Engine (Teacher)."""
        cfg = self.config
        signals = []
        
        for i in range(len(self.df)):
            if i < cfg.lookback:
                signals.append("HOLD")
                continue
            
            row = self.df.iloc[i]
            
            # Simple EMA crossover logic
            ema_fast = row.get("ema20", 0)
            ema_slow = row.get("ema50", 0)
            atr_val = row.get("atr14", 0)
            
            # Volatility filter
            atr_mean = self.df["atr14"].iloc[max(0, i-50):i].mean()
            vol_ok = atr_val > atr_mean * 0.7 if not pd.isna(atr_mean) else True
            
            # Generate signal
            if ema_fast > ema_slow and vol_ok:
                signals.append("BUY")
            elif ema_fast < ema_slow and vol_ok:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        self.df["signal"] = signals
        
        # Encode labels
        label_map = {"SELL": -1, "HOLD": 0, "BUY": 1}
        self.df["signal_code"] = self.df["signal"].map(label_map)
        
        print(f"📊 Generated labels:")
        print(f"   BUY:  {signals.count('BUY')}")
        print(f"   SELL: {signals.count('SELL')}")
        print(f"   HOLD: {signals.count('HOLD')}")
    
    def create_features_labels(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features (X) and labels (y).
        
        Returns:
            (features DataFrame, labels Series)
        """
        self.compute_indicators()
        self.generate_labels()
        
        # Feature columns
        feature_cols = [
            # Trend
            "ema20", "ema50", "ema_spread", "ema_spread_pct", "ema20_slope",
            # Volatility
            "atr14", "atr_ratio", "spread", "spread_ratio",
            # Momentum
            "rsi14", "macd", "macd_signal", "macd_hist",
            # Price
            "return_1", "return_5", "return_10", "bb_position",
            # Time
            "hour", "day_of_week",
        ]
        
        # Filter available columns
        available_cols = [c for c in feature_cols if c in self.df.columns]
        
        self.features = self.df[available_cols].fillna(0)
        self.labels = self.df["signal_code"]
        
        print(f"\n✅ Dataset ready:")
        print(f"   Features: {len(available_cols)} columns")
        print(f"   Samples:  {len(self.features)}")
        
        return self.features, self.labels
    
    def save_csv(self, path: str = "data/prepared_dataset.csv"):
        """Save full dataset for AI training."""
        # Select columns to save
        save_cols = [
            'time', 'open', 'high', 'low', 'close', 'volume',
            'ema20', 'ema50', 'ema_spread', 'ema_spread_pct', 'ema20_slope',
            'atr14', 'atr_ratio', 'spread', 'spread_ratio',
            'rsi14', 'macd', 'macd_signal', 'macd_hist',
            'return_1', 'return_5', 'return_10', 'bb_position',
            'hour', 'day_of_week',
            'signal', 'signal_code'
        ]
        
        available = [c for c in save_cols if c in self.df.columns]
        save_df = self.df[available].copy()
        
        save_df.to_csv(path, index=False)
        print(f"✅ Dataset saved: {path}")
        print(f"   Rows: {len(save_df)}")
        print(f"   Columns: {len(available)}")
    
    def get_train_test_split(self, test_ratio: float = 0.2) -> Tuple:
        """
        Split data for training and testing.
        
        Args:
            test_ratio: Fraction for test set
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if self.features is None:
            self.create_features_labels()
        
        split_idx = int(len(self.features) * (1 - test_ratio))
        
        X_train = self.features.iloc[:split_idx]
        X_test = self.features.iloc[split_idx:]
        y_train = self.labels.iloc[:split_idx]
        y_test = self.labels.iloc[split_idx:]
        
        print(f"📊 Train/Test Split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test


# =========================
# QUICK PREPARE FUNCTION
# =========================

def prepare_dataset_from_mt5(
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    bars: int = 1000,
    save_path: str = None
) -> DatasetPreparation:
    """
    Prepare dataset from MT5 data.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        bars: Number of bars
        save_path: Optional path to save CSV
        
    Returns:
        DatasetPreparation instance
    """
    from src.data.mt5_connector import MT5Connector
    
    mt5 = MT5Connector()
    mt5.connect()
    
    rates = mt5.get_rates(symbol, timeframe, bars)
    df = pd.DataFrame(rates)
    
    prep = DatasetPreparation(df)
    prep.create_features_labels()
    
    if save_path:
        prep.save_csv(save_path)
    
    return prep


def prepare_dataset_from_backtest(
    db_path: str = "data/backtest.db",
    csv_export: str = None,
    include_pnl: bool = True
) -> pd.DataFrame:
    """
    Prepare dataset from backtest database.
    
    Args:
        db_path: Path to backtest SQLite database
        csv_export: Optional path to export CSV
        include_pnl: Include PnL columns for reward learning
        
    Returns:
        DataFrame with features and labels (including outcome)
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    
    # Get all trades with outcomes
    trades_df = pd.read_sql_query("""
        SELECT 
            t.*,
            r.symbol,
            r.win_rate as report_win_rate
        FROM backtest_trades t
        JOIN backtest_reports r ON t.report_id = r.id
        ORDER BY t.id
    """, conn)
    
    conn.close()
    
    if trades_df.empty:
        print("⚠️ No trades in database")
        return pd.DataFrame()
    
    # Add label (win/loss based on PnL)
    trades_df["label"] = (trades_df["pnl"] > 0).astype(int)
    trades_df["signal_code"] = trades_df["direction"].map({"BUY": 1, "SELL": -1})
    
    # Create features from trade data
    trades_df["mae_ratio"] = trades_df["mae"] / trades_df["entry_price"]
    trades_df["mfe_ratio"] = trades_df["mfe"] / trades_df["entry_price"]
    trades_df["pnl_ratio"] = trades_df["pnl"] / trades_df["entry_price"]
    
    print(f"📊 Loaded from backtest database:")
    print(f"   Trades: {len(trades_df)}")
    print(f"   Symbols: {trades_df['symbol'].unique().tolist()}")
    print(f"   Win Rate: {trades_df['label'].mean():.1%}")
    
    if csv_export:
        trades_df.to_csv(csv_export, index=False)
        print(f"✅ Exported to: {csv_export}")
    
    return trades_df


def prepare_dataset_from_live_log(
    csv_path: str = "data/imitation_dataset.csv",
    add_outcome: bool = False
) -> pd.DataFrame:
    """
    Load dataset from live signal recording.
    
    Args:
        csv_path: Path to imitation_dataset.csv
        add_outcome: Add future return if available
        
    Returns:
        DataFrame with features and labels
    """
    df = pd.read_csv(csv_path)
    
    print(f"📊 Loaded from live log:")
    print(f"   Samples: {len(df)}")
    print(f"   BUY:  {len(df[df['action'] == 'BUY'])}")
    print(f"   SELL: {len(df[df['action'] == 'SELL'])}")
    print(f"   HOLD: {len(df[df['action'] == 'HOLD'])}")
    
    return df


# =========================
# TEST
# =========================

if __name__ == "__main__":
    print("🧪 Testing Dataset Preparation")
    print("=" * 50)
    
    try:
        # Try MT5
        prep = prepare_dataset_from_mt5(
            symbol="XAUUSD",
            bars=500,
            save_path="data/XAUUSD_dataset.csv"
        )
    except Exception as e:
        print(f"⚠️ MT5 not available: {e}")
        print("   Creating sample data...")
        
        # Create sample data
        np.random.seed(42)
        n = 300
        price = 2000 + np.cumsum(np.random.randn(n) * 10)
        
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="H"),
            "open": price,
            "high": price + np.abs(np.random.randn(n) * 5),
            "low": price - np.abs(np.random.randn(n) * 5),
            "close": price + np.random.randn(n) * 3,
            "volume": np.random.randint(100, 1000, n)
        })
        
        prep = DatasetPreparation(df)
        X, y = prep.create_features_labels()
        prep.save_csv("data/sample_dataset.csv")
        
        # Show sample
        print("\n📋 Sample Features:")
        print(X.head())
