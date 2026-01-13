# src/signals/imitation_dataset.py
"""
Imitation Dataset Generator
===========================
Generates dataset from SignalEngine V2 for AI Imitation Learning.

Features:
- Fetch OHLC data from MT5
- Calculate EMA, ATR indicators
- Generate labels from Rule-Based signals
- Export to CSV for training

Usage:
    python src/signals/imitation_dataset.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =========================
# INDICATOR FUNCTIONS
# =========================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 0.001)
    return 100 - (100 / (1 + rs))


# =========================
# IMITATION DATASET
# =========================

class ImitationDataset:
    """
    Generate dataset for AI Imitation Learning.
    Uses Rule-Based SignalEngine as Teacher.
    """
    
    def __init__(self, symbol: str = "XAUUSD", bars: int = 1000):
        self.symbol = symbol
        self.bars = bars
        self.df = None
    
    def fetch_from_mt5(self) -> pd.DataFrame:
        """Fetch OHLC data from MT5."""
        try:
            from src.data.mt5_connector import MT5Connector
            
            mt5 = MT5Connector()
            mt5.connect()
            
            rates = mt5.get_rates(self.symbol, "H1", self.bars)
            df = pd.DataFrame(rates)
            print(f"✅ Fetched {len(df)} bars from MT5")
            return df
        except Exception as e:
            print(f"⚠️ MT5 not available: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self, n: int = None) -> pd.DataFrame:
        """Create sample data for testing."""
        n = n or self.bars
        np.random.seed(42)
        price = 2000 + np.cumsum(np.random.randn(n) * 10)
        
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": price,
            "high": price + np.abs(np.random.randn(n) * 5),
            "low": price - np.abs(np.random.randn(n) * 5),
            "close": price + np.random.randn(n) * 3,
            "volume": np.random.randint(100, 1000, n)
        })
        print(f"📊 Created {n} sample bars")
        return df
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        close = df['close']
        
        # EMAs
        df["ema20"] = ema(close, 20)
        df["ema50"] = ema(close, 50)
        df["ema_spread"] = df["ema20"] - df["ema50"]
        
        # ATR
        df["atr14"] = atr(df, 14)
        df["atr_mean"] = df["atr14"].rolling(50).mean()
        
        # RSI
        df["rsi14"] = rsi(close, 14)
        
        # Trend flags
        df["trend_up"] = df["ema20"] > df["ema50"]
        df["volatility_ok"] = df["atr14"] > df["atr_mean"] * 0.7
        
        # Time features
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df["hour"] = df["time"].dt.hour
            df["day_of_week"] = df["time"].dt.dayofweek
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate labels from Rule-Based signal."""
        signals = []
        
        for i in range(len(df)):
            if i < 50:  # Warmup period
                signals.append("HOLD")
                continue
            
            row = df.iloc[i]
            trend_up = row.get("trend_up", False)
            vol_ok = row.get("volatility_ok", False)
            
            # Rule-Based signal logic
            if trend_up and vol_ok:
                signals.append("BUY")
            elif not trend_up and vol_ok:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        df["signal"] = signals
        
        # Encode labels
        label_map = {"SELL": 0, "HOLD": 1, "BUY": 2}
        df["label"] = df["signal"].map(label_map)
        
        return df
    
    def generate_dataset(self, use_mt5: bool = True) -> pd.DataFrame:
        """Generate full dataset."""
        print("=" * 50)
        print("📊 Generating Imitation Dataset")
        print("=" * 50)
        
        # Fetch data
        if use_mt5:
            df = self.fetch_from_mt5()
        else:
            df = self.create_sample_data()
        
        # Compute features
        df = self.compute_features(df)
        
        # Generate labels
        df = self.generate_labels(df)
        
        # Drop NaN
        df = df.dropna()
        
        self.df = df
        
        # Stats
        print(f"\n📈 Dataset Stats:")
        print(f"   Total samples: {len(df)}")
        print(f"   BUY:  {len(df[df['signal'] == 'BUY'])}")
        print(f"   SELL: {len(df[df['signal'] == 'SELL'])}")
        print(f"   HOLD: {len(df[df['signal'] == 'HOLD'])}")
        
        return df
    
    def export_csv(self, filepath: str = "data/imitation_full_dataset.csv") -> pd.DataFrame:
        """Export dataset to CSV."""
        if self.df is None:
            self.generate_dataset()
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.df.to_csv(filepath, index=False)
        print(f"\n✅ Dataset exported: {filepath}")
        
        return self.df


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    ds = ImitationDataset(symbol="XAUUSD", bars=500)
    df = ds.generate_dataset(use_mt5=True)
    ds.export_csv("data/imitation_full_dataset.csv")
    
    print("\n📋 Sample:")
    print(df[["close", "ema20", "ema50", "atr14", "signal"]].tail(10))
