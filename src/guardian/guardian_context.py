import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any

class GuardianContext:
    """
    Guardian Context Awareness
    ==========================
    Calculates macro-features for PPO V4 input.
    - Market Regime: 0.0 (Range) to 1.0 (Strong Trend)
    - Session Time: Normalized time of day
    - Volatility State: Relative to recent average
    """
    
    @staticmethod
    def get_context(df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract context features from dataframe.
        Expects DF with 'high', 'low', 'close', 'adx' (optional).
        """
        if df.empty:
            return {
                "market_regime": 0.5,
                "session_time": 0.5,
                "volatility_ratio": 1.0
            }
            
        # 1. Session Time (0.0 - 1.0)
        now = datetime.now(timezone.utc)
        # UTC 0 = 0.0, UTC 12 = 0.5, UTC 23:59 = ~1.0
        seconds_in_day = now.hour * 3600 + now.minute * 60 + now.second
        session_time = seconds_in_day / 86400.0
        
        # 2. Market Regime (ADX based or Choppiness)
        # If 'adx' exists, normalize 0-50 -> 0-1
        regime = 0.5
        if 'adx' in df.columns:
            adx = df.iloc[-1]['adx']
            # Sigmoid-like normalization: 25=0.5, 50=1.0, 0=0.0
            regime = np.clip(adx / 50.0, 0.0, 1.0)
        else:
            # Fallback: Simple Trend Efficiency
            # Body / Range (higher = stronger trend)
            sl = df.iloc[-10:] # last 10 bars
            total_dist = (sl['close'] - sl['open']).abs().sum()
            net_dist = abs(sl.iloc[-1]['close'] - sl.iloc[0]['open'])
            if total_dist > 0:
                regime = net_dist / total_dist # Efficiency Ratio
            
        # 3. Volatility (ATR Ratio)
        # Current ATR vs Avg ATR(50)
        vol_ratio = 1.0
        if 'atr' in df.columns:
            curr_atr = df.iloc[-1]['atr']
            avg_atr = df['atr'].rolling(50).mean().iloc[-1]
            if avg_atr > 0:
                vol_ratio = np.clip(curr_atr / avg_atr, 0.5, 2.0)
                
        return {
            "market_regime": float(regime),
            "session_time": float(session_time),
            "volatility_ratio": float(vol_ratio)
        }
