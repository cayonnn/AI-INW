# src/retrain/data_loader.py
"""
Data Loader for Retrain Pipeline
=================================

Loads training data from multiple sources:
- Live trades (trade_history.db or CSV)
- Shadow trades (shadow simulation)
- Score timeline
- Regime logs
"""

import os
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Optional, Any

from src.utils.logger import get_logger

logger = get_logger("DATA_LOADER")


def load_training_data(
    run_date: date = None,
    days_lookback: int = 7,
    data_dir: str = "data"
) -> Optional[Dict[str, Any]]:
    """
    Load training data for retraining.
    
    Args:
        run_date: End date for data
        days_lookback: Days of historical data
        data_dir: Data directory
        
    Returns:
        Dictionary with loaded dataframes
    """
    if run_date is None:
        run_date = date.today()
    
    start_date = run_date - timedelta(days=days_lookback)
    
    logger.info(f"Loading data from {start_date} to {run_date}")
    
    result = {
        "live": None,
        "shadow": None,
        "score": None,
        "regime": None,
        "start_date": str(start_date),
        "end_date": str(run_date),
    }
    
    # Load live trades
    live_path = os.path.join(data_dir, "live_trades.csv")
    if os.path.exists(live_path):
        try:
            df = pd.read_csv(live_path)
            if "date" in df.columns or "timestamp" in df.columns:
                date_col = "date" if "date" in df.columns else "timestamp"
                df[date_col] = pd.to_datetime(df[date_col])
                df = df[(df[date_col].dt.date >= start_date) & 
                       (df[date_col].dt.date <= run_date)]
            result["live"] = df
            logger.info(f"Loaded {len(df)} live trades")
        except Exception as e:
            logger.warning(f"Failed to load live trades: {e}")
    
    # Load shadow trades
    shadow_path = os.path.join(data_dir, "shadow_trades.csv")
    if os.path.exists(shadow_path):
        try:
            result["shadow"] = pd.read_csv(shadow_path)
            logger.info(f"Loaded shadow trades")
        except Exception as e:
            logger.warning(f"Failed to load shadow trades: {e}")
    
    # Load score timeline
    score_path = os.path.join(data_dir, "score_log.csv")
    if os.path.exists(score_path):
        try:
            result["score"] = pd.read_csv(score_path)
            logger.info(f"Loaded score log")
        except Exception as e:
            logger.warning(f"Failed to load score log: {e}")
    
    # Load regime logs
    regime_path = os.path.join(data_dir, "regime_log.csv")
    if os.path.exists(regime_path):
        try:
            result["regime"] = pd.read_csv(regime_path)
            logger.info(f"Loaded regime log")
        except Exception as e:
            logger.warning(f"Failed to load regime log: {e}")
    
    # Try to load from retrain daily data if CSVs not available
    retrain_dir = os.path.join(data_dir, "retrain")
    if result["live"] is None and os.path.exists(retrain_dir):
        result = _load_from_retrain_dir(retrain_dir, start_date, run_date)
    
    return result


def _load_from_retrain_dir(
    retrain_dir: str,
    start_date: date,
    end_date: date
) -> Dict[str, Any]:
    """Load from retrain daily JSON files."""
    import json
    
    result = {
        "live": None,
        "shadow": None,
        "score": None,
        "regime": None,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    
    all_trades = []
    all_scores = []
    regime_counts = {}
    
    current = start_date
    while current <= end_date:
        filename = f"daily_{current}.json"
        filepath = os.path.join(retrain_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                day_data = json.load(f)
            
            # Extract trades
            for trade in day_data.get("trades", []):
                trade["date"] = str(current)
                all_trades.append(trade)
            
            # Extract scores
            for score_entry in day_data.get("score_timeline", []):
                score_entry["date"] = str(current)
                all_scores.append(score_entry)
            
            # Aggregate regime
            for regime, count in day_data.get("regime_distribution", {}).items():
                regime_counts[regime] = regime_counts.get(regime, 0) + count
        
        current += timedelta(days=1)
    
    if all_trades:
        result["live"] = pd.DataFrame(all_trades)
        logger.info(f"Loaded {len(all_trades)} trades from daily files")
    
    if all_scores:
        result["score"] = pd.DataFrame(all_scores)
    
    if regime_counts:
        result["regime"] = pd.DataFrame([
            {"regime": k, "count": v} for k, v in regime_counts.items()
        ])
    
    return result


def create_sample_data(data_dir: str = "data") -> None:
    """Create sample data for testing."""
    import numpy as np
    from datetime import datetime
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample trades
    n_trades = 50
    trades = pd.DataFrame({
        "ticket": range(1, n_trades + 1),
        "symbol": ["XAUUSD"] * n_trades,
        "direction": np.random.choice(["BUY", "SELL"], n_trades),
        "pnl": np.random.randn(n_trades) * 10 + 5,
        "profit": np.random.randn(n_trades) * 10 + 5,
        "drawdown": np.random.uniform(0, 0.05, n_trades),
        "confidence": np.random.uniform(0.5, 0.9, n_trades),
        "regime": np.random.choice(["STRONG_TREND", "WEAK_TREND", "CHOP"], n_trades),
        "timestamp": pd.date_range(end=datetime.now(), periods=n_trades, freq='H'),
    })
    trades.to_csv(os.path.join(data_dir, "live_trades.csv"), index=False)
    
    # Sample scores
    scores = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=100, freq='30min'),
        "score": np.cumsum(np.random.randn(100) * 0.5) + 60,
        "profit": np.cumsum(np.random.randn(100) * 2),
        "drawdown": np.abs(np.cumsum(np.random.randn(100) * 0.3)),
    })
    scores.to_csv(os.path.join(data_dir, "score_log.csv"), index=False)
    
    logger.info(f"Created sample data in {data_dir}")
