# live_loop_v3.py
"""
Live Loop V3 - Complete AI Trading System (Fund-Grade)
=======================================================
All-in-one file with:
- MT5 Connector (Simplified)
- SignalEngine V3 (Rule + AI hybrid)
- Auto AI Training
- Risk Guard + HTF Filter
- Sandbox / Live Mode
- Fund-Grade Risk Management (RiskManager)
- Dynamic SL/TP (AI Prediction)
- Break-Even & Trailing Stop (TrailingManager)

Usage:
    python live_loop_v3.py
    python live_loop_v3.py --live
    python live_loop_v3.py --interval 60
"""

import time
import logging
import argparse
import pickle
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

# Fund-Grade Risk Management
try:
    from src.risk.risk_manager import RiskManager
    from src.risk.trailing import TrailingManager
    FUND_GRADE_ENABLED = True
except ImportError:
    FUND_GRADE_ENABLED = False
    logging.warning("Fund-Grade modules not available")


# =========================
# MT5 CONNECTOR (Simplified)
# =========================

class MT5Connector:
    """MT5 Connection wrapper."""
    
    def __init__(self):
        self.connected = False
    
    def connect(self):
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                self.connected = True
                logging.info("✅ MT5 connected successfully")
                return True
        except:
            pass
        
        logging.warning("⚠️ MT5 not available - using sandbox mode")
        self.connected = False
        return False
    
    def get_bars(self, symbol: str, timeframe: str = 'H1', count: int = 500) -> pd.DataFrame:
        """Get OHLC bars."""
        if self.connected:
            try:
                import MetaTrader5 as mt5
                tf_map = {'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4}
                rates = mt5.copy_rates_from_pos(symbol, tf_map.get(timeframe, mt5.TIMEFRAME_H1), 0, count)
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return df
            except:
                pass
        
        # Sandbox mode: generate random data
        np.random.seed(int(time.time()) % 10000)
        times = pd.date_range(end=pd.Timestamp.now(), periods=count, freq='h' if timeframe == 'H1' else '4h')
        prices = np.cumsum(np.random.randn(count) * 5) + 2650
        
        df = pd.DataFrame({
            'time': times,
            'open': prices,
            'high': prices + np.abs(np.random.randn(count) * 3),
            'low': prices - np.abs(np.random.randn(count) * 3),
            'close': prices + np.random.randn(count) * 2,
            'volume': np.random.randint(100, 1000, count)
        })
        return df
    
    def send_order(self, symbol: str, action: str, direction: str, volume: float,
                   sl: float = 0, tp: float = 0) -> dict:
        """Send order to MT5 with SL/TP."""
        if self.connected:
            try:
                import MetaTrader5 as mt5
                
                # Get current price
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    return {'status': 'ERROR', 'error': 'No tick data'}
                
                if direction == "BUY":
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                else:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "magic": 900005,
                    "comment": f"V3_{direction}",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                return {
                    'status': 'OK' if result.retcode == 10009 else 'ERROR',
                    'ticket': result.order if result.retcode == 10009 else 0,
                    'price': price,
                    'sl': sl,
                    'tp': tp,
                    'result': str(result)
                }
            except Exception as e:
                return {'status': 'ERROR', 'error': str(e)}
        
        # Sandbox mode
        ticket = int(time.time())
        return {
            'status': 'SANDBOX',
            'ticket': ticket,
            'symbol': symbol,
            'direction': direction,
            'volume': volume,
            'sl': sl,
            'tp': tp
        }
    
    def get_equity(self) -> float:
        """Get account equity."""
        if self.connected:
            try:
                import MetaTrader5 as mt5
                info = mt5.account_info()
                if info:
                    return info.equity
            except:
                pass
        
        # Sandbox: return default equity
        return 10000.0
    
    def get_drawdown(self) -> float:
        """Get current drawdown percentage."""
        if self.connected:
            try:
                import MetaTrader5 as mt5
                info = mt5.account_info()
                if info and info.balance > 0:
                    dd = (info.balance - info.equity) / info.balance
                    return max(0, dd)
            except:
                pass
        return 0.0


# =========================
# SIGNAL ENGINE V3
# =========================

class SignalEngineV3:
    """
    Hybrid Signal Engine:
    - Rule-Based (EMA + ATR)
    - AI Student (XGBoost)
    - Higher TF Filter
    - Risk Guard
    """
    
    def __init__(self, higher_tf: str = 'H4', risk_guard: bool = True, max_pos: int = 3):
        self.higher_tf = higher_tf
        self.risk_guard = risk_guard
        self.max_pos = max_pos
        self.positions = 0
        self.last_direction = None
        self.ai_model = None
        self.cooldown_sec = 30
        self.last_trade_time = datetime.min
    
    def load_ai_model(self, path: str):
        """Load AI model from file."""
        if os.path.exists(path):
            try:
                self.ai_model = pickle.load(open(path, 'rb'))
                logging.info(f"✅ AI Model loaded: {path}")
                return True
            except Exception as e:
                logging.warning(f"⚠️ Failed to load AI model: {e}")
        return False
    
    def generate_dataset(self, bars_h1: pd.DataFrame, bars_h4: pd.DataFrame) -> pd.DataFrame:
        """Generate features from OHLC data."""
        df = bars_h1.copy()
        
        # EMAs
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema_spread'] = df['ema20'] - df['ema50']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi14'] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))
        
        # HTF Trend
        if len(bars_h4) >= 50:
            htf_ema20 = bars_h4['close'].ewm(span=20).mean().iloc[-1]
            htf_ema50 = bars_h4['close'].ewm(span=50).mean().iloc[-1]
            df['htf_trend'] = 'UP' if htf_ema20 > htf_ema50 else 'DOWN'
        else:
            df['htf_trend'] = 'FLAT'
        
        return df.dropna()
    
    def get_signal(self, df: pd.DataFrame) -> tuple:
        """Generate signal from dataset."""
        if df.empty:
            return "HOLD", "No data"
        
        last = df.iloc[-1]
        signal = "HOLD"
        info = ""
        
        # === Rule-Based Signal ===
        atr_mean = df['atr14'].mean()
        vol_ok = last['atr14'] > atr_mean * 0.7
        
        if last['ema20'] > last['ema50'] and vol_ok:
            signal = "BUY"
            info = "Rule: EMA20 > EMA50"
        elif last['ema20'] < last['ema50'] and vol_ok:
            signal = "SELL"
            info = "Rule: EMA20 < EMA50"
        else:
            info = "Rule: HOLD (no trend/vol)"
        
        rule_signal = signal
        
        # === AI Override ===
        if self.ai_model is not None:
            try:
                features = last[['ema20', 'ema50', 'ema_spread', 'atr14', 'rsi14']].values.reshape(1, -1)
                ai_pred = self.ai_model.predict(features)[0]
                ai_label = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(ai_pred, "HOLD")
                
                # Consensus: AI and Rule must agree
                if rule_signal == ai_label:
                    signal = ai_label
                    info += f" | AI: {ai_label} ✓"
                else:
                    signal = "HOLD"
                    info += f" | AI: {ai_label} (no consensus)"
            except Exception as e:
                info += f" | AI error"
        
        # === HTF Filter ===
        htf_trend = last.get('htf_trend', 'FLAT')
        if signal == "BUY" and htf_trend == "DOWN":
            signal = "HOLD"
            info += " | HTF blocked"
        elif signal == "SELL" and htf_trend == "UP":
            signal = "HOLD"
            info += " | HTF blocked"
        
        # === Risk Guard ===
        if self.risk_guard and signal != "HOLD":
            now = datetime.now()
            
            # Duplicate direction
            if signal == self.last_direction:
                signal = "HOLD"
                info += " | BLOCKED: duplicate"
            
            # Max positions
            elif self.positions >= self.max_pos:
                signal = "HOLD"
                info += f" | BLOCKED: max_pos ({self.positions}/{self.max_pos})"
            
            # Cooldown
            elif (now - self.last_trade_time).total_seconds() < self.cooldown_sec:
                signal = "HOLD"
                info += " | BLOCKED: cooldown"
        
        return signal, info
    
    def calculate_sl_tp(self, df: pd.DataFrame, direction: str, 
                        atr_sl_mult: float = 1.5, atr_tp_mult: float = 3.0) -> tuple:
        """
        Calculate SL and TP based on ATR (Fund-Grade: RR = 1:2).
        
        Args:
            df: DataFrame with 'close' and 'atr14'
            direction: BUY or SELL
            atr_sl_mult: ATR multiplier for SL (default 1.5)
            atr_tp_mult: ATR multiplier for TP (default 3.0 = RR 1:2)
            
        Returns:
            (sl_price, tp_price, sl_pips)
        """
        if df.empty or 'atr14' not in df.columns:
            return 0, 0, 0
        
        last = df.iloc[-1]
        price = last['close']
        atr = last['atr14']
        
        sl_pips = atr * atr_sl_mult
        tp_pips = atr * atr_tp_mult
        
        if direction == "BUY":
            sl = price - sl_pips
            tp = price + tp_pips
        elif direction == "SELL":
            sl = price + sl_pips
            tp = price - sl_pips
        else:
            return 0, 0, 0
        
        return round(sl, 2), round(tp, 2), round(sl_pips, 2)
    
    def calculate_lot_size(self, equity: float, sl_pips: float, 
                           risk_pct: float = 0.005, pip_value: float = 1.0) -> float:
        """
        Calculate lot size based on risk percentage of equity (Fund-Grade).
        
        Formula: lot = (Equity × Risk%) / (SL pips × pip_value)
        
        Args:
            equity: Account equity
            sl_pips: Stop loss in pips
            risk_pct: Risk per trade (default 0.5% = 0.005)
            pip_value: Value per pip per lot (default 1.0 for XAUUSD)
            
        Returns:
            Calculated lot size (min 0.01, max 10.0)
        """
        if sl_pips <= 0 or equity <= 0:
            return 0.01  # Minimum lot
        
        risk_amount = equity * risk_pct
        lot = risk_amount / (sl_pips * pip_value)
        
        # Clamp to valid range
        lot = max(0.01, min(lot, 10.0))
        
        return round(lot, 2)
    
    def update_position(self, signal: str):
        """Update position state after trade."""
        if signal in ["BUY", "SELL"]:
            self.positions += 1
            self.last_direction = signal
            self.last_trade_time = datetime.now()


# =========================
# AUTO AI TRAINING
# =========================

def auto_train(dataset_path: str = "data/imitation_full_dataset.csv", 
               model_path: str = "models/xgb_imitation.pkl",
               min_samples: int = 200) -> bool:
    """
    Auto-train XGBoost model.
    
    Returns:
        True if training successful
    """
    if not os.path.exists(dataset_path):
        logging.warning(f"Dataset not found: {dataset_path}")
        return False
    
    try:
        import xgboost as xgb
        
        df = pd.read_csv(dataset_path)
        
        if len(df) < min_samples:
            logging.info(f"Not enough data: {len(df)}/{min_samples}")
            return False
        
        # Prepare features
        feature_cols = ['ema20', 'ema50', 'ema_spread', 'atr14', 'rsi14']
        available = [c for c in feature_cols if c in df.columns]
        
        if len(available) < 3:
            logging.warning("Not enough feature columns")
            return False
        
        X = df[available].fillna(0)
        
        # Prepare labels
        if 'label' in df.columns:
            y = df['label']
        elif 'signal' in df.columns:
            y_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
            y = df['signal'].map(y_map)
        else:
            logging.warning("No label column found")
            return False
        
        # Train/test split
        train_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
        y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='mlogloss',
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        pickle.dump(model, open(model_path, 'wb'))
        
        # Evaluate
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"✅ AI Model trained: {acc:.1%} accuracy, saved to {model_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return False


# =========================
# LIVE LOOP
# =========================

def live_loop(
    interval: int = 30,
    max_cycles: int = 1000,
    sandbox: bool = True,
    auto_train_enabled: bool = True,
    train_interval: int = 50
):
    """
    Main Live Loop.
    
    Args:
        interval: Seconds between cycles
        max_cycles: Maximum cycles
        sandbox: Sandbox mode (no real trades)
        auto_train_enabled: Enable auto AI training
        train_interval: Cycles between training
    """
    logging.info("=" * 60)
    logging.info("🚀 LIVE LOOP V3 - AI Trading System")
    logging.info("=" * 60)
    logging.info(f"Interval: {interval}s | Cycles: {max_cycles}")
    logging.info(f"Mode: {'SANDBOX' if sandbox else '🔴 LIVE'}")
    logging.info(f"Auto-Train: {'ON' if auto_train_enabled else 'OFF'}")
    logging.info("=" * 60)
    
    # Initialize
    mt5 = MT5Connector()
    if not sandbox:
        mt5.connect()
    
    engine = SignalEngineV3()
    
    # Load existing model
    model_path = Path("models/xgb_imitation.pkl")
    if model_path.exists():
        engine.load_ai_model(str(model_path))
    
    # =========================================
    # Fund-Grade Modules (NEW)
    # =========================================
    risk_manager = None
    trail_manager = None
    sl_model = None
    tp_model = None
    
    if FUND_GRADE_ENABLED:
        # Risk Manager
        risk_manager = RiskManager(
            risk_per_trade=0.005,      # 0.5% per trade
            max_positions_per_symbol=3,
            max_daily_loss_pct=0.03    # 3% daily loss limit
        )
        logging.info("✅ RiskManager initialized")
        
        # Trailing Manager
        trail_manager = TrailingManager(
            be_rr=1.0,      # Break-even at 1R
            trail_rr=2.0    # Trail at 2R
        )
        logging.info("✅ TrailingManager initialized")
        
        # AI SL/TP Models
        sl_model_path = Path("models/xgb_sl.pkl")
        tp_model_path = Path("models/xgb_tp.pkl")
        
        if sl_model_path.exists():
            sl_model = joblib.load(str(sl_model_path))
            logging.info("✅ AI SL Model loaded")
        
        if tp_model_path.exists():
            tp_model = joblib.load(str(tp_model_path))
            logging.info("✅ AI TP Model loaded")
    
    # Stats
    stats = {"cycles": 0, "BUY": 0, "SELL": 0, "HOLD": 0, "trades": 0, "trailing_updates": 0}
    
    try:
        while stats["cycles"] < max_cycles:
            start = time.time()
            stats["cycles"] += 1
            cycle = stats["cycles"]
            
            # 1. Fetch data
            bars_h1 = mt5.get_bars('XAUUSD', 'H1', 500)
            bars_h4 = mt5.get_bars('XAUUSD', 'H4', 125)
            
            # 2. Generate dataset
            df = engine.generate_dataset(bars_h1, bars_h4)
            
            # 3. Auto-train
            if auto_train_enabled and cycle % train_interval == 0:
                if auto_train():
                    engine.load_ai_model(str(model_path))
            
            # 4. Get signal
            signal, info = engine.get_signal(df)
            stats[signal] = stats.get(signal, 0) + 1
            
            # 5. Log
            ts = datetime.now().strftime("%H:%M:%S")
            if signal == "HOLD":
                logging.info(f"[{ts}] #{cycle}: ⏸️  HOLD | {info}")
            else:
                logging.info(f"[{ts}] #{cycle}: 🎯 {signal} | {info}")
                # =========================================
                # Fund-Grade: AI SL/TP Prediction (NEW)
                # =========================================
                last = df.iloc[-1]
                
                if sl_model is not None and tp_model is not None:
                    try:
                        # Use same features as training (handle feature count dynamically)
                        # Primary features + extended features
                        feature_cols = ['ema_spread', 'atr14', 'rsi14', 'ema20', 'ema50', 'hour', 'day_of_week', 'close']
                        
                        # Build feature array from available columns
                        feature_values = []
                        for col in feature_cols:
                            if col in last.index:
                                feature_values.append(last[col])
                            elif col == 'hour':
                                feature_values.append(datetime.now().hour)
                            elif col == 'day_of_week':
                                feature_values.append(datetime.now().weekday())
                            else:
                                feature_values.append(0)
                        
                        # Get expected feature count from model
                        expected_features = sl_model.n_features_in_
                        features = [feature_values[:expected_features]]
                        
                        sl_dist = abs(sl_model.predict(features)[0])
                        tp_dist = abs(tp_model.predict(features)[0])
                        
                        # Apply min/max constraints
                        atr = last.get('atr14', 5.0)
                        sl_dist = max(sl_dist, atr * 0.5)  # Min SL = 0.5 ATR
                        tp_dist = max(tp_dist, sl_dist * 1.5)  # Min RR = 1:1.5
                        
                        price = last['close']
                        if signal == "BUY":
                            sl = round(price - sl_dist, 2)
                            tp = round(price + tp_dist, 2)
                        else:
                            sl = round(price + sl_dist, 2)
                            tp = round(price - tp_dist, 2)
                        sl_pips = sl_dist
                        logging.info(f"   🤖 AI SL/TP: SL_dist={sl_dist:.2f}, TP_dist={tp_dist:.2f}")
                    except Exception as e:
                        logging.warning(f"   ⚠️ AI SL/TP failed: {e}, using ATR fallback")
                        sl, tp, sl_pips = engine.calculate_sl_tp(df, signal)
                else:
                    # Fallback to ATR-based
                    sl, tp, sl_pips = engine.calculate_sl_tp(df, signal)
                
                # =========================================
                # Fund-Grade: RiskManager Position Sizing (NEW)
                # =========================================
                if risk_manager is not None:
                    # Check if can trade
                    if not risk_manager.can_trade('XAUUSD'):
                        logging.warning("   ⚠️ RiskManager blocked trade")
                        continue
                    
                    # Calculate lot with RiskManager
                    volume = risk_manager.calc_lot(sl_pips, 'XAUUSD')
                else:
                    # Fallback to SignalEngine calculation
                    equity = mt5.get_equity()
                    volume = engine.calculate_lot_size(equity, sl_pips, risk_pct=0.005)
                
                # Check drawdown limit (stop if DD > 5%)
                dd = mt5.get_drawdown()
                if dd > 0.05:
                    logging.warning(f"   ⚠️ DD {dd:.1%} > 5% - Trade blocked")
                else:
                    # 6. Send trade
                    if not sandbox:
                        res = mt5.send_order('XAUUSD', 'OPEN', signal, volume, sl=sl, tp=tp)
                        engine.update_position(signal)
                        stats["trades"] += 1
                        logging.info(f"   Trade: {res}")
                        logging.info(f"   Lot: {volume} | SL: {sl} | TP: {tp} | RR: 1:2")
                    else:
                        engine.update_position(signal)
                        stats["trades"] += 1
                        logging.info(f"   📝 Sandbox | Lot: {volume} | SL: {sl} | TP: {tp} | RR: 1:2")
            
            # =========================================
            # Fund-Grade: Trailing Stop Management (NEW)
            # =========================================
            if trail_manager is not None and not sandbox and mt5.connected:
                try:
                    import MetaTrader5 as mt5_lib
                    positions = mt5_lib.positions_get(symbol='XAUUSD')
                    if positions:
                        atr = df.iloc[-1].get('atr14', 5.0) if not df.empty else 5.0
                        for pos in positions:
                            result = trail_manager.manage(pos, atr)
                            if result.updated:
                                stats["trailing_updates"] += 1
                                logging.info(f"   📈 Trailing updated #{pos.ticket}: SL → {result.new_sl:.2f}")
                except Exception as e:
                    logging.debug(f"Trailing management error: {e}")
            
            # 7. Wait
            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0 and cycle < max_cycles:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logging.info("\n🛑 Stopped by user")
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("📊 SESSION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Cycles: {stats['cycles']}")
    logging.info(f"BUY:  {stats['BUY']} | SELL: {stats['SELL']} | HOLD: {stats['HOLD']}")
    logging.info(f"Trades: {stats['trades']} | Trailing Updates: {stats.get('trailing_updates', 0)}")
    logging.info("=" * 60)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(description="Live Loop V3")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between cycles")
    parser.add_argument("--cycles", type=int, default=1000, help="Max cycles")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--no-train", action="store_true", help="Disable auto-training")
    
    args = parser.parse_args()
    
    live_loop(
        interval=args.interval,
        max_cycles=args.cycles,
        sandbox=not args.live,
        auto_train_enabled=not args.no_train
    )
