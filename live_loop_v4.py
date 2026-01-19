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
import csv
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib

# Dashboard Integration
try:
    from src.dashboard.workflow_state import get_workflow_client
    HAS_DASHBOARD = True
    wf = get_workflow_client()
except ImportError:
    HAS_DASHBOARD = False
    logging.warning("Dashboard module not found")


# Fund-Grade Risk Management
try:
    from src.risk.risk_manager import RiskManager
    from src.risk.trailing import TrailingManager
    # Guardian Margin Gate Integration
    try:
        from src.safety.guardian_margin_gate import GuardianMarginGate
        HAS_GUARDIAN = True
    except ImportError:
        HAS_GUARDIAN = False
        logging.warning("Guardian module not found")
        
    FUND_GRADE_ENABLED = True
except ImportError:
    FUND_GRADE_ENABLED = False
    logging.warning("Fund-Grade modules not available")



from src.safety.guardian_margin_gate import GuardianMarginGate
from src.rl.guardian_agent import get_guardian_agent
from src.safety.progressive_guard import get_progressive_guard
from src.safety.kill_switch import get_kill_switch
from src.utils.logger import get_logger

logger = get_logger("LIVE_V3")

def is_new_trading_day(last_day):
    """Check if we have crossed into a new UTC date."""
    today = datetime.now(timezone.utc).date()
    return last_day is None or today > last_day


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
    
        return Account()

    def symbol_info_tick(self, symbol: str):
        """Get current tick."""
        if self.connected:
            try:
                import MetaTrader5 as mt5
                return mt5.symbol_info_tick(symbol)
            except: pass
        return None

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
                
                # Check Filling Mode Support
                info = mt5.symbol_info(symbol)
                if info:
                    filling_mode = info.filling_mode
                    if filling_mode == 1:
                        request["type_filling"] = mt5.ORDER_FILLING_FOK
                    elif filling_mode == 2:
                        request["type_filling"] = mt5.ORDER_FILLING_IOC
                    elif filling_mode == 3: # FOK + IOC
                         request["type_filling"] = mt5.ORDER_FILLING_FOK # Prefer FOK
                    elif filling_mode == 0:
                        request["type_filling"] = mt5.ORDER_FILLING_RETURN

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

    def get_account_info(self):
        """Get full account info object."""
        if self.connected:
            try:
                import MetaTrader5 as mt5_lib
                return mt5_lib.account_info()
            except:
                pass
        return None


# Helper removed - imported from src.guardian.guardian_governance


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
            
            # 🔧 FIX: Check ACTUAL MT5 positions, not internal state
            has_same_dir_position = self._has_open_position_mt5(signal)
            actual_positions = self._count_open_positions_mt5()
            
            # Duplicate direction - ONLY block if position actually exists
            if has_same_dir_position and not self._allow_pyramid():
                signal = "HOLD"
                info += f" | BLOCKED: duplicate ({signal} position open)"
            
            # Max positions - use ACTUAL count from MT5
            elif actual_positions >= self.max_pos:
                signal = "HOLD"
                info += f" | BLOCKED: max_pos ({actual_positions}/{self.max_pos})"
            
            # Cooldown
            elif (now - self.last_trade_time).total_seconds() < self.cooldown_sec:
                signal = "HOLD"
                info += " | BLOCKED: cooldown"
        
        return signal, info
    
    def _has_open_position_mt5(self, direction: str) -> bool:
        """Check if there's an open position in given direction using MT5."""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol='XAUUSD')
            if positions is None or len(positions) == 0:
                return False
            
            for pos in positions:
                pos_dir = "BUY" if pos.type == 0 else "SELL"
                if pos_dir == direction:
                    return True
            return False
        except:
            return False
    
    def _count_open_positions_mt5(self) -> int:
        """Count actual open positions from MT5."""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol='XAUUSD')
            return len(positions) if positions else 0
        except:
            return self.positions  # Fallback to internal count
    
    def _allow_pyramid(self) -> bool:
        """Check if pyramiding (multiple same-direction positions) is allowed."""
        # For Aggressive profile, allow pyramiding
        try:
            from src.config.trading_profiles import get_active_profile
            profile = get_active_profile()
            return profile.entry.pyramid_allowed
        except:
            return False  # Conservative default
    
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
        
        return round(sl, 5), round(tp, 5), round(sl_pips, 5)
    
    def calculate_lot_size(self, equity: float, sl_dist: float, 
                           risk_pct: float = 0.005, symbol: str = "XAUUSD") -> float:
        """
        Calculate lot size based on risk percentage of equity.
        Uses MT5 symbol info for accurate TickValue.
        
        Args:
            equity: Account equity
            sl_dist: Stop loss distance in PRICE (not points)
            risk_pct: Risk per trade (default 0.5% = 0.005)
            symbol: Symbol to query tick value

            
        Returns:
            Calculated lot size (min 0.01, max 10.0)
        """
        if sl_dist <= 0 or equity <= 0:
            return 0.01

        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            if info and info.trade_tick_size > 0:
                tick_size = info.trade_tick_size
                tick_value = info.trade_tick_value
                # Formula: Lot = RiskMoney / (SL_Points * TickValue)
                # SL_Points = SL_Price_Dist / TickSize
                sl_points = sl_dist / tick_size
                risk_amount = equity * risk_pct
                
                # Avoid division by zero
                if sl_points * tick_value == 0:
                    return 0.01
                    
                lot = risk_amount / (sl_points * tick_value)
            else:
                # Fallback (Dangerous, assume XAUUSD-like 1.0)
                lot = (equity * risk_pct) / (sl_dist * 1.0) 
        except:
            lot = 0.01
        
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


# =========================================
# Multi-Symbol Logger
# =========================================
class MultiSymbolLogger:
    def __init__(self, filename="logs/guardian_multisymbol.csv"):
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = [
            "timestamp", "cycle", "symbol", "action", "confidence", 
            "equity", "margin_ratio", "block_reason", "daily_dd"
        ]
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def log(self, data):
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow({k: data.get(k, "") for k in self.headers})

def live_loop(
    interval: int = 30,
    max_cycles: int = 1000,
    sandbox: bool = True,
    auto_train_enabled: bool = True,
    train_interval: int = 50
):
    """
    Main Live Loop (Multi-Symbol Engine V4).
    """
    SYMBOLS = ["XAUUSD", "EURUSD", "BTCUSD", "GBPJPY"]
    
    # Initialize Multi-Symbol Logger
    ms_logger = MultiSymbolLogger()
    
    logging.info("=" * 60)
    logging.info("🚀 LIVE LOOP V4 - Multi-Symbol AI Engine")
    logging.info("=" * 60)
    logging.info(f"Interval: {interval}s | Cycles: {max_cycles}")
    logging.info(f"Symbols: {SYMBOLS}")
    logging.info(f"Mode: {'SANDBOX' if sandbox else '🔴 LIVE'}")
    logging.info("=" * 60)
    
    # Initialize
    mt5_lib = MT5Connector() # Renamed to avoid shadowing import
    if not sandbox:
        if not mt5_lib.connect():
             return

        # NEW CHECKS
        import MetaTrader5 as mt5
        
        # 1. Check Algo Trading (Blocking)
        while True:
            term_info = mt5.terminal_info()
            if term_info.trade_allowed:
                break
            
            # Prevent Watchdog Kill while waiting
            try: 
                with open("heartbeat.txt", "w") as f: f.write(str(time.time()))
            except: pass
            
            logging.critical("❌ ALGO TRADING DISABLED! Please enable 'Algo Trading' button in MT5.")
            logging.warning("   Waiting... (Checking again in 5s)")
            time.sleep(5)
        
        # 2. Validate Symbols (Remove Missing)
        valid_symbols = []
        for sym in SYMBOLS:
            info = mt5.symbol_info(sym)
            if not info:
                logging.error(f"❌ Symbol '{sym}' not found! Removing from list.")
                continue
            
            if not info.visible:
                if not mt5.symbol_select(sym, True):
                     logging.warning(f"⚠️ Could not select '{sym}'. Removing.")
                     continue
            
            valid_symbols.append(sym)
        
        if not valid_symbols:
            logging.critical("❌ No valid symbols found! Exiting.")
            return

        SYMBOLS = valid_symbols
        logging.info(f"✅ Active Symbols: {SYMBOLS}")
    
    # Initialize Engines & State
    
    # Get profile for max_pos
    try:
        from src.config.trading_profiles import get_active_profile
        _profile = get_active_profile()
        _max_pos = _profile.risk.max_open_positions
    except:
        _max_pos = 3  # Fallback
    
    # 🧠 Initialize Engines per Symbol
    engines = {sym: SignalEngineV3(max_pos=_max_pos) for sym in SYMBOLS}
    
    # =========================================
    # ⚙️ Load Trading Profile (Source of Truth)
    # =========================================
    from src.config.trading_profiles import get_active_profile
    active_profile = get_active_profile()
    logging.info(f"🔥 Trading Profile Loaded: {active_profile.name}")
    
    # 🛡️ Safety Systems (Global)
    progressive_guard = get_progressive_guard()
    from src.safety.progressive_guard import AlertThresholds
    progressive_guard.dd_thresholds = AlertThresholds(4.0, 6.0, 10.0)
    
    kill_switch = get_kill_switch()
    kill_switch.max_dd = active_profile.risk.hard_equity_dd_kill
    logging.info(f"🛡️ Kill Switch Initialized (MaxDD: {kill_switch.max_dd}%)")

    # =========================================
    # 🛡️ Guardian Governance (Global Account Level)
    # =========================================
    from src.guardian.guardian_governance import GuardianGovernance
    daily_limit_decimal = active_profile.risk.max_daily_loss / 100.0
    guardian_governance = GuardianGovernance(daily_dd_limit=daily_limit_decimal)
    
    # Bridge: Connect Governance to Margin Gate
    def update_dd_bridge(equity, balance, logger):
        mock_acc = type("Account", (), {"equity": equity, "balance": balance})
        blocked, reason = guardian_governance.check_daily_dd(mock_acc)
        return blocked, reason

    guardian_governance.update_dd_and_check = update_dd_bridge
    logging.info(f"🛡️ Guardian Governance Initialized (Limit: {active_profile.risk.max_daily_loss}%)")

    # 🤖 Guardian Agent (Rule-Based)
    try:
        from src.rl.guardian_agent import GuardianAgent
        guardian_agent = GuardianAgent(mode="advisor")
    except Exception as e:
        logging.error(f"Guardian Agent init failed: {e}")
        guardian_agent = None

    logging.info(f"🧠 Guardian Agent Initialized (Advisor Mode)")
    
    # 📊 Guardian Metrics Logger
    guardian_csv_logger = None
    try:
        from src.rl.guardian_logger import get_guardian_logger
        guardian_csv_logger = get_guardian_logger()
    except: pass
    
    # 🧠 PPO Guardian Advisor (Hybrid Mode)
    guardian_ppo = None
    try:
        from src.rl.guardian_ppo_infer import get_ppo_advisor
        guardian_ppo = get_ppo_advisor(enabled=True)
        if guardian_ppo.enabled:
            logging.info(f"🧠 Guardian PPO Advisor loaded")
    except: pass
    
    # 🧬 Guardian Hybrid Arbitration Layer
    guardian_hybrid = None
    if guardian_agent and guardian_ppo:
        from src.guardian.guardian_hybrid import GuardianHybrid
        guardian_hybrid = GuardianHybrid(guardian_agent, guardian_ppo, confidence_threshold=0.65)
        logging.info("🧬 Guardian Hybrid Layer initialized (Rule + PPO)")

    # 👻 Shadow Mode
    shadow_recorder = None
    try:
        from src.analysis.shadow_recorder import ShadowRecorder
        shadow_recorder = ShadowRecorder()
        logging.info("👻 Shadow Mode Simulator Initialized")
    except: pass
    
    # 🎛️ Guardian Auto-Tuner
    from src.guardian.guardian_autotuner import GuardianAutoTuner
    auto_tuner = GuardianAutoTuner()
    
    # 🔄 Daily Retrain Job
    retrain_job = None
    try:
        from src.retrain.daily_retrain_job import get_daily_retrain_job
        retrain_job = get_daily_retrain_job()
    except: pass

    # Load existing models
    model_path = Path("models/xgb_imitation.pkl")
    if model_path.exists():
        for engine in engines.values():
            engine.load_ai_model(str(model_path))
    
    # Fund-Grade Modules
    trail_manager = None
    if FUND_GRADE_ENABLED:
        trail_manager = TrailingManager(profile=active_profile)
        logging.info("✅ TrailingManager initialized (Global)")
    
    # Stats
    stats = {"cycles": 0, "BUY": 0, "SELL": 0, "HOLD": 0, "trades": 0}
    
    # 🛡️ Guardian State (Per Symbol)
    guardian_states = {
        sym: {
            "margin_block_count": 0,
            "dd_block_count": 0,
            "force_hold_until": 0.0,
            "last_block_reason": None,
            "escalation_count": 0,
        } for sym in SYMBOLS
    }
    
    # Global State
    global_dd = 0.0

    while stats["cycles"] < max_cycles:
        start_cycle_time = time.time()
        stats["cycles"] += 1
        cycle = stats["cycles"]
        
        # ==============================
        # 1. GLOBAL SAFETY CHECKS
        # ==============================
        guard_status = progressive_guard.get_status()
        if guard_status.get("kill_latched"):
            logging.critical("🔒 SYSTEM HALTED BY PROGRESSIVE GUARD")
            break

        # Account Check
        if mt5_lib.connected:
            acc = mt5_lib.get_account_info()
            if acc:
                if acc.equity < 50 or acc.margin_free <= 0:
                    logging.critical(f"🚨 ACCOUNT DEAD | Eq: {acc.equity} | FreeMargin: {acc.margin_free}")
                    time.sleep(interval)
                    continue
                
                # Governance Check (Global)
                blocked_by_gov, gov_reason = guardian_governance.update_dd_and_check(acc.equity, acc.balance, logging)
                global_dd = guardian_governance.current_dd
                
                if blocked_by_gov:
                    logging.critical(f"🛑 GOVERNANCE BLOCK: {gov_reason}")
                    time.sleep(interval)
                    continue

        # ==============================
        # 2. SYMBOL LOOP
        # ==============================
        for symbol in SYMBOLS:
            engine = engines[symbol]
            g_state = guardian_states[symbol]
            
            # --- Per-Symbol Logic ---
            
            # 1. Fetch Data
            try:
                bars_h1 = mt5_lib.get_bars(symbol, 'H1', 500)
                bars_h4 = mt5_lib.get_bars(symbol, 'H4', 125)
                
                if bars_h1 is None or len(bars_h1) < 50:
                    continue
            except Exception as e:
                logging.error(f"Data fetch fail {symbol}: {e}")
                continue

            # Shadow Update
            if shadow_recorder:
                tick = mt5_lib.symbol_info_tick(symbol)
                if tick:
                    shadow_recorder.update(tick.bid, tick.ask)

            # 2. Generate Features & Signal
            df = engine.generate_dataset(bars_h1, bars_h4)
            signal, info = engine.get_signal(df)
            
            # 3. Force Hold Check (Per Symbol)
            if time.time() < g_state.get("force_hold_until", 0):
                continue

            # 4. Guardian Hybrid Check
            if guardian_hybrid:
                # Construct State
                acc = mt5_lib.get_account_info() if mt5_lib.connected else None
                margin_ratio = (acc.margin_free / acc.balance) if acc and acc.balance > 0 else 1.0
                
                from src.guardian.guardian_context import GuardianContext
                context = GuardianContext.get_context(df)
                
                nav_state = {
                    "daily_dd": global_dd, # Shared Global DD
                    "margin_ratio": margin_ratio,
                    "chaos": 0, 
                    "step": cycle,
                    "margin_block_count": g_state.get("margin_block_count", 0),
                    "market_regime": context["market_regime"],
                    "session_time": context["session_time"],
                    "recent_win_rate": 0.5
                }
                
                hybrid_action, hybrid_reason = guardian_hybrid.decide(nav_state, signal)
                
                # Log to Multi-Symbol CSV
                ms_logger.log({
                    "cycle": cycle,
                    "symbol": symbol,
                    "action": hybrid_action,
                    "confidence": 0.0, # Placeholder until PPO returns prob
                    "equity": acc.equity if acc else 0,
                    "margin_ratio": margin_ratio,
                    "block_reason": hybrid_reason,
                    "daily_dd": global_dd
                })
                
                # 3. Apply Decision
                if hybrid_action in ["BLOCK", "FORCE_HOLD", "EMERGENCY_FREEZE"]:
                    logging.info(f"🧬 GUARDIAN BLOCK: {hybrid_reason}")
                    
                    if shadow_recorder and signal in ["BUY", "SELL"]:
                        try:
                            # 1. Get Price
                            tick_s = mt5_lib.symbol_info_tick(symbol)
                            if tick_s:
                                price = tick_s.ask if signal == "BUY" else tick_s.bid
                                atr = df.iloc[-1].get('atr14', 2.5)
                                sl_dist = atr * 2.0
                                tp_dist = atr * 4.0
                                
                                sl = price - sl_dist if signal == "BUY" else price + sl_dist
                                tp = price + tp_dist if signal == "BUY" else price - tp_dist
                                
                                shadow_recorder.place_trade(signal, price, sl, tp, 0.01, hybrid_reason, symbol)
                                logging.info(f"👻 Shadow Trade Recorded: {signal} @ {price:.2f}")
                        except Exception as e:
                             logging.error(f"Shadow Record Fail: {e}")

                    # BLOCK EXECUTION & UPDATE STATE
                    g_state["margin_block_count"] = g_state.get("margin_block_count", 0) + 1
                    g_state["last_block_reason"] = hybrid_reason
                    
                    if hybrid_action == "FORCE_HOLD":
                        g_state["force_hold_until"] = time.time() + 300 # 5 min
                    elif hybrid_action == "EMERGENCY_FREEZE":
                        g_state["force_hold_until"] = time.time() + 1800 # 30 min

                    continue 
                else:
                    # ALLOW: Reset block counts on successful pass
                    g_state["margin_block_count"] = 0
                    g_state["dd_block_count"] = 0

            # 5. Execute
            if signal in ["BUY", "SELL"]:
                # Calculate SL/TP
                atr = df.iloc[-1].get('atr14', 2.0)
                sl, tp, _ = engine.calculate_sl_tp(df, signal)
                
                # Dynamic Lot
                acc = mt5_lib.get_account_info()
                equity = acc.equity if acc else 1000
                lot = engine.calculate_lot_size(equity, atr * 1.5, symbol=symbol)
                
                logging.info(f"🚀 EXECUTE [{symbol}]: {signal} {lot} lots")
                
                if not sandbox:
                    res = mt5_lib.send_order(symbol, 'OPEN', signal, lot, sl=sl, tp=tp)
                    engine.update_position(signal)
                    stats["trades"] += 1
                    logging.info(f"   Trade: {res}")
                    logging.info(f"   Lot: {lot} | SL: {sl} | TP: {tp}")
                else:
                    engine.update_position(signal)
                    stats["trades"] += 1
                    logging.info(f"   📝 Sandbox | Lot: {lot}")

            # 6. Manage Trailing Stops
            if trail_manager and not sandbox:
                try:
                    positions = mt5_lib.positions_get(symbol=symbol)
                    if positions:
                        atr_val = df.iloc[-1].get('atr14', 1.0)
                        for pos in positions:
                            trail_manager.manage(pos, atr_val)
                except: pass

        # ==============================
        # END OF CYCLE
        # ==============================
        
        # 💓 Heartbeat & Summary
        try:
            with open("heartbeat.txt", "w") as f:
                f.write(str(time.time()))
        except: pass
        
        # 📊 Guardian Summary & Auto-Tuning (every 10 cycles)
        if cycle % 10 == 0:
            logging.info("-" * 40)
            logging.info(f"📊 SUMMARY #{cycle} | DD: {guardian_governance.current_dd:.2f}%")
            if auto_tuner:
                 comp_metrics = get_competition_metrics()
                 new_config = auto_tuner.tune(comp_metrics, cycle, profile=active_profile)
                 if new_config:
                     logging.info(f"🎛️ Auto-Tuner Adjustment: {new_config}")
                     if guardian_hybrid: guardian_hybrid.update_config(new_config)
            logging.info("-" * 40)
            logging.info("-" * 40)

        # Wait at end of cycle (Inside Loop)
        elapsed = time.time() - start_cycle_time
        sleep_time = max(1, interval - elapsed)
        time.sleep(sleep_time)




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
