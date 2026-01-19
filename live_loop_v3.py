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
        
        # === Risk Guard (Delegated to RiskManager later) ===
        # We only return the Raw Signal here. 
        # Position limits and duplicates are handled by RiskManager in the loop.
        
        return signal, info
    
        return signal, info
    
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
            y_map = {'SELL': 0, 'HOLD': 1, 'WAIT': 1, 'BUY': 2}
            y = df['signal'].map(y_map)
        else:
            logging.warning("No label column found")
            return False
            
        # UNIVERSAL CLEANING (Fix NaN issues)
        # Drop NaNs from y and align X
        valid_stats = y.notna()
        X = X[valid_stats]
        y = y[valid_stats]
        
        if len(y) < min_samples:
            logging.info(f"Not enough valid labels: {len(y)}/{min_samples}")
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
            verbosity=0,
            tree_method='hist',
            device='cuda'
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
    train_interval: int = 50,
    alpha_mode: str = "shadow"  # "shadow", "hybrid", "full"
):
    """
    Main Live Loop.
    
    Args:
        interval: Seconds between cycles
        max_cycles: Maximum cycles
        sandbox: Sandbox mode (no real trades)
        auto_train_enabled: Enable auto AI training
        train_interval: Cycles between training
        alpha_mode: Alpha operating mode ("shadow", "hybrid", "full")
    """
    logging.info("=" * 60)
    logging.info("🚀 LIVE LOOP V3 - AI Trading System")
    logging.info("=" * 60)
    logging.info(f"Interval: {interval}s | Cycles: {max_cycles}")
    logging.info(f"Mode: {'SANDBOX' if sandbox else '🔴 LIVE'}")
    logging.info(f"Auto-Train: {'ON' if auto_train_enabled else 'OFF'}")
    logging.info(f"Alpha Mode: {alpha_mode.upper()}")
    logging.info("=" * 60)
    
    # =========================================
    # 🧠 Alpha Hybrid Controller (Controlled Live)
    # =========================================
    alpha_hybrid_ctrl = None
    try:
        from src.rl.alpha_hybrid_controller import get_alpha_hybrid_controller
        alpha_hybrid_ctrl = get_alpha_hybrid_controller(mode=alpha_mode)
        logging.info(f"🧠 Alpha Hybrid Controller: mode={alpha_mode}")
    except Exception as e:
        logging.debug(f"Alpha Hybrid Controller init skipped: {e}")
    
    # Initialize
    mt5_lib = MT5Connector()
    if not sandbox:
        if not mt5_lib.connect():
             return

        # NEW CHECKS (Backported from V4)
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
    
    # Get profile for max_pos
    try:
        from src.config.trading_profiles import get_active_profile
        _profile = get_active_profile()
        _max_pos = _profile.risk.max_open_positions
    except:
        _max_pos = 3  # Fallback
    
    engine = SignalEngineV3(max_pos=_max_pos)
    
    # (Legacy Guardian Init Removed - See "Load Trading Profile" section below)
    
    # (Safety Systems Init Moved - See below)
    # =========================================
    # ⚙️ Load Trading Profile (Source of Truth)
    # =========================================
    from src.config.trading_profiles import get_active_profile
    active_profile = get_active_profile()
    logging.info(f"🔥 Trading Profile Loaded: {active_profile.name}")
    logging.info(f"   Risk: {active_profile.risk.max_daily_loss}% Daily Limit | Kill: {active_profile.risk.hard_equity_dd_kill}%")

    # 🛡️ Safety Systems (Synced with Profile)
    progressive_guard = get_progressive_guard()
    # Initial defaults from Competition Profile guidelines (4/6/10)
    # Note: These are actively tuned by auto-tuner later
    from src.safety.progressive_guard import AlertThresholds
    progressive_guard.dd_thresholds = AlertThresholds(4.0, 6.0, 10.0)
    
    kill_switch = get_kill_switch()
    kill_switch.max_dd = active_profile.risk.hard_equity_dd_kill
    logging.info(f"🛡️ Kill Switch Initialized (MaxDD: {kill_switch.max_dd}%)")

    # =========================================
    # 🛡️ Guardian Governance (Daily Reset & Latch)
    # =========================================
    from src.guardian.guardian_governance import GuardianGovernance
    
    # Initialize Governance from Profile
    # Convert % to decimal
    daily_limit_decimal = active_profile.risk.max_daily_loss / 100.0
    guardian_governance = GuardianGovernance(daily_dd_limit=daily_limit_decimal)
    
    # Bridge: Connect Governance to Margin Gate (for hard latching)
    def update_dd_bridge(equity, balance, logger):
        """Bridge Governance DD calculation -> Margin Gate Latch"""
        # Create a mock account object for compatibility
        mock_acc = type("Account", (), {"equity": equity, "balance": balance})
        
        blocked, reason = guardian_governance.check_daily_dd(mock_acc)
        
        # Sync DD to Margin Gate (it expects percent e.g. 10.5)
        if guardian:
             # Push calculated DD (decimal) converted to percent
             guardian.update_dd(guardian_governance.current_dd * 100)
            
        return blocked, reason

    # Attach bridge (Method injection for compatibility)
    guardian_governance.update_dd_and_check = update_dd_bridge
    
    logging.info(f"🛡️ Guardian Governance Initialized (Limit: {active_profile.risk.max_daily_loss}%)")
    
    # Initialize Margin Gate (Using Profile Limit)
    try:
        from src.safety.guardian_margin_gate import GuardianMarginGate
        
        limit_val = active_profile.risk.max_daily_loss # e.g. 14.0
        guardian = GuardianMarginGate(daily_loss_limit=limit_val/100.0) 
        
        # Inject the bridge
        guardian.check_governance_dd = update_dd_bridge
        
        # IMPORTANT: Margin buffer from competition config
        if hasattr(active_profile, "competition") and active_profile.competition.enabled:
             # Logic to tighten margin block if needed could go here
             pass 
             
        logging.info(f"🛡️ Guardian Gate Initialized (Limit: {limit_val}%)")
    except Exception as e:
        logging.error(f"Guardian init failed: {e}")
        guardian = None

    # 🤖 Guardian Agent (Rule-Based)
    try:
        from src.rl.guardian_agent import GuardianAgent
        def get_guardian_agent():
             # Fixed: GuardianAgent takes mode, not daily_loss_limit
             return GuardianAgent(mode="advisor")
        
        guardian_agent = get_guardian_agent()
    except Exception as e:
        logging.error(f"Guardian Agent init failed: {e}")
        guardian_agent = None

    last_trading_day = datetime.now(timezone.utc).date()
    logging.info(f"🧠 Guardian Agent Initialized (Advisor Mode)")
    
    # 📊 Guardian Metrics Logger
    guardian_csv_logger = None
    try:
        from src.rl.guardian_logger import get_guardian_logger
        guardian_csv_logger = get_guardian_logger()
        logging.info("📊 Guardian CSV Logger initialized")
    except Exception as e:
        logging.debug(f"Guardian CSV Logger not available: {e}")
    
    # 🧠 PPO Guardian Advisor (Hybrid Mode)
    guardian_ppo = None
    try:
        from src.rl.guardian_ppo_infer import get_ppo_advisor
        guardian_ppo = get_ppo_advisor(enabled=True)
        if guardian_ppo.enabled:
            logging.info(f"🧠 Guardian PPO Advisor loaded: {guardian_ppo.model_path}")
        else:
            logging.info("🧠 Guardian PPO Advisor disabled (no model)")
    except Exception as e:
        logging.debug(f"Guardian PPO Advisor not available: {e}")
    
    # =========================================
    # 🧬 Guardian Hybrid Arbitration Layer
    # =========================================
    guardian_hybrid = None
    if guardian_agent and guardian_ppo:
        from src.guardian.guardian_hybrid import GuardianHybrid
        guardian_hybrid = GuardianHybrid(guardian_agent, guardian_ppo, confidence_threshold=0.65)
        logging.info("🧬 Guardian Hybrid Layer initialized (Rule + PPO)")

    # =========================================
    # 👻 Shadow Mode (Phase 1)
    # =========================================
    shadow_recorder = None
    try:
        from src.analysis.shadow_recorder import ShadowRecorder
        shadow_recorder = ShadowRecorder()
        logging.info("👻 Shadow Mode Simulator Initialized (Tracking Blocks)")
    except Exception as e:
        logging.error(f"Shadow Recorder init failed: {e}")
    
    # =========================================
    # 🧠 Alpha PPO Shadow Mode (V1)
    # =========================================
    alpha_shadow = None
    try:
        from src.rl.alpha_shadow import get_alpha_shadow
        alpha_shadow = get_alpha_shadow(enabled=True)
        if alpha_shadow.ppo.enabled:
            logging.info("🧠 Alpha PPO Shadow Mode Initialized (Rule vs PPO comparison)")
        else:
            logging.info("🧠 Alpha PPO Shadow Mode Ready (waiting for model)")
    except Exception as e:
        logging.debug(f"Alpha Shadow init skipped: {e}")
    
    # =========================================
    # 🔌 PPO Live Switch (Confidence-Gated Execution)
    # =========================================
    ppo_live_switch = None
    try:
        from src.validation.ppo_live_switch import get_ppo_switch
        ppo_live_switch = get_ppo_switch()
        if alpha_mode == "full":
            ppo_live_switch.enable()
            logging.info("🔌 PPO Live Switch ENABLED (full mode)")
        else:
            logging.info(f"🔌 PPO Live Switch ready (mode={alpha_mode})")
    except Exception as e:
        logging.debug(f"PPO Live Switch init skipped: {e}")
    
    # =========================================
    # 🎛️ Guardian Auto-Tuner (Competition Mode)
    # =========================================
    from src.guardian.guardian_autotuner import GuardianAutoTuner
    auto_tuner = GuardianAutoTuner()
    logging.info("🎛️ Competition Auto-Tuner Initialized (Adaptive Governance)")

    def get_competition_metrics():
        """Fetch live metrics for Auto-Tuner."""
        metrics = {
            "win_rate": 0.5,
            "avg_r": 0.0,
            "current_dd": guardian_state.get("current_dd", 0.0),
            "block_rate": 0.0,
            "equity": 1000.0 # Default
        }
        
        if not mt5_lib.connected:
            return metrics
            
        try:
            # Equity check
            acc = mt5.get_account_info()
            if acc:
                metrics["equity"] = acc.equity
                
            # Fetch deals for today
            now = datetime.now()
            start = datetime(now.year, now.month, now.day)
            deals = mt5_lib.history_deals_get(start, now + timedelta(days=1))
            
            if deals and len(deals) > 0:
                profits = [d.profit for d in deals if d.entry == mt5_lib.DEAL_ENTRY_OUT]
                if profits:
                    wins = len([p for p in profits if p > 0])
                    metrics["win_rate"] = wins / len(profits)
                    
                    # Avg R approximation (Profit / Avg Loss)
                    losses = [abs(p) for p in profits if p < 0]
                    avg_loss = sum(losses) / len(losses) if losses else 1.0
                    avg_win = sum([p for p in profits if p > 0]) / len(profits) if wins > 0 else 0.0
                    if avg_loss > 0:
                        metrics["avg_r"] = avg_win / avg_loss
            
            # Block rate
            total_decisions = guardian_ppo.decisions if guardian_ppo else 1
            if total_decisions > 0:
                metrics["block_rate"] = guardian_state.get("total_blocks", 0) / total_decisions
                
        except Exception as e:
            logging.debug(f"Metric fetch error: {e}")
            
        return metrics
    
    # 🔄 Daily Retrain Job (23:00 scheduled)
    retrain_job = None
    try:
        from src.retrain.daily_retrain_job import get_daily_retrain_job
        retrain_job = get_daily_retrain_job()
        logging.info("🔄 Daily Retrain Job initialized (runs after 23:00)")
    except Exception as e:
        logging.debug(f"Daily Retrain Job not available: {e}")
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
    
    tp_model = None
    
    # 🔥 Single Source of Truth: Load Profile ONCE
    from src.config.trading_profiles import get_active_profile
    active_profile = get_active_profile()
    logging.info(f"🔥 Trading Profile: {active_profile.name}")
    
    if FUND_GRADE_ENABLED:
        # Risk Manager
        risk_manager = RiskManager(
            risk_per_trade=active_profile.risk.risk_per_trade / 100,  # Convert % to decimal
            max_positions_per_symbol=active_profile.risk.max_open_positions,
            max_daily_loss_pct=active_profile.risk.max_daily_loss / 100
        )
        logging.info("✅ RiskManager initialized")
        
        # Trailing Manager - USE PROFILE! (No hardcoded values)
        trail_manager = TrailingManager(profile=active_profile)
        logging.info("✅ TrailingManager initialized")
        
        # Validate config consistency
        assert trail_manager.be_rr == active_profile.trailing.be_trigger_r, \
            f"Config Drift! TrailingManager BE={trail_manager.be_rr} != Profile BE={active_profile.trailing.be_trigger_r}"
        
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
    
    # ===============================
    # 🛡️ Guardian Governance State
    # ===============================
    guardian_state = {
        "margin_block_count": 0,
        "dd_block_count": 0,
        "force_hold_until": 0.0,
        "last_block_reason": None,
        "escalation_count": 0,
        "total_blocks": 0,
    }
    
    try:
        guardian_latched_warned = False
        while stats["cycles"] < max_cycles:
            start = time.time()

            stats["cycles"] += 1
            cycle = stats["cycles"]
            
            if HAS_DASHBOARD:
                wf.start_cycle("XAUUSD")
            


            # ==============================
            # 🛡️ PROGRESSIVE GUARD (Absolute Authority)
            # ==============================
            guard_status = progressive_guard.get_status()
            if guard_status.get("kill_latched"):
                logging.critical("🔒 SYSTEM HALTED BY PROGRESSIVE GUARD (Kill Switch Active)")
                if HAS_DASHBOARD:
                    wf.update_step('init', 'ERROR', "SYSTEM HALTED: Progressive Guard Kill Switch")
                break  # STOP LOOP



            # ==============================
            # ==============================
            # 🔄 DAILY DD CHECK & RESET (Governance Layer)
            # ==============================
            if guardian_governance:
                try:
                    if mt5_lib.connected:
                        acc = mt5_lib.get_account_info()
                        if acc:
                            # ===============================
                            # 🔒 ACCOUNT UNUSABLE HARD GATE
                            # ===============================
                            equity = acc.equity
                            free_margin = acc.margin_free
                            
                            if equity < 50 or free_margin <= 0:
                                logging.critical(f"🚨 ACCOUNT UNUSABLE | equity={equity:.2f}, free_margin={free_margin:.2f}")
                                if guardian_csv_logger:
                                    guardian_csv_logger.log({
                                        "action": "HARD_BLOCK",
                                        "block_reason": "ACCOUNT_DEAD",
                                        "equity": equity,
                                        "daily_dd": 0.0,
                                        "margin_ratio": 0.0,
                                        "cycle": cycle
                                    })
                                stats["HOLD"] += 1
                                if HAS_DASHBOARD:
                                    wf.update_step('init', 'ERROR', "ACCOUNT DEAD: Equity < $50 or No Margin")
                                time.sleep(interval)
                                continue

                            # Use bridge to update DD and check for hard latch
                            # This handles start-of-day snapshot automatically
                            blocked_by_gov, gov_reason = guardian_governance.update_dd_and_check(
                                acc.equity, acc.balance, logging
                            )
                            
                            # Update local state for logging
                            guardian_state["current_dd"] = guardian_governance.current_dd
                            
                            if blocked_by_gov:
                                # Governance Hard Latch Triggered
                                logging.critical(f"🛑 GOVERNANCE BLOCK: {gov_reason}")
                                stats["HOLD"] += 1
                                if HAS_DASHBOARD:
                                    wf.update_step('init', 'ERROR', f"BLOCKED: {gov_reason}")
                                
                                # Governance block happens BEFORE signal generation
                                # Cannot record shadow trade here - no signal yet
                                
                                time.sleep(interval)
                                continue # SKIP EVERYTHING ELSE
                except Exception as e:
                    logging.error(f"Governance Check Error: {e}")

            # 1. Fetch data
            if HAS_DASHBOARD: wf.update_step('data', 'RUNNING', 'Fetching OHLCV...')
            bars_h1 = mt5_lib.get_bars('XAUUSD', 'H1', 500)

            bars_h4 = mt5_lib.get_bars('XAUUSD', 'H4', 125)
            bars_h4 = mt5_lib.get_bars('XAUUSD', 'H4', 125)
            
            # 👻 SHADOW UPDATE
            if shadow_recorder:
                tick = mt5_lib.symbol_info_tick('XAUUSD')
                if tick:
                    shadow_recorder.update(tick.bid, tick.ask)

            if HAS_DASHBOARD: wf.update_step('data', 'COMPLETED')
            
            # 2. Generate dataset
            if HAS_DASHBOARD: wf.update_step('features', 'RUNNING', 'Generating features...')
            df = engine.generate_dataset(bars_h1, bars_h4)
            if HAS_DASHBOARD: wf.update_step('features', 'COMPLETED')
            
            # 2.5 Save Data (Fix Stale Data Issue)
            try:
                # Save just the last row (latest complete candle)
                latest_row = df.iloc[[-1]] 
                
                # Append to dataset (Main Archive)
                dataset_path = "data/imitation_full_dataset.csv"
                header = not os.path.exists(dataset_path)
                latest_row.to_csv(dataset_path, mode='a', header=header, index=False)
                
                # Append to Retrain Inbox (For DailyRetrainJob)
                retrain_inbox = "data/retrain/stream.csv"
                os.makedirs("data/retrain", exist_ok=True)
                header_retrain = not os.path.exists(retrain_inbox)
                latest_row.to_csv(retrain_inbox, mode='a', header=header_retrain, index=False)
                
                # logging.debug(f"Saved data point: {latest_row.index[-1]}")
            except Exception as e:
                logging.error(f"Data save failed: {e}")
            
            # 3. Auto-train
            if auto_train_enabled and cycle % train_interval == 0:
                if HAS_DASHBOARD: wf.update_step('ai', 'RUNNING', 'Auto-training model...')
                if auto_train():
                    engine.load_ai_model(str(model_path))
                if HAS_DASHBOARD: wf.update_step('ai', 'COMPLETED')
            
            # 4. Get signal
            if HAS_DASHBOARD: wf.update_step('fusion', 'RUNNING', 'Analyzing signals...')
            signal, info = engine.get_signal(df)
            stats[signal] = stats.get(signal, 0) + 1
            if HAS_DASHBOARD: wf.update_step('fusion', 'COMPLETED', f"Signal: {signal}")
            
            # ==============================
            # 🧠 ALPHA PPO SHADOW COMPARISON
            # ==============================
            ppo_action = "HOLD"
            ppo_conf = 0.0
            agreed = True
            
            if alpha_shadow:
                try:
                    tick = mt5_lib.symbol_info_tick('XAUUSD') if mt5_lib.connected else None
                    current_price = tick.bid if tick else df.iloc[-1].get('close', 0)
                    
                    # Get market data for state injection
                    last_row = df.iloc[-1]
                    ema20 = last_row.get('ema20', last_row.get('EMA20', 0))
                    ema50 = last_row.get('ema50', last_row.get('EMA50', 0))
                    atr = last_row.get('atr14', last_row.get('atr', 2.5))
                    rsi_val = last_row.get('rsi14', last_row.get('rsi', 50))
                    spread = last_row.get('spread', 0.1)
                    
                    ppo_action, ppo_conf, agreed = alpha_shadow.compare(
                        rule_action=signal,
                        cycle=cycle,
                        market_row=last_row,
                        open_positions=len(mt5.positions_get() or []) if mt5_lib.connected else 0,
                        floating_dd=guardian_state.get("current_dd", 0.0),
                        guardian_state=0 if not guardian or guardian.allow_trade() else 2,
                        current_price=current_price
                    )
                    
                    # ============================================================
                    # 📊 Shadow Reward Evaluation (DD avoided / Missed Profit)
                    # ============================================================
                    if alpha_shadow.ppo.enabled and hasattr(alpha_shadow, 'ppo'):
                        try:
                            from src.rl.alpha_env import AlphaTradingEnv, AlphaEnvConfig
                            
                            # Create or reuse shadow env for step simulation
                            if not hasattr(alpha_shadow, '_shadow_env'):
                                alpha_shadow._shadow_env = AlphaTradingEnv(
                                    config=AlphaEnvConfig(max_steps=99999)
                                )
                                alpha_shadow._shadow_env.reset()
                            
                            shadow_env = alpha_shadow._shadow_env
                            
                            # Inject current market state
                            shadow_env.update_market_state(
                                ema_fast_diff=np.clip((ema20 - ema50) / max(atr, 1), -1, 1),
                                rsi=(rsi_val - 50) / 50,
                                atr=np.clip(atr / current_price * 100, -1, 1),
                                spread=np.clip(spread / max(atr, 0.1), -1, 1),
                                time_of_day=datetime.now().hour / 24.0,
                                open_positions=len(mt5.positions_get() or []) if mt5_lib.connected else 0,
                                floating_dd=guardian_state.get("current_dd", 0.0),
                                guardian_state=0 if not guardian or guardian.allow_trade() else 2
                            )
                            
                            # Get action mapping
                            action_map = {"HOLD": 0, "BUY": 1, "SELL": 2}
                            ppo_action_idx = action_map.get(ppo_action, 0)
                            
                            # Simulate step (NO REAL EXECUTION)
                            _, shadow_reward, _, _, shadow_info = shadow_env.step(ppo_action_idx)
                            
                            # Log shadow metrics to Guardian CSV
                            if guardian_csv_logger:
                                guardian_csv_logger.log({
                                    "action": "SHADOW_EVAL",
                                    "rule_signal": signal,
                                    "ppo_action": ppo_action,
                                    "ppo_conf": f"{ppo_conf:.2f}",
                                    "agreed": agreed,
                                    "shadow_reward": f"{shadow_reward:.3f}",
                                    "floating_dd": guardian_state.get("current_dd", 0.0),
                                    "cycle": cycle
                                })
                            
                            # Log insight every 25 cycles
                            if cycle % 25 == 0:
                                outcome_icon = "✅" if agreed else "❌"
                                logging.info(
                                    f"📊 Shadow | Rule={signal:4s} PPO={ppo_action:4s} ({ppo_conf:.0%}) "
                                    f"{outcome_icon} | Reward={shadow_reward:+.2f}"
                                )
                                
                        except Exception as e:
                            logging.debug(f"Shadow env step failed: {e}")
                    
                    # Log summary every 50 cycles
                    if cycle % 50 == 0:
                        logging.info(alpha_shadow.summary())
                        
                except Exception as e:
                    logging.debug(f"Alpha Shadow comparison failed: {e}")

            # ==============================
            # 🧠 ALPHA HYBRID DECISION (Controlled Live)
            # ==============================
            original_signal = signal  # Keep original for logging
            alpha_source = "RULE"
            
            # ==============================
            # 🔌 PPO LIVE SWITCH DECISION
            # ==============================
            if ppo_live_switch and alpha_mode == "full":
                try:
                    # Get account state for rollback checks
                    acc_state = {
                        "intraday_dd": guardian_state.get("current_dd", 0),
                        "margin_level": (acc.margin_free / max(acc.balance, 0.01)) if acc else 1.5
                    }
                    
                    # Get PPO decision with automatic fallback
                    obs = np.array([0.0] * 10)  # Simplified obs
                    ppo_signal, alpha_source, switch_info = ppo_live_switch.get_decision(
                        obs=obs,
                        rule_signal=original_signal,
                        guardian_state=guardian_state,
                        account_state=acc_state
                    )
                    
                    if alpha_source == "PPO":
                        signal = ppo_signal
                        logging.info(
                            f"🔌 PPO Live: {signal} (conf={switch_info.get('confidence', 0):.0%}, was Rule={original_signal})"
                        )
                    else:
                        logging.debug(f"PPO fallback: {switch_info.get('fallback_reason', 'unknown')}")
                        
                except Exception as e:
                    logging.debug(f"PPO Live Switch error: {e}")
            
            # Legacy Hybrid Controller (for hybrid mode)
            elif alpha_hybrid_ctrl and alpha_mode == "hybrid":
                try:
                    from src.rl.alpha_decision import create_alpha_decision
                    
                    # Build PPO decision object
                    action_map = {"HOLD": 0, "BUY": 1, "SELL": 2}
                    ppo_action_int = action_map.get(ppo_action, 0)
                    
                    ppo_decision = create_alpha_decision(
                        action=ppo_action_int,
                        confidence=ppo_conf,
                        risk_score=guardian_state.get("current_dd", 0) * 2,  # Scale DD to risk
                        regime="TREND" if signal != "HOLD" else "RANGE",
                        reason=f"PPO conf={ppo_conf:.0%}",
                        ema_signal=original_signal
                    )
                    
                    # Get hybrid decision
                    final_action, alpha_source, hybrid_info = alpha_hybrid_ctrl.decide(
                        ppo_decision=ppo_decision,
                        rule_signal=original_signal,
                        guardian_state=guardian_state,
                        market_state={
                            "margin_ratio": acc.margin_free / max(acc.balance, 1) if acc else 1.0,
                            "regime": "TREND" if signal != "HOLD" else "RANGE"
                        } if mt5_lib.connected else {}
                    )
                    
                    # Update signal if PPO is used
                    if alpha_source == "PPO":
                        signal = final_action
                        logging.info(
                            f"🧠 Alpha Hybrid: Using PPO decision {signal} "
                            f"(conf={ppo_conf:.0%}, was Rule={original_signal})"
                        )
                    elif alpha_source == "GUARDIAN_OVERRIDE":
                        signal = "HOLD"
                        logging.info(
                            f"🛡️ Alpha blocked by Guardian: {hybrid_info.get('blocked_reason', 'unknown')}"
                        )
                        
                except Exception as e:
                    logging.debug(f"Alpha Hybrid decision failed: {e}")

            # ==============================
            # 🛡️ FORCE_HOLD CHECK (Escalation)
            # ==============================
            if time.time() < guardian_state.get("force_hold_until", 0):
                remaining = int(guardian_state["force_hold_until"] - time.time())
                logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] #{cycle}: 🧊 FORCE_HOLD active ({remaining}s remaining)")
                stats["HOLD"] += 1
                time.sleep(interval)
                continue
            
            # ==============================
            # 🧬 GUARDIAN HYBRID ARBITRATION (Rule + PPO)
            # ==============================
            risk_multiplier = 1.0
            if guardian_hybrid:
                # 1. Construct State
                acc = mt5_lib.get_account_info() if mt5_lib.connected else None
                current_dd = guardian_state.get("current_dd", 0.0)
                margin_ratio = (acc.margin_free / acc.balance) if acc and acc.balance > 0 else 1.0
                
                # V4 Context
                from src.guardian.guardian_context import GuardianContext
                context = GuardianContext.get_context(df)
                
                nav_state = {
                    "daily_dd": current_dd,
                    "margin_ratio": margin_ratio,
                    "chaos": 0, 
                    "step": cycle,
                    "margin_block_count": guardian_state.get("margin_block_count", 0),
                    # V4 Features
                    "market_regime": context["market_regime"],
                    "session_time": context["session_time"],
                    "recent_win_rate": 0.5 # Placeholder (connected to Auto-Tuner stats later)
                }
                
                # 2. Decide
                hybrid_action, hybrid_reason = guardian_hybrid.decide(nav_state, signal)
                
                # 3. Apply Decision
                if hybrid_action == "BLOCK" or hybrid_action == "FORCE_HOLD" or hybrid_action == "EMERGENCY_FREEZE":
                    logging.info(f"🧬 GUARDIAN BLOCK: {hybrid_reason}")
                    
                    # 👻 SHADOW RECORDING (Hybrid Block)
                    # 👻 SHADOW RECORDING (Hybrid Block)
                    if shadow_recorder and signal in ["BUY", "SELL"]:
                         try:
                             # 1. Get Price
                             tick_s = mt5_lib.symbol_info_tick("XAUUSD")
                             if tick_s:
                                 price = tick_s.ask if signal == "BUY" else tick_s.bid
                                 
                                 # 2. Calc Shadow SL/TP (Approximate based on ATR)
                                 # Fallback to fixed 5.0 (500 pts) SL if ATR missing
                                 atr = df.iloc[-1].get('atr', 2.5) 
                                 sl_dist = atr * 2.0
                                 tp_dist = atr * 4.0
                                 
                                 if signal == "BUY":
                                     sl = price - sl_dist
                                     tp = price + tp_dist
                                 else:
                                     sl = price + sl_dist
                                     tp = price - tp_dist
                                     
                                 # 3. Record
                                 shadow_recorder.place_trade(
                                     signal=signal,
                                     price=price,
                                     sl=sl,
                                     tp=tp,
                                     volume=0.01,
                                     reason=hybrid_reason
                                 )
                                 logging.info(f"👻 Shadow Trade Recorded: {signal} @ {price:.2f}")
                         except Exception as e:
                             logging.error(f"Shadow Record Fail: {e}")

                    # BLOCK EXECUTION
                    stats["HOLD"] += 1
                    time.sleep(interval)
                    continue 
                         
                    if HAS_DASHBOARD:
                        wf.update_step('init', 'ERROR', f"Guardian Block: {hybrid_reason}")
                    stats["HOLD"] += 1
                    
                    # Defer "continue" to allow Shadow Recording if we want accurate SL/TP?
                    # No, let's record with a placeholder or simple logic for now to avoid refactoring flow.
                    if shadow_recorder and signal in ["BUY", "SELL"]:
                         try:
                             tick = mt5.symbol_info_tick('XAUUSD')
                             price = tick.ask if signal == "BUY" else tick.bid
                             # Provisional SL/TP (e.g. 500 pips)
                             sl_dist = 5.0 # $5
                             tp_dist = 10.0 # $10
                             sl = price - sl_dist if signal == "BUY" else price + sl_dist
                             tp = price + tp_dist if signal == "BUY" else price - tp_dist
                             shadow_recorder.place_trade(signal, price, sl, tp, 0.01)
                         except: pass

                    time.sleep(interval)
                    continue
                
                elif hybrid_action == "REDUCE_RISK":
                    risk_multiplier = 0.5
                    logging.info(f"🧬 GUARDIAN ADVICE: REDUCE RISK ({hybrid_reason})")
            
            # 5. Log
            ts = datetime.now().strftime("%H:%M:%S")
            if signal == "HOLD":
                logging.info(f"[{ts}] #{cycle}: ⏸️  HOLD | {info}")
                if HAS_DASHBOARD: wf.add_log("INFO", f"HOLD: {info}")
            else:
                logging.info(f"[{ts}] #{cycle}: 🎯 {signal} | {info}")
                if HAS_DASHBOARD: wf.add_log("INFO", f"SIGNAL {signal}: {info}")
                
                # ==============================
                # 🧠 GUARDIAN AGENT OVERSIGHT
                # ==============================
                if HAS_DASHBOARD: wf.update_step('gate', 'RUNNING', 'Guardian review...')
                
                # Quick account snapshot
                acc_state = {}
                if mt5_lib.connected:
                    a = mt5_lib.get_account_info()
                    if a: acc_state = {"equity": a.equity, "margin_free": a.margin_free}
                
                guardian_status = guardian.status() if guardian else {}

                guardian_decision = guardian_agent.evaluate(
                    signal=signal,
                    account_state=acc_state,
                    guardian_state=guardian_status
                )
                
                if guardian_decision == "BLOCK":
                    # Track block in governance state
                    guardian_state["total_blocks"] += 1
                    guardian_state["margin_block_count"] += 1  # Count as margin-related
                    guardian_state["last_block_reason"] = "GUARDIAN_AGENT"
                    
                    logging.warning(f"🛡️ GuardianAgent blocked signal (block #{guardian_state['margin_block_count']})")
                    
                    # ESCALATION: After 3 blocks, trigger FORCE_HOLD for 5 minutes
                    if guardian_state["margin_block_count"] >= 3 and guardian_state["force_hold_until"] < time.time():
                        guardian_state["escalation_count"] += 1
                        guardian_state["force_hold_until"] = time.time() + 300  # 5 minutes
                        logging.critical(f"🧊 FORCE_HOLD ESCALATION: 3+ blocks detected (5 min cooldown)")
                    
                    if HAS_DASHBOARD: 
                        wf.update_step('gate', 'WARNING', 'Blocked by GuardianAgent')
                        wf.add_log('WARNING', '🛡️ GuardianAgent blocked trade')
                    time.sleep(interval)
                    continue
                
                # ==============================
                # 🧠 PPO GUARDIAN ADVISOR (SOFT)
                # ==============================
                if guardian_ppo and guardian_ppo.enabled:
                    ppo_state = {
                        "daily_dd": guardian_state.get("current_dd", 0),
                        "margin_ratio": acc_state.get("margin_free", 0) / max(acc_state.get("equity", 1), 1) if acc_state else 1.0,
                        "free_margin_ratio": acc_state.get("margin_free", 0) / max(acc_state.get("equity", 1), 1) if acc_state else 1.0,
                        "chaos": 1 if guardian_state.get("last_block_reason") else 0,
                    }
                    
                    ppo_action, ppo_conf = guardian_ppo.decide(ppo_state)
                    
                    if ppo_action != "ALLOW" and ppo_conf >= guardian_ppo.confidence_threshold:
                        guardian_state["total_blocks"] += 1
                        guardian_state["last_block_reason"] = f"PPO_{ppo_action}"
                        
                        logging.info(f"🧠 PPO Advisor: {ppo_action} (conf={ppo_conf:.2f})")
                        
                        if HAS_DASHBOARD:
                            wf.add_log('INFO', f'🧠 PPO: {ppo_action} (conf={ppo_conf:.2f})')
                        
                        time.sleep(interval)
                        continue
                
                # =========================================
                # Fund-Grade: AI SL/TP Prediction (NEW)
                # =========================================
                if HAS_DASHBOARD: wf.update_step('risk', 'RUNNING', 'Calculating Risk & SL/TP...')
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
                    # Fallback Logic using Profile
                    atr = last.get('atr14', 5.0)
                    sl_mult = active_profile.sltp.atr_multiplier_sl
                    sl_dist = atr * sl_mult
                    tp_dist = sl_dist * active_profile.sltp.default_rr
                    
                    price = last['close']
                    if signal == "BUY":
                        sl = round(price - sl_dist, 5)
                        tp = round(price + tp_dist, 5)
                    else:
                        sl = round(price + sl_dist, 5)
                        tp = round(price - tp_dist, 5)
                    sl_pips = sl_dist
                
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
                    logging.error("RiskManager is REQUIRED for live_loop_v3.py")
                    continue
                
                # Apply Guardian Risk Multiplier
                if risk_multiplier < 1.0:
                    volume = round(volume * risk_multiplier, 2)
                    logging.info(f"   📉 Guardian reduced volume to {volume} (x{risk_multiplier})")
                
                if HAS_DASHBOARD: wf.update_step('risk', 'COMPLETED', f"Lot: {volume}")

                # =========================================
                # 🛡️ GUARDIAN MARGIN PRE-CHECK (NEW)
                # =========================================
                if guardian:
                    from src.safety.guardian_margin_gate import GuardianDecision
                    margin_check = guardian.evaluate(volume, signal)
                    
                    if margin_check.decision == GuardianDecision.BLOCK:
                        guardian_state["margin_block_count"] += 1
                        guardian_state["last_block_reason"] = margin_check.reason
                        
                        logging.warning(f"   🛑 MARGIN BLOCK: {margin_check.reason} (count: {guardian_state['margin_block_count']})")
                        
                        # ESCALATION: After 3 blocks, trigger FORCE_HOLD for 5 minutes
                        if guardian_state["margin_block_count"] >= 3:
                            guardian_state["force_hold_until"] = time.time() + 300  # 5 minutes
                            logging.critical(f"   🧊 FORCE_HOLD ESCALATION: Margin starvation detected (5 min cooldown)")
                            if HAS_DASHBOARD:
                                wf.update_step('execution', 'ERROR', "FORCE_HOLD: Margin starvation")
                                wf.add_log('CRITICAL', '🧊 FORCE_HOLD: Margin starvation (5 min)')
                        else:
                            if HAS_DASHBOARD: 
                                wf.update_step('execution', 'ERROR', f"Margin Block: {margin_check.reason}")
                                wf.add_log('WARNING', f"Trade blocked: {margin_check.reason}")
                        
                        time.sleep(interval)  # Wait before next cycle
                        continue
                    
                    elif margin_check.decision == GuardianDecision.CLAMP:
                        # Reset block count on successful clamp
                        guardian_state["margin_block_count"] = 0
                        logging.warning(f"   ⚠️ MARGIN CLAMP: {volume} → {margin_check.allowed_lot}")
                        volume = margin_check.allowed_lot
                        if HAS_DASHBOARD: wf.add_log('WARNING', f"Lot clamped: {volume}")
                    
                    else:
                        # ALLOW: Reset block counts on successful margin check
                        guardian_state["margin_block_count"] = 0
                        guardian_state["dd_block_count"] = 0
                        guardian_state["last_block_reason"] = None

                # 6. Send trade
                if HAS_DASHBOARD: wf.update_step('execution', 'RUNNING', f"Sending {signal}...")
                if not sandbox:
                    res = mt5_lib.send_order('XAUUSD', 'OPEN', signal, volume, sl=sl, tp=tp)
                    engine.update_position(signal)
                    stats["trades"] += 1
                    logging.info(f"   Trade: {res}")
                    logging.info(f"   Lot: {volume} | SL: {sl} | TP: {tp} | RR: 1:2")
                    
                    # Reset governance counters on successful trade
                    guardian_state["margin_block_count"] = 0
                    guardian_state["dd_block_count"] = 0
                    guardian_state["last_block_reason"] = None
                    
                    # Notify components
                    guardian_agent.observe_trade(res)
                else:
                    engine.update_position(signal)
                    stats["trades"] += 1
                    logging.info(f"   📝 Sandbox | Lot: {volume} | SL: {sl} | TP: {tp} | RR: 1:2")
                    if HAS_DASHBOARD: wf.update_step('execution', 'COMPLETED', 'Sandbox Trade Sent')

            
            # =========================================
            # Fund-Grade: Trailing Stop Management (NEW)
            # =========================================
            if HAS_DASHBOARD: wf.update_step('trailing', 'RUNNING', 'Scanning positions...')
            if trail_manager is not None and not sandbox and mt5_lib.connected:
                try:

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
            if HAS_DASHBOARD: wf.update_step('analytics', 'COMPLETED', 'Cycle Finished')
            
            # 💓 HEARTBEAT (For Watchdog)
            try:
                with open("heartbeat.txt", "w") as f:
                    f.write(str(time.time()))
            except Exception as e:
                logging.error(f"Heartbeat write failed: {e}")
            
            # 📊 Guardian Summary & Auto-Tuning (every 10 cycles)
            if cycle % 10 == 0:
                freeze_remaining = int(guardian_state["force_hold_until"] - time.time()) if guardian_state["force_hold_until"] > time.time() else 0
                logging.info(
                    f"📊 Guardian Summary | "
                    f"blocks={guardian_state['total_blocks']} | "
                    f"margin={guardian_state['margin_block_count']} | "
                    f"dd={guardian_state['dd_block_count']} | "
                    f"escalations={guardian_state['escalation_count']} | "
                    f"freeze={freeze_remaining}s"
                )
                
                # 📊 CSV Export
                if guardian_csv_logger:
                    try:
                        acc = mt5_lib.get_account_info() if mt5_lib.connected else None
                        guardian_csv_logger.log({
                            "cycle": cycle,
                            "margin_ratio": acc.margin_free / acc.equity if acc and acc.equity > 0 else 0,
                            "daily_dd": guardian_state.get("current_dd", 0),
                            "equity": acc.equity if acc else 0,
                            "action": guardian_state.get("last_block_reason", "ALLOW"),
                            "block_reason": guardian_state.get("last_block_reason", ""),
                            "escalation": freeze_remaining > 0,
                            "block_count": guardian_state["margin_block_count"],
                            # Competition Metrics
                            "source": "Guardian",
                            "event": "ACCOUNT_DEAD" if (acc and (acc.equity < 50 or acc.margin_free <= 0)) else ("BLOCK" if guardian_state.get("last_block_reason") else "ACTIVE"),
                            "potential_dd": 0.0, # Placeholder until simulation integrated
                            "missed_profit": 0.0, # Placeholder
                            # PPO Attention (Mock for dashboard demo)
                            "att_equity": 0.3,
                            "att_margin": 0.4 if (acc and acc.margin_free < 0) else 0.1,
                            "att_dd": 0.4 if guardian_state.get("current_dd", 0) > 0.05 else 0.1,
                            "att_error": 0.1,
                            "att_latency": 0.1
                        })
                    except Exception as e:
                        logging.debug(f"CSV log error: {e}")

                # 🎛️ Run Auto-Tuner (Competition Logic)
                try:
                    comp_metrics = get_competition_metrics()
                    # Pass active_profile to enforce "Competition Envelope"
                    new_config = auto_tuner.tune(comp_metrics, cycle, profile=_profile)
                    
                    # Apply new config dynamically
                    if guardian_governance:
                        guardian_governance.daily_dd_limit = new_config["daily_dd_limit"]
                    
                    if guardian: # Margin Gate
                        guardian.daily_loss_limit = new_config["daily_dd_limit"]
                    
                    if progressive_guard:
                        from src.safety.progressive_guard import AlertThresholds
                        progressive_guard.dd_thresholds = AlertThresholds(
                            level_1=new_config["progressive_l1"] * 100, # Convert to %
                            level_2=new_config["progressive_l2"] * 100,
                            level_3=new_config["progressive_l4"] * 100  # Kill switch
                        )
                    
                    if guardian_hybrid:
                        guardian_hybrid.theta = new_config["ppo_confidence"]
                        
                    # logging.info(f"🎛️ Auto-Tuned: DD_Limit={new_config['daily_dd_limit']:.0%}, PPO_Conf={new_config['ppo_confidence']:.2f}")
                except Exception as e:
                    logging.error(f"Auto-Tuner Error: {e}")
                
                # 🔄 Daily Retrain Check (after 23:00)
                if retrain_job and retrain_job.should_run():
                    logging.info("🔄 Daily retrain triggered (23:00+)...")
                    try:
                        result = retrain_job.run()
                        if result.success:
                            logging.info(f"✅ Retrain complete: improvement={result.score_improvement:+.1%}, deployed={result.deployed}")
                        else:
                            logging.warning(f"⚠️ Retrain skipped: {result.errors}")
                    except Exception as e:
                        logging.error(f"Retrain error: {e}")
            
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
    parser.add_argument("--alpha-mode", choices=["shadow", "hybrid", "full"], default="shadow",
                       help="Alpha PPO mode: shadow (no exec), hybrid (confident exec), full (all exec)")
    
    args = parser.parse_args()
    
    live_loop(
        interval=args.interval,
        max_cycles=args.cycles,
        sandbox=not args.live,
        auto_train_enabled=not args.no_train,
        alpha_mode=getattr(args, 'alpha_mode', 'shadow')
    )
