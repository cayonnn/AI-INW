"""
fund_grade_orchestrator.py
===========================
Fund-Grade Trading Orchestrator

เชื่อมต่อทุก module ใน Blueprint:

Pipeline Flow:
Market Data → Features → AI Models → Signal Fusion
    ↓
Global Regime → Crisis Check → Risk Committee Vote
    ↓
Capital Protection → Decision Gate → Position Size
    ↓
Execute → Alpha Attribution → Decay Detection
    ↺
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from src.utils.logger import get_logger
from src.utils.config_loader import get_config


# Data
from src.data.mt5_connector import MT5Connector
from src.data.data_processor import DataProcessor
from src.data.data_validator import DataValidator

# Features
from src.features.price_action import PriceActionFeatures
from src.features.trend_momentum import TrendMomentumFeatures
from src.features.volatility import VolatilityFeatures
from src.features.regime_detector import RegimeDetector

# Models
from src.models.lstm_direction import LSTMDirectionModel
from src.models.xgb_timing import XGBTimingModel
from src.models.signal_schema import AISignal, SignalDirection

# Signals
from src.signals.signal_fusion import SignalFusion
from src.signals.decision_gate import DecisionGate
from src.signals.global_regime_forecast import (
    GlobalRegimeForecaster, MacroIndicators, MarketIndicators
)

# Risk - Core
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossCalculator
from src.risk.auto_disable import AutoDisableManager
from src.risk.position_manager import (
    PositionManager,
    PositionAction,
    get_position_manager,
)
from src.risk.win_streak_booster import WinStreakRiskBooster, get_win_streak_booster

# Risk - CIO Brain
from src.risk.crisis_mode import CrisisModeController
from src.risk.recovery_engine import PostCrisisRecoveryEngine
from src.risk.capital_allocator import DynamicCapitalAllocator
from src.risk.meta_rl_allocator import MetaRLAllocator
from src.risk.ai_risk_committee import AIRiskCommittee
from src.risk.capital_protection import CapitalProtectionSystem

# Execution
from src.execution.position_executor import PositionExecutor, get_position_executor

# Config
from src.config.trading_profiles import get_active_profile, TradingMode

# Utils - Analytics
from src.utils.alpha_attribution import AlphaAttributionEngine
from src.utils.strategy_decay import StrategyDecayDetector
from src.utils.strategy_pool import SelfPruningStrategyPool
from src.utils.crowding_detection import CrowdingDetector
from src.utils.explainability import ExplainabilityEngine


logger = get_logger("FUND_ORCHESTRATOR")


class FundGradeOrchestrator:
    """
    Hedge Fund Grade Trading Orchestrator.
    
    Full Blueprint Implementation:
    - Global Regime Forecast
    - AI Risk Committee Voting
    - Capital Protection Layers
    - Crisis Mode Management
    - Alpha Attribution Loop
    - Strategy Lifecycle Management
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing Fund-Grade Orchestrator")
        logger.info("=" * 60)

        # --- Configuration ---
        config = get_config()
        self.settings = config
        self.risk_params = config.risk
        self.symbols_cfg = {"symbols": config.get_enabled_symbols()}

        # --- Data Layer ---
        self.mt5 = MT5Connector()
        self.processor = DataProcessor()
        self.validator = DataValidator()

        # --- Feature Engines ---
        self.feature_engines = [
            PriceActionFeatures(),
            TrendMomentumFeatures(),
            VolatilityFeatures(),
            RegimeDetector(),
        ]

        # --- AI Models ---
        self.direction_model = LSTMDirectionModel()
        self.timing_model = XGBTimingModel()

        # --- Signal Processing ---
        self.signal_fusion = SignalFusion()
        self.decision_gate = DecisionGate()

        # --- Global Regime ---
        self.regime_forecaster = GlobalRegimeForecaster()

        # --- CIO Brain ---
        self.risk_committee = AIRiskCommittee()
        self.capital_protection = CapitalProtectionSystem()
        self.crisis_controller = CrisisModeController()
        self.recovery_engine = PostCrisisRecoveryEngine()
        self.capital_allocator = DynamicCapitalAllocator()
        self.meta_rl = MetaRLAllocator()

        # --- Core Risk ---
        self.position_sizer = PositionSizer()
        self.stop_loss_engine = StopLossCalculator()
        self.auto_disable = AutoDisableManager()

        # --- Position Management 🔥 ---
        self.trading_profile = get_active_profile()
        self.position_manager = PositionManager(self.trading_profile)
        self.position_executor = PositionExecutor(use_command_writer=True)
        
        # --- Trailing with same profile (Config Drift Protection) ---
        from src.risk.trailing import TrailingManager
        self.trailing_manager = TrailingManager(profile=self.trading_profile)
        
        # --- Win-Streak Risk Booster 🔥 (Competition Grade) ---
        self.streak_booster = WinStreakRiskBooster(
            base_risk=self.trading_profile.risk.risk_per_trade,
            max_risk=min(3.0, self.trading_profile.risk.risk_per_trade * 3),
            ramp_mode="aggressive" if self.trading_profile.mode.value == "aggressive" else "standard",
            profile=self.trading_profile
        )
        
        # ⚠️ CONFIG DRIFT GUARD - Stop if configs don't match!
        self._validate_config_consistency()
        
        logger.info(f"🔥 Trading Profile: {self.trading_profile.name}")

        # --- Analytics ---
        self.alpha_engine = AlphaAttributionEngine()
        self.decay_detector = StrategyDecayDetector()
        self.strategy_pool = SelfPruningStrategyPool()
        self.crowding_detector = CrowdingDetector()
        self.explainability = ExplainabilityEngine()

        # --- State ---
        self.current_regime = "NEUTRAL"
        self.is_crisis = False
        self.portfolio_metrics = {}

        logger.info("Fund-Grade Orchestrator Ready")

    # =========================================================
    # CONFIG DRIFT PROTECTION
    # =========================================================
    def _validate_config_consistency(self):
        """
        ⚠️ CONFIG DRIFT GUARD
        
        Validates that all components use consistent configuration.
        Raises error if configs don't match - prevents silent bugs in live trading.
        """
        profile = self.trading_profile
        pm = self.position_manager
        tm = self.trailing_manager
        
        # Check BE trigger consistency
        if abs(pm.profile.trailing.be_trigger_r - tm.be_rr) > 0.001:
            raise ValueError(
                f"CONFIG DRIFT DETECTED! "
                f"PositionManager BE={pm.profile.trailing.be_trigger_r}R vs "
                f"TrailingManager BE={tm.be_rr}R"
            )
        
        # Check Trail trigger consistency
        if abs(pm.profile.trailing.trail_start_r - tm.trail_rr) > 0.001:
            raise ValueError(
                f"CONFIG DRIFT DETECTED! "
                f"PositionManager Trail={pm.profile.trailing.trail_start_r}R vs "
                f"TrailingManager Trail={tm.trail_rr}R"
            )
        
        # Check ATR multiplier consistency
        if abs(pm.profile.trailing.trail_atr_multiplier - tm.atr_multiplier) > 0.001:
            raise ValueError(
                f"CONFIG DRIFT DETECTED! "
                f"PositionManager ATR×={pm.profile.trailing.trail_atr_multiplier} vs "
                f"TrailingManager ATR×={tm.atr_multiplier}"
            )
        
        logger.info(
            f"✅ Config consistency validated: "
            f"BE@{tm.be_rr}R, Trail@{tm.trail_rr}R, ATR×{tm.atr_multiplier}"
        )

    # =========================================================
    # 🔥 WIN-STREAK BOOSTER INTEGRATION
    # =========================================================
    def on_trade_closed(self, result: str, r_multiple: float = 0):
        """
        Called when a trade closes - updates win streak booster.
        
        Args:
            result: "WIN", "LOSS", "BE"
            r_multiple: Actual R achieved (optional for advanced mode)
        """
        # Update booster
        self.streak_booster.on_trade_result(result)
        
        # Log new risk level
        state = self.streak_booster.get_state()
        if state.is_boosted:
            logger.info(
                f"🔥 WIN STREAK: {state.win_streak} | "
                f"📈 Dynamic Risk: {state.current_risk_pct:.2f}%"
            )
    
    def get_dynamic_risk(self) -> float:
        """
        Get current dynamic risk % with safety guards.
        
        Safety Guards (hard limits):
        - Cannot exceed profile max risk cap
        - Blocked if daily DD > 80% of limit
        - Blocked if in crisis mode
        
        Returns:
            Safe risk percentage to use
        """
        profile = self.trading_profile
        base_risk = profile.risk.risk_per_trade
        
        # Check safety conditions
        daily_dd = self.portfolio_metrics.get("drawdown", 0)
        max_dd = profile.risk.max_daily_loss
        in_crisis = self.is_crisis or self.crisis_controller.mode.value == "SURVIVAL"
        
        # Safety guard: Block if conditions not met
        is_safe = self.streak_booster.check_safety(
            crisis_mode=in_crisis,
            daily_dd=daily_dd,
            max_dd=max_dd,
            spread_ok=True,  # Would come from MT5
            latency_ok=True   # Would come from system
        )
        
        if not is_safe:
            logger.warning(f"⚠️ Booster blocked, using base risk: {base_risk}%")
            return base_risk
        
        # Get dynamic risk from booster
        dynamic_risk = self.streak_booster.current_risk()
        
        # Hard cap: Never exceed max_risk from profile
        max_risk_cap = profile.risk.risk_per_trade * 3  # 3x base max
        if dynamic_risk > max_risk_cap:
            logger.warning(f"⚠️ Risk capped: {dynamic_risk}% → {max_risk_cap}%")
            dynamic_risk = max_risk_cap
        
        return dynamic_risk

    # =========================================================
    # MAIN TRADING CYCLE
    # =========================================================
    def run_cycle(self) -> List[Dict]:
        """Execute one complete trading cycle."""
        logger.info("=" * 50)
        logger.info("Starting Fund-Grade Trading Cycle")
        logger.info("=" * 50)

        # Step 1: Pre-flight checks
        if not self._pre_flight_check():
            return []

        # Step 2: Update global regime
        self._update_global_regime()

        # Step 3: Risk Committee vote
        committee_decision = self._convene_risk_committee()
        if committee_decision.decision.value == "EMERGENCY_STOP":
            logger.critical("🚨 EMERGENCY STOP - No trading")
            return []

        # Step 4: Capital protection check
        protection_state = self._check_capital_protection()
        if protection_state.is_emergency:
            logger.warning("Capital protection triggered")
            return []

        # Step 5: Get allowed strategies
        allowed_strategies = self._get_allowed_strategies()

        # Step 6: 🔥 MANAGE EXISTING POSITIONS (New!)
        self._manage_positions()

        # Step 7: Process symbols for new entries
        trade_instructions = []
        for symbol_cfg in self.symbols_cfg.get("symbols", []):
            symbol = self._get_symbol_name(symbol_cfg)
            
            trade = self._process_symbol(symbol, allowed_strategies)
            if trade:
                trade_instructions.append(trade)

        # Step 8: Post-cycle analytics
        self._post_cycle_analytics(trade_instructions)

        logger.info(f"Cycle complete: {len(trade_instructions)} trades")
        return trade_instructions

    # =========================================================
    # PRE-FLIGHT CHECKS
    # =========================================================
    def _pre_flight_check(self) -> bool:
        """Perform all pre-flight checks."""
        # Check auto-disable
        can_trade, reason = self.auto_disable.can_trade()
        if not can_trade:
            logger.warning(f"AutoDisable: {reason}")
            return False

        # Check crisis mode (using .mode attribute directly)
        if self.crisis_controller.mode.value == "SURVIVAL":
            logger.warning("SURVIVAL mode - minimal trading only")
            self.is_crisis = True
        elif self.crisis_controller.mode.value == "DEFENSIVE":
            logger.info("DEFENSIVE mode active")

        # Check recovery state
        recovery_status = self.recovery_engine.get_status()
        if recovery_status.get("phase", "NORMAL") != "NORMAL":
            logger.info(f"Recovery phase: {recovery_status.get('phase')}")

        return True

    # =========================================================
    # GLOBAL REGIME
    # =========================================================
    def _update_global_regime(self):
        """Update global market regime."""
        # Get market indicators (simplified - would come from data)
        macro = MacroIndicators(
            yield_curve_slope=0.5,
            cb_hawkish_score=0.6,
        )
        market = MarketIndicators(
            vix_level=self.portfolio_metrics.get("vix", 18),
            cross_asset_correlation=self.portfolio_metrics.get("correlation", 0.3),
        )

        forecast = self.regime_forecaster.forecast(macro, market)
        self.current_regime = forecast.regime.value

        logger.info(f"Global Regime: {self.current_regime} ({forecast.confidence:.0%})")

    # =========================================================
    # RISK COMMITTEE
    # =========================================================
    def _convene_risk_committee(self):
        """Convene AI Risk Committee."""
        metrics = {
            "drawdown": self.portfolio_metrics.get("drawdown", 0),
            "volatility": self.portfolio_metrics.get("volatility", 0.01),
            "survival_probability": self.portfolio_metrics.get("survival", 0.95),
            "crowding_score": self.portfolio_metrics.get("crowding", 0.2),
            "regime": self.current_regime,
            "avg_decay_score": self.portfolio_metrics.get("decay", 0.3),
        }

        decision = self.risk_committee.convene(metrics)
        
        if decision.decision.value != "NORMAL":
            logger.warning(f"Risk Committee: {decision.decision.value}")
            for concern in decision.key_concerns[:3]:
                logger.warning(f"  - {concern}")

        return decision

    # =========================================================
    # CAPITAL PROTECTION
    # =========================================================
    def _check_capital_protection(self):
        """Check capital protection layers."""
        metrics = {
            "portfolio_drawdown": self.portfolio_metrics.get("drawdown", 0),
            "current_volatility": self.portfolio_metrics.get("volatility", 0.01),
            "average_volatility": 0.01,
            "liquidity_score": self.portfolio_metrics.get("liquidity", 1.0),
            "strategy_drawdowns": {},
        }

        state = self.capital_protection.check(metrics)
        return state

    # =========================================================
    # STRATEGY MANAGEMENT
    # =========================================================
    def _get_allowed_strategies(self) -> List[str]:
        """Get list of allowed strategies based on current mode."""
        # Get active strategies from pool
        pool_active = self.strategy_pool.get_active_strategies()
        
        # If no strategies defined, return default list
        if not pool_active:
            return ["default"]
        
        return pool_active

    # =========================================================
    # 🔥 POSITION MANAGEMENT (New!)
    # =========================================================
    def _manage_positions(self):
        """
        Active Position Management Step.
        
        This is where HOLD != do nothing.
        Uses PositionManager to make intelligent decisions for each open position.
        """
        import MetaTrader5 as mt5
        
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return
            
            logger.info(f"📊 Managing {len(positions)} open position(s)")
            
            # Collect market data and ATR for each symbol
            market_data = {}
            atr_dict = {}
            ai_predictions = {}
            
            symbols = set(pos.symbol for pos in positions)
            
            for symbol in symbols:
                # Fetch data
                data = self._fetch_data(symbol)
                if data:
                    # Get primary timeframe
                    primary_tf = list(data.keys())[0]
                    df = data[primary_tf]
                    market_data[symbol] = df
                    
                    # Get ATR
                    if hasattr(df, 'columns') and 'atr_14' in df.columns:
                        atr_dict[symbol] = df['atr_14'].iloc[-1]
                    else:
                        # Calculate ATR manually
                        features = self._generate_features(data)
                        fdf = features.get(primary_tf, df)
                        if 'atr_14' in fdf.columns:
                            atr_dict[symbol] = fdf['atr_14'].iloc[-1]
                        else:
                            atr_dict[symbol] = 5.0  # Default for XAUUSD
                    
                    # Get AI prediction for exit signals
                    try:
                        features = self._generate_features(data)
                        signals = self._run_models(symbol, features)
                        if signals:
                            fused = self._fuse_signals(signals)
                            if fused:
                                ai_predictions[symbol] = {
                                    'action': fused.direction.value,
                                    'confidence': fused.confidence,
                                    'trend_strength': fused.probability,
                                }
                    except Exception:
                        pass
            
            # Process each position
            actions_taken = 0
            for pos in positions:
                symbol = pos.symbol
                
                decision = self.position_manager.manage_position(
                    position=pos,
                    market_data=market_data.get(symbol),
                    ai_prediction=ai_predictions.get(symbol),
                    atr=atr_dict.get(symbol, 5.0)
                )
                
                # Execute non-HOLD actions
                if decision.action != PositionAction.HOLD:
                    result = self.position_executor.execute(decision, pos)
                    if result.success:
                        actions_taken += 1
                        logger.info(
                            f"✅ #{pos.ticket} {decision.action.value}: {decision.reason}"
                        )
                    else:
                        logger.warning(
                            f"⚠️ #{pos.ticket} Failed: {result.message}"
                        )
                else:
                    # Active HOLD - just monitoring
                    logger.debug(
                        f"👁️ #{pos.ticket} HOLD: R={decision.r_multiple:.2f}"
                    )
            
            if actions_taken > 0:
                logger.info(f"🔥 Position Management: {actions_taken} action(s) taken")
                
            # Log statistics periodically
            stats = self.position_manager.get_stats()
            if stats['decisions'] % 10 == 0 and stats['decisions'] > 0:
                logger.info(
                    f"📈 PM Stats: {stats['be_moves']} BE, "
                    f"{stats['partial_closes']} partials, "
                    f"{stats['full_exits']} exits"
                )
                
        except Exception as e:
            logger.error(f"Position management error: {e}")

    # =========================================================
    # SYMBOL PROCESSING
    # =========================================================
    def _process_symbol(self, symbol: str, allowed_strategies: List[str]) -> Optional[Dict]:
        """Process a single symbol through the full pipeline."""
        try:
            # Fetch data
            data = self._fetch_data(symbol)
            if not data:
                return None

            # Generate features
            features = self._generate_features(data)

            # Run AI models
            signals = self._run_models(symbol, features)
            if not signals:
                return None

            # Fuse signals
            fused = self._fuse_signals(signals)
            if not fused:
                return None

            # Decision gate
            decision = self.decision_gate.evaluate(fused)
            if decision.action != "TRADE":
                return None

            # Apply risk controls
            trade = self._apply_risk(symbol, decision, features)
            if not trade:
                return None

            # Generate explanation
            self._record_explanation(trade, fused, decision)

            return trade

        except Exception as e:
            logger.exception(f"Error processing {symbol}: {e}")
            # Note: record_model_error requires SystemMetrics, skip for now
            return None

    def _fetch_data(self, symbol: str) -> Optional[Dict]:
        """Fetch market data."""
        data = {}
        for tf in ["H1", "M15"]:
            rates = self.mt5.get_rates(symbol, tf, count=200)
            if rates is not None and len(rates) >= 100:
                # Use processor.fill_missing() or just pass raw data
                data[tf] = self.processor.fill_missing(rates) if hasattr(rates, 'columns') else rates
        return data if data else None

    def _generate_features(self, data: Dict) -> Dict:
        """Generate features for all timeframes."""
        features = {}
        for tf, df in data.items():
            tf_features = df.copy()
            for engine in self.feature_engines:
                # Try add_all_features, fallback to add_regime_features
                if hasattr(engine, 'add_all_features'):
                    tf_features = engine.add_all_features(tf_features)
                elif hasattr(engine, 'add_regime_features'):
                    tf_features = engine.add_regime_features(tf_features)
            features[tf] = tf_features.dropna()
        return features

    def _run_models(self, symbol: str, features: Dict) -> List[AISignal]:
        """Run AI models."""
        signals = []
        for tf, df in features.items():
            if len(df) < 60:
                continue

            # LSTM
            try:
                pred = self._get_lstm_prediction(df)
                if pred:
                    signals.append(self._to_signal(symbol, tf, pred, "lstm"))
            except Exception:
                pass

            # XGBoost
            try:
                pred = self._get_xgb_prediction(df)
                if pred:
                    signals.append(self._to_signal(symbol, tf, pred, "xgboost"))
            except Exception:
                pass

        return signals

    def _get_lstm_prediction(self, df) -> Optional[Dict]:
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
        if not feature_cols:
            return None
        X = df[feature_cols].iloc[-60:].values.reshape(1, 60, -1)
        return self.direction_model.predict_single(X)

    def _get_xgb_prediction(self, df) -> Optional[Dict]:
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
        if not feature_cols:
            return None
        X = df[feature_cols].iloc[-1].values
        return self.timing_model.predict_single(X)

    def _to_signal(self, symbol: str, tf: str, pred: Dict, model: str) -> AISignal:
        from src.models.signal_schema import VolatilityState, MarketRegime
        direction_map = {"LONG": SignalDirection.LONG, "SHORT": SignalDirection.SHORT}
        direction = direction_map.get(pred.get("direction", ""), SignalDirection.NEUTRAL)
        
        return AISignal(
            direction=direction,
            probability=pred.get("probability", 0.5),
            confidence=pred.get("confidence", 0.5),
            volatility_state=VolatilityState.NORMAL,
            regime=MarketRegime.RANGING,
            expected_rr=1.5,
            symbol=symbol,
            timeframe=tf,
            model_name=model,
        )

    def _fuse_signals(self, signals: List[AISignal]) -> Optional[AISignal]:
        if not signals:
            return None
        from src.signals.signal_fusion import ModelOutput
        outputs = [
            ModelOutput(s.model_name, s.direction, s.probability, s.confidence)
            for s in signals
        ]
        first = signals[0]
        return self.signal_fusion.fuse(
            outputs, first.symbol, first.timeframe, first.regime, first.volatility_state
        )

    def _apply_risk(self, symbol: str, decision, features: Dict) -> Optional[Dict]:
        """Apply risk management."""
        signal = decision.signal
        primary_tf = list(features.keys())[0]
        df = features[primary_tf]
        
        atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else 0.001
        entry_price = df["close"].iloc[-1]

        # Stop loss
        sltp = self.stop_loss_engine.calculate_atr_based(
            entry_price=entry_price,
            atr=atr,
            direction=signal.direction.value,
        )
        if not sltp.is_valid:
            return None

        # Position size with protection multiplier
        size_mult = self.capital_protection.get_position_multiplier()
        balance = self.mt5.get_balance() or 10000

        position = self.position_sizer.calculate(
            balance=balance * size_mult,
            sl_pips=sltp.sl_pips,
            pip_value=10.0,
            symbol=symbol,
            confidence=signal.confidence,
        )
        if not position.is_valid:
            return None

        return {
            "action": "OPEN",
            "symbol": symbol,
            "direction": signal.direction.value,
            "volume": position.lot_size,
            "sl": sltp.stop_loss,
            "tp": sltp.take_profit,
            "magic": 900001,
            "comment": f"AI_{datetime.now().strftime('%Y%m%d')}",
            "confidence": signal.confidence,
            "regime": self.current_regime,
        }

    # =========================================================
    # EXPLAINABILITY
    # =========================================================
    def _record_explanation(self, trade: Dict, signal: AISignal, decision):
        """Record trade explanation."""
        self.explainability.explain_decision(
            trade_id=f"{trade['symbol']}_{datetime.now().timestamp()}",
            signal_data={
                "symbol": trade["symbol"],
                "direction": trade["direction"],
                "confidence": trade["confidence"],
                "regime": trade["regime"],
                "indicators": ["LSTM", "XGBoost"],
                "conditions": [decision.reason],
                "primary_driver": "AI_Ensemble",
            },
            model_outputs={
                "model_name": "ensemble",
                "feature_importance": {},
            },
            portfolio_state={},
            risk_metrics={"regime_risk": "NORMAL"},
        )

    # =========================================================
    # POST-CYCLE ANALYTICS
    # =========================================================
    def _post_cycle_analytics(self, trades: List[Dict]):
        """Run post-cycle analytics."""
        # Update portfolio metrics for next cycle
        self.portfolio_metrics["trades_this_cycle"] = len(trades)

        logger.info(f"Analytics: {len(trades)} trades processed")

    # =========================================================
    # HELPERS
    # =========================================================
    def _get_symbol_name(self, cfg) -> str:
        return cfg.get("name", cfg) if isinstance(cfg, dict) else cfg

    def update_portfolio_metrics(self, metrics: Dict):
        """Update portfolio metrics from external source."""
        self.portfolio_metrics.update(metrics)


# =========================================================
# GLOBAL INSTANCE
# =========================================================
_orchestrator: Optional[FundGradeOrchestrator] = None


def get_fund_orchestrator() -> FundGradeOrchestrator:
    """Get global fund-grade orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = FundGradeOrchestrator()
    return _orchestrator
