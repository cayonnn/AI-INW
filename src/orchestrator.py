"""
orchestrator.py
================
Central brain of the AI-assisted MT5 trading system.

Responsibilities:
- Load configuration
- Pull & validate data
- Generate features
- Run AI / rule-based models
- Fuse signals
- Apply risk controls
- Emit final trade decisions (to MT5 EA or execution layer)
"""

from datetime import datetime
from typing import Dict, List, Optional

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

from src.data.mt5_connector import MT5Connector
from src.data.data_processor import DataProcessor
from src.data.data_validator import DataValidator

from src.features.price_action import PriceActionFeatures
from src.features.trend_momentum import TrendMomentumFeatures
from src.features.volatility import VolatilityFeatures
from src.features.regime_detector import RegimeDetector

from src.models.lstm_direction import LSTMDirectionModel
from src.models.xgb_timing import XGBTimingModel
from src.models.model_hotswap import ModelHotSwap
from src.models.signal_schema import AISignal

from src.signals.multi_timeframe import MultiTimeframeAnalyzer
from src.signals.signal_fusion import SignalFusion
from src.signals.decision_gate import DecisionGate

from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossCalculator
from src.risk.auto_disable import AutoDisableManager


logger = get_logger("ORCHESTRATOR")


class TradingOrchestrator:
    """
    High-level controller coordinating the entire trading pipeline.
    
    Pipeline Flow:
    Market Data → Features → AI Models → Signal Fusion → Decision Gate → Risk → Execute
    """

    def __init__(self):
        logger.info("Initializing Trading Orchestrator")

        # --- Load configuration ---
        self.settings = load_config("config/settings.yaml")
        self.risk_params = load_config("config/risk_params.yaml")
        self.symbols_cfg = load_config("config/symbols.yaml")

        # --- Core services ---
        self.mt5 = MT5Connector()
        self.processor = DataProcessor()
        self.validator = DataValidator()

        # --- Feature generators ---
        self.feature_engines = [
            PriceActionFeatures(),
            TrendMomentumFeatures(),
            VolatilityFeatures(),
            RegimeDetector(),
        ]

        # --- Models ---
        self.direction_model = LSTMDirectionModel()
        self.timing_model = XGBTimingModel()
        self.model_hotswap = ModelHotSwap()

        # --- Signal processing ---
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.signal_fusion = SignalFusion()
        self.decision_gate = DecisionGate()

        # --- Risk management ---
        self.position_sizer = PositionSizer()
        self.stop_loss_engine = StopLossCalculator()
        self.auto_disable = AutoDisableManager()

        logger.info("Trading Orchestrator ready")

    # ---------------------------------------------------------
    # Main cycle
    # ---------------------------------------------------------
    def run_cycle(self) -> List[Dict]:
        """
        Executes one full decision cycle.
        Returns finalized trade instructions.
        """
        logger.info("Starting trading cycle")
        trade_instructions = []

        # Check if trading is allowed
        can_trade, reason = self.auto_disable.can_trade()
        if not can_trade:
            logger.warning(f"Trading disabled: {reason}")
            return trade_instructions

        # Process each symbol
        symbols = self.symbols_cfg.get("symbols", [])
        for symbol_cfg in symbols:
            symbol = symbol_cfg.get("name", symbol_cfg) if isinstance(symbol_cfg, dict) else symbol_cfg
            timeframes = symbol_cfg.get("timeframes", ["H1"]) if isinstance(symbol_cfg, dict) else ["H1"]

            try:
                logger.info(f"Processing symbol: {symbol}")
                
                # Pipeline execution
                raw_data = self._fetch_data(symbol, timeframes)
                if not raw_data:
                    continue
                    
                features = self._generate_features(raw_data)
                signals = self._run_models(symbol, features)
                
                fused_signal = self._fuse_signals(signals)
                decision = self._decision(symbol, fused_signal)

                if decision is None:
                    continue

                trade = self._apply_risk(symbol, decision, features)
                if trade:
                    trade_instructions.append(trade)

            except Exception as e:
                logger.exception(f"Error processing {symbol}: {e}")
                self.auto_disable.record_model_error()

        logger.info(f"Trading cycle completed: {len(trade_instructions)} trades")
        return trade_instructions

    # ---------------------------------------------------------
    # Internal pipeline steps
    # ---------------------------------------------------------
    def _fetch_data(self, symbol: str, timeframes: List[str]) -> Optional[Dict]:
        """Fetch and validate market data."""
        data = {}
        for tf in timeframes:
            rates = self.mt5.get_rates(symbol, tf, count=200)
            if rates is None or len(rates) < 100:
                logger.warning(f"Insufficient data for {symbol} {tf}")
                continue
            
            # Validate data quality
            report = self.validator.validate(rates)
            if not report.is_valid:
                logger.warning(f"Data validation failed for {symbol} {tf}")
                continue
                
            data[tf] = self.processor.fill_missing_values(rates)
        
        return data if data else None

    def _generate_features(self, data: Dict) -> Dict:
        """Generate features for all timeframes."""
        features = {}
        for tf, df in data.items():
            tf_features = df.copy()
            for engine in self.feature_engines:
                tf_features = engine.add_all_features(tf_features)
            features[tf] = tf_features.dropna()
        return features

    def _run_models(self, symbol: str, features: Dict) -> List[AISignal]:
        """Run AI models on features."""
        signals = []

        for tf, df in features.items():
            if len(df) < 60:
                continue
                
            # LSTM direction prediction
            try:
                direction_pred = self._get_lstm_prediction(df)
                if direction_pred:
                    signals.append(self._pred_to_signal(symbol, tf, direction_pred, "lstm"))
            except Exception as e:
                logger.error(f"LSTM prediction error: {e}")

            # XGBoost timing prediction
            try:
                timing_pred = self._get_xgb_prediction(df)
                if timing_pred:
                    signals.append(self._pred_to_signal(symbol, tf, timing_pred, "xgboost"))
            except Exception as e:
                logger.error(f"XGBoost prediction error: {e}")

        return signals

    def _get_lstm_prediction(self, df) -> Optional[Dict]:
        """Get LSTM model prediction."""
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
        if len(feature_cols) == 0:
            return None
        X = df[feature_cols].iloc[-60:].values.reshape(1, 60, -1)
        return self.direction_model.predict_single(X)

    def _get_xgb_prediction(self, df) -> Optional[Dict]:
        """Get XGBoost model prediction."""
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
        if len(feature_cols) == 0:
            return None
        X = df[feature_cols].iloc[-1].values
        return self.timing_model.predict_single(X)

    def _pred_to_signal(self, symbol: str, tf: str, pred: Dict, model_name: str) -> AISignal:
        """Convert prediction dict to AISignal."""
        from src.models.signal_schema import SignalDirection, VolatilityState, MarketRegime
        
        direction_map = {
            "LONG": SignalDirection.LONG,
            "SHORT": SignalDirection.SHORT,
            "NEUTRAL": SignalDirection.NEUTRAL,
            "LONG_ENTRY": SignalDirection.LONG,
            "SHORT_ENTRY": SignalDirection.SHORT,
            "NO_ENTRY": SignalDirection.NEUTRAL,
        }
        
        direction = direction_map.get(pred.get("direction", pred.get("prediction", "NEUTRAL")), SignalDirection.NEUTRAL)
        
        return AISignal(
            direction=direction,
            probability=pred.get("probability", 0.5),
            confidence=pred.get("confidence", 0.5),
            volatility_state=VolatilityState.NORMAL,
            regime=MarketRegime.RANGING,
            expected_rr=1.5,
            symbol=symbol,
            timeframe=tf,
            model_name=model_name,
        )

    def _fuse_signals(self, signals: List[AISignal]) -> Optional[AISignal]:
        """Fuse multiple signals into one."""
        if not signals:
            return None
        
        from src.signals.signal_fusion import ModelOutput
        
        outputs = [
            ModelOutput(
                model_name=s.model_name,
                direction=s.direction,
                probability=s.probability,
                confidence=s.confidence,
            )
            for s in signals
        ]
        
        # Use first signal's metadata
        first = signals[0]
        return self.signal_fusion.fuse(
            outputs, first.symbol, first.timeframe, first.regime, first.volatility_state
        )

    def _decision(self, symbol: str, signal: Optional[AISignal]):
        """Apply decision gate."""
        if signal is None:
            return None
            
        decision = self.decision_gate.evaluate(signal)
        
        if decision.action == "TRADE":
            logger.info(f"Decision for {symbol}: {decision.signal.direction.value}")
            return decision
        else:
            logger.debug(f"No trade for {symbol}: {decision.reason}")
            return None

    def _apply_risk(self, symbol: str, decision, features: Dict) -> Optional[Dict]:
        """Apply risk management and prepare trade."""
        signal = decision.signal
        
        # Get ATR for SL calculation
        primary_tf = list(features.keys())[0]
        df = features[primary_tf]
        atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else 0.001
        
        # Calculate SL/TP
        entry_price = df["close"].iloc[-1]
        sltp = self.stop_loss_engine.calculate_atr_based(
            entry_price=entry_price,
            atr=atr,
            direction=signal.direction.value,
        )
        
        if not sltp.is_valid:
            logger.warning(f"SL/TP calculation failed for {symbol}")
            return None
        
        # Calculate position size
        balance = self.mt5.get_balance() or 10000
        position = self.position_sizer.calculate(
            balance=balance,
            sl_pips=sltp.sl_pips,
            pip_value=10.0,
            symbol=symbol,
            confidence=signal.confidence,
        )
        
        if not position.is_valid:
            logger.warning(f"Position sizing failed for {symbol}: {position.rejection_reason}")
            return None

        trade = {
            "symbol": symbol,
            "action": signal.direction.value,
            "volume": position.lot_size,
            "stop_loss": sltp.stop_loss,
            "take_profit": sltp.take_profit,
            "risk_pct": position.risk_pct,
            "confidence": signal.confidence,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"Trade prepared: {symbol} {signal.direction.value} {position.lot_size} lots")
        return trade


# ---------------------------------------------------------
# Global instance
# ---------------------------------------------------------
_orchestrator: Optional[TradingOrchestrator] = None


def get_orchestrator() -> TradingOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TradingOrchestrator()
    return _orchestrator
