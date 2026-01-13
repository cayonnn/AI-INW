"""
alpha_dashboard.py
===================
Explainable Alpha Dashboard

สิ่งที่ "กองทุนจริงขาดไม่ได้"

ตอบคำถามเหล่านี้ได้ทันที:
- ทำไมวันนี้กำไร
- กำไรจากอะไร
- เสี่ยงตรงไหน
- ถ้าตลาดเปลี่ยนจะเป็นยังไง

Dashboard ไม่ใช่โชว์ แต่ feed กลับเข้า engine
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from src.utils.logger import get_logger

logger = get_logger("ALPHA_DASHBOARD")


@dataclass
class PnLBreakdown:
    """PnL breakdown by source."""
    total_pnl: float
    market_beta: float           # From market movement
    trend_alpha: float           # From trend following
    volatility_alpha: float      # From volatility plays
    execution_alpha: float       # From good execution
    residual: float              # Unexplained


@dataclass
class StrategyHealth:
    """Strategy health summary."""
    name: str
    alpha_status: str            # +, -, 0
    decay_level: str             # Low, Medium, High
    status: str                  # Active, At-Risk, Reduced, Frozen
    capital_pct: float


@dataclass
class RegimePerformance:
    """Performance by regime."""
    regime: str
    best_strategy: str
    worst_strategy: str
    avg_pnl: float


@dataclass
class DashboardSnapshot:
    """Complete dashboard snapshot."""
    timestamp: datetime
    
    # PnL Breakdown
    pnl_breakdown: PnLBreakdown
    
    # Strategy Health
    strategy_health: List[StrategyHealth]
    
    # Regime Matrix
    regime_performance: List[RegimePerformance]
    
    # Risk Status
    current_drawdown: float
    crisis_mode: bool
    recovery_state: str
    
    # Alerts
    alerts: List[str]
    
    # Recommendations
    recommendations: List[str]


class ExplainableAlphaDashboard:
    """
    Explainable Alpha Dashboard.
    
    Provides real-time insight into system performance and decisions.
    """

    def __init__(self):
        self.snapshots: List[DashboardSnapshot] = []
        self.current_snapshot: Optional[DashboardSnapshot] = None
        
        # Connected components (set during integration)
        self.alpha_attribution = None
        self.strategy_pool = None
        self.decay_detector = None
        self.crisis_controller = None
        self.recovery_engine = None

    # -------------------------------------------------
    # Main snapshot generation
    # -------------------------------------------------
    def generate_snapshot(self,
                         trades: List[Any],
                         alpha_results: Dict[str, Any],
                         decay_results: Dict[str, Any],
                         pool_status: Dict,
                         system_metrics: Dict) -> DashboardSnapshot:
        """
        Generate complete dashboard snapshot.
        """
        # PnL Breakdown
        pnl_breakdown = self._calculate_pnl_breakdown(alpha_results)
        
        # Strategy Health
        strategy_health = self._build_strategy_health(alpha_results, decay_results, pool_status)
        
        # Regime Performance
        regime_performance = self._analyze_regime_performance(trades)
        
        # Risk Status
        current_dd = system_metrics.get("drawdown", 0.0)
        crisis_mode = system_metrics.get("crisis_mode", False)
        recovery_state = system_metrics.get("recovery_state", "NORMAL")
        
        # Generate alerts
        alerts = self._generate_alerts(decay_results, current_dd, alpha_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            decay_results, alpha_results, crisis_mode
        )
        
        snapshot = DashboardSnapshot(
            timestamp=datetime.now(),
            pnl_breakdown=pnl_breakdown,
            strategy_health=strategy_health,
            regime_performance=regime_performance,
            current_drawdown=current_dd,
            crisis_mode=crisis_mode,
            recovery_state=recovery_state,
            alerts=alerts,
            recommendations=recommendations,
        )
        
        self.current_snapshot = snapshot
        self.snapshots.append(snapshot)
        
        # Keep last 100 snapshots
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]
        
        return snapshot

    # -------------------------------------------------
    # PnL Analysis
    # -------------------------------------------------
    def _calculate_pnl_breakdown(self, alpha_results: Dict) -> PnLBreakdown:
        """Calculate PnL breakdown from alpha attribution."""
        total = 0.0
        market_beta = 0.0
        trend_alpha = 0.0
        vol_alpha = 0.0
        execution = 0.0
        residual = 0.0
        
        for strategy, result in alpha_results.items():
            if hasattr(result, 'total_return'):
                total += result.total_return
            if hasattr(result, 'factor_exposures'):
                for exp in result.factor_exposures:
                    if exp.factor.value == "MARKET":
                        market_beta += exp.contribution
                    elif exp.factor.value == "TREND":
                        trend_alpha += exp.contribution
                    elif exp.factor.value == "VOLATILITY":
                        vol_alpha += exp.contribution
            if hasattr(result, 'alpha'):
                residual += result.alpha
        
        return PnLBreakdown(
            total_pnl=total,
            market_beta=market_beta,
            trend_alpha=trend_alpha,
            volatility_alpha=vol_alpha,
            execution_alpha=execution,
            residual=residual,
        )

    # -------------------------------------------------
    # Strategy Health
    # -------------------------------------------------
    def _build_strategy_health(self, alpha_results: Dict, 
                               decay_results: Dict,
                               pool_status: Dict) -> List[StrategyHealth]:
        """Build strategy health summary."""
        health_list = []
        
        # Get all strategies
        all_strategies = set(alpha_results.keys()) | set(decay_results.keys())
        
        for name in all_strategies:
            alpha = alpha_results.get(name)
            decay = decay_results.get(name)
            
            # Determine alpha status
            if alpha and hasattr(alpha, 'alpha'):
                alpha_status = "+" if alpha.alpha > 0 else "-" if alpha.alpha < 0 else "0"
            else:
                alpha_status = "?"
            
            # Determine decay level
            if decay and hasattr(decay, 'decay_score'):
                if decay.decay_score < 0.3:
                    decay_level = "Low"
                elif decay.decay_score < 0.6:
                    decay_level = "Medium"
                else:
                    decay_level = "High"
            else:
                decay_level = "Unknown"
            
            # Get status from pool
            status = "Unknown"
            capital_pct = 0.0
            by_state = pool_status.get("by_state", {})
            for state, strategies in by_state.items():
                if name in strategies:
                    status = state
                    break
            
            health_list.append(StrategyHealth(
                name=name,
                alpha_status=alpha_status,
                decay_level=decay_level,
                status=status,
                capital_pct=capital_pct,
            ))
        
        return health_list

    # -------------------------------------------------
    # Regime Analysis
    # -------------------------------------------------
    def _analyze_regime_performance(self, trades: List[Any]) -> List[RegimePerformance]:
        """Analyze performance by regime."""
        regimes = ["TRENDING", "RANGING", "VOLATILE", "CRISIS"]
        
        performance = []
        for regime in regimes:
            regime_trades = [t for t in trades if hasattr(t, 'regime') and t.regime == regime]
            
            if not regime_trades:
                continue
            
            # Group by strategy
            by_strategy = {}
            for t in regime_trades:
                if hasattr(t, 'strategy') and hasattr(t, 'pnl'):
                    if t.strategy not in by_strategy:
                        by_strategy[t.strategy] = 0.0
                    by_strategy[t.strategy] += t.pnl
            
            if by_strategy:
                best = max(by_strategy, key=by_strategy.get)
                worst = min(by_strategy, key=by_strategy.get)
                avg_pnl = sum(by_strategy.values()) / len(by_strategy)
                
                performance.append(RegimePerformance(
                    regime=regime,
                    best_strategy=best,
                    worst_strategy=worst,
                    avg_pnl=avg_pnl,
                ))
        
        return performance

    # -------------------------------------------------
    # Alerts & Recommendations
    # -------------------------------------------------
    def _generate_alerts(self, decay_results: Dict, 
                        drawdown: float,
                        alpha_results: Dict) -> List[str]:
        """Generate system alerts."""
        alerts = []
        
        # Decay alerts
        for name, result in decay_results.items():
            if hasattr(result, 'status'):
                if result.status.value == "CRITICAL":
                    alerts.append(f"🚨 {name}: Critical decay detected")
                elif result.status.value == "DECAYING":
                    alerts.append(f"⚠️ {name}: Alpha decay confirmed")
        
        # Drawdown alerts
        if drawdown < -0.05:
            alerts.append(f"📉 Drawdown: {drawdown:.1%}")
        if drawdown < -0.08:
            alerts.append("🔴 Approaching max drawdown limit")
        
        # Alpha confidence alerts
        for name, result in alpha_results.items():
            if hasattr(result, 'alpha_confidence') and result.alpha_confidence < 0.3:
                alerts.append(f"⚠️ {name}: Low alpha confidence ({result.alpha_confidence:.0%})")
        
        return alerts

    def _generate_recommendations(self, decay_results: Dict,
                                  alpha_results: Dict,
                                  crisis_mode: bool) -> List[str]:
        """Generate action recommendations."""
        recommendations = []
        
        if crisis_mode:
            recommendations.append("🛡️ Crisis mode active: Focus on capital preservation")
            recommendations.append("📉 Reduce all positions to minimum")
        
        # Strategy-specific
        for name, result in decay_results.items():
            if hasattr(result, 'recommendation'):
                if result.recommendation == "FREEZE_IMMEDIATELY":
                    recommendations.append(f"❄️ Freeze {name} immediately")
                elif result.recommendation == "REDUCE_CAPITAL_50%":
                    recommendations.append(f"📉 Reduce {name} capital by 50%")
        
        # Alpha-based
        for name, result in alpha_results.items():
            if hasattr(result, 'status'):
                if result.status.value == "STRONG":
                    recommendations.append(f"📈 Consider increasing {name} allocation")
        
        return recommendations

    # -------------------------------------------------
    # Display helpers
    # -------------------------------------------------
    def get_summary_text(self) -> str:
        """Get text summary for logging."""
        if not self.current_snapshot:
            return "No snapshot available"
        
        s = self.current_snapshot
        
        lines = [
            "=" * 50,
            "ALPHA DASHBOARD SUMMARY",
            "=" * 50,
            f"Timestamp: {s.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "",
            "📊 PnL Breakdown:",
            f"  Total: {s.pnl_breakdown.total_pnl:.4f}",
            f"  Market Beta: {s.pnl_breakdown.market_beta:.4f}",
            f"  Trend Alpha: {s.pnl_breakdown.trend_alpha:.4f}",
            f"  Residual: {s.pnl_breakdown.residual:.4f}",
            "",
            "🏥 Strategy Health:",
        ]
        
        for h in s.strategy_health[:5]:  # Top 5
            lines.append(f"  {h.name}: Alpha={h.alpha_status}, Decay={h.decay_level}, Status={h.status}")
        
        lines.extend([
            "",
            f"⚠️ Alerts: {len(s.alerts)}",
        ])
        for alert in s.alerts[:3]:
            lines.append(f"  {alert}")
        
        lines.extend([
            "",
            f"💡 Recommendations: {len(s.recommendations)}",
        ])
        for rec in s.recommendations[:3]:
            lines.append(f"  {rec}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)

    def get_json_snapshot(self) -> Dict:
        """Get snapshot as JSON-serializable dict."""
        if not self.current_snapshot:
            return {}
        
        s = self.current_snapshot
        
        return {
            "timestamp": s.timestamp.isoformat(),
            "pnl": {
                "total": s.pnl_breakdown.total_pnl,
                "market_beta": s.pnl_breakdown.market_beta,
                "trend_alpha": s.pnl_breakdown.trend_alpha,
                "volatility_alpha": s.pnl_breakdown.volatility_alpha,
                "residual": s.pnl_breakdown.residual,
            },
            "strategies": [
                {
                    "name": h.name,
                    "alpha": h.alpha_status,
                    "decay": h.decay_level,
                    "status": h.status,
                }
                for h in s.strategy_health
            ],
            "risk": {
                "drawdown": s.current_drawdown,
                "crisis_mode": s.crisis_mode,
                "recovery_state": s.recovery_state,
            },
            "alerts": s.alerts,
            "recommendations": s.recommendations,
        }
