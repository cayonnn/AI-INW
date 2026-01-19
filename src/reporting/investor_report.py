# src/reporting/investor_report.py
"""
Investor-Grade Reporting Engine
================================

Auto-generates institutional-quality reports.

Reports:
    - Monthly Performance
    - Risk Attribution
    - Drawdown Explained
    - Guardian Intervention Summary
    - Capital Allocation Decisions

Paper Statement:
    "Our system generates investor-grade reports explaining
     risk decisions in natural language."
"""

import os
import sys
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("INVESTOR_REPORT")


@dataclass
class PerformanceMetrics:
    """Key performance metrics."""
    period: str
    start_equity: float
    end_equity: float
    pnl: float
    pnl_pct: float
    max_dd: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    
    def to_dict(self) -> Dict:
        return {
            "period": self.period,
            "pnl": f"${self.pnl:+,.2f}",
            "return": f"{self.pnl_pct:+.2%}",
            "max_dd": f"{self.max_dd:.2%}",
            "sharpe": f"{self.sharpe_ratio:.2f}",
            "win_rate": f"{self.win_rate:.1%}",
            "trades": self.total_trades
        }


@dataclass 
class RiskAttribution:
    """Risk attribution breakdown."""
    dd_avoided_by_guardian: float
    dd_avoided_by_meta: float
    dd_from_alpha: float
    capital_lockdown_time_pct: float
    risk_events: int


class InvestorReportEngine:
    """
    Generates investor-grade reports.
    
    Features:
        - Natural language explanations
        - Risk attribution
        - Guardian impact analysis
        - PDF/HTML export ready
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("📊 InvestorReportEngine initialized")
    
    def generate_monthly_report(
        self,
        metrics: PerformanceMetrics,
        risk: RiskAttribution,
        decisions: List[Dict] = None
    ) -> str:
        """Generate monthly performance report."""
        report = []
        
        # Header
        report.append("=" * 60)
        report.append(f"📊 MONTHLY PERFORMANCE REPORT - {metrics.period}")
        report.append("=" * 60)
        report.append("")
        
        # Performance Summary
        report.append("## Performance Summary")
        report.append(f"• Net P&L: ${metrics.pnl:+,.2f} ({metrics.pnl_pct:+.2%})")
        report.append(f"• Maximum Drawdown: {metrics.max_dd:.2%}")
        report.append(f"• Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        report.append(f"• Win Rate: {metrics.win_rate:.1%}")
        report.append(f"• Total Trades: {metrics.total_trades}")
        report.append("")
        
        # Risk Attribution
        report.append("## Risk Attribution")
        total_dd_avoided = risk.dd_avoided_by_guardian + risk.dd_avoided_by_meta
        report.append(f"• Drawdown Avoided by Guardian: {risk.dd_avoided_by_guardian:.1%}")
        report.append(f"• Drawdown Avoided by Meta-Controller: {risk.dd_avoided_by_meta:.1%}")
        report.append(f"• Total Risk Mitigation: {total_dd_avoided:.1%}")
        report.append("")
        
        # Natural Language Insights
        report.append("## Key Insights")
        report.extend(self._generate_insights(metrics, risk))
        report.append("")
        
        # Guardian Summary
        report.append("## Guardian Intervention Summary")
        report.append(f"• Risk Events Handled: {risk.risk_events}")
        report.append(f"• Capital Lockdown Time: {risk.capital_lockdown_time_pct:.1%}")
        
        if risk.dd_avoided_by_guardian > 0.05:
            report.append("")
            report.append(f"> \"**{risk.dd_avoided_by_guardian:.0%}** of potential drawdown was")
            report.append(f">  avoided by the Guardian system during high-risk periods.\"")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _generate_insights(
        self,
        metrics: PerformanceMetrics,
        risk: RiskAttribution
    ) -> List[str]:
        """Generate natural language insights."""
        insights = []
        
        # Performance insight
        if metrics.pnl_pct > 0.05:
            insights.append(f"• Strong performance with {metrics.pnl_pct:.1%} return.")
        elif metrics.pnl_pct > 0:
            insights.append(f"• Positive return of {metrics.pnl_pct:.1%} maintained.")
        else:
            insights.append(f"• Challenging period with {metrics.pnl_pct:.1%} return.")
        
        # Risk insight
        total_avoided = risk.dd_avoided_by_guardian + risk.dd_avoided_by_meta
        if total_avoided > 0.10:
            insights.append(f"• Risk systems prevented significant losses ({total_avoided:.0%} DD avoided).")
        
        # Guardian insight
        if risk.dd_avoided_by_guardian > risk.dd_from_alpha:
            insights.append("• Guardian outperformed Alpha in risk-adjusted terms.")
        
        # Sharpe insight
        if metrics.sharpe_ratio > 2.0:
            insights.append(f"• Excellent risk-adjusted returns (Sharpe {metrics.sharpe_ratio:.1f}).")
        elif metrics.sharpe_ratio > 1.0:
            insights.append(f"• Good risk-adjusted performance (Sharpe {metrics.sharpe_ratio:.1f}).")
        
        return insights
    
    def generate_risk_report(self, risk: RiskAttribution) -> str:
        """Generate detailed risk attribution report."""
        report = [
            "=" * 60,
            "🛡️ RISK ATTRIBUTION REPORT",
            "=" * 60,
            "",
            "## Drawdown Prevention",
            f"• Guardian blocks: {risk.dd_avoided_by_guardian:.1%} saved",
            f"• Meta pauses: {risk.dd_avoided_by_meta:.1%} saved",
            f"• Alpha losses: {risk.dd_from_alpha:.1%}",
            "",
            "## Intervention Analysis",
            f"• Total risk events: {risk.risk_events}",
            f"• Lockdown time: {risk.capital_lockdown_time_pct:.1%} of period",
            "",
            "## Paper Claim",
            "> \"The system autonomously prevented {:.0%} of potential".format(
                risk.dd_avoided_by_guardian + risk.dd_avoided_by_meta
            ),
            ">  drawdown through hierarchical risk governance.\"",
            "",
            "=" * 60
        ]
        return "\n".join(report)
    
    def save_report(self, report: str, filename: str):
        """Save report to file."""
        path = f"{self.output_dir}/{filename}"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📄 Report saved: {path}")
        return path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    engine = InvestorReportEngine()
    
    # Sample data
    metrics = PerformanceMetrics(
        period="January 2026",
        start_equity=10000,
        end_equity=10850,
        pnl=850,
        pnl_pct=0.085,
        max_dd=0.045,
        sharpe_ratio=2.3,
        win_rate=0.62,
        total_trades=47
    )
    
    risk = RiskAttribution(
        dd_avoided_by_guardian=0.12,
        dd_avoided_by_meta=0.05,
        dd_from_alpha=0.045,
        capital_lockdown_time_pct=0.08,
        risk_events=7
    )
    
    report = engine.generate_monthly_report(metrics, risk)
    print(report)
    
    engine.save_report(report, "monthly_jan2026.txt")
