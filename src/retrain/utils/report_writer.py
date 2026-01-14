# src/retrain/utils/report_writer.py
"""
Report Writer - Markdown Report Generation
===========================================

Generates markdown reports for daily retraining.
"""

import os
from datetime import date, datetime
from typing import Dict, Any, Optional

from src.utils.logger import get_logger

logger = get_logger("REPORT_WRITER")

REPORTS_DIR = "reports"


def write_report(
    best_config: Dict,
    run_date: date,
    data: Dict = None,
    old_config: Dict = None
) -> str:
    """
    Write markdown retrain report.
    
    Args:
        best_config: Best candidate configuration
        run_date: Date of retrain
        data: Training data used
        old_config: Previous configuration
        
    Returns:
        Report file path
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    date_str = run_date.strftime("%Y%m%d")
    filepath = os.path.join(REPORTS_DIR, f"retrain_{date_str}.md")
    
    # Extract values
    score = best_config.get("score", 0)
    max_dd = best_config.get("max_dd", 0) * 100
    conf = best_config.get("confidence", {})
    pyramid = best_config.get("pyramid", {})
    
    # Get old values
    old_conf = old_config.get("confidence", {}) if old_config else {}
    old_pyramid = old_config.get("pyramid", {}) if old_config else {}
    old_score = old_config.get("score", 0) if old_config else 0
    
    report = f"""# Daily Retrain Report - {run_date}

## Summary

| Metric | Old | New | Delta |
|--------|-----|-----|-------|
| Score | {old_score:.2f} | {score:.2f} | {score - old_score:+.2f} |
| Max DD | - | {max_dd:.1f}% | - |

## New Configuration

### Confidence Engine

| Parameter | Old | New |
|-----------|-----|-----|
| High Threshold | {old_conf.get('high', '-')} | {conf.get('high', 0.82):.3f} |
| Mid Threshold | {old_conf.get('mid', '-')} | {conf.get('mid', 0.66):.3f} |
| Low Threshold | {old_conf.get('low', '-')} | {conf.get('low', 0.50):.3f} |

### Pyramid Manager

| Parameter | Old | New |
|-----------|-----|-----|
| R1 Multiplier | {old_pyramid.get('r1', '-')} | {pyramid.get('r1', 1.0):.2f} |
| R2 Multiplier | {old_pyramid.get('r2', '-')} | {pyramid.get('r2', 0.7):.2f} |
| R3 Multiplier | {old_pyramid.get('r3', '-')} | {pyramid.get('r3', 0.4):.2f} |

"""

    # Bayesian Outcome section
    pass_type = best_config.get("pass_type", "unknown")
    mode = best_config.get("mode", "shadow")
    atr = best_config.get("atr", {})
    pyramid_levels = pyramid.get("levels", 1)
    
    pass_icon = "✅" if pass_type == "hard" else "⚠️" if pass_type == "soft" else "❌"
    
    report += f"""## Bayesian Outcome

| Metric | Value |
|--------|-------|
| Best Raw Score | {score:.2f} |
| Hard-pass | {'✅' if pass_type == 'hard' else '❌'} |
| Soft-pass | {'✅' if pass_type == 'soft' else '❌' if pass_type == 'hard' else '⚠️'} |
| Mode | {mode} |
| Action | {'Live deployment' if pass_type == 'hard' else 'Shadow deployment'} |

## Risk Flags

"""
    
    # Add risk flags
    risk_flags = []
    if atr.get("sl_mult", 1) < 0.8:
        risk_flags.append("🔴 Tight SL multiplier")
    if atr.get("tp_mult", 2) > 3:
        risk_flags.append("🟡 High TP multiplier")
    if pyramid_levels >= 3:
        risk_flags.append("🟡 Maximum pyramid depth")
    if max_dd > 6:
        risk_flags.append("🔴 High max drawdown")
    
    if risk_flags:
        for flag in risk_flags:
            report += f"- {flag}\n"
    else:
        report += "- ✅ No significant risk flags\n"
    
    report += f"""
## Recommendation

"""
    if pass_type == "hard":
        report += "✅ Ready for live deployment\n"
    else:
        report += "- Shadow test 3 days before promotion\n"
        if max_dd > 5:
            report += "- Monitor drawdown closely\n"
        if pyramid_levels >= 3:
            report += "- Consider reducing pyramid if DD > 5%\n"
    
    # Decision section
    improvement = score - old_score if old_score else score
    decision = "Approved" if improvement > 0 else "Needs Review"
    
    report += f"""
## Decision

**Status**: {'✅ ' + decision if improvement > 0 else '⚠️ ' + decision}
**Activation**: {'Next trading day' if pass_type == 'hard' else 'After 3-day shadow test'}

---
*Generated: {datetime.now().isoformat()}*
*Pipeline: Competition-Grade Retrain v2.0*
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report written: {filepath}")
    
    return filepath


def write_comparison_report(
    candidates: list,
    run_date: date
) -> str:
    """Write comparison report for multiple candidates."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    date_str = run_date.strftime("%Y%m%d")
    filepath = os.path.join(REPORTS_DIR, f"candidates_{date_str}.md")
    
    report = f"""# Candidate Comparison - {run_date}

## All Candidates

| Rank | Score | Max DD | Conf High | Pyramid R2 |
|------|-------|--------|-----------|------------|
"""
    
    for i, c in enumerate(candidates[:10], 1):
        conf = c.get("confidence", {})
        pyramid = c.get("pyramid", {})
        report += (
            f"| {i} | {c.get('score', 0):.2f} | "
            f"{c.get('max_dd', 0)*100:.1f}% | "
            f"{conf.get('high', 0):.3f} | "
            f"{pyramid.get('r2', 0):.2f} |\n"
        )
    
    report += f"\n---\n*Generated: {datetime.now().isoformat()}*\n"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return filepath
