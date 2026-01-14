# src/retrain/promotion.py
"""
Profile Promotion - Shadow to Live
===================================

Handles promotion of shadow profiles to live based on:
- Score comparison
- Drawdown safety
- Consistency over time
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("PROMOTION")


# Promotion thresholds
MIN_CONSECUTIVE_DAYS = 3
SCORE_EPSILON = 0.05  # 5% improvement required
MAX_DD_THRESHOLD = 0.08  # 8% max DD allowed


def eligible_for_promotion(
    shadow_scores: List[Dict],
    live_scores: List[Dict],
    epsilon: float = SCORE_EPSILON,
    min_days: int = MIN_CONSECUTIVE_DAYS
) -> bool:
    """
    Check if shadow profile is eligible for promotion.
    
    Rules:
    - Shadow score > Live score + epsilon for N consecutive days
    - Shadow max_dd < Live max_dd
    
    Args:
        shadow_scores: List of shadow score dicts
        live_scores: List of live score dicts
        epsilon: Minimum improvement margin
        min_days: Minimum consecutive days required
        
    Returns:
        True if eligible for promotion
    """
    if len(shadow_scores) < min_days or len(live_scores) < min_days:
        logger.info(f"Not enough data: shadow={len(shadow_scores)}, live={len(live_scores)}")
        return False
    
    # Compare last N days
    for s, l in zip(shadow_scores[-min_days:], live_scores[-min_days:]):
        shadow_score = s.get("score", 0)
        live_score = l.get("score", 0)
        shadow_dd = s.get("dd", s.get("max_dd", 0))
        live_dd = l.get("dd", l.get("max_dd", 0))
        
        # Score must exceed by epsilon
        if shadow_score <= live_score * (1 + epsilon):
            logger.info(f"Score not enough: {shadow_score:.2f} vs {live_score:.2f}")
            return False
        
        # DD must be lower
        if shadow_dd >= live_dd:
            logger.info(f"DD not better: {shadow_dd:.1%} vs {live_dd:.1%}")
            return False
    
    logger.info("✅ Shadow eligible for promotion!")
    return True


def calculate_promotion_score(
    shadow_scores: List[Dict],
    live_scores: List[Dict]
) -> Dict[str, Any]:
    """
    Calculate promotion readiness score.
    
    Returns detailed analysis of shadow vs live performance.
    """
    if not shadow_scores or not live_scores:
        return {
            "ready": False,
            "reason": "Insufficient data",
            "shadow_avg": 0,
            "live_avg": 0,
            "delta": 0,
        }
    
    shadow_avg = sum(s.get("score", 0) for s in shadow_scores) / len(shadow_scores)
    live_avg = sum(l.get("score", 0) for l in live_scores) / len(live_scores)
    
    shadow_dd_max = max(s.get("dd", s.get("max_dd", 0)) for s in shadow_scores)
    live_dd_max = max(l.get("dd", l.get("max_dd", 0)) for l in live_scores)
    
    delta = shadow_avg - live_avg
    dd_improvement = live_dd_max - shadow_dd_max
    
    eligible = eligible_for_promotion(shadow_scores, live_scores)
    
    return {
        "ready": eligible,
        "shadow_avg_score": round(shadow_avg, 2),
        "live_avg_score": round(live_avg, 2),
        "score_delta": round(delta, 2),
        "shadow_max_dd": round(shadow_dd_max * 100, 1),
        "live_max_dd": round(live_dd_max * 100, 1),
        "dd_improvement": round(dd_improvement * 100, 1),
        "days_analyzed": min(len(shadow_scores), len(live_scores)),
    }


def promote_shadow_to_live(
    shadow_profile_path: str,
    active_profile_path: str = "profiles/active.json"
) -> bool:
    """
    Promote shadow profile to live.
    
    Args:
        shadow_profile_path: Path to shadow profile YAML
        active_profile_path: Path to active profile marker
        
    Returns:
        True if promotion successful
    """
    import json
    
    if not os.path.exists(shadow_profile_path):
        logger.error(f"Shadow profile not found: {shadow_profile_path}")
        return False
    
    try:
        # Extract profile name and version
        basename = os.path.basename(shadow_profile_path)
        parts = basename.replace(".yaml", "").split("_v")
        
        if len(parts) != 2:
            logger.error(f"Invalid profile name: {basename}")
            return False
        
        name, version = parts[0], int(parts[1])
        
        # Update active profile
        with open(active_profile_path, 'w') as f:
            json.dump({
                "name": name,
                "version": version,
                "file": basename,
                "promoted_at": datetime.now().isoformat(),
                "was_shadow": True,
            }, f, indent=2)
        
        logger.info(f"✅ Promoted {name} v{version} to live")
        return True
        
    except Exception as e:
        logger.error(f"Promotion failed: {e}")
        return False


def create_promotion_report(analysis: Dict) -> str:
    """Create markdown promotion report."""
    status = "✅ APPROVED" if analysis["ready"] else "⏸ PENDING"
    
    report = f"""# Promotion Analysis

## Status: {status}

| Metric | Shadow | Live | Delta |
|--------|--------|------|-------|
| Avg Score | {analysis['shadow_avg_score']} | {analysis['live_avg_score']} | {analysis['score_delta']:+.2f} |
| Max DD | {analysis['shadow_max_dd']}% | {analysis['live_max_dd']}% | {analysis['dd_improvement']:+.1f}% |

**Days Analyzed:** {analysis['days_analyzed']}

---
*Generated: {datetime.now().isoformat()}*
"""
    return report
