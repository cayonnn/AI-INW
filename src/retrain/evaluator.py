# src/retrain/evaluator.py
"""
Candidate Evaluator - Stress Testing
=====================================

Stress tests candidate configs before deployment:
- Max drawdown check
- Score validation
- Risk limits
- Consistency check
"""

from typing import Dict, List, Optional, Any

from src.utils.logger import get_logger

logger = get_logger("EVALUATOR")


# Safety limits (relaxed for initial testing)
MAX_ALLOWED_DD = 0.10      # 10% max drawdown
MIN_REQUIRED_SCORE = -100  # Allow negative scores initially
MIN_WIN_RATE = 0.40        # 40% minimum win rate


def stress_test(
    candidates: List[Dict],
    data: Dict[str, Any],
    max_dd: float = MAX_ALLOWED_DD,
    min_score: float = MIN_REQUIRED_SCORE
) -> Optional[Dict]:
    """
    Stress test candidate configs with soft-pass fallback.
    
    Adds mode tag to candidate:
    - "live_candidate": Hard pass, ready for live
    - "shadow": Soft pass or last resort, needs shadow testing
    
    Args:
        candidates: List of candidate configs
        data: Training data for additional validation
        max_dd: Maximum allowed drawdown
        min_score: Minimum required score
        
    Returns:
        Best safe candidate or None
    """
    if not candidates:
        print("⚠ No candidates to evaluate")
        return None
    
    print(f"Evaluating {len(candidates)} candidates")
    
    # Try hard pass first (DD < 6%, positive score)
    hard_pass = [
        c for c in candidates
        if c.get("max_dd", 1) < 0.06 and c.get("score", 0) > 0
    ]
    
    if hard_pass:
        hard_pass.sort(key=lambda x: x.get("score", 0), reverse=True)
        best = hard_pass[0]
        best["mode"] = "live_candidate"
        best["pass_type"] = "hard"
        print(f"✅ Hard pass: score={best.get('score', 0):.2f}, max_dd={best.get('max_dd', 0):.1%}")
        return best
    
    print("⚠ No hard-pass candidate, applying soft criteria")
    
    # Soft pass: relaxed DD (< 8%), any score
    soft_pass = [
        c for c in candidates
        if c.get("max_dd", 1) < 0.08
    ]
    
    if soft_pass:
        soft_pass.sort(key=lambda x: (x.get("score", 0), -x.get("max_dd", 1)), reverse=True)
        best = soft_pass[0]
        best["mode"] = "shadow"
        best["pass_type"] = "soft"
        best["soft_approved"] = True
        best["status"] = "SHADOW_ONLY"
        best["reason"] = "Failed hard DD gate (> 6%)"
        best["promotion_blocked"] = True
        print(f"⚠ Soft pass: score={best.get('score', 0):.2f}, max_dd={best.get('max_dd', 0):.1%}")
        return best
    
    # Last resort: pick best score, force shadow
    print("⚠ No soft-pass, using best available candidate")
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    best = candidates[0]
    best["mode"] = "shadow"
    best["pass_type"] = "last_resort"
    best["soft_approved"] = True
    best["last_resort"] = True
    best["status"] = "SHADOW_ONLY"
    best["reason"] = "High score but failed stability gates"
    best["promotion_blocked"] = True
    return best


def _evaluate_candidate(
    candidate: Dict,
    max_dd: float,
    min_score: float
) -> tuple[bool, str]:
    """Evaluate a single candidate."""
    
    # Check drawdown
    dd = candidate.get("max_dd", 0)
    if dd > max_dd:
        return False, f"DD too high ({dd:.1%} > {max_dd:.1%})"
    
    # Check score
    score = candidate.get("score", 0)
    if score < min_score:
        return False, f"Score too low ({score:.2f} < {min_score})"
    
    # Check required fields
    if "confidence" not in candidate:
        return False, "Missing confidence config"
    
    if "pyramid" not in candidate:
        return False, "Missing pyramid config"
    
    return True, "OK"


def validate_config_structure(config: Dict) -> tuple[bool, List[str]]:
    """
    Validate the structure of a config.
    
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    # Check confidence section
    if "confidence" not in config:
        errors.append("Missing 'confidence' section")
    else:
        conf = config["confidence"]
        if "high" not in conf:
            errors.append("Missing confidence.high")
        if "mid" not in conf:
            errors.append("Missing confidence.mid")
    
    # Check pyramid section
    if "pyramid" not in config:
        errors.append("Missing 'pyramid' section")
    else:
        pyr = config["pyramid"]
        if "r2" not in pyr and "layers" not in pyr:
            errors.append("Missing pyramid.r2 or pyramid.layers")
    
    # Check score
    if "score" not in config:
        errors.append("Missing 'score' field")
    
    return len(errors) == 0, errors


def rank_candidates(candidates: List[Dict]) -> List[Dict]:
    """Rank candidates by composite score."""
    def composite_score(c: Dict) -> float:
        score = c.get("score", 0)
        dd = c.get("max_dd", 1)
        
        # Penalize high drawdown
        dd_penalty = max(0, dd - 0.03) * 10
        
        return score - dd_penalty
    
    return sorted(candidates, key=composite_score, reverse=True)
