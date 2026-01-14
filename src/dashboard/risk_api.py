# src/dashboard/risk_api.py
"""
Effective Risk Dashboard API
============================

API endpoints for real-time risk visualization:
- /risk/effective - Full risk stack breakdown
- /risk/shadow - Shadow candidate status
- /risk/tournament - Tournament leaderboard
"""

from fastapi import APIRouter
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analytics.risk_stack import get_risk_stack
from src.shadow.tournament import get_tournament
from src.registry.promotion_engine import get_promotion_engine

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/effective")
async def get_effective_risk() -> Dict[str, Any]:
    """
    Get effective risk stack breakdown.
    
    Shows all risk multipliers and final effective risk.
    """
    try:
        stack = get_risk_stack()
        snapshot = stack.get_snapshot()
        
        return {
            "status": "ok",
            "data": {
                "base_risk": snapshot.base_risk,
                "win_streak_mult": snapshot.streak_mult,
                "confidence_mult": snapshot.confidence_mult,
                "mode_mult": snapshot.mode_mult,
                "score_cap": snapshot.score_cap,
                "pyramid_mult": snapshot.pyramid_mult,
                "effective_risk": snapshot.effective_risk,
                "breakdown": snapshot.breakdown,
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": _get_default_risk_stack()
        }


@router.get("/shadow")
async def get_shadow_status() -> Dict[str, Any]:
    """
    Get shadow candidate status.
    
    Shows current shadow profiles and promotion eligibility.
    """
    try:
        tournament = get_tournament()
        engine = get_promotion_engine()
        
        shadows = []
        for name, entry in tournament.shadows.items():
            shadows.append({
                "name": name,
                "score": entry.score,
                "max_dd": entry.max_dd,
                "win_days": entry.win_days,
                "eligible": tournament._is_eligible(entry, tournament.live) if tournament.live else False,
            })
        
        return {
            "status": "ok",
            "data": {
                "live_profile": engine.current_live,
                "shadows": shadows,
                "promotion_candidate": tournament.get_promotion_candidate(),
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": {"shadows": []}
        }


@router.get("/tournament")
async def get_tournament_leaderboard() -> Dict[str, Any]:
    """
    Get tournament leaderboard.
    
    Shows all profiles ranked by score.
    """
    try:
        tournament = get_tournament()
        leaderboard = tournament.get_leaderboard()
        
        return {
            "status": "ok",
            "data": {
                "leaderboard": leaderboard,
                "live_profile": tournament.live.profile_id if tournament.live else None,
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": {"leaderboard": []}
        }


@router.get("/governance")
async def get_governance_state() -> Dict[str, Any]:
    """
    Get promotion governance state.
    
    Shows current live, previous (rollback), and state history.
    """
    try:
        engine = get_promotion_engine()
        state = engine.get_state()
        
        return {
            "status": "ok",
            "data": state
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def _get_default_risk_stack() -> Dict[str, float]:
    """Default risk stack for fallback."""
    return {
        "base_risk": 2.0,
        "win_streak_mult": 1.0,
        "confidence_mult": 1.0,
        "mode_mult": 1.0,
        "score_cap": 1.0,
        "pyramid_mult": 1.0,
        "effective_risk": 2.0,
    }


# For standalone testing
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Risk Dashboard API")
    app.include_router(router)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
