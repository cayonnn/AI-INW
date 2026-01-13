"""
ai_risk_committee.py
=====================
AI Risk Committee (Voting System)

Committee Members:
- Risk Engine, Stress Sim, Crowding, Regime, Alpha Monitor

Decisions: NORMAL, CAUTION, RISK_OFF, EMERGENCY_STOP
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List
from src.utils.logger import get_logger

logger = get_logger("AI_RISK_COMMITTEE")


class CommitteeDecision(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    RISK_OFF = "RISK_OFF"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class VoteType(str, Enum):
    APPROVE = "APPROVE"
    CONCERN = "CONCERN"
    REJECT = "REJECT"
    EMERGENCY = "EMERGENCY"


@dataclass
class Vote:
    member: str
    vote_type: VoteType
    confidence: float
    reason: str


@dataclass
class CommitteeMeeting:
    decision: CommitteeDecision
    votes: List[Vote]
    key_concerns: List[str]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class AIRiskCommittee:
    """AI Risk Committee with multi-member voting."""

    def __init__(self):
        self.members = ["RiskEngine", "StressSim", "Crowding", "Regime", "Alpha"]
        self.meeting_history: List[CommitteeMeeting] = []
        self.emergency_threshold = 2

    def convene(self, metrics: Dict) -> CommitteeMeeting:
        """Convene meeting and get decision."""
        votes = []
        
        # Risk Engine vote
        dd = metrics.get("drawdown", 0)
        if dd < -0.10:
            votes.append(Vote("RiskEngine", VoteType.EMERGENCY, 0.95, f"DD={dd:.1%}"))
        elif dd < -0.05:
            votes.append(Vote("RiskEngine", VoteType.REJECT, 0.8, f"DD={dd:.1%}"))
        else:
            votes.append(Vote("RiskEngine", VoteType.APPROVE, 0.8, "DD normal"))
        
        # Stress Sim vote
        survival = metrics.get("survival_probability", 1.0)
        if survival < 0.80:
            votes.append(Vote("StressSim", VoteType.EMERGENCY, 0.9, f"Survival={survival:.0%}"))
        elif survival < 0.90:
            votes.append(Vote("StressSim", VoteType.REJECT, 0.8, f"Survival={survival:.0%}"))
        else:
            votes.append(Vote("StressSim", VoteType.APPROVE, 0.8, "Stress OK"))
        
        # Crowding vote
        crowding = metrics.get("crowding_score", 0)
        if crowding > 0.7:
            votes.append(Vote("Crowding", VoteType.REJECT, 0.85, f"Crowding={crowding:.0%}"))
        elif crowding > 0.5:
            votes.append(Vote("Crowding", VoteType.CONCERN, 0.7, f"Crowding={crowding:.0%}"))
        else:
            votes.append(Vote("Crowding", VoteType.APPROVE, 0.75, "Crowding OK"))
        
        # Regime vote
        regime = metrics.get("regime", "NEUTRAL")
        if regime == "CRISIS":
            votes.append(Vote("Regime", VoteType.EMERGENCY, 0.9, "Crisis regime"))
        elif regime == "RISK_OFF":
            votes.append(Vote("Regime", VoteType.REJECT, 0.8, "Risk-off regime"))
        else:
            votes.append(Vote("Regime", VoteType.APPROVE, 0.7, f"Regime={regime}"))
        
        # Alpha vote
        decay = metrics.get("avg_decay_score", 0)
        if decay > 0.7:
            votes.append(Vote("Alpha", VoteType.REJECT, 0.8, f"Decay={decay:.0%}"))
        elif decay > 0.5:
            votes.append(Vote("Alpha", VoteType.CONCERN, 0.7, f"Decay={decay:.0%}"))
        else:
            votes.append(Vote("Alpha", VoteType.APPROVE, 0.75, "Alpha healthy"))
        
        # Count votes
        emergency = sum(1 for v in votes if v.vote_type == VoteType.EMERGENCY)
        reject = sum(1 for v in votes if v.vote_type == VoteType.REJECT)
        concern = sum(1 for v in votes if v.vote_type == VoteType.CONCERN)
        
        # Decision
        if emergency >= self.emergency_threshold:
            decision = CommitteeDecision.EMERGENCY_STOP
        elif emergency >= 1 or reject >= 2:
            decision = CommitteeDecision.RISK_OFF
        elif concern + reject > 2:
            decision = CommitteeDecision.CAUTION
        else:
            decision = CommitteeDecision.NORMAL
        
        concerns = [v.reason for v in votes if v.vote_type != VoteType.APPROVE]
        actions = self._get_actions(decision)
        
        meeting = CommitteeMeeting(decision, votes, concerns, actions)
        self.meeting_history.append(meeting)
        
        logger.info(f"Committee: {decision.value}")
        return meeting

    def _get_actions(self, decision: CommitteeDecision) -> List[str]:
        if decision == CommitteeDecision.EMERGENCY_STOP:
            return ["Close all positions", "Disable trading", "Lock system"]
        elif decision == CommitteeDecision.RISK_OFF:
            return ["Reduce positions 70%", "Disable entries", "Defensive only"]
        elif decision == CommitteeDecision.CAUTION:
            return ["Reduce size 30%", "Increase SL buffer", "Monitor"]
        return ["Normal operation"]

    def get_status(self) -> Dict:
        last = self.meeting_history[-1] if self.meeting_history else None
        return {
            "meetings": len(self.meeting_history),
            "last_decision": last.decision.value if last else None,
        }
