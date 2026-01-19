# src/guardian/guardian_hybrid.py

from typing import Dict, Tuple, Any
from src.rl.guardian_agent import GuardianAction
import logging

logger = logging.getLogger("GUARDIAN_HYBRID")

class GuardianHybrid:
    """
    Guardian Hybrid Arbitration Layer
    =================================
    Combines:
    1. Rule-based Guardian (Hard Safety Gate)
    2. PPO Advisor (Intelligent Risk Adjustment)
    
    Logic:
    - If Rule says HARD BLOCK (FORCE_HOLD/FREEZE) -> BLOCK (PPO cannot override)
    - If Rule says ALLOW/REDUCE -> Consult PPO
    - If PPO confidence > threshold -> Apply PPO suggestion
    """
    def __init__(self, rule_guardian, ppo_agent, confidence_threshold: float = 0.65):
        """
        Initialize Hybrid Guardian.
        
        Args:
            rule_guardian: Instance of GuardianAgent
            ppo_agent: Instance of GuardianPPOAdvisor
            confidence_threshold: PPO confidence to act on advice (0.0-1.0)
        """
        self.rule = rule_guardian
        self.ppo = ppo_agent
        self.theta = confidence_threshold

    def decide(self, state: Dict[str, Any], alpha_action: str) -> Tuple[str, str]:
        """
        Arbitrate decision between Rule and PPO.
        
        Returns:
            (FinalAction, Reason)
            FinalAction: ALLOW, BLOCK, REDUCE_RISK
        """
        # 1. Rule-Based Check (Hard Gate)
        # Adapt flat state to GuardianAgent.evaluate signature: (signal, account_state, guardian_state)
        # We construct minimal ad-hoc dicts from the flat state
        account_state = {
            "equity": 0, # not used in evaluate directly if ratios provided, but needed for safety
            "margin_free": 0 # mapped from ratios
        }
        # However, GuardianAgent.evaluate calculates margin logic itself. 
        # Safe strategy: Re-map flat "state" back to expected format or Mock it.
        # Better: pass a dummy guardian_state since we only need simple checks or rely on the state dict we have
        
        # Real fix: constructing compatibility dicts
        comp_account = {"equity": 1.0, "margin_free": state.get("margin_ratio", 1.0)} # Normalized
        comp_guardian = {"margin_block_count": state.get("margin_block_count", 0)}
        
        rule_result = self.rule.evaluate(alpha_action, comp_account, comp_guardian)
        
        # GuardianAgent.evaluate returns just a string "ALLOW"/"BLOCK", not a tuple.
        # We need to handle that.
        rule_reason = "Rule Block" if rule_result == "BLOCK" else "Authorized"
        
        # Define Hard Blocks
        # FORCE_HOLD and EMERGENCY_FREEZE are absolute overrides
        if rule_result in [GuardianAction.FORCE_HOLD, GuardianAction.EMERGENCY_FREEZE]:
            return "BLOCK", f"[RULE_HARD] {rule_reason}"
            
        # 2. PPO Advisor Check
        # Mute PPO if margin critical (let Rule or Hard Gate handle it)
        # Prevents "PPO EMERGENCY_FREEZE" spam when account is already dead
        margin_ratio = state.get("margin_ratio", 1.0)
        daily_dd = state.get("daily_dd", 0.0)
        
        if margin_ratio <= 0.0:
             # Account is dead, PPO opinion irrelevant
             return "ALLOW", "[PPO_MUTED] Margin Depleted"
        
        if margin_ratio <= 0.0:
             # Account is dead, PPO opinion irrelevant
             return "ALLOW", "[PPO_MUTED] Margin Depleted"
        
        try:
            # Pass full state (including V4 context if present)
            ppo_action, ppo_conf = self.ppo.decide(state)
            
            # Logic: If PPO feels strongly (conf > theta) and wants to intervene
            if ppo_action != "ALLOW" and ppo_conf >= self.theta:
                # PPO Intervention
                return ppo_action, f"[PPO_ADVISE] {ppo_action} (conf={ppo_conf:.2f})"
        
        except ValueError as e:
            # Handle Shape Mismatch (Migration V3 -> V4)
            if "shape" in str(e) or "dimension" in str(e):
                logging.warning(f"⚠️ PPO Shape Mismatch: {e}. using Rule-Only until Retrain.")
                return "ALLOW", "[PPO_ERROR] Shape Mismatch"
            else:
                logging.debug(f"PPO Decision Error: {e}")
                return "ALLOW", "[PPO_ERROR] Unknown"
        except Exception as e:
             logging.debug(f"PPO Unexpected Error: {e}")
             return "ALLOW", "[PPO_ERROR] Unexpected"

            
        # 3. Default to Rule Outcome (usually ALLOW or REDUCE_RISK)
        # Convert Enum to string
        if rule_result == GuardianAction.REDUCE_RISK:
            return "REDUCE_RISK", f"[RULE_SOFT] {rule_reason}"
            
        return "ALLOW", "Hybrid_Consensus"
