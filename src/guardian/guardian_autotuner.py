# src/guardian/guardian_autotuner.py

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("GUARDIAN_TUNER")

class GuardianAutoTuner:
    """
    Guardian Auto-Tuner (Competition Grade)
    =======================================
    Dynamically adjusts risk governance thresholds based on live performance.
    
    States:
    - 🟢 AGGRESSIVE (Winning): Relax limits to maximize score
    - 🟡 NEUTRAL (Stable): Standard competition settings
    - 🔴 DEFENSIVE (Losing): Tighten limits to survive
    
    Metrics Monitored:
    - Win Rate (last N trades)
    - Average R-Multiple
    - Rolling Drawdown
    - Guardian Block Rate
    """
    
    def __init__(self):
        self.state = "NEUTRAL"
        self.last_update_cycle = 0
        self.update_interval = 10  # Tune every 10 cycles (approx 5 min)
        
        # Default Competition Configs
        self.config = {
            "daily_dd_limit": 0.14,      # Start at 14%
            "margin_buffer_pct": 2.0,    # Block if free margin < 2%
            "ppo_confidence": 0.65,      # Standard confidence
            "progressive_l1": 0.04,      # 4% WARN
            "progressive_l2": 0.06,      # 6% SCALE DOWN
            "progressive_l3": 0.08,      # 8% FREEZE
            "progressive_l4": 0.10,      # 10% KILL
        }

    def tune(self, metrics: Dict[str, float], cycle: int, profile: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run auto-tuning logic with Competition Envelope.
        
        Args:
            metrics: Live metrics (win_rate, equity, etc.)
            cycle: Current loop cycle
            profile: TradingProfile object (Source of Truth for Envelope)
            
        Returns:
            New configuration dict
        """
        win_rate = metrics.get("win_rate", 0.5)
        avg_r = metrics.get("avg_r", 0.0)
        dd = metrics.get("current_dd", 0.0)
        block_rate = metrics.get("block_rate", 0.0)
        equity = metrics.get("equity", 1000.0)
        
        # 1. SURVIVAL SAFETY FLOOR
        if equity < 100:
            if self.state != "SURVIVAL":
                 logger.critical(f"🎛️ Auto-Tuner: SURVIVAL MODE ACTIVATED (Equity ${equity:.2f} < $100)")
            self.state = "SURVIVAL"
            return {
                "daily_dd_limit": 0.05,
                "margin_buffer_pct": 10.0,
                "ppo_confidence": 0.95,
                "progressive_l1": 0.02,
                "progressive_l2": 0.03,
                "progressive_l3": 0.05,
                "progressive_l4": 0.07,
                "allow_trading": False
            }
            
        # 2. DETERMINE STATE
        if win_rate < 0.45 or dd > 0.12 or block_rate > 0.30:
            new_state = "DEFENSIVE"
        elif win_rate > 0.55 and avg_r > 0.4 and dd < 0.08:
            new_state = "AGGRESSIVE"
        else:
            new_state = "NEUTRAL"
            
        # 3. APPLY CONFIG BY STATE
        if new_state == "DEFENSIVE":
            self.config.update({
                "daily_dd_limit": 0.12,
                "margin_buffer_pct": 5.0,
                "ppo_confidence": 0.75,
                "progressive_l1": 0.03,
                "progressive_l2": 0.05,
                "progressive_l3": 0.07,
                "progressive_l4": 0.09,
            })
            
        elif new_state == "AGGRESSIVE":
            target_limit = 0.16
            target_margin = 1.5
            
            # 🟢 Check Governance Envelope (Constitution)
            if profile and hasattr(profile, "envelope"):
                # 1. Max DD Limit
                env_min_dd, env_max_dd = profile.envelope.max_daily_loss_range
                env_max_dd = env_max_dd / 100.0
                if target_limit > env_max_dd:
                    logger.warning(f"🎛️ Auto-Tuner: Aggressive MaxDD ({target_limit*100}%) clamped by Envelope ({env_max_dd*100}%)")
                    target_limit = env_max_dd

                # 2. Mix Margin Buffer
                env_min_margin, env_max_margin = profile.envelope.margin_buffer_range
                if target_margin < env_min_margin:
                     logger.warning(f"🎛️ Auto-Tuner: Aggressive Margin ({target_margin}%) clamped by Envelope ({env_min_margin}%)")
                     target_margin = env_min_margin

            self.config.update({
                "daily_dd_limit": target_limit,
                "margin_buffer_pct": target_margin,
                "ppo_confidence": 0.60,
                "progressive_l1": 0.05,
                "progressive_l2": 0.07,
                "progressive_l3": 0.09,
                "progressive_l4": 0.12,
            })
            
        else: # NEUTRAL
            base_limit = 0.14
            if profile: 
                base_limit = profile.risk.max_daily_loss / 100.0
                
            self.config.update({
                "daily_dd_limit": base_limit,
                "margin_buffer_pct": 2.0,
                "ppo_confidence": 0.65,
                "progressive_l1": 0.04,
                "progressive_l2": 0.06,
                "progressive_l3": 0.08,
                "progressive_l4": 0.10,
            })

        if new_state != self.state:
            logger.info(f"🎛️ Auto-Tuner: Switched to {new_state} Mode")
            self.state = new_state
            
        return self.config
