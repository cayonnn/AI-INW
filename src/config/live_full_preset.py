# src/config/live_full_preset.py
"""
LIVE_FULL Preset
=================

Full capability live trading preset.
All intelligence enabled, all safety active.

"เปิดสมองเต็ม แต่ยังไม่เผาบัญชี"
"""

# =============================================================================
# LIVE_FULL PRESET
# =============================================================================

LIVE_FULL = {
    "name": "LIVE_FULL",
    "description": "Full capability live trading",
    "account": "MT5_DEMO",
    
    # --- Alpha PPO ---
    "alpha": {
        "enabled": True,
        "confidence_min": 0.55,
        "model_path": "models/alpha_ppo_v1_FINAL.zip",
    },
    
    # --- Risk ---
    "risk": {
        "per_trade": 0.5,         # 0.5%
        "max_positions": 2,
        "daily_dd_limit": 10.0,   # 10%
        "hard_latch": True,
    },
    
    # --- Guardian ---
    "guardian": {
        "rule_based": True,
        "ppo_enabled": True,
        "hard_override": True,
    },
    
    # --- Intelligence ---
    "auto_train": True,
    "auto_tuner": True,
    "kill_switch": True,
    
    # --- Monitoring ---
    "dashboard": True,
    "decision_log": True,
}


def get_live_full_config():
    """Get LIVE_FULL configuration."""
    return LIVE_FULL.copy()


def print_config():
    """Print configuration summary."""
    print("=" * 60)
    print("🔥 LIVE_FULL PRESET")
    print("=" * 60)
    print(f"""
MODE                  = LIVE_FULL
ACCOUNT               = MT5_DEMO

Alpha PPO             = ENABLED
Alpha confidence min  = {LIVE_FULL['alpha']['confidence_min']}

Risk per trade        = {LIVE_FULL['risk']['per_trade']}%
Max positions         = {LIVE_FULL['risk']['max_positions']}

Guardian Rule         = {'ON' if LIVE_FULL['guardian']['rule_based'] else 'OFF'}
Guardian PPO          = {'ON' if LIVE_FULL['guardian']['ppo_enabled'] else 'OFF'}

Daily DD limit        = {LIVE_FULL['risk']['daily_dd_limit']}%
Hard latch            = {'ON' if LIVE_FULL['risk']['hard_latch'] else 'OFF'}

Auto-Train            = {'ON' if LIVE_FULL['auto_train'] else 'OFF'}
Auto-Tuner            = {'ON' if LIVE_FULL['auto_tuner'] else 'OFF'}
    """)
    print("=" * 60)


if __name__ == "__main__":
    print_config()
