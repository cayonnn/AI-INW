# src/config/competition_full_live.py
"""
COMPETITION_FULL_LIVE Configuration
=====================================

Final Live Config for Competition + Real Trading.

Philosophy:
    - Alpha กล้าตัดสินใจ
    - Guardian ปกป้องทุน
    - ระบบเรียนรู้ต่อเนื่อง

Usage:
    python live_loop_v3.py --live --profile COMPETITION_FULL_LIVE
"""

# ================================
# TRADING PROFILE: COMPETITION
# ================================

PROFILE_NAME = "COMPETITION_FULL_LIVE"

SYMBOLS = ["XAUUSD"]
TIMEFRAME = "M5"
INTERVAL_SEC = 30
MAX_CYCLES = 1000

ACCOUNT_MODE = "DEMO"   # เปลี่ยนเป็น "REAL" เมื่อพร้อม

# ================================
# ALPHA PPO V1
# ================================

ALPHA_MODE = "LIVE"

ALPHA_MODEL_PATH = "models/alpha_ppo_v1_20260119_172015_FINAL.zip"

ALPHA_CONFIDENCE_MIN = 0.55
ALPHA_ACTION_SPACE = {
    0: "HOLD",
    1: "BUY",
    2: "SELL"
}

ALPHA_COOLDOWN_SEC = 60

# ================================
# GUARDIAN GOVERNANCE
# ================================

GUARDIAN_MODE = "HYBRID"

GUARDIAN_RULE_BASED = True
GUARDIAN_PPO_ENABLED = True

GUARDIAN_PPO_MODEL = "models/guardian_ppo_v3_20260115_1857.zip"
GUARDIAN_PPO_CONFIDENCE_MIN = 0.70

# ================================
# RISK MANAGEMENT
# ================================

RISK_PER_TRADE = 0.5        # %
MAX_OPEN_POSITIONS = 2

DAILY_DD_LIMIT = 0.10       # 10%
HARD_LATCH_ON_DD = True

FREE_MARGIN_MIN_RATIO = 0.02   # 2%
MARGIN_BUFFER_RATIO = 1.10     # 110%

KILL_SWITCH = {
    "MAX_DD": 0.03,        # 3% instant kill
    "MAX_ERRORS": 3
}

# ================================
# AUTO TUNER
# ================================

AUTO_TUNER_ENABLED = True

AUTO_TUNER_RULES = {
    "LOW_VOL": {
        "risk_mult": 0.8,
        "guardian_strict": 1.2
    },
    "HIGH_VOL": {
        "risk_mult": 1.2,
        "guardian_strict": 0.9
    }
}

AUTO_TUNER_INTERVAL_SEC = 300

# ================================
# AUTO TRAINING
# ================================

AUTO_TRAIN_ENABLED = True

TRAIN_DATA_PATH = "data/retrain"
MIN_TRADES_FOR_RETRAIN = 5

RETRAIN_TIME = "23:00"
AUTO_DEPLOY_NEW_MODEL = False   # Manual review first

# ================================
# LOGGING
# ================================

CSV_LOG_PATH = "logs/guardian_metrics.csv"

LOG_FIELDS = [
    "timestamp",
    "alpha_action",
    "guardian_action",
    "guardian_reason",
    "dd_current",
    "dd_avoided",
    "freeze_time",
    "equity",
    "balance"
]

# ================================
# DASHBOARD
# ================================

DASHBOARD_ENABLED = True
DASHBOARD_MODE = "LIVE"

STREAMLIT_PORT = 8501
SOCKET_PUSH_INTERVAL = 1.0


# ================================
# FULL CONFIG DICT
# ================================

COMPETITION_FULL_LIVE = {
    "profile": PROFILE_NAME,
    "symbols": SYMBOLS,
    "timeframe": TIMEFRAME,
    "interval": INTERVAL_SEC,
    "account_mode": ACCOUNT_MODE,
    
    "alpha": {
        "mode": ALPHA_MODE,
        "model_path": ALPHA_MODEL_PATH,
        "confidence_min": ALPHA_CONFIDENCE_MIN,
        "cooldown_sec": ALPHA_COOLDOWN_SEC,
    },
    
    "guardian": {
        "mode": GUARDIAN_MODE,
        "rule_based": GUARDIAN_RULE_BASED,
        "ppo_enabled": GUARDIAN_PPO_ENABLED,
        "ppo_model": GUARDIAN_PPO_MODEL,
        "confidence_min": GUARDIAN_PPO_CONFIDENCE_MIN,
    },
    
    "risk": {
        "per_trade": RISK_PER_TRADE,
        "max_positions": MAX_OPEN_POSITIONS,
        "daily_dd_limit": DAILY_DD_LIMIT,
        "hard_latch": HARD_LATCH_ON_DD,
        "kill_switch": KILL_SWITCH,
    },
    
    "auto_tuner": {
        "enabled": AUTO_TUNER_ENABLED,
        "rules": AUTO_TUNER_RULES,
        "interval": AUTO_TUNER_INTERVAL_SEC,
    },
    
    "auto_train": {
        "enabled": AUTO_TRAIN_ENABLED,
        "data_path": TRAIN_DATA_PATH,
        "min_trades": MIN_TRADES_FOR_RETRAIN,
        "retrain_time": RETRAIN_TIME,
        "auto_deploy": AUTO_DEPLOY_NEW_MODEL,
    },
    
    "logging": {
        "csv_path": CSV_LOG_PATH,
        "fields": LOG_FIELDS,
    },
    
    "dashboard": {
        "enabled": DASHBOARD_ENABLED,
        "mode": DASHBOARD_MODE,
        "port": STREAMLIT_PORT,
    }
}


def get_config():
    """Get full competition config."""
    return COMPETITION_FULL_LIVE


def print_config():
    """Print config summary."""
    print("=" * 60)
    print(f"🔥 {PROFILE_NAME}")
    print("=" * 60)
    print(f"""
MODE            = {ACCOUNT_MODE}
SYMBOLS         = {SYMBOLS}

ALPHA PPO       = {ALPHA_MODE}
CONFIDENCE      = {ALPHA_CONFIDENCE_MIN}

GUARDIAN        = {GUARDIAN_MODE}
RULE-BASED      = {'ON' if GUARDIAN_RULE_BASED else 'OFF'}
PPO             = {'ON' if GUARDIAN_PPO_ENABLED else 'OFF'}

RISK/TRADE      = {RISK_PER_TRADE}%
MAX POSITIONS   = {MAX_OPEN_POSITIONS}
DAILY DD LIMIT  = {DAILY_DD_LIMIT * 100}%

AUTO-TRAIN      = {'ON' if AUTO_TRAIN_ENABLED else 'OFF'}
AUTO-TUNER      = {'ON' if AUTO_TUNER_ENABLED else 'OFF'}
DASHBOARD       = {'ON' if DASHBOARD_ENABLED else 'OFF'}
    """)
    print("=" * 60)


if __name__ == "__main__":
    print_config()
