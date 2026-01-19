# 🚀 Alpha PPO V1 Roadmap

> **Evolution: From Rule-Based Alpha to RL-Powered Alpha**

---

## 🎯 Goal

Replace the current **Rule + XGBoost Alpha** with a **PPO-based Alpha Agent** that can learn to trade directly from market data, while maintaining Guardian oversight.

---

## 📊 Current State vs Target State

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT (Hybrid Alpha)                    │
├─────────────────────────────────────────────────────────────┤
│  Rule Strategy (EMA)  →  XGBoost Imitation  →  Candidate    │
│      (Hardcoded)            (Supervised)        (Passive)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
                          EVOLUTION
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    TARGET (PPO Alpha V1)                     │
├─────────────────────────────────────────────────────────────┤
│       PPO Agent (Learns Profit)  →  Direct Decision         │
│         (Reinforcement)               (Active)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Alpha PPO V1 Design

### Observation Space (10-dim)

```python
observation = [
    # Technical Indicators
    ema20,           # EMA 20 normalized
    ema50,           # EMA 50 normalized
    ema_slope,       # Rate of EMA change
    atr,             # ATR (volatility)
    atr_pct,         # ATR as % of price
    rsi,             # RSI (momentum)
    
    # Market Context
    spread,          # Current spread
    time_of_day,     # Trading session (0-1)
    market_regime,   # Trend/Range/Volatile (encoded)
    
    # Account State
    open_positions,  # Current position count
]
```

### Action Space (3 discrete actions)

```python
class AlphaAction(Enum):
    HOLD = 0   # No trade
    BUY = 1    # Open long position
    SELL = 2   # Open short position
```

### Reward Function (Competition-Optimized)

```python
def compute_reward(trade_result, dd_snapshot, prev_dd):
    """
    Competition-style reward: Maximize profit, minimize drawdown.
    """
    reward = 0.0
    
    # 1. Profit Component (+)
    if trade_result.profit > 0:
        reward += trade_result.profit * 0.1  # Scaled
    else:
        reward += trade_result.profit * 0.15  # Penalize losses more
    
    # 2. Drawdown Penalty (-)
    dd_increase = max(0, dd_snapshot - prev_dd)
    reward -= dd_increase * 5.0  # Heavy penalty
    
    # 3. Overtrading Penalty (-)
    if trade_count_today > 10:
        reward -= 0.5  # Discourage excessive trading
    
    # 4. Consistency Bonus (+)
    if win_rate_7d > 0.55:
        reward += 0.2
    
    return reward
```

---

## 🏗️ Architecture with Guardian

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ALPHA PPO V1 SYSTEM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐                                            │
│  │   PPO Alpha Agent    │  ← Learns trading strategy                 │
│  │   (Profit Seeker)    │                                            │
│  └──────────┬───────────┘                                            │
│             ↓                                                        │
│       BUY / SELL / HOLD                                              │
│             ↓                                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    GUARDIAN LAYER                             │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  Rule Guardian (CANNOT BE BYPASSED)                     │  │    │
│  │  │  • Daily DD Limit                                       │  │    │
│  │  │  • Margin Check                                         │  │    │
│  │  │  • Error Streak                                         │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  PPO Guardian (Advisory)                                 │  │    │
│  │  │  • High-confidence override                             │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────┘    │
│             ↓                                                        │
│       EXECUTE / BLOCK                                                │
│             ↓                                                        │
│  ┌──────────────────────┐                                            │
│  │     MT5 Execution    │                                            │
│  └──────────────────────┘                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

> **Important**: Alpha PPO cannot bypass Guardian. Guardian = Risk Manager, Alpha = Trader.

---

## 🧪 Training Setup

### Environment: `AlphaTradingEnv`

```python
class AlphaTradingEnv(gym.Env):
    """
    Trading environment for Alpha PPO training.
    """
    
    def __init__(self, df, initial_balance=1000):
        self.df = df
        self.balance = initial_balance
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
    
    def step(self, action):
        # Execute action
        # Calculate reward
        # Check done condition
        return obs, reward, done, truncated, info
    
    def reset(self):
        # Reset to initial state
        return obs, info
```

### Training Script

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = DummyVecEnv([lambda: AlphaTradingEnv(df_train)])

# Train PPO
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=500_000)
model.save("models/alpha_ppo_v1")
```

---

## 📅 Implementation Phases

### Phase 1: Environment Design (1 week)
- [ ] Define observation space
- [ ] Define action space
- [ ] Implement reward function
- [ ] Create `AlphaTradingEnv` class

### Phase 2: Training Pipeline (1 week)
- [ ] Historical data preparation
- [ ] Cross-validation setup
- [ ] Training loop with checkpoints
- [ ] TensorBoard logging

### Phase 3: Integration (3 days)
- [ ] Create `src/rl/alpha_ppo_infer.py`
- [ ] Integrate into `live_loop_v3.py`
- [ ] Shadow mode testing

### Phase 4: Validation (1 week)
- [ ] Backtesting on held-out data
- [ ] Paper trading (sandbox mode)
- [ ] Compare vs current Rule+XGB

---

## 📊 Evolution Roadmap

| Stage | Alpha | Guardian | Status |
|-------|-------|----------|--------|
| **V1 (Legacy)** | Rule + XGB | Rule-based only | ✅ Passed |
| **V2 (Current)** | Rule + XGB | Rule + PPO (Hybrid) | ✅ Active |
| **V2.5 (Shadow)** | PPO Alpha (Shadow Mode) | Rule + PPO | 🔄 In Progress |
| **V3 (Target)** | PPO Alpha (Live) | PPO Guardian | 📅 Pending Promotion |
| **Fund-Grade** | Portfolio PPO | Central Risk Brain | 🎯 Vision |

> **Current Status**: V2 Hybrid Guardian with Alpha PPO V1 running in Shadow Mode for evaluation.

---

## 🔒 Safety Constraints

Even with PPO Alpha, these HARD rules remain:

```python
# Guardian CANNOT be bypassed by Alpha PPO
GUARDIAN_RULES = [
    "Daily DD >= 10% → LATCH all trading",
    "Free Margin < 5% equity → BLOCK",
    "3+ consecutive blocks → FORCE_HOLD 5 min",
    "Account equity < $50 → EMERGENCY_FREEZE"
]
```

---

## 🏆 Paper Statement (V1 Complete)

> *"Our evolved system replaces rule-based signal generation with a reinforcement learning agent trained via Proximal Policy Optimization. The PPO Alpha learns directly from market dynamics while a hierarchical Guardian maintains capital preservation constraints, demonstrating safe RL deployment in live financial markets."*

---

## 📁 Planned Files

| File | Purpose |
|------|---------|
| `src/rl/alpha_ppo_env.py` | Training environment |
| `src/rl/alpha_ppo_train.py` | Training script |
| `src/rl/alpha_ppo_infer.py` | Inference for live trading |
| `models/alpha_ppo_v1.zip` | Trained model |
| `training/alpha_ppo/` | Training logs & checkpoints |
