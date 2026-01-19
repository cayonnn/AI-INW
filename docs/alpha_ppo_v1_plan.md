# Alpha PPO V1 Implementation Plan

## Overview
Transition from Rule + XGBoost Alpha to PPO-based Alpha for autonomous entry/exit decisions.

---

## Phase 1: Environment Setup (Week 1)

### 1.1 Create AlphaGymEnv

```python
# src/rl/alpha_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AlphaGymEnv(gym.Env):
    """
    Gymnasium environment for Alpha PPO training.
    
    The agent learns to:
    - Identify profitable entry opportunities
    - Time exits for maximum profit
    - Avoid false signals
    
    Guardian layer is simulated during training
    to ensure realistic reward signals.
    """
    
    def __init__(self, data, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.data = data
        
        # Observation: 10-dimensional state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
        
        # Action: HOLD(0), BUY(1), SELL(2)
        self.action_space = spaces.Discrete(3)
        
        # Episode state
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.total_pnl = 0.0
        
    def _get_obs(self):
        """Build observation from current market state."""
        row = self.data.iloc[self.current_step]
        
        return np.array([
            row.get('ema20_norm', 0),
            row.get('ema50_norm', 0),
            row.get('ema_slope', 0),
            row.get('atr_norm', 0),
            row.get('atr_pct', 0),
            row.get('rsi_norm', 0),
            row.get('spread_norm', 0),
            self.position,
            row.get('hour_norm', 0),
            self.total_pnl / 1000  # Normalized P/L
        ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 50  # Skip warmup
        self.position = 0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        
        return self._get_obs(), {}
    
    def step(self, action):
        reward = 0.0
        row = self.data.iloc[self.current_step]
        price = row['close']
        
        # Execute action
        if action == 1 and self.position == 0:  # BUY
            self.position = 1
            self.entry_price = price
            reward -= 0.001  # Transaction cost
            
        elif action == 2 and self.position == 0:  # SELL
            self.position = -1
            self.entry_price = price
            reward -= 0.001
            
        elif action == 0 and self.position != 0:  # Close position
            pnl = (price - self.entry_price) * self.position
            self.total_pnl += pnl
            reward += pnl / 10  # Normalized reward
            self.position = 0
            self.entry_price = 0.0
        
        # Advance
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, done, truncated, {}
```

### 1.2 Reward Function

```python
def alpha_reward(trade_result, state):
    """
    Asymmetric reward optimized for competition metrics.
    
    Priorities:
    1. Profit maximization
    2. Drawdown minimization
    3. Consistency (Sharpe)
    """
    base = trade_result.profit_pnl / state.equity * 100
    
    # Amplify losses (asymmetric)
    if base < 0:
        base *= 1.5
    
    # Consistency bonus
    if state.trades_today > 0:
        win_rate = state.wins_today / state.trades_today
        if win_rate > 0.6:
            base += 0.05
    
    # Overtrading penalty
    if state.trades_today > 10:
        base -= 0.02 * (state.trades_today - 10)
    
    return base
```

---

## Phase 2: Training Pipeline (Week 2-3)

### 2.1 Data Preparation

```python
# training/prepare_alpha_data.py

def prepare_training_data(csv_path, lookback=100):
    """
    Prepare feature-rich dataset for Alpha PPO training.
    
    Features:
    - Technical indicators (EMA, ATR, RSI)
    - Normalized to [-1, 1] range
    - Includes labels from Triple Barrier
    """
    df = pd.read_csv(csv_path)
    
    # Calculate indicators
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['atr'] = calculate_atr(df, 14)
    df['rsi'] = calculate_rsi(df, 14)
    
    # Normalize
    # ...
    
    return df
```

### 2.2 Training Script

```python
# training/train_alpha_ppo.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_alpha_ppo(
    data_path: str,
    timesteps: int = 500_000,
    save_path: str = "models/alpha_ppo"
):
    """Train Alpha PPO model."""
    
    data = prepare_training_data(data_path)
    
    env = DummyVecEnv([lambda: AlphaGymEnv(data)])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/alpha_ppo_tb/"
    )
    
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    
    return model
```

---

## Phase 3: Integration (Week 4)

### 3.1 Alpha PPO Inference Module

```python
# src/rl/alpha_ppo_infer.py

class AlphaPPOAgent:
    """
    Alpha agent using trained PPO model.
    
    Replaces Rule + XGBoost strategy in live_loop_v3.py.
    """
    
    def __init__(self, model_path="models/alpha_ppo"):
        self.model = PPO.load(model_path)
        self.mode = "ppo"
    
    def predict(self, state: np.ndarray) -> tuple:
        """
        Get action and confidence.
        
        Returns:
            (action, confidence): 0=HOLD, 1=BUY, 2=SELL
        """
        action, _states = self.model.predict(state, deterministic=True)
        
        # Get action probability for confidence
        obs = state.reshape(1, -1)
        action_probs = self.model.policy.get_distribution(
            self.model.policy.obs_to_tensor(obs)[0]
        ).distribution.probs[0]
        
        confidence = float(action_probs[action])
        
        return int(action), confidence
```

### 3.2 Live Loop Integration

```python
# In live_loop_v3.py

# Replace this section:
# signal = signal_engine.get_signal(df)

# With:
if USE_ALPHA_PPO:
    from src.rl.alpha_ppo_infer import AlphaPPOAgent
    alpha_agent = AlphaPPOAgent("models/alpha_ppo")
    
    state = build_alpha_state(df, account)
    action, confidence = alpha_agent.predict(state)
    
    signal = {0: "HOLD", 1: "BUY", 2: "SELL"}[action]
    
    # Still pass through Guardian!
    guardian_result = guardian_hybrid.decide(state_dict, signal)
```

---

## Phase 4: Validation (Week 5-6)

### 4.1 Backtest Comparison

| Metric | Rule + XGB | Alpha PPO V1 | Target |
|--------|------------|--------------|--------|
| Total Return | ? | ? | +20% |
| Max Drawdown | ? | ? | <15% |
| Win Rate | ? | ? | >55% |
| Sharpe Ratio | ? | ? | >1.0 |

### 4.2 Paper Trading

1. Run both systems in parallel
2. Compare signal quality
3. Measure Guardian override rate

---

## Key Constraints

> [!WARNING]
> **Guardian Cannot Be Bypassed**
> 
> Even with Alpha PPO, the Guardian layer (Rule + PPO) 
> retains veto power over all trade proposals.

> [!IMPORTANT]
> **Gradual Rollout**
> 
> Week 1: Shadow mode (Alpha PPO suggests, Rule+XGB executes)
> Week 2: Split traffic (50/50)
> Week 3: Full Alpha PPO (with Guardian oversight)

---

## Success Criteria

- [ ] AlphaGymEnv passes sanity checks
- [ ] Training converges (reward > 0 after 100k steps)
- [ ] Backtest beats Rule+XGB by 5%+
- [ ] Paper trade: <20% Guardian override rate
- [ ] Live integration without crashes
