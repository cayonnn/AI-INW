# src/rl/alpha_curriculum.py
"""
Alpha PPO Curriculum Learning
=============================

Progressive training system for Alpha PPO.

Stages:
    0. Imitation: Learn from Rule + Guardian approved trades
    1. Shadow: No execution, learn Guardian patterns
    2. Hybrid: Limited authority, 50% lot size
    3. Adversarial: Train against Guardian logic
    4. Live: Micro-capital real execution

Paper Statement:
    "We employ curriculum learning to train our Alpha agent, progressing
    from behavioral cloning through shadow evaluation to controlled live
    trading, ensuring stable policy development and capital preservation."
"""

import os
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Any, Callable
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("ALPHA_CURRICULUM")


class TrainingStage(IntEnum):
    """Training curriculum stages."""
    IMITATION = 0
    SHADOW = 1
    HYBRID = 2
    ADVERSARIAL = 3
    LIVE = 4


@dataclass
class StageConfig:
    """Configuration for each training stage."""
    name: str
    stage: TrainingStage
    
    # Authority
    can_execute: bool = False
    max_confidence_required: float = 0.0  # 0 = no requirement
    lot_size_multiplier: float = 0.0  # 0 = no trading
    max_positions: int = 0
    
    # Training params
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    
    # Reward weights
    reward_profit: float = 0.1
    reward_guardian_approve: float = 0.2
    reward_guardian_block: float = -0.2
    reward_dd_avoided: float = 0.3
    reward_overtrade: float = -0.1
    reward_hold_in_risk: float = 0.05
    
    # Promotion criteria
    min_episodes: int = 100
    min_win_rate: float = 0.50
    max_dd: float = 0.15
    min_guardian_compliance: float = 0.70


# =============================================================================
# Stage Configurations
# =============================================================================

CURRICULUM_STAGES: Dict[TrainingStage, StageConfig] = {
    TrainingStage.IMITATION: StageConfig(
        name="Imitation Learning",
        stage=TrainingStage.IMITATION,
        can_execute=False,
        total_timesteps=50_000,
        learning_rate=1e-3,
        batch_size=128,
        reward_profit=0.0,  # No profit reward in imitation
        reward_guardian_approve=0.3,  # High reward for matching approved trades
        min_episodes=50,
        min_win_rate=0.0,  # No win rate requirement
    ),
    
    TrainingStage.SHADOW: StageConfig(
        name="Shadow PPO",
        stage=TrainingStage.SHADOW,
        can_execute=False,
        total_timesteps=200_000,
        learning_rate=3e-4,
        reward_profit=0.1,
        reward_guardian_approve=0.2,
        reward_guardian_block=-0.2,
        reward_dd_avoided=0.3,
        min_episodes=200,
        min_win_rate=0.48,
        min_guardian_compliance=0.60,
    ),
    
    TrainingStage.HYBRID: StageConfig(
        name="Hybrid PPO",
        stage=TrainingStage.HYBRID,
        can_execute=True,
        max_confidence_required=0.60,
        lot_size_multiplier=0.5,  # 50% of normal lot
        max_positions=1,
        total_timesteps=300_000,
        learning_rate=1e-4,
        reward_profit=0.15,
        reward_guardian_approve=0.15,
        reward_guardian_block=-0.25,
        reward_dd_avoided=0.25,
        min_episodes=300,
        min_win_rate=0.52,
        max_dd=0.10,
        min_guardian_compliance=0.70,
    ),
    
    TrainingStage.ADVERSARIAL: StageConfig(
        name="Adversarial PPO",
        stage=TrainingStage.ADVERSARIAL,
        can_execute=True,
        max_confidence_required=0.55,
        lot_size_multiplier=0.75,
        max_positions=2,
        total_timesteps=500_000,
        learning_rate=5e-5,
        reward_profit=0.15,
        reward_guardian_approve=0.1,
        reward_guardian_block=-0.3,  # Higher penalty
        reward_dd_avoided=0.35,
        min_episodes=500,
        min_win_rate=0.54,
        max_dd=0.08,
        min_guardian_compliance=0.80,
    ),
    
    TrainingStage.LIVE: StageConfig(
        name="Live Micro-Capital",
        stage=TrainingStage.LIVE,
        can_execute=True,
        max_confidence_required=0.50,
        lot_size_multiplier=1.0,
        max_positions=3,
        total_timesteps=100_000,  # Fewer steps, real market
        learning_rate=1e-5,  # Very slow learning
        reward_profit=0.2,
        reward_guardian_approve=0.1,
        reward_guardian_block=-0.2,
        reward_dd_avoided=0.3,
        min_episodes=100,
        min_win_rate=0.55,
        max_dd=0.05,  # Very strict
        min_guardian_compliance=0.85,
    ),
}


@dataclass
class StageMetrics:
    """Metrics tracked for each stage."""
    stage: TrainingStage
    episodes: int = 0
    total_reward: float = 0.0
    win_rate: float = 0.0
    max_dd: float = 0.0
    guardian_compliance: float = 0.0
    avg_confidence: float = 0.0
    
    # Detailed stats
    trades_executed: int = 0
    trades_blocked: int = 0
    dd_avoided_count: int = 0
    overtrade_count: int = 0
    
    def update(self, episode_result: Dict):
        """Update metrics from episode result."""
        self.episodes += 1
        self.total_reward += episode_result.get("total_reward", 0)
        
        # Running averages
        alpha = 0.1
        self.win_rate = (1 - alpha) * self.win_rate + alpha * episode_result.get("win_rate", 0.5)
        self.max_dd = max(self.max_dd, episode_result.get("max_dd", 0))
        self.guardian_compliance = (
            (1 - alpha) * self.guardian_compliance + 
            alpha * episode_result.get("guardian_compliance", 1.0)
        )
        self.avg_confidence = (
            (1 - alpha) * self.avg_confidence +
            alpha * episode_result.get("avg_confidence", 0.5)
        )
        
        # Counters
        self.trades_executed += episode_result.get("trades", 0)
        self.trades_blocked += episode_result.get("blocks", 0)
        self.dd_avoided_count += episode_result.get("dd_avoided", 0)
        self.overtrade_count += episode_result.get("overtrades", 0)
    
    def check_promotion(self, config: StageConfig) -> tuple:
        """
        Check if stage criteria are met for promotion.
        
        Returns:
            (ready, blockers) - ready=True if can promote
        """
        blockers = []
        
        if self.episodes < config.min_episodes:
            blockers.append(f"Episodes: {self.episodes}/{config.min_episodes}")
        
        if self.win_rate < config.min_win_rate:
            blockers.append(f"Win rate: {self.win_rate:.1%}/{config.min_win_rate:.1%}")
        
        if self.max_dd > config.max_dd:
            blockers.append(f"Max DD: {self.max_dd:.1%} > {config.max_dd:.1%}")
        
        if self.guardian_compliance < config.min_guardian_compliance:
            blockers.append(
                f"Guardian compliance: {self.guardian_compliance:.1%}/"
                f"{config.min_guardian_compliance:.1%}"
            )
        
        return len(blockers) == 0, blockers


class CurriculumManager:
    """
    Manages curriculum progression for Alpha PPO training.
    
    Tracks metrics, handles promotion between stages,
    and provides stage-specific configurations.
    """
    
    def __init__(self, starting_stage: TrainingStage = TrainingStage.IMITATION):
        self.current_stage = starting_stage
        self.metrics: Dict[TrainingStage, StageMetrics] = {}
        self.stage_history: List[Dict] = []
        
        # Initialize metrics for all stages
        for stage in TrainingStage:
            self.metrics[stage] = StageMetrics(stage=stage)
        
        logger.info(f"📚 Curriculum initialized at stage: {starting_stage.name}")
    
    @property
    def config(self) -> StageConfig:
        """Get current stage configuration."""
        return CURRICULUM_STAGES[self.current_stage]
    
    @property
    def stage_name(self) -> str:
        """Get current stage name."""
        return self.config.name
    
    def get_reward_function(self) -> Callable:
        """
        Get reward function for current stage.
        
        Returns a function that calculates reward from step info.
        """
        cfg = self.config
        
        def reward_fn(info: Dict) -> float:
            reward = 0.0
            
            # Profit component
            profit = info.get("profit", 0)
            if profit > 0:
                reward += profit * cfg.reward_profit
            elif profit < 0:
                reward += profit * cfg.reward_profit * 1.5  # Penalize losses more
            
            # Guardian components
            if info.get("guardian_approved", False):
                reward += cfg.reward_guardian_approve
            if info.get("guardian_blocked", False):
                reward += cfg.reward_guardian_block
            if info.get("dd_avoided", False):
                reward += cfg.reward_dd_avoided
            
            # Behavioral components
            if info.get("overtrade", False):
                reward += cfg.reward_overtrade
            if info.get("hold_in_risk", False):
                reward += cfg.reward_hold_in_risk
            
            return reward
        
        return reward_fn
    
    def update_metrics(self, episode_result: Dict):
        """Update current stage metrics."""
        self.metrics[self.current_stage].update(episode_result)
    
    def check_promotion(self) -> tuple:
        """
        Check if ready to promote to next stage.
        
        Returns:
            (ready, next_stage, blockers)
        """
        if self.current_stage == TrainingStage.LIVE:
            return False, None, ["Already at final stage"]
        
        metrics = self.metrics[self.current_stage]
        config = self.config
        
        ready, blockers = metrics.check_promotion(config)
        
        next_stage = TrainingStage(self.current_stage + 1) if ready else None
        
        return ready, next_stage, blockers
    
    def promote(self) -> bool:
        """
        Promote to next stage if criteria are met.
        
        Returns:
            True if promotion occurred
        """
        ready, next_stage, blockers = self.check_promotion()
        
        if not ready:
            logger.warning(f"❌ Cannot promote. Blockers: {blockers}")
            return False
        
        # Record history
        self.stage_history.append({
            "from_stage": self.current_stage.name,
            "to_stage": next_stage.name,
            "metrics": self.get_metrics_summary(),
            "timestamp": None  # Will be set by caller
        })
        
        self.current_stage = next_stage
        logger.info(f"🎓 PROMOTED to stage: {next_stage.name}")
        
        return True
    
    def force_stage(self, stage: TrainingStage):
        """Force set stage (for testing or rollback)."""
        old_stage = self.current_stage
        self.current_stage = stage
        logger.warning(f"⚠️ Stage forced: {old_stage.name} → {stage.name}")
    
    def rollback(self):
        """Rollback to previous stage (safety mechanism)."""
        if self.current_stage > TrainingStage.SHADOW:
            old_stage = self.current_stage
            self.current_stage = TrainingStage(self.current_stage - 1)
            logger.warning(f"⏪ ROLLBACK: {old_stage.name} → {self.current_stage.name}")
            return True
        return False
    
    def get_metrics_summary(self) -> Dict:
        """Get current stage metrics summary."""
        m = self.metrics[self.current_stage]
        cfg = self.config
        
        return {
            "stage": self.current_stage.name,
            "episodes": m.episodes,
            "win_rate": f"{m.win_rate:.1%}",
            "max_dd": f"{m.max_dd:.1%}",
            "guardian_compliance": f"{m.guardian_compliance:.1%}",
            "avg_confidence": f"{m.avg_confidence:.1%}",
            "trades": m.trades_executed,
            "blocks": m.trades_blocked,
            "dd_avoided": m.dd_avoided_count,
            "promotion_ready": m.check_promotion(cfg)[0]
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        m = self.metrics[self.current_stage]
        ready, _, blockers = self.check_promotion()
        
        return (
            f"📚 Stage: {self.stage_name} | "
            f"Episodes: {m.episodes} | "
            f"WR: {m.win_rate:.0%} | "
            f"DD: {m.max_dd:.1%} | "
            f"Guardian: {m.guardian_compliance:.0%} | "
            f"{'✅ Ready' if ready else '❌ ' + str(len(blockers)) + ' blockers'}"
        )


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha PPO Curriculum Test")
    print("=" * 60)
    
    curriculum = CurriculumManager(starting_stage=TrainingStage.IMITATION)
    
    print(f"\nStarting stage: {curriculum.stage_name}")
    print(f"Config: {curriculum.config}")
    
    # Simulate training episodes
    for episode in range(60):
        result = {
            "total_reward": np.random.uniform(0, 2),
            "win_rate": 0.48 + episode * 0.001,
            "max_dd": 0.05 + np.random.uniform(0, 0.02),
            "guardian_compliance": 0.55 + episode * 0.005,
            "avg_confidence": 0.6,
            "trades": np.random.randint(0, 5),
            "blocks": np.random.randint(0, 2),
            "dd_avoided": np.random.randint(0, 2),
            "overtrades": np.random.randint(0, 1)
        }
        
        curriculum.update_metrics(result)
        
        if episode % 20 == 0:
            print(f"\nEpisode {episode}: {curriculum.summary()}")
    
    # Check promotion
    ready, next_stage, blockers = curriculum.check_promotion()
    print(f"\n--- Promotion Check ---")
    print(f"Ready: {ready}")
    print(f"Blockers: {blockers}")
    
    if ready:
        curriculum.promote()
        print(f"\nNew stage: {curriculum.stage_name}")
    
    print("\n" + "=" * 60)
