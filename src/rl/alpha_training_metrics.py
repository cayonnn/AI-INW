# src/rl/alpha_training_metrics.py
"""
Alpha PPO Training Metrics
==========================

Metrics collection and visualization for training progress.

Features:
    - Stage-wise metrics tracking
    - Reward curve plotting
    - Guardian interaction analysis
    - Paper-ready figures
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("TRAINING_METRICS")


@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""
    timestamp: str
    stage: int
    episode: int
    
    # Performance
    total_reward: float
    profit: float
    trades: int
    wins: int
    
    # Risk
    max_dd: float
    avg_dd: float
    
    # Guardian interaction
    guardian_approvals: int
    guardian_blocks: int
    dd_avoided: int
    
    # Alpha behavior
    confidence_mean: float
    confidence_std: float
    hold_ratio: float


class TrainingMetricsCollector:
    """
    Collects and stores training metrics for analysis.
    """
    
    def __init__(self, log_dir: str = "logs/alpha_training"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics_file = f"{log_dir}/training_metrics.csv"
        self.summary_file = f"{log_dir}/training_summary.json"
        
        self.episodes: List[EpisodeMetrics] = []
        self.stage_summaries: Dict[int, Dict] = {}
        
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not Path(self.metrics_file).exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "stage", "episode",
                    "total_reward", "profit", "trades", "wins",
                    "max_dd", "avg_dd",
                    "guardian_approvals", "guardian_blocks", "dd_avoided",
                    "confidence_mean", "confidence_std", "hold_ratio"
                ])
    
    def log_episode(self, metrics: EpisodeMetrics):
        """Log a single episode's metrics."""
        self.episodes.append(metrics)
        
        # Append to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.timestamp, metrics.stage, metrics.episode,
                f"{metrics.total_reward:.4f}", f"{metrics.profit:.4f}",
                metrics.trades, metrics.wins,
                f"{metrics.max_dd:.4f}", f"{metrics.avg_dd:.4f}",
                metrics.guardian_approvals, metrics.guardian_blocks, metrics.dd_avoided,
                f"{metrics.confidence_mean:.4f}", f"{metrics.confidence_std:.4f}",
                f"{metrics.hold_ratio:.4f}"
            ])
    
    def get_stage_summary(self, stage: int) -> Dict:
        """Get summary statistics for a stage."""
        stage_episodes = [e for e in self.episodes if e.stage == stage]
        
        if not stage_episodes:
            return {}
        
        rewards = [e.total_reward for e in stage_episodes]
        profits = [e.profit for e in stage_episodes]
        wins = sum(e.wins for e in stage_episodes)
        trades = sum(e.trades for e in stage_episodes)
        
        return {
            "stage": stage,
            "episodes": len(stage_episodes),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "total_profit": sum(profits),
            "win_rate": wins / max(trades, 1),
            "avg_trades_per_episode": trades / len(stage_episodes),
            "guardian_block_rate": (
                sum(e.guardian_blocks for e in stage_episodes) /
                max(sum(e.guardian_blocks + e.guardian_approvals for e in stage_episodes), 1)
            ),
            "dd_avoided_total": sum(e.dd_avoided for e in stage_episodes)
        }
    
    def save_summary(self):
        """Save training summary to JSON."""
        all_stage_summaries = {}
        for stage in set(e.stage for e in self.episodes):
            all_stage_summaries[stage] = self.get_stage_summary(stage)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_episodes": len(self.episodes),
            "stages": all_stage_summaries
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Training summary saved to {self.summary_file}")
    
    def generate_report(self) -> str:
        """Generate text report for paper."""
        lines = [
            "=" * 60,
            "📊 ALPHA PPO TRAINING REPORT",
            "=" * 60,
            f"Total Episodes: {len(self.episodes)}",
            ""
        ]
        
        for stage in sorted(set(e.stage for e in self.episodes)):
            summary = self.get_stage_summary(stage)
            lines.append(f"--- Stage {stage} ---")
            lines.append(f"  Episodes: {summary['episodes']}")
            lines.append(f"  Avg Reward: {summary['avg_reward']:.3f}")
            lines.append(f"  Win Rate: {summary['win_rate']:.1%}")
            lines.append(f"  Guardian Block Rate: {summary['guardian_block_rate']:.1%}")
            lines.append(f"  DD Avoided: {summary['dd_avoided_total']}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def paper_figures_guide() -> str:
    """Guide for generating paper figures."""
    return """
📊 PAPER FIGURES GUIDE (Alpha PPO Training)
============================================

Required Figures:

1. Figure: Reward Curve by Stage
   - X: Episode
   - Y: Cumulative Reward
   - Lines: Stage 0, 1, 2, 3 (different colors)
   - File: logs/alpha_training/training_metrics.csv

2. Figure: Alpha vs Rule Win Rate
   - X: Training Episode (or Time)
   - Y: Win Rate (%)
   - Lines: Alpha PPO, Rule Baseline
   - Show improvement over training

3. Figure: Guardian Interaction Rate
   - X: Stage
   - Y: Block Rate (%)
   - Bars: Guardian blocks decreasing per stage
   - Message: "Alpha learns to respect Guardian"

4. Figure: DD Avoided Analysis
   - X: Stage
   - Y: Count of DD avoided events
   - Bars: Increasing per stage
   - Message: "Curriculum teaches risk avoidance"

5. Figure: Confidence Distribution
   - X: Confidence Score
   - Y: Frequency
   - Histogram per stage
   - Show: Distribution shifts right with training

Commands to generate data:
    python src/rl/alpha_trainer.py --auto-curriculum

Data location:
    logs/alpha_training/training_metrics.csv
    logs/alpha_training/training_summary.json
"""


# CLI
if __name__ == "__main__":
    print(paper_figures_guide())
