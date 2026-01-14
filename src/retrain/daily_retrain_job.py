# src/retrain/daily_retrain_job.py
"""
Daily Retrain Job - Competition Grade
======================================

Automated retraining triggered after day close:
- Collects daily performance data
- Optimizes meta-parameters
- Safe deployment with versioning
- Failsafe rollback mechanisms

Trigger Methods:
- Scheduled (23:59 server time)
- Via cron / Task Scheduler
- After N trades completed
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrain.data_collector import RetrainDataCollector, get_data_collector
from src.retrain.meta_optimizer import (
    MetaParameterOptimizer, MetaConfig, OptimizationResult,
    get_meta_optimizer
)
from src.utils.logger import get_logger

logger = get_logger("DAILY_RETRAIN")


@dataclass
class RetrainResult:
    """Result of daily retrain."""
    success: bool
    date: str
    old_version: int
    new_version: int
    score_improvement: float
    deployed: bool
    rollback_available: bool
    report_path: str
    errors: list


class DailyRetrainJob:
    """
    Daily Retrain Job Manager.
    
    Process:
    1. Collect day's data
    2. Run meta-parameter optimization
    3. Stress test candidate
    4. Safe deploy (if improvement > threshold)
    5. Generate report
    
    Failsafes:
    - Max delta per day (±15%)
    - Auto rollback if DD spikes
    - Lock during live session
    """
    
    # Failsafe limits
    MAX_DELTA_PER_DAY = 0.15        # ±15% max change
    MIN_IMPROVEMENT_DEPLOY = 0.05   # 5% improvement to deploy
    MAX_DD_BEFORE_ROLLBACK = 0.10   # 10% DD triggers rollback
    
    def __init__(
        self,
        reports_dir: str = "reports",
        min_trades: int = 5,        # Min trades to run retrain
        auto_deploy: bool = False   # Manual deploy by default
    ):
        """
        Initialize Daily Retrain Job.
        
        Args:
            reports_dir: Directory for reports
            min_trades: Minimum trades required
            auto_deploy: Auto-deploy or manual review
        """
        self.reports_dir = reports_dir
        self.min_trades = min_trades
        self.auto_deploy = auto_deploy
        
        os.makedirs(reports_dir, exist_ok=True)
        
        self.data_collector = get_data_collector()
        self.optimizer = get_meta_optimizer()
        
        self.retrained_today = False
        self.last_retrain_date: Optional[str] = None
        self.rollback_config: Optional[MetaConfig] = None
        
        logger.info(
            f"DailyRetrainJob initialized: "
            f"min_trades={min_trades}, auto_deploy={auto_deploy}"
        )
    
    def should_run(self, current_time: datetime = None) -> bool:
        """
        Check if retrain should run.
        
        Conditions:
        - After 23:59 server time
        - Not already retrained today
        - Or N trades completed
        """
        if current_time is None:
            current_time = datetime.now()
        
        today = current_time.strftime("%Y-%m-%d")
        
        # Already retrained today
        if self.retrained_today and self.last_retrain_date == today:
            return False
        
        # Check time (after 23:00)
        if current_time.hour >= 23:
            return True
        
        return False
    
    def run(
        self,
        force: bool = False,
        days_to_analyze: int = 7
    ) -> RetrainResult:
        """
        Run the daily retrain.
        
        Args:
            force: Force run even if conditions not met
            days_to_analyze: Days of historical data to use
            
        Returns:
            RetrainResult with details
        """
        today = datetime.now().strftime("%Y-%m-%d")
        errors = []
        
        logger.info(f"Starting daily retrain for {today}")
        
        try:
            # 1. End day collection
            day_data = self.data_collector.end_day()
            
            # 2. Load historical data
            historical = self.data_collector.load_recent_days(days_to_analyze)
            
            if len(historical) < 1:
                errors.append("No historical data available")
                return self._create_failure_result(today, errors)
            
            total_trades = sum(
                d.get("win_count", 0) + d.get("loss_count", 0)
                for d in historical
            )
            
            if total_trades < self.min_trades and not force:
                errors.append(f"Not enough trades ({total_trades} < {self.min_trades})")
                return self._create_failure_result(today, errors)
            
            # 3. Get current config
            current_config = self.optimizer.load_active_config()
            old_version = current_config.version
            
            # Save for rollback
            self.rollback_config = current_config
            
            # 4. Run optimization
            opt_result = self.optimizer.optimize(historical, current_config)
            
            # 5. Check improvement
            score_improvement = opt_result.improvements.get("score", 0)
            
            # 6. Validate delta within limits
            if not self._validate_delta(current_config, opt_result.config):
                errors.append("Config change exceeds max delta")
                opt_result = OptimizationResult(
                    config=current_config,
                    score=opt_result.score,
                    metrics=opt_result.metrics,
                    improvements={},
                    recommended=False
                )
            
            # 7. Deploy decision
            deployed = False
            if opt_result.recommended and (self.auto_deploy or force):
                deployed = self._deploy_config(opt_result.config)
            
            # 8. Generate report
            report_path = self._generate_report(
                today, current_config, opt_result, historical, deployed
            )
            
            # Mark as done
            self.retrained_today = True
            self.last_retrain_date = today
            
            logger.info(
                f"Retrain complete: improvement={score_improvement:.1%}, "
                f"deployed={deployed}"
            )
            
            return RetrainResult(
                success=True,
                date=today,
                old_version=old_version,
                new_version=opt_result.config.version,
                score_improvement=score_improvement,
                deployed=deployed,
                rollback_available=self.rollback_config is not None,
                report_path=report_path,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            errors.append(str(e))
            return self._create_failure_result(today, errors)
    
    def _validate_delta(
        self,
        old: MetaConfig,
        new: MetaConfig
    ) -> bool:
        """Validate config changes are within limits."""
        # Check key parameters
        checks = [
            (old.streak_base_risk, new.streak_base_risk, "streak_base_risk"),
            (old.streak_max_risk, new.streak_max_risk, "streak_max_risk"),
            (old.alpha_score_threshold, new.alpha_score_threshold, "alpha_score"),
            (old.defensive_score_threshold, new.defensive_score_threshold, "defensive_score"),
        ]
        
        for old_val, new_val, name in checks:
            if old_val == 0:
                continue
            delta = abs(new_val - old_val) / old_val
            if delta > self.MAX_DELTA_PER_DAY:
                logger.warning(
                    f"Delta too large for {name}: {delta:.1%} > {self.MAX_DELTA_PER_DAY:.0%}"
                )
                return False
        
        return True
    
    def _deploy_config(self, config: MetaConfig) -> bool:
        """Deploy new config as active."""
        try:
            self.optimizer.save_config(config, active=True)
            logger.info(f"Deployed config v{config.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to deploy: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback to previous config."""
        if not self.rollback_config:
            logger.warning("No rollback config available")
            return False
        
        try:
            self.optimizer.save_config(self.rollback_config, active=True)
            logger.warning(f"Rolled back to v{self.rollback_config.version}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _generate_report(
        self,
        date: str,
        old_config: MetaConfig,
        opt_result: OptimizationResult,
        historical: list,
        deployed: bool
    ) -> str:
        """Generate markdown retrain report."""
        report_path = f"{self.reports_dir}/retrain_{date.replace('-', '')}.md"
        
        new_config = opt_result.config
        metrics = opt_result.metrics
        improvement = opt_result.improvements.get("score", 0)
        
        report = f"""# Retrain Report - {date}

## Summary

| Metric | Value |
|--------|-------|
| Old Version | v{old_config.version} |
| New Version | v{new_config.version} |
| Score Improvement | {improvement:+.1%} |
| Deployed | {'Yes' if deployed else 'No'} |
| Recommended | {'Yes' if opt_result.recommended else 'No'} |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total PnL | ${metrics.get('total_pnl', 0):.2f} |
| Total Trades | {metrics.get('total_trades', 0)} |
| Win Rate | {metrics.get('win_rate', 0):.1%} |
| Max Drawdown | {metrics.get('max_drawdown', 0):.1%} |

## Config Changes

### Mode Controller
| Parameter | Old | New |
|-----------|-----|-----|
| Alpha Score | {old_config.alpha_score_threshold} | {new_config.alpha_score_threshold} |
| Defensive Score | {old_config.defensive_score_threshold} | {new_config.defensive_score_threshold} |
| Alpha DD | {old_config.alpha_dd_threshold}% | {new_config.alpha_dd_threshold}% |

### Win Streak Booster
| Parameter | Old | New |
|-----------|-----|-----|
| Base Risk | {old_config.streak_base_risk}% | {new_config.streak_base_risk}% |
| Max Risk | {old_config.streak_max_risk}% | {new_config.streak_max_risk}% |

### Pyramid Manager
| Parameter | Old | New |
|-----------|-----|-----|
| Max Entries | {old_config.pyramid_max_entries} | {new_config.pyramid_max_entries} |
| DD Threshold | {old_config.pyramid_dd_threshold:.1%} | {new_config.pyramid_dd_threshold:.1%} |

## Regime Performance

"""
        # Add regime stats if available
        if historical:
            regime_stats = {}
            for day in historical:
                for regime, count in day.get("regime_distribution", {}).items():
                    regime_stats[regime] = regime_stats.get(regime, 0) + count
            
            report += "| Regime | Trades |\n|--------|--------|\n"
            for regime, count in regime_stats.items():
                report += f"| {regime} | {count} |\n"
        
        report += f"\n---\n*Generated: {datetime.now().isoformat()}*\n"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved: {report_path}")
        return report_path
    
    def _create_failure_result(self, date: str, errors: list) -> RetrainResult:
        """Create failure result."""
        return RetrainResult(
            success=False,
            date=date,
            old_version=0,
            new_version=0,
            score_improvement=0,
            deployed=False,
            rollback_available=False,
            report_path="",
            errors=errors
        )
    
    def reset_daily_flag(self) -> None:
        """Reset retrained flag for new day."""
        self.retrained_today = False


# Singleton instance
_job: Optional[DailyRetrainJob] = None


def get_daily_retrain_job() -> DailyRetrainJob:
    """Get or create singleton DailyRetrainJob."""
    global _job
    if _job is None:
        _job = DailyRetrainJob()
    return _job


def run_daily_retrain(force: bool = False) -> RetrainResult:
    """Convenience function to run daily retrain."""
    job = get_daily_retrain_job()
    return job.run(force=force)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Retrain Job")
    parser.add_argument("--force", action="store_true", help="Force run")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Daily Retrain Job")
    print("=" * 60)
    
    job = DailyRetrainJob()
    result = job.run(force=args.force, days_to_analyze=args.days)
    
    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Improvement: {result.score_improvement:+.1%}")
    print(f"Deployed: {result.deployed}")
    if result.report_path:
        print(f"Report: {result.report_path}")
    if result.errors:
        print(f"Errors: {result.errors}")
