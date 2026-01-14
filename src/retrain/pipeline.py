# src/retrain/pipeline.py
"""
Retrain Pipeline - Competition Grade
=====================================

Full retraining pipeline:
1. Load training data
2. Run optimizer
3. Stress test candidates
4. Deploy best candidate
5. Write report
"""

import os
import sys
from datetime import date, datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrain.data_loader import load_training_data
from src.retrain.evaluator import stress_test
from src.retrain.deployer import deploy_candidate
from src.retrain.utils.report_writer import write_report
from src.utils.logger import get_logger

logger = get_logger("RETRAIN_PIPELINE")


def run_pipeline(
    run_date: date = None,
    use_bayesian: bool = True,
    auto_deploy: bool = False,
    days_lookback: int = 7
) -> Dict[str, Any]:
    """
    Run the full retrain pipeline.
    
    Args:
        run_date: Date to run for (default: today)
        use_bayesian: Use Bayesian optimizer (else grid search)
        auto_deploy: Auto-deploy best candidate
        days_lookback: Days of data to use
        
    Returns:
        Pipeline result with best config and report path
    """
    if run_date is None:
        run_date = date.today()
    
    logger.info(f"Starting retrain pipeline for {run_date}")
    
    result = {
        "success": False,
        "date": str(run_date),
        "best_config": None,
        "deployed": False,
        "report_path": None,
        "errors": []
    }
    
    try:
        # 1. Load training data
        logger.info("Step 1: Loading training data...")
        data = load_training_data(run_date, days_lookback)
        
        if data is None or data.get("live") is None:
            result["errors"].append("No training data available")
            logger.warning("No training data found")
            return result
        
        # 2. Run optimizer
        logger.info("Step 2: Running optimizer...")
        if use_bayesian:
            try:
                from src.retrain.optimizers.bayesian_score_optimizer import optimize
                candidates = optimize(data)
            except ImportError:
                # Fallback to grid search
                from src.retrain.optimizers.grid_optimizer import optimize_grid
                candidates = optimize_grid(data)
        else:
            from src.retrain.optimizers.grid_optimizer import optimize_grid
            candidates = optimize_grid(data)
        
        if not candidates:
            result["errors"].append("No candidates generated")
            return result
        
        logger.info(f"Generated {len(candidates)} candidates")
        
        # 3. Stress test candidates
        logger.info("Step 3: Stress testing candidates...")
        best = stress_test(candidates, data)
        
        if best is None:
            result["errors"].append("No candidate passed stress test")
            return result
        
        result["best_config"] = best
        logger.info(f"Best candidate score: {best.get('score', 0):.2f}")
        
        # 4. Deploy best candidate
        if auto_deploy:
            logger.info("Step 4: Deploying candidate...")
            deployed = deploy_candidate(best, active=False)
            result["deployed"] = deployed
        else:
            logger.info("Step 4: Skipping auto-deploy (manual review)")
        
        # 5. Write report
        logger.info("Step 5: Writing report...")
        report_path = write_report(best, run_date, data)
        result["report_path"] = report_path
        
        result["success"] = True
        logger.info(f"Pipeline complete: {report_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        result["errors"].append(str(e))
    
    return result


def run_quick_validation(config: Dict) -> bool:
    """Quick validation of a config without full pipeline."""
    required_keys = ["confidence", "pyramid", "score"]
    return all(k in config for k in required_keys)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrain Pipeline")
    parser.add_argument("--date", type=str, help="Date (YYYY-MM-DD)")
    parser.add_argument("--bayesian", action="store_true", help="Use Bayesian")
    parser.add_argument("--deploy", action="store_true", help="Auto deploy")
    args = parser.parse_args()
    
    run_date = date.fromisoformat(args.date) if args.date else None
    
    result = run_pipeline(
        run_date=run_date,
        use_bayesian=args.bayesian,
        auto_deploy=args.deploy
    )
    
    print(f"\nResult: {'SUCCESS' if result['success'] else 'FAILED'}")
    if result["report_path"]:
        print(f"Report: {result['report_path']}")
    if result["errors"]:
        print(f"Errors: {result['errors']}")
