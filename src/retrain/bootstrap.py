# src/retrain/bootstrap.py
"""
Retrain Bootstrap - Auto Dependencies
======================================

Automatically installs required dependencies for retrain.
Falls back gracefully if internet not available.
"""

import subprocess
import importlib.util
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("BOOTSTRAP")


def ensure_skopt() -> bool:
    """
    Ensure scikit-optimize is installed.
    
    Returns:
        True if skopt is available
    """
    # Check if already installed
    if importlib.util.find_spec("skopt"):
        logger.info("✅ skopt already installed")
        return True
    
    logger.warning("⚠ skopt not found, attempting auto-install...")
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "scikit-optimize", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )
        logger.info("✅ skopt installed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("❌ skopt install timed out")
        return False
        
    except Exception as e:
        logger.error(f"❌ skopt install failed: {e}")
        return False


def ensure_dependencies() -> dict:
    """
    Ensure all retrain dependencies are available.
    
    Returns:
        Dict with dependency status
    """
    status = {
        "skopt": ensure_skopt(),
        "pandas": importlib.util.find_spec("pandas") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "yaml": importlib.util.find_spec("yaml") is not None,
    }
    
    all_ok = all(status.values())
    
    if all_ok:
        logger.info("✅ All retrain dependencies ready")
    else:
        missing = [k for k, v in status.items() if not v]
        logger.warning(f"⚠ Missing dependencies: {missing}")
    
    return status


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Retrain Bootstrap")
    print("=" * 50)
    
    status = ensure_dependencies()
    
    for dep, ok in status.items():
        icon = "✅" if ok else "❌"
        print(f"{icon} {dep}")
