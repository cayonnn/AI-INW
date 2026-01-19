
import unittest
import json
import sys
import os
import datetime

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_verification():
    """Run all tests in tests/ directory and save report."""
    
    # 1. Discover Tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    if not os.path.exists(start_dir):
        print(f"Error: {start_dir} directory not found.")
        return
        
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 2. Run Tests
    result = unittest.TestResult()
    suite.run(result)
    
    # 3. Compile Report
    report = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "details": []
    }
    
    # Add failure details
    for failure in result.failures:
        report["details"].append({
            "test": str(failure[0]),
            "status": "FAIL",
            "message": str(failure[1])
        })
        
    for error in result.errors:
        report["details"].append({
            "test": str(error[0]),
            "status": "ERROR",
            "message": str(error[1])
        })
        
    # 4. Save to JSON
    os.makedirs("logs", exist_ok=True)
    with open("logs/verification_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    run_verification()
