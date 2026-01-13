# src/core/__init__.py
"""
Core Trading Infrastructure - PRODUCTION READY
===============================================

Fund-Grade execution components:
- ExecutionContract: Hard validation for all orders
- CapitalAllocator: Risk-based lot sizing
- SafetyChecklist: Pre-live validation
"""

from .execution_contract import (
    ExecutionContract,
    ExecutionBlocked,
    TradeIntent,
    validate_and_send
)
from .capital_allocator import (
    CapitalAllocator,
    AllocationResult
)
from .safety_checklist import (
    SafetyChecklist,
    CheckResult
)
from .parity_checker import (
    compare_decisions,
    check_parity,
    ParityResult,
    ParityMonitor
)
from .capital_allocation_engine import (
    CapitalAllocationEngine,
    MarketRegime
)

__all__ = [
    # Execution
    "ExecutionContract",
    "ExecutionBlocked",
    "TradeIntent",
    "validate_and_send",
    # Capital
    "CapitalAllocator",
    "AllocationResult",
    "CapitalAllocationEngine",
    "MarketRegime",
    # Safety
    "SafetyChecklist",
    "CheckResult",
    # Parity
    "compare_decisions",
    "check_parity",
    "ParityResult",
    "ParityMonitor",
]
