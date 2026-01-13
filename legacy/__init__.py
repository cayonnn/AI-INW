# legacy/__init__.py
"""
⚠️ LEGACY MODULES – DO NOT USE

This folder contains deprecated logic.
All execution MUST go through:

- LiveLoopV3
- RiskManager
- TrailingManager
- AI SLTP Models

Usage of legacy modules in production is PROHIBITED.
"""

raise ImportError(
    "❌ Legacy modules are deprecated. Use LiveLoopV3 stack only.\n"
    "See: live_loop_v3.py, src/risk/risk_manager.py, src/risk/trailing.py"
)
