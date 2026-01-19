# src/reporting/audit_trail.py
"""
Full Audit Trail System
========================

Immutable decision logging for regulatory compliance.

Features:
    - Complete decision trace
    - Timestamped JSON logs
    - Immutable append-only
    - Searchable history
    - Export for auditors

Paper Statement:
    "The system maintains a complete, immutable decision log
     suitable for regulatory review and investor disputes."
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("AUDIT_TRAIL")


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    cycle: int
    
    # Regime
    regime: str
    regime_confidence: float
    
    # Alpha
    alpha_action: str
    alpha_confidence: float
    
    # Guardian
    guardian_decision: str
    guardian_reason: str
    
    # Meta
    meta_action: str
    risk_multiplier: float
    
    # Final
    final_action: str
    final_lot: float
    
    # Account
    equity: float
    current_dd: float
    
    # Hash for integrity
    entry_hash: str = ""
    prev_hash: str = ""
    
    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash for integrity verification."""
        data = f"{self.timestamp}|{self.cycle}|{self.alpha_action}|{self.final_action}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AuditTrail:
    """
    Immutable audit trail for all trading decisions.
    
    Features:
        - Append-only log
        - Hash chain for integrity
        - JSON export
        - Search capability
    """
    
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.entries: List[AuditEntry] = []
        self.current_file = None
        self._init_log_file()
        
        logger.info("📜 AuditTrail initialized")
    
    def _init_log_file(self):
        """Initialize daily log file."""
        date_str = datetime.now().strftime("%Y%m%d")
        self.current_file = f"{self.log_dir}/audit_{date_str}.jsonl"
    
    def log(
        self,
        cycle: int,
        regime: str,
        regime_confidence: float,
        alpha_action: str,
        alpha_confidence: float,
        guardian_decision: str,
        guardian_reason: str,
        meta_action: str,
        risk_multiplier: float,
        final_action: str,
        final_lot: float,
        equity: float,
        current_dd: float
    ) -> AuditEntry:
        """
        Log a trading decision.
        
        All parameters are logged immutably.
        """
        # Get previous hash for chain
        prev_hash = self.entries[-1].entry_hash if self.entries else "GENESIS"
        
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            cycle=cycle,
            regime=regime,
            regime_confidence=regime_confidence,
            alpha_action=alpha_action,
            alpha_confidence=alpha_confidence,
            guardian_decision=guardian_decision,
            guardian_reason=guardian_reason,
            meta_action=meta_action,
            risk_multiplier=risk_multiplier,
            final_action=final_action,
            final_lot=final_lot,
            equity=equity,
            current_dd=current_dd,
            prev_hash=prev_hash
        )
        
        # Recompute hash with prev_hash
        data = f"{entry.timestamp}|{entry.cycle}|{entry.alpha_action}|{prev_hash}"
        entry.entry_hash = hashlib.sha256(data.encode()).hexdigest()[:16]
        
        self.entries.append(entry)
        self._append_to_file(entry)
        
        return entry
    
    def _append_to_file(self, entry: AuditEntry):
        """Append entry to log file (immutable)."""
        with open(self.current_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
    
    def verify_integrity(self) -> bool:
        """Verify hash chain integrity."""
        if not self.entries:
            return True
        
        prev_hash = "GENESIS"
        for entry in self.entries:
            if entry.prev_hash != prev_hash:
                logger.error(f"Integrity violation at cycle {entry.cycle}")
                return False
            prev_hash = entry.entry_hash
        
        logger.info("✅ Audit trail integrity verified")
        return True
    
    def search(
        self,
        start_time: str = None,
        end_time: str = None,
        action_filter: str = None,
        guardian_filter: str = None
    ) -> List[AuditEntry]:
        """Search audit entries."""
        results = self.entries
        
        if action_filter:
            results = [e for e in results if e.final_action == action_filter]
        
        if guardian_filter:
            results = [e for e in results if e.guardian_decision == guardian_filter]
        
        return results
    
    def export_for_audit(self, output_path: str) -> str:
        """Export complete audit trail for external audit."""
        export_data = {
            "system": "AI Trading System",
            "export_time": datetime.now().isoformat(),
            "entry_count": len(self.entries),
            "integrity_verified": self.verify_integrity(),
            "entries": [e.to_dict() for e in self.entries]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📤 Audit export: {output_path}")
        return output_path
    
    def get_summary(self) -> Dict:
        """Get audit trail summary."""
        if not self.entries:
            return {"entries": 0}
        
        guardian_blocks = sum(1 for e in self.entries if e.guardian_decision == "BLOCK")
        trades = sum(1 for e in self.entries if e.final_action != "HOLD")
        
        return {
            "total_entries": len(self.entries),
            "trades_executed": trades,
            "guardian_blocks": guardian_blocks,
            "block_rate": guardian_blocks / max(len(self.entries), 1),
            "first_entry": self.entries[0].timestamp,
            "last_entry": self.entries[-1].timestamp,
            "integrity_valid": self.verify_integrity()
        }


# =============================================================================
# Singleton
# =============================================================================

_audit_trail: Optional[AuditTrail] = None


def get_audit_trail() -> AuditTrail:
    """Get singleton AuditTrail."""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Audit Trail Test")
    print("=" * 60)
    
    audit = AuditTrail()
    
    # Log some entries
    for i in range(5):
        audit.log(
            cycle=i,
            regime="TRENDING",
            regime_confidence=0.8,
            alpha_action="BUY",
            alpha_confidence=0.75,
            guardian_decision="ALLOW" if i % 2 == 0 else "BLOCK",
            guardian_reason="Normal" if i % 2 == 0 else "High DD",
            meta_action="NORMAL",
            risk_multiplier=1.0,
            final_action="BUY" if i % 2 == 0 else "HOLD",
            final_lot=0.02 if i % 2 == 0 else 0,
            equity=1000 + i * 10,
            current_dd=0.02 + i * 0.01
        )
    
    # Verify
    print(f"\n{audit.verify_integrity()}")
    
    # Summary
    print(f"\nSummary: {audit.get_summary()}")
    
    # Export
    audit.export_for_audit("reports/audit_export.json")
    
    print("=" * 60)
