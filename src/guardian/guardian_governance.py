# src/guardian/guardian_governance.py

from datetime import datetime, date
import logging
from typing import Tuple, Optional

# Configure logger
logger = logging.getLogger("GUARDIAN_GOV")

class GuardianGovernance:
    """
    Guardian Governance Layer
    -------------------------
    Responsible for:
    1. Daily Reset Logic
    2. Start-of-Day Balance Snapshot
    3. Daily Drawdown Calculation & Latching
    
    This acts as the supreme governance authority for daily risk limits.
    """
    def __init__(self, daily_dd_limit: float = 0.10):
        """
        Initialize Guardian Governance.
        
        Args:
            daily_dd_limit: Max daily drawdown as a decimal (0.10 = 10%)
        """
        self.daily_dd_limit = daily_dd_limit
        self.current_day: Optional[date] = None
        self.start_balance: Optional[float] = None
        self.hard_latched = False
        self.latch_reason: Optional[str] = None
        self.current_dd = 0.0

    def _today(self) -> date:
        """Get current UTC date."""
        return datetime.utcnow().date()

    def reset_if_new_day(self, account):
        """
        Check for new day and reset tracking if needed.
        Should be called at the start of every cycle.
        """
        today = self._today()

        if self.current_day != today:
            self.current_day = today
            self.start_balance = float(account.balance)
            self.hard_latched = False
            self.latch_reason = None
            self.current_dd = 0.0

            logger.info("=" * 40)
            logger.info(f"📊 GUARDIAN DAILY RESET | {today}")
            logger.info(f"   Start Balance: ${self.start_balance:.2f}")
            logger.info("=" * 40)

    def check_daily_dd(self, account) -> Tuple[bool, Optional[str]]:
        """
        Check if Daily Drawdown limit is exceeded.
        
        Returns:
            (bool, reason): True if blocked (latched), False if allowed.
        """
        # 1. New Day Check
        self.reset_if_new_day(account)

        # 2. Check Valid State
        if self.start_balance is None:
            # Fallback for mock/error states, just capture current
            self.start_balance = float(account.balance)
            return False, None

        # 3. Check Latch
        if self.hard_latched:
            return True, self.latch_reason

        # 4. Calculate Daily DD
        # Logic: (StartBalance - CurrentEquity) / StartBalance
        equity = float(account.equity)
        
        if self.start_balance > 0:
            self.current_dd = (self.start_balance - equity) / self.start_balance
        else:
            self.current_dd = 0.0
            
        self.current_dd = max(0.0, self.current_dd)

        # 5. Evaluate Limit
        if self.current_dd >= self.daily_dd_limit:
            self.hard_latched = True
            self.latch_reason = (
                f"Daily DD limit exceeded: "
                f"{self.current_dd*100:.2f}% >= {self.daily_dd_limit*100:.2f}%"
            )

            logger.critical("=" * 50)
            logger.critical("🔒 GUARDIAN HARD LATCH TRIGGERED")
            logger.critical(f"   Reason: {self.latch_reason}")
            logger.critical("   Action: ALL TRADING BLOCKED UNTIL NEXT DAY")
            logger.critical("=" * 50)

            return True, self.latch_reason

        return False, None
