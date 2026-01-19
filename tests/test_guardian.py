
import unittest
from unittest.mock import MagicMock
from src.guardian.guardian_hybrid import GuardianHybrid
from src.rl.guardian_agent import GuardianAction

class TestGuardianHybrid(unittest.TestCase):
    def setUp(self):
        # Mocks
        self.rule_guardian = MagicMock()
        self.ppo_agent = MagicMock()
        self.hybrid = GuardianHybrid(
            self.rule_guardian, 
            self.ppo_agent, 
            confidence_threshold=0.6
        )
        
        # Default mock returns
        self.rule_guardian.evaluate.return_value = "ALLOW"
        self.ppo_agent.decide.return_value = ("ALLOW", 0.0)

    def test_rule_hard_block_overrides_ppo(self):
        """Rule BLOCK should override PPO ALLOW"""
        self.rule_guardian.evaluate.return_value = GuardianAction.FORCE_HOLD
        self.ppo_agent.decide.return_value = ("ALLOW", 0.99)
        
        action, reason = self.hybrid.decide({}, "BUY")
        
        self.assertEqual(action, "BLOCK")
        self.assertIn("RULE_HARD", reason)

    def test_ppo_intervention(self):
        """PPO BLOCK with high confidence should intervene"""
        self.rule_guardian.evaluate.return_value = "ALLOW"
        self.ppo_agent.decide.return_value = ("REDUCE_RISK", 0.8) # > 0.6
        
        action, reason = self.hybrid.decide({"margin_ratio": 1.0}, "BUY")
        
        self.assertEqual(action, "REDUCE_RISK")
        self.assertIn("PPO_ADVISE", reason)

    def test_ppo_low_confidence_ignored(self):
        """PPO BLOCK with low confidence should be ignored"""
        self.rule_guardian.evaluate.return_value = "ALLOW"
        self.ppo_agent.decide.return_value = ("REDUCE_RISK", 0.4) # < 0.6
        
        action, reason = self.hybrid.decide({"margin_ratio": 1.0}, "BUY")
        
        self.assertEqual(action, "ALLOW")

if __name__ == '__main__':
    unittest.main()
