
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.rl.guardian_ppo_infer import GuardianPPOAdvisor

class TestGuardianPPO(unittest.TestCase):
    @patch('stable_baselines3.PPO')
    def test_initialization(self, mock_ppo):
        """Test that model loads correctly"""
        mock_ppo.load.return_value = MagicMock()
        advisor = GuardianPPOAdvisor(model_path="dummy.zip", enabled=True)
        self.assertIsNotNone(advisor.model)

    def test_disabled_advisor(self):
        """Test disabled advisor returns ALLOW"""
        advisor = GuardianPPOAdvisor(enabled=False)
        action, conf = advisor.decide({})
        self.assertEqual(action, "ALLOW")
        self.assertEqual(conf, 0.0)

    @patch('stable_baselines3.PPO')
    def test_decision_logic(self, mock_ppo):
        """Test inference logic (Mocking SB3 internals)"""
        # Mock Policy
        mock_model = MagicMock()
        mock_policy = MagicMock()
        mock_dist = MagicMock()
        
        mock_ppo.load.return_value = mock_model
        mock_model.policy = mock_policy
        mock_policy.get_distribution.return_value = mock_dist
        
        # Mock Probabilities: [ALLOW, REDUCE, HOLD, FREEZE]
        # Make it predict index 1 (REDUCE) with 0.8 confidence
        mock_dist.distribution.probs.detach().cpu().numpy.return_value = [[0.1, 0.8, 0.05, 0.05]]
        
        advisor = GuardianPPOAdvisor(model_path="dummy.zip", enabled=True)
        # Mock obs_to_tensor to avoid torch error
        mock_policy.obs_to_tensor.return_value = (None, None)
        
        action, conf = advisor.decide({"daily_dd": 0.01})
        
        self.assertEqual(action, "REDUCE_RISK")
        self.assertAlmostEqual(conf, 0.8)

if __name__ == '__main__':
    unittest.main()
