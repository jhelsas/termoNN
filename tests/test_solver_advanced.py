import torch
from src.pinn.solver import train
from tests.base_test import PINNTestCase

class TestSolverAdvanced(PINNTestCase):
    def test_history_tracking_adam(self):
        """Verifies that the Adam stage correctly tracks loss and weight history."""
        config = {
            "adam_epochs": 10,
            "lbfgs_epochs": 0,
            "use_self_adaptive_weights": True,
            "adaptive_weight_every": 2
        }
        model, history = train(config=config)
        
        self.assertIn("adam", history)
        self.assertEqual(len(history["adam"]["loss"]), 10)
        self.assertEqual(len(history["adam"]["lambda_bc"]), 10)
        # Check if values are populated
        self.assertTrue(all(isinstance(v, float) for v in history["adam"]["loss"]))

    def test_history_tracking_lbfgs(self):
        """Verifies that the L-BFGS stage tracks loss residues."""
        config = {
            "adam_epochs": 0,
            "lbfgs_epochs": 5
        }
        model, history = train(config=config)
        
        self.assertIn("lbfgs", history)
        self.assertEqual(len(history["lbfgs"]["loss"]), 5)
        self.assertEqual(len(history["lbfgs"]["loss_pde"]), 5)

    def test_lbfgs_point_regeneration(self):
        """
        Verifies that L-BFGS stage respects different point counts from config.
        Since we can't easily peek into the closure tensors from outside, 
        we check if the config keys are handled.
        """
        # This is a smoke test to ensure the regeneration logic doesn't crash
        config = {
            "adam_epochs": 1,
            "lbfgs_epochs": 1,
            "lbfgs_points_domain": 100,
            "lbfgs_points_bc": 50
        }
        # Should not raise exception
        model, history = train(config=config)
        self.assertIsNotNone(model)

    def test_adaptive_weight_clamping(self):
        """Ensures that self-adaptive weights are clamped as expected."""
        # We trigger a weight update and verify history values are within reasonable bounds
        config = {
            "adam_epochs": 5,
            "use_self_adaptive_weights": True,
            "adaptive_weight_every": 1,
            "lambda_bc": 1.0 # Start small to see update
        }
        model, history = train(config=config)
        for l_bc in history["adam"]["lambda_bc"]:
            self.assertLessEqual(l_bc, 2000.0)
            self.assertGreaterEqual(l_bc, 0.0)
