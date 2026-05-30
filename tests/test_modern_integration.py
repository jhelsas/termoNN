import torch
from src.pinn.solver import train
from src.core.geometry import PolygonDomain
from tests.base_test import PINNTestCase

class TestModernFeaturesIntegration(PINNTestCase):
    def test_train_with_sobolev_and_multiscale(self):
        """Integration: Verifies training loop runs with Sobolev and Multi-scale features."""
        outer = [(0, 0), (1, 0), (1, 1), (0, 1)]
        domain = PolygonDomain(outer)
        
        config = {
            "num_layers": 2,
            "hidden_dim": 32, # Must be multiple of 8 (2*4scales) for our multiscale logic
            "adam_epochs": 5,
            "lbfgs_epochs": 1,
            "use_sobolev": True,
            "use_fourier_features": True
        }
        
        bc_fn = lambda x, y: torch.zeros_like(x).unsqueeze(1)
        
        # This smoke test ensures no shape mismatches or autograd depth issues
        model, history = train(domain=domain, bc_fn=bc_fn, config=config)
        
        self.assertIn("adam", history)
        self.assertEqual(len(history["adam"]["loss"]), 5)
        self.assertTrue(model.use_fourier_features)
        self.assertTrue(hasattr(model, 'B_list'))

    def test_sobolev_config_toggle(self):
        """Integration: Verifies Sobolev loss can be toggled via config."""
        outer = [(0, 0), (1, 0), (1, 1), (0, 1)]
        domain = PolygonDomain(outer)
        
        # Run 1 epoch without Sobolev
        c1 = {"adam_epochs": 1, "lbfgs_epochs": 0, "use_sobolev": False, "hidden_dim": 16}
        # Run 1 epoch with Sobolev
        c2 = {"adam_epochs": 1, "lbfgs_epochs": 0, "use_sobolev": True, "hidden_dim": 16}
        
        bc_fn = lambda x, y: torch.zeros_like(x).unsqueeze(1)
        
        # Should both run without error
        train(domain=domain, bc_fn=bc_fn, config=c1)
        train(domain=domain, bc_fn=bc_fn, config=c2)
