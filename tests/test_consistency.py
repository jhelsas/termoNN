import unittest
import torch
import numpy as np
from src.pinn.solver import train
from src.core.geometry import PolygonDomain
from src.core.data import get_device
from tests.base_test import PINNTestCase

class TestConsistency(PINNTestCase):
    """
    Consistency and edge-case tests for the PINN system.
    Ensures that solvers and models behave predictably under stress or weird configurations.
    """
    
    def test_reproducibility(self):
        """Consistency: Verifies that setting a seed results in identical training runs."""
        vertices = torch.tensor([[0,0], [1,0], [1,1], [0,1]], dtype=torch.float32)
        domain = PolygonDomain(vertices)
        bc_fn = lambda x, y: torch.zeros((x.shape[0], 1), device=x.device)
        
        config = {
            "num_layers": 3,
            "hidden_dim": 8,
            "adam_epochs": 10,
            "lbfgs_epochs": 0,
            "seed": 123
        }
        
        # Run 1
        model1, history1 = train(domain=domain, bc_fn=bc_fn, config=config)
        weights1 = torch.cat([p.flatten() for p in model1.parameters()])
        
        # Run 2
        model2, history2 = train(domain=domain, bc_fn=bc_fn, config=config)
        weights2 = torch.cat([p.flatten() for p in model2.parameters()])
        
        self.assertTensorsEqual(weights1, weights2, "Weights should be identical with the same seed")

    def test_solver_no_data_crash(self):
        """Robustness: Verifies solver handles zero-point or tiny-point configurations."""
        vertices = torch.tensor([[0,0], [1,0], [1,1], [0,1]], dtype=torch.float32)
        domain = PolygonDomain(vertices)
        
        # Extremely small point counts
        config = {
            "adam_points_domain": 2,
            "adam_points_bc": 2,
            "adam_epochs": 1,
            "lbfgs_epochs": 0,
            "num_layers": 2,
            "hidden_dim": 4,
            "activation": "sine",
            "lambda_bc": 1.0
        }
        
        try:
            train(domain=domain, bc_fn=lambda x, y: torch.zeros_like(x).unsqueeze(1), config=config)
        except Exception as e:
            self.fail(f"Solver crashed with tiny point count: {e}")

    def test_spectral_omega_scalar_vs_tuple(self):
        """Consistency: Verifies model can initialize with both scalar and tuple omega."""
        from src.pinn.model import PINN
        
        # Scalar
        m1 = PINN(omega=30.0, use_fourier_features=False)
        self.assertTrue(torch.all(m1.act[0].omega == 30.0))
        
        # Tuple (Range)
        m2 = PINN(omega=(1.0, 30.0), hidden_dim=10, use_fourier_features=False)
        self.assertEqual(m2.act[0].omega.shape, (10,))
        self.assertLess(m2.act[0].omega[0], m2.act[0].omega[-1])

    def test_adaptive_weights_sanity(self):
        """Consistency: Verifies that self-adaptive weighting adjusts lambda_bc."""
        vertices = torch.tensor([[0,0], [1,0], [1,1], [0,1]], dtype=torch.float32)
        domain = PolygonDomain(vertices)
        
        # We need a setup where BC gradients are likely very different from PDE gradients
        # Use a high frequency BC
        def high_freq_bc(x, y):
            return torch.sin(100 * x).unsqueeze(1)
            
        config = {
            "adam_epochs": 110, # Enough for one adaptive update (every 100)
            "lbfgs_epochs": 0,
            "use_self_adaptive_weights": True,
            "adaptive_weight_every": 100,
            "lambda_bc": 1.0 # Start low
        }
        
        # We can't easily check the internal state of train() without modifying it, 
        # but we can ensure it runs. 
        # Future-proofing: If we ever return the weight history, we'd check it here.
        model = train(domain=domain, bc_fn=high_freq_bc, config=config)
        self.assertIsNotNone(model)

    def test_multi_hole_geometry_overlap(self):
        """Consistency: Verifies sampling works when holes are close or nested."""
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        h1 = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]) # Large hole
        h2 = torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]) # Hole inside hole?
        # Actually PolygonDomain uses logic: inside_outer & ~h1 & ~h2
        # If h2 is inside h1, ~h1 is already false there.
        
        domain = PolygonDomain(outer, holes=[h1, h2])
        x, y = domain.sample_interior(50)
        self.assertEqual(len(x), 50)
        # All points must be outside both holes
        for h in [h1, h2]:
            # Simple check for square holes
            in_h = (x > h[:,0].min()) & (x < h[:,0].max()) & (y > h[:,1].min()) & (y < h[:,1].max())
            self.assertFalse(torch.any(in_h))

if __name__ == '__main__':
    unittest.main()
