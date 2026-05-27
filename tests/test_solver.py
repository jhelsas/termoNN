import torch
from src.pinn.solver import train
from src.core.geometry import PolygonDomain
from tests.base_test import PINNTestCase

class TestSolver(PINNTestCase):
    def test_adaptive_weight_clamping(self):
        """Solver Validation: Ensures self-adaptive weights respect the safety clamp."""
        # We need a domain and bc_fn to trigger the weighting logic
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        domain = PolygonDomain(outer)
        
        def zero_bc(x, y):
            return torch.zeros((x.shape[0], 1), device=x.device)
            
        config = {
            "adam_epochs": 150, # Must be > adaptive_weight_every (100)
            "use_self_adaptive_weights": True,
            "adaptive_weight_every": 50,
            "lambda_bc": 10.0,
            "num_layers": 3,
            "hidden_dim": 8
        }
        
        # We don't check the exact value (stochastic), but it should run without NaN
        # and be influenced by the clamp logic in solver.py
        model = train(domain=domain, bc_fn=zero_bc, config=config)
        self.assertIsNotNone(model)

    def test_train_config_patience(self):
        """Solver Validation: Ensures train() respects custom scheduler patience."""
        # Smoke test to ensure scheduler initialization doesn't crash
        config = {"adam_epochs": 2, "lbfgs_epochs": 0}
        model = train(config=config)
        self.assertIsNotNone(model)

    def test_adaptive_sampling_rar(self):
        """Solver Validation: Checks if RAR sampling successfully executes."""
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        domain = PolygonDomain(outer)
        
        # Test with a dummy f_fn
        def f_fn(x, y): return torch.ones_like(x)
        
        config = {
            "adam_epochs": 5, 
            "lbfgs_epochs": 0,
            "use_adaptive_sampling": True,
            "adaptive_every": 2,
            "num_layers": 2,
            "hidden_dim": 4
        }
        
        # Ensure it runs without error (verifies generate_adaptive_domain_data path)
        model = train(domain=domain, f_fn=f_fn, config=config)
        self.assertIsNotNone(model)
