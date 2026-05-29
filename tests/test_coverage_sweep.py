import torch
import pytest
from src.pinn.solver import train
from src.pinn.physics import poisson_loss
from src.core.geometry import PolygonDomain
from src.core.data import get_device, generate_domain_data
from tests.base_test import PINNTestCase

class TestCoverageSweep(PINNTestCase):
    def test_solver_full_adaptive_path(self):
        """HITS: src/pinn/solver.py: 97-104, 120, 145.
        Triggers range loss weighting and L-BFGS range loss.
        """
        config = {
            "adam_epochs": 5,
            "lbfgs_epochs": 2,
            "use_self_adaptive_weights": True,
            "lambda_range": 10.0,
            "use_adaptive_sampling": True,
            "adaptive_every": 2
        }
        # Run training - this hits the adaptive weight branches for range loss
        model, history = train(config=config)
        self.assertIn("lambda_range", history["adam"])
        self.assertTrue(len(history["lbfgs"]["loss_range"]) > 0)

    def test_geometry_remainder_logic(self):
        """HITS: src/core/geometry.py: 120-121.
        Uses a prime number of points to trigger the segment remainder logic.
        """
        square = [(0,0), (1,0), (1,1), (0,1)]
        domain = PolygonDomain(square)
        # 17 is prime, won't divide evenly into 4 segments
        x, y, _, _ = domain.sample_boundary(17)
        self.assertEqual(len(x), 17)

    def test_physics_poisson_none_source(self):
        """HITS: src/pinn/physics.py: 59.
        Tests poisson_loss behavior with no source function.
        """
        from src.pinn.model import PINN
        model = PINN()
        x, y = torch.rand(10), torch.rand(10)
        # Should default to Laplace (f=0)
        loss = poisson_loss(model, x, y, f_fn=None)
        self.assertGreater(loss.item(), 0)

    def test_data_device_fallbacks(self):
        """HITS: src/core/data.py: 21-23.
        Force get_device to evaluate branches.
        """
        from src.core.data import get_device
        dev = get_device()
        self.assertIsInstance(dev, torch.device)
