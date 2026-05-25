import torch
from src.core.data import generate_domain_data, generate_boundary_data, generate_adaptive_domain_data, set_seed
from src.core.geometry import PolygonDomain
from src.pinn.model import PINN
from tests.base_test import PINNTestCase
import numpy as np

class TestUtils(PINNTestCase):
    def test_domain_sampling(self):
        """Verifies domain points are within [0, 1]x[0, 1]."""
        x, y = generate_domain_data(100, device=self.device)
        self.assertTrue(torch.all(x >= 0) and torch.all(x <= 1))
        self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))

    def test_domain_sampling_count(self):
        """Ensures the correct number of domain points are generated."""
        n = 123
        x, y = generate_domain_data(n, device=self.device)
        self.assertEqual(len(x), n)
        self.assertEqual(len(y), n)

    def test_boundary_sampling_count(self):
        """Ensures the correct number of boundary points are generated."""
        n = 400
        x, y, u = generate_boundary_data(n, device=self.device)
        self.assertEqual(len(x), n)
        self.assertEqual(len(y), n)
        self.assertEqual(u.shape, (n, 1))

    def test_boundary_sampling(self):
        """Verifies boundary points are actually on the edges."""
        x_bc, y_bc, _ = generate_boundary_data(100, device=self.device)
        on_boundary = (x_bc == 0) | (x_bc == 1) | (y_bc == 0) | (y_bc == 1)
        self.assertTrue(torch.all(on_boundary))

    def test_reproducibility(self):
        """Ensures set_seed results in identical data generation."""
        set_seed(123)
        x1, y1 = generate_domain_data(10, device=self.device)
        set_seed(123)
        x2, y2 = generate_domain_data(10, device=self.device)
        self.assertTensorsEqual(x1, x2)
        self.assertTensorsEqual(y1, y2)

    def test_boundary_values_range(self):
        """Checks if BC values u_bc are within [0, 1]."""
        _, _, u_bc = generate_boundary_data(200, device=self.device)
        self.assertTrue(torch.all(u_bc >= 0.0) and torch.all(u_bc <= 1.0))

    def test_boundary_distribution(self):
        """Ensures points are distributed across all 4 boundaries."""
        n_points = 400
        x_bc, y_bc, _ = generate_boundary_data(n_points, device=self.device)
        expected = n_points // 4
        self.assertEqual((x_bc == 0).sum().item(), expected)
        self.assertEqual((x_bc == 1).sum().item(), expected)
        self.assertEqual((y_bc == 0).sum().item(), expected)
        self.assertEqual((y_bc == 1).sum().item(), expected)

    def test_reproducibility_with_polygon(self):
        """Ensures set_seed results in identical polygon sampling."""
        outer = [(0,0), (2,1), (1,2)]
        domain = PolygonDomain(outer)
        
        set_seed(42)
        x1, y1 = generate_domain_data(10, device=self.device, domain=domain)
        set_seed(42)
        x2, y2 = generate_domain_data(10, device=self.device, domain=domain)
        self.assertTensorsEqual(x1, x2)
        self.assertTensorsEqual(y1, y2)

    def test_domain_default_fallback(self):
        """Verifies that passing domain=None still works (backward compatibility)."""
        x, y = generate_domain_data(10, device=self.device, domain=None)
        self.assertEqual(x.shape, (10,))
        self.assertTrue(torch.all(x >= 0) and torch.all(x <= 1))

    def test_adaptive_sampling_rar(self):
        """Data Validation: Verifies RAR adaptive sampling logic."""
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        domain = PolygonDomain(outer)
        model = PINN(hidden_dim=10).to(self.device)
        
        n_points = 50
        # Should return exactly n_points
        x, y = generate_adaptive_domain_data(model, n_points, domain=domain, device=self.device)
        self.assertEqual(len(x), n_points)
        self.assertTrue(torch.all(domain.is_inside(x, y)))

    def test_adaptive_sampling_determinism(self):
        """Data Validation: Verifies RAR is deterministic with set_seed."""
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        domain = PolygonDomain(outer)
        model = PINN(hidden_dim=5).to(self.device)
        
        set_seed(123)
        x1, y1 = generate_adaptive_domain_data(model, 10, domain=domain, device=self.device)
        set_seed(123)
        x2, y2 = generate_adaptive_domain_data(model, 10, domain=domain, device=self.device)
        self.assertTensorsEqual(x1, x2)

    def test_adaptive_sampling_error_targeting(self):
        """Data Validation: Verifies RAR actually picks points with higher PDE error."""
        # Square domain
        outer = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        domain = PolygonDomain(outer)
        
        # Model that outputs 0 everywhere
        model = PINN(hidden_dim=10).to(self.device)
        for p in model.parameters():
            torch.nn.init.zeros_(p)
            
        # Source term: f=1 for x > 0.5, f=0 for x <= 0.5
        # Since u=0, residue = |0 - f|. Residue is 1 for x > 0.5.
        def f_fn(x, y):
            f = torch.zeros_like(x)
            f[x > 0.5] = 1.0
            return f
            
        n_points = 100
        x, y = generate_adaptive_domain_data(model, n_points, domain=domain, f_fn=f_fn, device=self.device)
        
        # All points should ideally be in the high-error region (x > 0.5)
        self.assertTrue(torch.all(x > 0.5))

    def test_adaptive_sampling_range_targeting(self):
        """Data Validation: Verifies RAR picks points violating the Maximum Principle."""
        outer = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        domain = PolygonDomain(outer)
        
        class OutOfRangeModel(torch.nn.Module):
            def forward(self, coords):
                # u = 2.0 for x > 0.5 (out of [0, 1]), u = 0.5 elsewhere
                u = torch.ones((coords.shape[0], 1), device=coords.device) * 0.5
                u[coords[:, 0] > 0.5] = 2.0
                return u
        
        model = OutOfRangeModel().to(self.device)
        config = {"lambda_range": 100.0}
        
        n_points = 50
        x, y = generate_adaptive_domain_data(model, n_points, domain=domain, config=config, device=self.device)
        
        # RAR should prioritize the out-of-range region
        self.assertTrue(torch.all(x > 0.5))
