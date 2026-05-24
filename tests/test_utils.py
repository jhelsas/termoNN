import torch
from src.utils import generate_domain_data, generate_boundary_data, set_seed
from tests.base_test import PINNTestCase

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

    def test_sampling_gradient_isolation(self):
        """Ensures sampled points do not have accidental gradients."""
        x, y = generate_domain_data(10, device=self.device)
        self.assertFalse(x.requires_grad)
        x_bc, _, _ = generate_boundary_data(10, device=self.device)
        self.assertFalse(x_bc.requires_grad)
