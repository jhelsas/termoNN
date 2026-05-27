import torch
import unittest
from src.pinn.physics import range_loss, poisson_loss
from src.pinn.model import PINN
from tests.base_test import PINNTestCase

class TestPhysicsAdvanced(PINNTestCase):
    def test_range_loss_behavior(self):
        """Tests that range_loss correctly identifies violations."""
        # Case 1: No violation
        u_safe = torch.tensor([[0.1], [0.5], [0.9]], device=self.device)
        loss_safe = range_loss(u_safe, min_val=0.0, max_val=1.0)
        self.assertEqual(loss_safe.item(), 0.0)
        
        # Case 2: Upper violation
        u_high = torch.tensor([[1.1]], device=self.device)
        loss_high = range_loss(u_high, min_val=0.0, max_val=1.0)
        self.assertAlmostEqual(loss_high.item(), 0.1**2)
        
        # Case 3: Lower violation
        u_low = torch.tensor([[-0.2]], device=self.device)
        loss_low = range_loss(u_low, min_val=0.0, max_val=1.0)
        self.assertAlmostEqual(loss_low.item(), 0.2**2)

    def test_poisson_loss_known_solution(self):
        """Tests poisson_loss against a quadratic function with known Delta u."""
        # u(x, y) = x^2 + y^2  => u_xx = 2, u_yy = 2 => Delta u = 4
        # We define a 'model' that always returns x^2 + y^2
        class QuadModel(torch.nn.Module):
            def forward(self, coords):
                return (coords[:, 0]**2 + coords[:, 1]**2).unsqueeze(1)
        
        model = QuadModel()
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        
        # f(x, y) = 4
        f_fn = lambda x, y: torch.full_like(x, 4.0)
        
        loss = poisson_loss(model, x, y, f_fn=f_fn)
        self.assertLess(loss.item(), 1e-6)
        
        # f(x, y) = 0 (should fail)
        loss_fail = poisson_loss(model, x, y, f_fn=None)
        self.assertAlmostEqual(loss_fail.item(), 4.0**2)

    def test_grad_nan_safety(self):
        """Physics Validation: Checks that autograd doesn't crash on zero-grad paths."""
        model = PINN(num_layers=2, hidden_dim=4).to(self.device)
        # Constant inputs
        x = torch.zeros(5, device=self.device)
        y = torch.zeros(5, device=self.device)
        
        # Should execute without error
        loss = poisson_loss(model, x, y)
        self.assertTrue(torch.isfinite(loss))

    def test_boundary_gradient_loss(self):
        """Physics Validation: Checks the tangential derivative penalty."""
        from src.pinn.physics import boundary_gradient_loss
        
        # Solution u = x. grad u = (1, 0).
        class LinearModel(torch.nn.Module):
            def forward(self, coords):
                return coords[:, 0:1]
        
        model = LinearModel()
        
        # Boundary at y=0, x from 0 to 1. Normal is (0, -1). Tangent is (1, 0).
        x_bc = torch.linspace(0, 1, 10, device=self.device)
        y_bc = torch.zeros_like(x_bc)
        nx = torch.zeros_like(x_bc)
        ny = torch.full_like(x_bc, -1.0)
        
        # Tangent derivative = u_x * tx + u_y * ty = 1 * 1 + 0 * 0 = 1.
        # Loss should be 1.0.
        loss = boundary_gradient_loss(model, x_bc, y_bc, nx, ny)
        self.assertAlmostEqual(loss.item(), 1.0)
        
        # Boundary at x=0, y from 0 to 1. Normal is (-1, 0). Tangent is (0, -1).
        # Tangent derivative = u_x * tx + u_y * ty = 1 * 0 + 0 * -1 = 0.
        # Loss should be 0.0.
        x_bc2 = torch.zeros(10, device=self.device)
        y_bc2 = torch.linspace(0, 1, 10, device=self.device)
        nx2 = torch.full_like(x_bc2, -1.0)
        ny2 = torch.zeros_like(x_bc2)
        
        loss2 = boundary_gradient_loss(model, x_bc2, y_bc2, nx2, ny2)
        self.assertAlmostEqual(loss2.item(), 0.0)
