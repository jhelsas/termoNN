import unittest
import torch
from src.pinn.model import PINN
from src.pinn.physics import range_loss, poisson_loss
from tests.base_test import PINNTestCase

class TestPhysicsStability(PINNTestCase):
    """
    Advanced stability tests for PINN physics constraints.
    Focuses on gradient behavior, adaptive weight stability, and high-frequency sources.
    """

    def test_range_loss_gradient_direction(self):
        """Stability: Verify range_loss gradients point in the direction of the valid range."""
        # Setup: A point u=1.5 with range [0.0, 1.0]. 
        # The loss is (1.5-1.0)^2 = 0.25. 
        # The gradient dL/du should be 2*(1.5-1.0) = 1.0.
        # Since u = model(x), a positive gradient dL/du means we need to DECREASE u to minimize loss.
        
        u = torch.tensor([[1.5]], requires_grad=True, device=self.device)
        loss = range_loss(u, min_val=0.0, max_val=1.0)
        loss.backward()
        
        # dL/du should be positive (1.0) because we are above the max_val
        self.assertGreater(u.grad.item(), 0)
        self.assertAlmostEqual(u.grad.item(), 1.0)
        
        # Setup: A point u=-0.5 with range [0.0, 1.0].
        # dL/du should be negative (-1.0) because we need to INCREASE u to minimize loss.
        u_low = torch.tensor([[-0.5]], requires_grad=True, device=self.device)
        loss_low = range_loss(u_low, min_val=0.0, max_val=1.0)
        loss_low.backward()
        
        self.assertLess(u_low.grad.item(), 0)
        self.assertAlmostEqual(u_low.grad.item(), -1.0)

    def test_poisson_residue_high_frequency_source(self):
        """Stability: Verify the Poisson residue computation is stable with high-frequency sources."""
        # u(x, y) = sin(kx) / k^2  => u_x = cos(kx)/k => u_xx = -sin(kx)
        # Delta u = -sin(kx)
        k = 100.0
        def f_fn(x, y):
            return -torch.sin(k * x)
            
        class HighFreqModel(torch.nn.Module):
            def forward(self, coords):
                return (torch.sin(k * coords[:, 0]) / (k**2)).unsqueeze(1)
        
        model = HighFreqModel()
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        
        loss = poisson_loss(model, x, y, f_fn=f_fn)
        # Loss should be zero (or very close to machine epsilon)
        self.assertLess(loss.item(), 1e-4)

    def test_adaptive_weight_warmup_bounds(self):
        """Stability: Ensure that the gradient-based weighting logic (from solver.py) is numerically bounded."""
        # Simulate the logic inside solver.py's adaptive weighting
        model = PINN(hidden_dim=10, num_layers=3, activation='sine').to(self.device)
        
        x = torch.rand(100, device=self.device)
        y = torch.rand(100, device=self.device)
        
        # 1. Compute PDE grads
        model.zero_grad()
        loss_pde = poisson_loss(model, x, y)
        loss_pde.backward(retain_graph=True)
        grads_pde = [p.grad.abs().max() for p in model.parameters() if p.grad is not None]
        max_grad_pde = torch.stack(grads_pde).max()
        
        # 2. Compute BC grads
        model.zero_grad()
        u_pred = model(torch.stack([x, y], dim=1))
        loss_bc = torch.mean(u_pred**2) # Mock BC loss
        loss_bc.backward()
        grads_bc = [p.grad.abs().mean() for p in model.parameters() if p.grad is not None]
        mean_grad_bc = torch.stack(grads_bc).mean()
        
        # 3. Compute weight
        lambda_bc = max_grad_pde / (mean_grad_bc + 1e-8)
        
        # Verify lambda_bc is finite and not NaN
        self.assertTrue(torch.isfinite(lambda_bc))
        self.assertGreater(lambda_bc.item(), 0)
        
    def test_gradient_consistency_with_no_source(self):
        """Stability: Verify that poisson_loss(f=None) is identical to laplace_loss."""
        from src.pinn.physics import laplace_loss
        model = PINN(hidden_dim=10, num_layers=3, activation='sine').to(self.device)
        x = torch.rand(50, device=self.device)
        y = torch.rand(50, device=self.device)
        
        l1 = poisson_loss(model, x, y, f_fn=None)
        l2 = laplace_loss(model, x, y)
        
        self.assertAlmostEqual(l1.item(), l2.item(), places=6)

if __name__ == '__main__':
    unittest.main()
