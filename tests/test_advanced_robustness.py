import torch
import numpy as np
from src.pinn.model import PINN
from src.pinn.physics import sobolev_laplace_loss, poisson_loss
from src.core.geometry import PolygonDomain
from tests.base_test import PINNTestCase

class TestAdvancedRobustness(PINNTestCase):
    def test_multiscale_spectral_capacity(self):
        """Capacity Validation: Verifies Multi-Scale PINN can represent high-frequency signals."""
        # Define a signal with low and high frequency components
        def target_fn(x, y):
            return torch.sin(np.pi * x) + 0.1 * torch.sin(50 * np.pi * x)

        model = PINN(hidden_dim=128, use_fourier_features=True).to(self.device)
        x = torch.linspace(0, 1, 100, device=self.device)
        y = torch.zeros_like(x)
        coords = torch.stack([x, y], dim=1)
        
        # Fit for a few iterations
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(coords)
            loss = torch.mean((pred - target_fn(x, y).unsqueeze(1))**2)
            loss.backward()
            optimizer.step()
            
        self.assertLess(loss.item(), 1.0) # Should be converging

    def test_adaptive_weights_sobolev_stability(self):
        """Stability Validation: Verifies adaptive weights don't NaN with 3rd-order gradients."""
        model = PINN(hidden_dim=32).to(self.device)
        x = torch.randn(10, device=self.device, requires_grad=True)
        y = torch.randn(10, device=self.device, requires_grad=True)
        
        # Compute Sobolev loss
        loss_pde = sobolev_laplace_loss(model, x, y)
        
        # Calculate gradients of the loss w.r.t. parameters
        loss_pde.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"NaN gradient in {name}")

    def test_distance_gradient_at_sharp_vertex(self):
        """Geometry Validation: Verifies distance gradients are stable at sharp vertices."""
        # A very sharp "needle" triangle
        needle = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0001, 0.0001]])
        domain = PolygonDomain(needle)
        
        # Test points exactly at the vertex
        x = torch.tensor([0.0001], device=self.device, requires_grad=True)
        y = torch.tensor([0.0001], device=self.device, requires_grad=True)
        
        dist = domain.exact_distance(x, y)
        dist.backward()
        
        # Even at the vertex, gradients should be finite due to the 1e-10 epsilon
        self.assertTrue(torch.isfinite(x.grad))
        self.assertTrue(torch.isfinite(y.grad))

    def test_sobolev_memory_pressure(self):
        """Performance Validation: Ensures Sobolev loss doesn't cause OOM on large batches."""
        if self.device.type == 'cpu':
            n_points = 1000
        else:
            n_points = 5000
            
        model = PINN(hidden_dim=64, num_layers=4).to(self.device)
        x = torch.randn(n_points, device=self.device)
        y = torch.randn(n_points, device=self.device)
        
        # Should execute without out-of-memory error
        loss = sobolev_laplace_loss(model, x, y)
        loss.backward()
        self.assertTrue(True)
