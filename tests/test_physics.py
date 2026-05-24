import torch
import torch.nn as nn
import numpy as np
from src.physics import laplace_loss, boundary_loss, poisson_loss
from src.utils import generate_domain_data
from tests.base_test import PINNTestCase

class TestPhysics(PINNTestCase):
    def test_laplace_loss_zeros(self):
        """Physics Validation: Linear functions (u=ax+by+c) must have zero Laplace loss."""
        model = nn.Linear(2, 1).to(self.device)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)
        
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_laplace_loss_linear_sloped(self):
        """Physics Validation: u = 3x + 4y + 5 must have zero Laplace loss."""
        class SlopedLinearModel(nn.Module):
            def forward(self, x):
                return (3 * x[:, 0] + 4 * x[:, 1] + 5).unsqueeze(1)
        
        model = SlopedLinearModel().to(self.device)
        x = torch.linspace(0, 1, 15, device=self.device)
        y = torch.linspace(0, 1, 15, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_laplace_loss_quad_identity(self):
        """Physics Validation: u = x^2 + y^2 => u_xx + u_yy = 2 + 2 = 4. Loss = 4^2 = 16."""
        class QuadModel(nn.Module):
            def forward(self, x):
                return (x[:, 0]**2 + x[:, 1]**2).unsqueeze(1)
        
        model = QuadModel().to(self.device)
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 16.0, places=4)

    def test_laplace_loss_saddle(self):
        """Physics Validation: u = x^2 - y^2 is harmonic => u_xx + u_yy = 2 - 2 = 0."""
        class SaddleModel(nn.Module):
            def forward(self, x):
                return (x[:, 0]**2 - x[:, 1]**2).unsqueeze(1)
        
        model = SaddleModel().to(self.device)
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_boundary_loss_perfect_fit(self):
        """Verifies BC loss is zero when the model matches target values."""
        model = lambda x: torch.ones((x.shape[0], 1), device=self.device)
        x_bc = torch.zeros(10, device=self.device)
        y_bc = torch.zeros(10, device=self.device)
        u_bc = torch.ones((10, 1), device=self.device)
        loss = boundary_loss(model, x_bc, y_bc, u_bc)
        self.assertEqual(loss.item(), 0.0)

    def test_laplace_harmonic_function(self):
        """Physics Validation: Analytical harmonic function must have 0 loss."""
        class HarmonicModel(nn.Module):
            def forward(self, x):
                return (torch.sin(x[:, 0]) * torch.cosh(x[:, 1])).unsqueeze(1)
        
        model = HarmonicModel().to(self.device)
        x = torch.linspace(0, 0.5, 20, device=self.device)
        y = torch.linspace(0, 0.5, 20, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_hessian_independence(self):
        """Verifies that u_xx and u_yy are calculated from correct dimensions."""
        class XYModel(nn.Module):
            def forward(self, x):
                return (x[:, 0]**2).unsqueeze(1) # u_xx=2, u_yy=0
        
        model = XYModel().to(self.device)
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 4.0, places=5)

    def test_physics_batch_consistency(self):
        """Ensures the loss is the mean of individual point losses."""
        class QuadModel(nn.Module):
            def forward(self, x): return (x[:, 0]**2).unsqueeze(1)
        
        model = QuadModel().to(self.device)
        x = torch.tensor([0.1, 0.5], device=self.device)
        y = torch.tensor([0.1, 0.5], device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertEqual(loss.item(), 4.0)

    def test_numerical_stability_small_values(self):
        """Checks for stability with small input coordinates."""
        from src.model import PINN
        model = PINN().to(self.device)
        x = torch.full((10,), 1e-6, device=self.device)
        y = torch.full((10,), 1e-6, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertTrue(torch.isfinite(loss))

    def test_laplace_loss_large_coordinates(self):
        """Physics Validation: Residue calculation should be invariant to coordinate shift."""
        class ShiftedQuad(nn.Module):
            def forward(self, x):
                # u = (x-100)^2 + (y-100)^2 => u_xx+u_yy = 4
                return ((x[:, 0]-100)**2 + (x[:, 1]-100)**2).unsqueeze(1)
        
        model = ShiftedQuad().to(self.device)
        x = torch.linspace(100, 101, 10, device=self.device)
        y = torch.linspace(100, 101, 10, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 16.0, places=4)

    def test_poisson_loss_with_source(self):
        """Physics Validation: u = x^2 => u_xx=2, u_yy=0. f=2. Residue = 2+0-2 = 0."""
        class XQuad(nn.Module):
            def forward(self, x): return (x[:, 0]**2).unsqueeze(1)
        
        model = XQuad().to(self.device)
        f_fn = lambda x, y: torch.full_like(x, 2.0)
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        loss = poisson_loss(model, x, y, f_fn=f_fn)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_poisson_sine_source(self):
        """Physics Validation: u = sin(x)sin(y) => u_xx + u_yy = -2sin(x)sin(y). f = -2sin(x)sin(y)."""
        class SineSineModel(nn.Module):
            def forward(self, x):
                return (torch.sin(x[:, 0]) * torch.sin(x[:, 1])).unsqueeze(1)
        
        model = SineSineModel().to(self.device)
        f_fn = lambda x, y: -2 * torch.sin(x) * torch.sin(y)
        x = torch.linspace(0, np.pi, 20, device=self.device)
        y = torch.linspace(0, np.pi, 20, device=self.device)
        loss = poisson_loss(model, x, y, f_fn=f_fn)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_poisson_variable_source(self):
        """Physics Validation: u = x^3 + y^3 => u_xx + u_yy = 6x + 6y. f = 6x + 6y."""
        class CubicModel(nn.Module):
            def forward(self, x):
                return (x[:, 0]**3 + x[:, 1]**3).unsqueeze(1)
        
        model = CubicModel().to(self.device)
        f_fn = lambda x, y: 6*x + 6*y
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        loss = poisson_loss(model, x, y, f_fn=f_fn)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
