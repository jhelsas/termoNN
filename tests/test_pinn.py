import torch
import unittest
import torch.nn as nn
from src.model import PINN
from src.physics import laplace_loss, boundary_loss
from src.utils import generate_domain_data, generate_boundary_data, get_device

class TestPINN(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

    def test_model_output_shape(self):
        """Verifies the MLP returns the correct tensor shape (Batch x 1)."""
        model = PINN(input_dim=2, output_dim=1).to(self.device)
        test_input = torch.randn(10, 2, device=self.device)
        output = model(test_input)
        self.assertEqual(output.shape, (10, 1))

    def test_laplace_loss_zeros(self):
        """Physics Validation: Linear functions must have zero Laplace loss."""
        model = nn.Linear(2, 1).to(self.device)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)
        
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_laplace_loss_non_zero(self):
        """Physics Validation: Non-Laplacian functions must have non-zero loss."""
        class QuadModel(nn.Module):
            def forward(self, x):
                return (x[:, 0]**2 + x[:, 1]**2).unsqueeze(1)
        
        model = QuadModel().to(self.device)
        x = torch.linspace(0, 1, 10, device=self.device)
        y = torch.linspace(0, 1, 10, device=self.device)
        loss = laplace_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 16.0, places=4)

    def test_boundary_loss_perfect_fit(self):
        """Verifies BC loss is zero when the model matches target boundary values."""
        model = lambda x: torch.ones((x.shape[0], 1), device=self.device)
        x_bc = torch.zeros(10, device=self.device)
        y_bc = torch.zeros(10, device=self.device)
        u_bc = torch.ones((10, 1), device=self.device)
        loss = boundary_loss(model, x_bc, y_bc, u_bc)
        self.assertEqual(loss.item(), 0.0)

    def test_data_generation(self):
        """Verifies the utility functions sample points correctly."""
        x, y = generate_domain_data(100, device=self.device)
        self.assertEqual(x.device.type, self.device.type)
        
        x_bc, y_bc, u_bc = generate_boundary_data(40, device=self.device)
        self.assertEqual(u_bc.device.type, self.device.type)
        self.assertTrue(torch.all(u_bc >= 0))

    def test_boundary_accuracy(self):
        """Verifies that boundary points are actually on the edges of [0, 1]x[0, 1]."""
        x_bc, y_bc, _ = generate_boundary_data(100, device=self.device)
        # For each point, either x or y must be 0 or 1
        on_boundary = (x_bc == 0) | (x_bc == 1) | (y_bc == 0) | (y_bc == 1)
        self.assertTrue(torch.all(on_boundary), "Some BC points are not on the boundary edges.")

    def test_domain_range(self):
        """Ensures domain points are within the [0, 1] range."""
        x, y = generate_domain_data(100, device=self.device)
        self.assertTrue(torch.all(x >= 0) and torch.all(x <= 1))
        self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))

    def test_activation_differentiability(self):
        """Ensures the model activation allows non-zero second derivatives."""
        model = PINN().to(self.device)
        x = torch.linspace(0, 1, 10, requires_grad=True, device=self.device)
        y = torch.linspace(0, 1, 10, requires_grad=True, device=self.device)
        coords = torch.stack([x, y], dim=1)
        u = model(coords)
        u_x = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 0]
        u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
        
        # In a randomly initialized Tanh network, u_xx should not be identically zero
        self.assertFalse(torch.all(u_xx == 0), "Second derivative is zero. Check activation function (e.g., ReLU is bad for PINNs).")

    def test_plotting_smoke(self):
        """Smoke test to ensure plotting runs and saves a file."""
        import os
        from main import plot_results
        model = PINN().to(self.device)
        plot_results(model)
        self.assertTrue(os.path.exists("solution.png"))

    def test_gradient_flow(self):
        """Integration Test: Confirms weights update after a single backward pass."""
        model = PINN(hidden_dim=5).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_params = [p.clone().detach() for p in model.parameters()]
        
        x, y = generate_domain_data(10, device=self.device)
        loss = laplace_loss(model, x, y)
        loss.backward()
        optimizer.step()
        
        # Check that at least some parameters changed
        updated = False
        for p_init, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_new):
                updated = True
                break
        self.assertTrue(updated, "Model parameters were not updated after training step.")

    def test_model_determinism(self):
        """Ensures the model produces same output for same input (no stochastic layers)."""
        model = PINN().to(self.device)
        x = torch.randn(5, 2, device=self.device)
        out1 = model(x)
        out2 = model(x)
        self.assertTrue(torch.equal(out1, out2), "Model output is non-deterministic (check for Dropout or BatchNorm).")

    def test_physics_loss_gradient_existence(self):
        """Ensures the laplace_loss itself has a gradient for at least some parameters."""
        model = PINN().to(self.device)
        x, y = generate_domain_data(10, device=self.device)
        loss = laplace_loss(model, x, y)
        loss.backward()
        
        has_grad = any(param.grad is not None for param in model.parameters())
        self.assertTrue(has_grad, "No parameters received gradients from laplace_loss.")

if __name__ == "__main__":
    unittest.main()
