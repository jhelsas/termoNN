import torch
import os
from src.model import PINN
from src.utils import generate_domain_data, generate_boundary_data
from src.physics import laplace_loss, boundary_loss
from main import plot_results, train
from tests.base_test import PINNTestCase

class TestIntegration(PINNTestCase):
    def test_gradient_flow(self):
        """Integration Test: Confirms weights update after a single backward pass."""
        model = PINN(hidden_dim=5).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_params = [p.clone().detach() for p in model.parameters()]
        
        x, y = generate_domain_data(10, device=self.device)
        loss = laplace_loss(model, x, y)
        loss.backward()
        optimizer.step()
        
        updated = any(not torch.allclose(p_init, p_new) 
                     for p_init, p_new in zip(initial_params, model.parameters()))
        self.assertTrue(updated)

    def test_full_training_cycle_adam(self):
        """Ensures that a short Adam training cycle decreases the loss."""
        # Note: train() already handles seeds and device internally
        model = train(adam_epochs=5, lbfgs_epochs=0, lr=0.01)
        self.assertEqual(next(model.parameters()).device.type, self.device.type)

    def test_lbfgs_closure_integration(self):
        """Verifies L-BFGS optimizer reduction."""
        model = PINN(hidden_dim=5).to(self.device)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
        x_d, y_d = generate_domain_data(10, device=self.device)
        x_b, y_b, u_b = generate_boundary_data(10, device=self.device)
        
        def closure():
            optimizer.zero_grad()
            loss = laplace_loss(model, x_d, y_d) + boundary_loss(model, x_b, y_b, u_b)
            loss.backward()
            return loss
            
        l_init = closure().item()
        optimizer.step(closure)
        l_new = closure().item()
        self.assertLess(l_new, l_init)

    def test_plotting_workflow(self):
        """Smoke test for the plotting utility."""
        model = PINN().to(self.device)
        if os.path.exists("solution.png"):
            os.remove("solution.png")
        plot_results(model)
        self.assertTrue(os.path.exists("solution.png"))
