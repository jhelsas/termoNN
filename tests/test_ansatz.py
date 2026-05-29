import torch
from src.pinn.model import PINN, ExactBoundaryAnsatz
from src.core.geometry import PolygonDomain
from tests.base_test import PINNTestCase

class TestAnsatz(PINNTestCase):
    def setUp(self):
        super().setUp()
        # Create a simple nested square domain
        outer = [(0, 0), (2, 0), (2, 2), (0, 2)]
        inner = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
        self.domain = PolygonDomain(outer, holes=[inner], device=self.device)
        self.core_model = PINN(hidden_dim=10, num_layers=2).to(self.device)
        self.ansatz = ExactBoundaryAnsatz(self.core_model, self.domain, bc_mode='nested').to(self.device)

    def test_ansatz_outer_boundary(self):
        """Verifies u=0 exactly on the outer boundary."""
        # Points on the outer boundary
        x = torch.tensor([0.0, 1.0, 2.0, 1.0], device=self.device)
        y = torch.tensor([1.0, 0.0, 1.0, 2.0], device=self.device)
        coords = torch.stack([x, y], dim=1)
        
        u = self.ansatz(coords)
        # Because we added 1e-10 inside the sqrt for gradient safety, 
        # the exact distance at boundary is sqrt(1e-10) ~ 1e-5.
        # So we relax the tolerance slightly to account for the gradient-safe distance.
        self.assertTrue(torch.allclose(u, torch.zeros_like(u), atol=1e-4))

    def test_ansatz_inner_boundary(self):
        """Verifies u=1 exactly on the inner boundary."""
        # Points on the inner boundary
        x = torch.tensor([0.5, 1.0, 1.5, 1.0], device=self.device)
        y = torch.tensor([1.0, 0.5, 1.0, 1.5], device=self.device)
        coords = torch.stack([x, y], dim=1)
        
        u = self.ansatz(coords)
        self.assertTrue(torch.allclose(u, torch.ones_like(u), atol=1e-4))

    def test_ansatz_interior_gradients(self):
        """Verifies that the Ansatz preserves autograd for the PDE."""
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        y = torch.tensor([1.0], device=self.device, requires_grad=True)
        coords = torch.stack([x, y], dim=1)
        
        u = self.ansatz(coords)
        grads = torch.autograd.grad(u, coords, create_graph=True)[0]
        
        self.assertTrue(torch.isfinite(grads).all())
        self.assertFalse(torch.all(grads == 0)) # Should have non-zero gradient inside

    def test_ansatz_invalid_mode(self):
        """Ensures an unsupported BC mode raises NotImplementedError."""
        invalid_ansatz = ExactBoundaryAnsatz(self.core_model, self.domain, bc_mode='unknown')
        coords = torch.tensor([[1.0, 1.0]], device=self.device)
        with self.assertRaises(NotImplementedError):
            invalid_ansatz(coords)
