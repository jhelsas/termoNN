import unittest
import numpy as np
import torch
from src.core.geometry import PolygonDomain
from src.fem.solver import solve_fem
from tests.base_test import PINNTestCase

class TestFEM(PINNTestCase):
    """
    Tests for the Traditional Finite Element Method solver.
    Verifies that the numerical ground truth is accurate and stable.
    """
    def test_fem_laplace_linear(self):
        """FEM Validation: Linear solution on a square."""
        outer = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.]])
        domain = PolygonDomain(outer)
        
        # BC: u = x (satisfied by Laplace)
        def bc_fn(x, y):
            return x.unsqueeze(1)
            
        mesh, u = solve_fem(domain, bc_fn, n_interior=100, n_boundary=40)
        
        # In a linear case, nodes at (x, y) should have u approx x
        nodes = mesh.p.T # (N, 2)
        x_coords = nodes[:, 0]
        
        # Check mean absolute error
        error = np.mean(np.abs(u - x_coords))
        self.assertLess(error, 0.05)

    def test_fem_poisson_constant_source(self):
        """FEM Validation: Poisson with constant source."""
        outer = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.]])
        domain = PolygonDomain(outer)
        
        # u = x^2 => u_xx = 2, u_yy = 0 => f = 2
        def bc_fn(x, y):
            return (x**2).unsqueeze(1)
        def f_fn(x, y):
            return torch.full_like(x, 2.0)
            
        mesh, u = solve_fem(domain, bc_fn, f_fn=f_fn, n_interior=200, n_boundary=50)
        
        nodes = mesh.p.T
        x_coords = nodes[:, 0]
        analytic = x_coords**2
        
        error = np.mean(np.abs(u - analytic))
        self.assertLess(error, 0.05)

    def test_fem_hole_masking(self):
        """FEM Validation: Verify solving works on a domain with a hole."""
        outer = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.]])
        hole = torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
        domain = PolygonDomain(outer, holes=[hole])
        
        def bc_fn(x, y):
            return torch.zeros_like(x).unsqueeze(1)
            
        mesh, u = solve_fem(domain, bc_fn, n_interior=100, n_boundary=40)
        
        # Verify solution exists and matches boundary nodes
        self.assertGreater(len(u), 0)
        self.assertEqual(len(u), mesh.p.shape[1])

    def test_fem_output_format(self):
        """FEM Validation: Check return types and shapes."""
        outer = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.]])
        domain = PolygonDomain(outer)
        bc_fn = lambda x, y: torch.zeros_like(x).unsqueeze(1)
        
        mesh, u = solve_fem(domain, bc_fn)
        from skfem import MeshTri
        self.assertIsInstance(mesh, MeshTri)
        self.assertIsInstance(u, np.ndarray)
        self.assertEqual(u.ndim, 1)

if __name__ == '__main__':
    unittest.main()
