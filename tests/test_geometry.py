import torch
import numpy as np
from src.utils import PolygonDomain, generate_domain_data, generate_boundary_data
from tests.base_test import PINNTestCase

class TestGeometry(PINNTestCase):
    """
    Tests specific to non-convex polygon domains and complex sampling logic.
    """
    def test_polygon_domain_is_inside(self):
        """Tests the point-in-polygon logic for a square with a hole."""
        outer = torch.tensor([(0, 0), (1, 0), (1, 1), (0, 1)])
        hole = torch.tensor([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
        domain = PolygonDomain(outer, holes=[hole], device=self.device)
        
        # Points inside
        px = torch.tensor([0.2], device=self.device)
        py = torch.tensor([0.2], device=self.device)
        self.assertTrue(domain.is_inside(px, py)[0])
        
        # Points in the hole
        px_hole = torch.tensor([0.5], device=self.device)
        py_hole = torch.tensor([0.5], device=self.device)
        self.assertFalse(domain.is_inside(px_hole, py_hole)[0])

    def test_polygon_domain_sampling(self):
        """Verifies sampling respects the polygon boundaries and holes."""
        outer = torch.tensor([(0, 0), (2, 0), (2, 2), (0, 2)])
        hole = torch.tensor([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        domain = PolygonDomain(outer, holes=[hole], device=self.device)
        
        n = 500
        x, y = domain.sample_interior(n)
        
        self.assertEqual(len(x), n)
        self.assertEqual(x.device.type, self.device.type)
        
        inside = domain.is_inside(x, y)
        self.assertTrue(torch.all(inside))

    def test_polygon_boundary_sampling(self):
        """Verifies boundary sampling returns points on the edges."""
        outer = torch.tensor([(0, 0), (1, 0), (1, 1), (0, 1)])
        domain = PolygonDomain(outer, device=self.device)
        
        n = 100
        x, y = domain.sample_boundary(n)
        
        on_edge = (torch.abs(x - 0) < 1e-6) | (torch.abs(x - 1) < 1e-6) | \
                  (torch.abs(y - 0) < 1e-6) | (torch.abs(y - 1) < 1e-6)
        self.assertTrue(torch.all(on_edge))

    def test_non_convex_sampling_success(self):
        """Ensures rejection sampling works for a non-convex 'C' shape."""
        c_shape = torch.tensor([(0,0), (2,0), (2,2), (0,2), (0,1.5), (1.5,1.5), (1.5,0.5), (0,0.5)])
        domain = PolygonDomain(c_shape, device=self.device)
        
        n = 100
        x, y = domain.sample_interior(n)
        self.assertTrue(torch.all(domain.is_inside(x, y)))

    def test_custom_bc_fn_mapping(self):
        """Tests that generate_boundary_data correctly applies a custom bc_fn on a polygon."""
        outer = [(0, 0), (1, 0), (1, 1), (0, 1)]
        domain = PolygonDomain(outer)
        
        def constant_bc(x, y):
            return torch.ones((x.shape[0], 1), device=x.device) * 5.0
            
        _, _, u_bc = generate_boundary_data(50, device=self.device, domain=domain, bc_fn=constant_bc)
        self.assertTrue(torch.all(u_bc == 5.0))
