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
        """Verifies boundary sampling returns points on the edges and correct IDs."""
        outer = torch.tensor([(0, 0), (1, 0), (1, 1), (0, 1)])
        domain = PolygonDomain(outer, device=self.device)
        
        n = 100
        x, y, b_ids = domain.sample_boundary(n)
        
        # Every point is on edge
        on_edge = (torch.abs(x - 0) < 1e-6) | (torch.abs(x - 1) < 1e-6) | \
                  (torch.abs(y - 0) < 1e-6) | (torch.abs(y - 1) < 1e-6)
        self.assertTrue(torch.all(on_edge))
        # Every point belongs to polygon 0
        self.assertTrue(torch.all(b_ids == 0))

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

    def test_koch_snowflake_generation(self):
        """Verifies Koch Snowflake vertices are generated and closed."""
        from src.utils import generate_koch_snowflake
        order = 2
        vertices = generate_koch_snowflake(order=order, scale=1.0)
        
        # Order N snowflake has 3 * 4^N vertices
        expected_len = 3 * (4**order)
        self.assertEqual(len(vertices), expected_len)
        
        # Check coordinates are finite
        self.assertTrue(torch.isfinite(vertices).all())
        
        # Ensure it fits roughly in the specified scale
        self.assertTrue(vertices.max() <= 1.5)
        self.assertTrue(vertices.min() >= -0.5)

    def test_koch_snowflake_domain(self):
        """Verifies we can sample inside a Koch Snowflake."""
        from src.utils import generate_koch_snowflake
        vertices = generate_koch_snowflake(order=2)
        domain = PolygonDomain(vertices)
        
        n = 50
        x, y = domain.sample_interior(n, device=self.device)
        self.assertEqual(len(x), n)
        self.assertTrue(torch.all(domain.is_inside(x, y)))

    def test_multi_hole_id_assignment(self):
        """Geometry Validation: Verifies that multiple holes get unique sequential IDs."""
        outer = torch.tensor([[0,0], [10,0], [10,10], [0,10]])
        h1 = torch.tensor([[1,1], [2,1], [2,2], [1,2]])
        h2 = torch.tensor([[3,3], [4,3], [4,4], [3,4]])
        domain = PolygonDomain(outer, holes=[h1, h2])
        
        # Sample points and check IDs
        _, _, b_ids = domain.sample_boundary(100)
        unique_ids = torch.unique(b_ids).cpu().numpy().tolist()
        self.assertIn(0, unique_ids) # Outer
        self.assertIn(1, unique_ids) # Hole 1
        self.assertIn(2, unique_ids) # Hole 2

    def test_polygon_device_consistency(self):
        """Geometry Validation: Ensures sampled points are on the domain's device."""
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        domain = PolygonDomain(outer, device=self.device)
        x, y = domain.sample_interior(10)
        self.assertEqual(x.device.type, self.device.type)
        
        # Test boundary sampling device
        xb, yb, bids = domain.sample_boundary(10)
        self.assertEqual(xb.device.type, self.device.type)

    def test_bc_fn_signature_detection(self):
        """Integration Validation: Verifies that generate_boundary_data detects b_ids in signature."""
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        domain = PolygonDomain(outer)
        
        # Function WITH b_ids
        def bc_with_ids(x, y, b_ids):
            return torch.ones_like(x).unsqueeze(1)
            
        # Function WITHOUT b_ids
        def bc_without_ids(x, y):
            return torch.zeros_like(x).unsqueeze(1)
            
        # These should run without TypeError
        from src.utils import generate_boundary_data
        _ = generate_boundary_data(10, domain=domain, bc_fn=bc_with_ids)
        _ = generate_boundary_data(10, domain=domain, bc_fn=bc_without_ids)
