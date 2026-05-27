import unittest

import torch
import numpy as np
from src.core.geometry import PolygonDomain, generate_koch_snowflake
from src.fem.solver import solve_fem
from src.pinn.solver import train
from tests.base_test import PINNTestCase

class TestFEMComparison(PINNTestCase):
    def test_fem_square_laplace(self):
        """Compare PINN and FEM on a simple square domain (Laplace)."""
        vertices = torch.tensor([[0,0], [1,0], [1,1], [0,1]], dtype=torch.float32)
        domain = PolygonDomain(vertices)
        
        def bc_fn(x, y):
            # u(x, y) = sin(pi*x) * sinh(pi*y) / sinh(pi)
            # Solves Delta u = 0 with u(x, 1) = sin(pi*x)
            return (torch.sin(torch.pi * x) * torch.sinh(torch.pi * y) / torch.sinh(torch.tensor(torch.pi))).unsqueeze(1)

        # 1. Solve with FEM
        mesh, u_fem = solve_fem(domain, bc_fn, resolution=40)
        
        # 2. Solve with PINN (very small model for quick test)
        config = {
            "num_layers": 3,
            "hidden_dim": 16,
            "activation": "sine",
            "adam_epochs": 500,
            "lbfgs_epochs": 100,
        }
        model = train(domain=domain, bc_fn=bc_fn, config=config)
        
        # 3. Compare at interior points
        x_test, y_test = domain.sample_interior(200)
        u_pinn = model(torch.stack([x_test, y_test], dim=1).to(model.device)).detach().cpu().numpy().flatten()
        
        # Simple interpolation for comparison
        from scipy.interpolate import griddata
        u_fem_interp = griddata(mesh.p.T, u_fem, (x_test.cpu().numpy(), y_test.cpu().numpy()), method='linear')
        
        # Mask NaNs from interpolation (if any points were slightly outside due to triangulation)
        mask = ~np.isnan(u_fem_interp)
        mse = np.mean((u_pinn[mask] - u_fem_interp[mask])**2)
        
        print(f"MSE PINN vs FEM (Square): {mse:.6f}")
        self.assertLess(mse, 0.01)

    def test_fem_snowflake_harmonic(self):
        """Compare PINN and FEM on a Koch Snowflake."""
        vertices = generate_koch_snowflake(order=2, scale=1.0, center=(0.5, 0.5))
        domain = PolygonDomain(vertices)
        
        def bc_harmonic(x, y):
            # u(x, y) = x^2 - y^2 (harmonic: Delta u = 0)
            return (x**2 - y**2).unsqueeze(1)

        # 1. Solve with FEM
        mesh, u_fem = solve_fem(domain, bc_harmonic, resolution=30)
        
        # 2. Solve with PINN
        config = {
            "num_layers": 4,
            "hidden_dim": 32,
            "activation": "sine",
            "adam_epochs": 800,
            "lbfgs_epochs": 200,
        }
        model = train(domain=domain, bc_fn=bc_harmonic, config=config)
        
        # 3. Compare
        x_test, y_test = domain.sample_interior(200)
        u_pinn = model(torch.stack([x_test, y_test], dim=1).to(model.device)).detach().cpu().numpy().flatten()
        
        from scipy.interpolate import griddata
        u_fem_interp = griddata(mesh.p.T, u_fem, (x_test.cpu().numpy(), y_test.cpu().numpy()), method='linear')
        
        mask = ~np.isnan(u_fem_interp)
        mse = np.mean((u_pinn[mask] - u_fem_interp[mask])**2)
        
        print(f"MSE PINN vs FEM (Snowflake): {mse:.6f}")
        # Snowflake is harder, higher tolerance
        self.assertLess(mse, 0.05)

if __name__ == '__main__':
    unittest.main()
