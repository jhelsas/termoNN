import torch
import numpy as np
import matplotlib.pyplot as plt
from src.core.geometry import PolygonDomain, generate_koch_snowflake
from src.core.fem import solve_fem, interpolate_fem
from src.pinn.solver import train
from src.core.viz import plot_results
from src.core.data import get_device

class FEMWrapper(torch.nn.Module):
    """Wraps FEM solution to be used by PINN visualization tools."""
    def __init__(self, mesh, u_fem, device='cpu'):
        super().__init__()
        self.mesh = mesh
        self.u_fem = u_fem
        self.device_val = device
        # Dummy parameter to make viz.py happy (it calls .device on parameters)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, coords):
        # coords is (N, 2)
        x, y = coords[:, 0], coords[:, 1]
        u_interp = interpolate_fem(self.mesh, self.u_fem, x, y)
        # Convert NaNs to 0 for visualization outside domain (viz.py masks anyway)
        u_interp = np.nan_to_num(u_interp)
        return torch.tensor(u_interp, dtype=torch.float32, device=coords.device).unsqueeze(1)

def run_snowflake_fem_plot():
    print("\n--- FEM on Koch Snowflake ---")
    vertices = generate_koch_snowflake(order=3, scale=1.0, center=(0.5, 0.5))
    domain = PolygonDomain(vertices)
    
    def bc_harmonic(x, y):
        return (torch.sin(torch.pi * (x-0.5)) * torch.cosh(torch.pi * (y-0.5))).unsqueeze(1)

    mesh, u_fem = solve_fem(domain, bc_harmonic, resolution=60)
    fem_model = FEMWrapper(mesh, u_fem, device=get_device())
    
    plot_results(fem_model, domain=domain, filename='fem_snowflake.png')
    print("FEM result saved to fem_snowflake.png")

def run_nested_comparison():
    print("\n--- Nested Snowflake Annulus Comparison ---")
    # Reduced fractal order for faster execution in this environment
    outer = generate_koch_snowflake(order=2, scale=1.0, center=(0.5, 0.5))
    inner = generate_koch_snowflake(order=1, scale=0.4, center=(0.5, 0.5))
    domain = PolygonDomain(outer, holes=[inner])
    
    def bc_nested(x, y, b_ids=None):
        u = torch.zeros((x.shape[0], 1), device=x.device)
        if b_ids is not None:
            # ID 0 is outer snowflake, ID 1 is the hole (set to 1.0)
            u[b_ids == 1] = 1.0
        return u

    # 1. FEM Solution
    print("Solving with FEM...")
    mesh, u_fem = solve_fem(domain, bc_nested, resolution=30)
    fem_model = FEMWrapper(mesh, u_fem, device=get_device())
    plot_results(fem_model, domain=domain, filename='nested_fem.png', resolution=100)

    # 2. PINN Solution
    print("Solving with PINN...")
    config = {
        "num_layers": 3,
        "hidden_dim": 32,
        "activation": "sine",
        "adam_epochs": 300,
        "lbfgs_epochs": 50,
        "lambda_bc": 50.0,
    }
    pinn_model = train(domain=domain, bc_fn=bc_nested, config=config)
    plot_results(pinn_model, domain=domain, filename='nested_pinn.png', resolution=100)

    # 3. Error Comparison
    x_test, y_test = domain.sample_interior(500)
    coords = torch.stack([x_test, y_test], dim=1)
    
    u_pinn = pinn_model(coords).detach().cpu().numpy().flatten()
    u_fem_interp = interpolate_fem(mesh, u_fem, x_test, y_test)
    
    mask = ~np.isnan(u_fem_interp)
    u_pinn_m = u_pinn[mask]
    u_fem_m = u_fem_interp[mask]
    
    if len(u_pinn_m) > 0:
        mse = np.mean((u_pinn_m - u_fem_m)**2)
        max_err = np.max(np.abs(u_pinn_m - u_fem_m))
        print(f"\nNested Annulus Results:")
        print(f"  - MSE: {mse:.8f}")
        print(f"  - Max Absolute Error: {max_err:.8f}")

        # 4. Plot Absolute Error
        plt.figure(figsize=(10, 8), dpi=150)
        sc = plt.scatter(x_test.cpu()[mask], y_test.cpu()[mask], c=np.abs(u_pinn_m - u_fem_m), 
                    cmap='inferno', s=5, alpha=0.8)
        plt.colorbar(sc, label='Absolute Error |u_pinn - u_fem|')
        
        v = domain.vertices.cpu().numpy()
        plt.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), 'k-', lw=1)
        for hole in domain.holes:
            vh = hole.cpu().numpy()
            plt.plot(np.append(vh[:, 0], vh[0, 0]), np.append(vh[:, 1], vh[0, 1]), 'k-', lw=1)
            
        plt.title('PINN vs FEM Error Distribution')
        plt.savefig('nested_error.png')
        plt.close()
    
    print("Comparison plots saved: nested_fem.png, nested_pinn.png, nested_error.png")

if __name__ == "__main__":
    run_snowflake_fem_plot()
    run_nested_comparison()
