import torch

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from src.core.geometry import PolygonDomain, generate_koch_snowflake
from src.fem.solver import solve_fem
from src.pinn.solver import train
from src.core.viz import plot_results
from src.core.data import get_device
from scipy.interpolate import griddata

# Shared Domain Definition
def get_nested_domain():
    outer = generate_koch_snowflake(order=2, scale=1.0, center=(0.5, 0.5))
    inner = generate_koch_snowflake(order=1, scale=0.4, center=(0.5, 0.5))
    return PolygonDomain(outer, holes=[inner])

def bc_nested(x, y, b_ids=None):
    u = torch.zeros((x.shape[0], 1), device=x.device)
    if b_ids is not None:
        u[b_ids == 1] = 1.0
    return u

class FEMWrapper(torch.nn.Module):
    def __init__(self, mesh_p, u_fem, device='cpu'):
        super().__init__()
        self.mesh_p = mesh_p
        self.u_fem = u_fem
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, coords):
        x, y = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy()
        u_interp = griddata(self.mesh_p, self.u_fem, (x, y), method='linear')
        u_interp = np.nan_to_num(u_interp)
        return torch.tensor(u_interp, dtype=torch.float32, device=coords.device).unsqueeze(1)

    @property
    def device(self):
        return self.dummy.device

def generate_fem_solution():
    print("--- 1. Generating FEM Solution ---")
    domain = get_nested_domain()
    mesh, u_fem = solve_fem(domain, bc_nested, resolution=60)
    
    # Save to disk
    np.savez('fem_data.npz', mesh_p=mesh.p.T, u_fem=u_fem)
    
    # Plot
    fem_model = FEMWrapper(mesh.p.T, u_fem, device=get_device()).to(get_device())
    plot_results(fem_model, domain=domain, filename='nested_fem.png', resolution=250)
    print("FEM solution saved to fem_data.npz and nested_fem.png")

def generate_pinn_solution():
    print("--- 2. Generating PINN Solution ---")
    domain = get_nested_domain()
    config = {
        "num_layers": 4,
        "hidden_dim": 64,
        "activation": "sine",
        "adaptive_activations": False, 
        "use_fourier_features": False,
        "omega": 30.0,
        "output_transform": "sigmoid", 
        "adam_epochs": 2000,
        "lbfgs_epochs": 1000,
        "adam_lr": 0.001,
        "lambda_bc": 200.0,
        "lambda_range": 0.0, 
        "lambda_grad_bc": 0.0,
        "use_adaptive_sampling": False,
        "use_self_adaptive_weights": False,
        "seed": 42
    }
    pinn_model, history = train(domain=domain, bc_fn=bc_nested, config=config)
    
    # Sample and save high-resolution grid for error comparison later
    x_test, y_test = domain.sample_interior(5000)
    coords = torch.stack([x_test, y_test], dim=1).to(get_device())
    u_pinn = pinn_model(coords).detach().cpu().numpy().flatten()
    
    np.savez('pinn_data.npz', x_test=x_test.cpu().numpy(), y_test=y_test.cpu().numpy(), u_pinn=u_pinn)
    
    plot_results(pinn_model, domain=domain, filename='nested_pinn.png', resolution=250)
    print("PINN solution saved to pinn_data.npz and nested_pinn.png")

def compare_solutions():
    print("--- 3. Comparing Solutions ---")
    if not os.path.exists('fem_data.npz') or not os.path.exists('pinn_data.npz'):
        print("Error: Missing data files. Run FEM and PINN generation first.")
        return

    fem_data = np.load('fem_data.npz')
    pinn_data = np.load('pinn_data.npz')

    mesh_p = fem_data['mesh_p']
    u_fem = fem_data['u_fem']
    x_test = pinn_data['x_test']
    y_test = pinn_data['y_test']
    u_pinn = pinn_data['u_pinn']

    # Interpolate FEM onto PINN test points
    u_fem_interp = griddata(mesh_p, u_fem, (x_test, y_test), method='linear')
    
    mask = ~np.isnan(u_fem_interp)
    u_pinn_m = u_pinn[mask]
    u_fem_m = u_fem_interp[mask]

    if len(u_pinn_m) > 0:
        mse = np.mean((u_pinn_m - u_fem_m)**2)
        max_err = np.max(np.abs(u_pinn_m - u_fem_m))
        print(f"\nNested Annulus Results:")
        print(f"  - MSE: {mse:.8f}")
        print(f"  - Max Absolute Error: {max_err:.8f}")

        plt.figure(figsize=(10, 8), dpi=150)
        sc = plt.scatter(x_test[mask], y_test[mask], c=np.abs(u_pinn_m - u_fem_m), 
                    cmap='inferno', s=5, alpha=0.8)
        plt.colorbar(sc, label='Absolute Error |u_pinn - u_fem|')
        
        domain = get_nested_domain()
        v = domain.vertices.cpu().numpy()
        plt.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), 'k-', lw=1)
        for hole in domain.holes:
            vh = hole.cpu().numpy()
            plt.plot(np.append(vh[:, 0], vh[0, 0]), np.append(vh[:, 1], vh[0, 1]), 'k-', lw=1)
            
        plt.title('PINN vs FEM Error Distribution')
        plt.savefig('nested_error.png')
        plt.close()
        print("Comparison plot saved to nested_error.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINN vs FEM comparison modularly.")
    parser.add_argument('--step', type=str, choices=['fem', 'pinn', 'compare', 'all'], default='all',
                        help="Which step to run: fem, pinn, compare, or all (default)")
    args = parser.parse_args()

    if args.step in ['fem', 'all']:
        generate_fem_solution()
    if args.step in ['pinn', 'all']:
        generate_pinn_solution()
    if args.step in ['compare', 'all']:
        compare_solutions()
