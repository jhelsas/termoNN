import torch
import warnings
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from src.pinn.solver import train
from src.core.viz import plot_results, plot_history, plot_comparison
from src.core.geometry import PolygonDomain, generate_koch_snowflake
from src.core.data import get_device
from src.fem.solver import solve_fem
from scipy.interpolate import griddata

# Suppress benign CUDA/cuBLAS context warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

def bc_harmonic(x, y):
    # u(x, y) = sin(pi*(x-0.5)) * cosh(pi*(y-0.5))
    return (torch.sin(torch.pi * (x-0.5)) * torch.cosh(torch.pi * (y-0.5))).unsqueeze(1)

def bc_nested(x, y, b_ids=None):
    u = torch.zeros((x.shape[0], 1), device=x.device)
    if b_ids is not None:
        # ID 0 is outer snowflake, ID 1 is the hole
        u[b_ids == 1] = 1.0
    return u

def solve_koch_snowflake_example():
    """Example: Solving Laplace Equation on a Koch Snowflake fractal."""
    print("--- Koch Snowflake Harmonic Example ---")
    vertices = generate_koch_snowflake(order=3, scale=1.0, center=(0.5, 0.5))
    domain = PolygonDomain(vertices)
    
    config = {
        "num_layers": 4,
        "hidden_dim": 32,
        "activation": "sine",
        "adam_epochs": 1500,
        "lbfgs_epochs": 300,
    }

    print("\n--- Scenario: harmonic ---")
    model, history = train(domain=domain, bc_fn=bc_harmonic, config=config)
    plot_results(model, domain=domain, filename='snowflake_harmonic.png')
    plot_history(history, filename='snowflake_harmonic_history.png')

def solve_nested_snowflakes_example(config=None):
    """Example: Solving Laplace Equation in a domain bounded by two Koch Snowflakes."""
    print("\n--- Nested Koch Snowflakes Example ---")
    outer = generate_koch_snowflake(order=2, scale=1.0, center=(0.5, 0.5))
    inner = generate_koch_snowflake(order=1, scale=0.4, center=(0.5, 0.5))
    domain = PolygonDomain(outer, holes=[inner])
    
    if config is None:
        config = {
            "num_layers": 4,
            "hidden_dim": 64,
            "activation": "sine",
            "adam_epochs": 2000,
            "lbfgs_epochs": 500,
            "use_self_adaptive_weights": False,
        }

    model, history = train(domain=domain, bc_fn=bc_nested, config=config)
    plot_results(model, domain=domain, filename='nested_snowflakes.png')
    plot_history(history, filename='nested_snowflakes_history.png')
    return model, domain

def compare_pinn_fem(config=None):
    """Benchmarks PINN against FEM on the nested snowflake domain."""
    print("\n--- PINN vs FEM Benchmark ---")
    
    # 1. Geometry and BCs
    outer = generate_koch_snowflake(order=2, scale=1.0, center=(0.5, 0.5))
    inner = generate_koch_snowflake(order=1, scale=0.4, center=(0.5, 0.5))
    domain = PolygonDomain(outer, holes=[inner])
    
    # 2. Solve with FEM
    print("Step 1: Solving with FEM...")
    mesh, u_fem = solve_fem(domain, bc_nested, resolution=60)
    
    # 3. Solve with PINN
    print("Step 2: Solving with PINN...")
    if config is None:
        config = {
            "num_layers": 4,
            "hidden_dim": 64,
            "activation": "sine",
            "use_self_adaptive_weights": True,
            "adam_epochs": 2000,
            "lbfgs_epochs": 1000,
            "lambda_bc": 200.0,
            "use_adaptive_sampling": True,
            "adaptive_every": 100
        }
    
    pinn_model, history = train(domain=domain, bc_fn=bc_nested, config=config)
    
    # 4. Comparison and Visualization
    print("Step 3: Comparing and Visualizing Results...")
    plot_comparison(pinn_model, mesh, u_fem, domain, filename='comparison_nested.png', resolution=250)
    plot_history(history, filename='comparison_nested_history.png')
    
    x_test, y_test = domain.sample_interior(5000)
    coords = torch.stack([x_test, y_test], dim=1).to(get_device())
    u_pinn = pinn_model(coords).detach().cpu().numpy().flatten()
    
    u_fem_interp = griddata(mesh.p.T, u_fem, (x_test.cpu().numpy(), y_test.cpu().numpy()), method='linear')
    mask = ~np.isnan(u_fem_interp)
    
    mse = np.mean((u_pinn[mask] - u_fem_interp[mask])**2)
    print(f"Benchmark Result -> MSE: {mse:.8e}")
    
    # Visual comparison
    plot_results(pinn_model, domain=domain, filename='nested_pinn.png', resolution=250)
    
    plt.figure(figsize=(10, 8), dpi=150)
    plt.scatter(x_test.cpu().numpy()[mask], y_test.cpu().numpy()[mask], 
                c=np.abs(u_pinn[mask] - u_fem_interp[mask]), cmap='inferno', s=5)
    plt.colorbar(label='Absolute Error')
    plt.title(f'PINN vs FEM Error (MSE: {mse:.2e})')
    plt.savefig('nested_error.png')
    plt.close()
    print("Plots saved: nested_pinn.png, nested_error.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN Laplace Solver Entry Point")
    parser.add_argument('--mode', type=str, choices=['snowflake', 'nested', 'compare'], default='nested',
                        help="Execution mode (default: nested)")
    args = parser.parse_args()

    # Stabilized configuration with Vertex Pinning and Reduced Spectral Ringing
    config = {
        "num_layers": 4,             
        "hidden_dim": 64,            
        "activation": "sine",
        "adaptive_activations": False, 
        "use_fourier_features": False, # Disabled to reduce spectral ringing at corners
        "omega": 15.0,               # Lower omega for smoother function approximation
        "adam_epochs": 4000,         
        "lbfgs_epochs": 800,        
        "adam_lr": 0.001,            # Increased slightly for stable convergence
        "lbfgs_points_domain": 5000, 
        "lbfgs_points_bc": 3000,
        "lambda_bc": 200.0,          
        "lambda_range": 100.0,       
        "lambda_grad_bc": 2.0,       
        "use_adaptive_sampling": True, 
        "adaptive_every": 100,       
        "use_self_adaptive_weights": True, 
        "adaptive_weight_every": 100,      
        "use_sobolev": True,
        "sobolev_h1_weight": 1e-4, # Reduced from 0.1 to prevent gradient explosion
    }

    if args.mode == 'snowflake':
        solve_koch_snowflake_example()
    elif args.mode == 'compare':
        compare_pinn_fem(config=config)
    else:
        solve_nested_snowflakes_example(config=config)
