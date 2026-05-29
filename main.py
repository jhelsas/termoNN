import torch
import warnings
from src.pinn.solver import train
from src.core.viz import plot_results
from src.core.geometry import PolygonDomain, generate_koch_snowflake

# Suppress benign CUDA/cuBLAS context warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

def solve_koch_snowflake_example():
    """Example: Solving Laplace Equation on a Koch Snowflake fractal."""
    print("--- Koch Snowflake Harmonic Example ---")
    vertices = generate_koch_snowflake(order=3, scale=1.0, center=(0.5, 0.5))
    domain = PolygonDomain(vertices)
    
    def bc_harmonic(x, y):
        # u(x, y) = sin(pi*(x-0.5)) * cosh(pi*(y-0.5))
        return (torch.sin(torch.pi * (x-0.5)) * torch.cosh(torch.pi * (y-0.5))).unsqueeze(1)

    # Simplified config for basic example
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

def solve_nested_snowflakes_example(config=None):
    """Example: Solving Laplace Equation in a domain bounded by two Koch Snowflakes."""
    print("\n--- Nested Koch Snowflakes Example ---")
    outer = generate_koch_snowflake(order=3, scale=1.0, center=(0.5, 0.5))
    inner = generate_koch_snowflake(order=2, scale=0.4, center=(0.5, 0.5))
    domain = PolygonDomain(outer, holes=[inner])
    
    def bc_nested(x, y, b_ids=None):
        u = torch.zeros((x.shape[0], 1), device=x.device)
        if b_ids is not None:
            # ID 0 is outer snowflake, ID 1 is the hole
            u[b_ids == 1] = 1.0
        return u

    model, history = train(domain=domain, bc_fn=bc_nested, config=config)
    plot_results(model, domain=domain, filename='nested_snowflakes.png')

if __name__ == "__main__":
    # "Self-Adaptive-Spectral" Breakthrough configuration
    config = {
        "num_layers": 6,             
        "hidden_dim": 128,           
        "activation": "sine",
        "adaptive_activations": True, 
        "omega": (1.0, 30.0),        
        "adam_epochs": 4000,         
        "lbfgs_epochs": 1500,        
        "adam_lr": 0.0005,           
        "lbfgs_points_domain": 4000, 
        "lbfgs_points_bc": 2000,
        "lambda_bc": 500.0,          
        "lambda_range": 200.0,       
        "use_adaptive_sampling": True, 
        "adaptive_every": 200,
        "use_self_adaptive_weights": True, 
        "adaptive_weight_every": 100,      
    }

    # choose one of the examples to run
    # solve_koch_snowflake_example()
    solve_nested_snowflakes_example(config=config)
