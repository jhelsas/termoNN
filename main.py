import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
from src.model import PINN
from src.physics import laplace_loss, boundary_loss, poisson_loss
from src.utils import generate_domain_data, generate_boundary_data, set_seed, get_device, PolygonDomain

# Suppress benign CUDA/cuBLAS context warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

def train(domain=None, bc_fn=None, f_fn=None, config=None) -> torch.nn.Module:
    """
    Trains the PINN model using a two-stage approach.
    Now uses a configuration dictionary for architecture and resolution.
    """
    # Default configuration
    default_config = {
        "num_layers": 4,
        "hidden_dim": 20,
        "adam_epochs": 2000,
        "lbfgs_epochs": 500,
        "adam_lr": 0.001,
        "adam_points_domain": 2000,
        "adam_points_bc": 400,
        "lbfgs_points_domain": 3000,
        "lbfgs_points_bc": 600,
        "lambda_bc": 10.0,
        "seed": 42
    }
    
    if config:
        default_config.update(config)
    cfg = default_config

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Training on device: {device}")
    
    # Architecture defined by config
    model = PINN(
        hidden_dim=cfg["hidden_dim"], 
        num_layers=cfg["num_layers"]
    ).to(device)
    
    # Stage 1: Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=cfg["adam_lr"])
    
    print(f"--- Stage 1: Adam Optimization ({cfg['adam_epochs']} epochs) ---")
    for epoch in range(cfg["adam_epochs"]):
        x_domain, y_domain = generate_domain_data(cfg["adam_points_domain"], device=device, domain=domain)
        x_bc, y_bc, u_bc = generate_boundary_data(cfg["adam_points_bc"], device=device, domain=domain, bc_fn=bc_fn)
        
        optimizer_adam.zero_grad()
        loss_pde = poisson_loss(model, x_domain, y_domain, f_fn=f_fn)
        loss_bc = boundary_loss(model, x_bc, y_bc, u_bc)
        total_loss = loss_pde + cfg["lambda_bc"] * loss_bc
        
        total_loss.backward()
        optimizer_adam.step()
        
        if epoch % 500 == 0:
            print(f"Adam Epoch {epoch:4d} | Loss: {total_loss.item():.6f} "
                  f"(PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f})")

    # Stage 2: L-BFGS (Fine-tuning)
    print(f"--- Stage 2: L-BFGS Fine-tuning ({cfg['lbfgs_epochs']} iterations) ---")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1, 
        max_iter=20, 
        tolerance_grad=1e-7, 
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    # Static set for L-BFGS resolution defined by config
    x_domain, y_domain = generate_domain_data(cfg["lbfgs_points_domain"], device=device, domain=domain)
    x_bc, y_bc, u_bc = generate_boundary_data(cfg["lbfgs_points_bc"], device=device, domain=domain, bc_fn=bc_fn)

    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde = poisson_loss(model, x_domain, y_domain, f_fn=f_fn)
        loss_bc = boundary_loss(model, x_bc, y_bc, u_bc)
        total_loss = loss_pde + cfg["lambda_bc"] * loss_bc
        total_loss.backward()
        return total_loss

    for epoch in range(cfg["lbfgs_epochs"]):
        loss = optimizer_lbfgs.step(closure)
        if epoch % 100 == 0:
            print(f"L-BFGS Epoch {epoch:3d} | Loss: {loss.item():.8f}")
            
    return model

def plot_results(model: torch.nn.Module, domain=None, filename='solution.png') -> None:
    """Generates and saves the solution plot, masking areas outside the domain."""
    device = next(model.parameters()).device
    
    # Determine bounds
    if domain:
        min_x, max_x = domain.min_x.cpu().item(), domain.max_x.cpu().item()
        min_y, max_y = domain.min_y.cpu().item(), domain.max_y.cpu().item()
        # Add a small margin
        margin = 0.1
        min_x, max_x = min_x - margin, max_x + margin
        min_y, max_y = min_y - margin, max_y + margin
    else:
        min_x, max_x, min_y, max_y = 0, 1, 0, 1

    x = np.linspace(min_x, max_x, 150)
    y = np.linspace(min_y, max_y, 150)
    X, Y = np.meshgrid(x, y)
    
    # Flatten and convert to torch
    coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), 
                         dtype=torch.float32, 
                         device=device)
    
    model.eval()
    with torch.no_grad():
        u_pred_flat = model(coords).cpu().numpy()
        
    u_pred = u_pred_flat.reshape(150, 150)
    
    # Mask points outside the domain
    if domain:
        px = torch.tensor(X.ravel(), dtype=torch.float32, device=domain.device)
        py = torch.tensor(Y.ravel(), dtype=torch.float32, device=domain.device)
        mask = domain.is_inside(px, py).reshape(150, 150).cpu().numpy()
        u_pred[~mask] = np.nan
        
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, u_pred, levels=50, cmap='viridis')
    plt.colorbar(label='u(x, y)')
    plt.title('PINN Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(filename)
    plt.close()
    print(f"Result saved to {filename}")

def solve_koch_snowflake_example():
    """
    Example: Solving Laplace Equation on a Koch Snowflake fractal.
    """
    from src.utils import generate_koch_snowflake, PolygonDomain
    
    print("--- Koch Snowflake Harmonic Example ---")
    vertices = generate_koch_snowflake(order=3, scale=1.0, center=(0.5, 0.5))
    domain = PolygonDomain(vertices)
    
    # BC Option: Harmonic
    def bc_harmonic(x, y):
        # u(x, y) = sin(pi*(x-0.5)) * cosh(pi*(y-0.5))
        return (torch.sin(np.pi * (x-0.5)) * torch.cosh(np.pi * (y-0.5))).unsqueeze(1)

    print("\n--- Scenario: harmonic ---")
    model = train(domain=domain, bc_fn=bc_harmonic, adam_epochs=1200, lbfgs_epochs=200)
    plot_results(model, domain=domain, filename='snowflake_harmonic.png')

def solve_nested_snowflakes_example(config=None):
    """
    Example: Solving Laplace Equation in a domain bounded by two Koch Snowflakes.
    """
    from src.utils import generate_koch_snowflake, PolygonDomain
    
    print("\n--- Nested Koch Snowflakes Example ---")
    
    outer_vertices = generate_koch_snowflake(order=3, scale=1.0, center=(0.5, 0.5))
    inner_vertices = generate_koch_snowflake(order=2, scale=0.4, center=(0.5, 0.5))
    domain = PolygonDomain(outer_vertices, holes=[inner_vertices])
    
    def bc_nested(x, y):
        u = torch.zeros((x.shape[0], 1), device=x.device)
        dist = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        u[dist < 0.4] = 1.0 
        return u

    print("Training on Nested Snowflake domain...")
    model = train(domain=domain, bc_fn=bc_nested, config=config)
    
    plot_results(model, domain=domain, filename='nested_snowflakes.png')
    print("Nested fractal solution saved to nested_snowflakes.png")

if __name__ == "__main__":
    # Example Configuration: tunable architecture and resolution
    config = {
        "num_layers": 5,           # Deeper network for complex fractals
        "hidden_dim": 32,          # Wider layers
        "adam_epochs": 1000,       # Reduced for quick demonstration
        "lbfgs_epochs": 200,
        "adam_points_domain": 2500, # Higher resolution sampling
        "adam_points_bc": 500,
    }

    solve_nested_snowflakes_example(config=config)
