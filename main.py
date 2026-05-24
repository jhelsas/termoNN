import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
from src.model import PINN
from src.physics import laplace_loss, boundary_loss, poisson_loss, range_loss
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
        "activation": "sine",
        "omega": 30.0,
        "adam_epochs": 2000,
        "lbfgs_epochs": 500,
        "adam_lr": 0.001,
        "adam_points_domain": 2000,
        "adam_points_bc": 400,
        "lbfgs_points_domain": 3000,
        "lbfgs_points_bc": 600,
        "lambda_bc": 10.0,
        "lambda_range": 0.0,  # Penalty for range violations (off by default)
        "seed": 42
    }
    
    if config:
        default_config.update(config)
    cfg = default_config

    # Sanitation and Sanity Checks
    required_keys = ["hidden_dim", "num_layers", "activation", "lambda_bc"]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")
    
    # Support for multi-frequency log printing
    omega_desc = str(cfg.get('omega', 30.0))
    
    print(f"--- PINN Configuration ---")
    print(f"  - Model: {cfg['num_layers']} layers, {cfg['hidden_dim']} units, {cfg['activation']} activation")
    print(f"  - Omega: {omega_desc}")
    print(f"  - Weighting: lambda_bc={cfg['lambda_bc']}, lambda_range={cfg.get('lambda_range', 0.0)}")

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Training on device: {device}")
    
    # Architecture defined by config
    model = PINN(
        hidden_dim=cfg["hidden_dim"], 
        num_layers=cfg["num_layers"],
        activation=cfg["activation"],
        omega=cfg["omega"]
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
        
        if cfg["lambda_range"] > 0:
            # Predict on domain to check range
            coords_d = torch.stack([x_domain, y_domain], dim=1)
            u_domain = model(coords_d)
            loss_r = range_loss(u_domain)
            total_loss += cfg["lambda_range"] * loss_r
        
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
        
        loss_r_val = 0.0
        if cfg["lambda_range"] > 0:
            # Check range on BOTH domain and boundary points
            coords_d = torch.stack([x_domain, y_domain], dim=1)
            coords_b = torch.stack([x_bc, y_bc], dim=1)
            u_all = torch.cat([model(coords_d), model(coords_b)], dim=0)
            loss_r = range_loss(u_all)
            total_loss += cfg["lambda_range"] * loss_r
            loss_r_val = loss_r.item()
            
        total_loss.backward()
        # Attach individual losses
        total_loss.pde = loss_pde.item()
        total_loss.bc = loss_bc.item()
        total_loss.range = loss_r_val
        return total_loss

    for epoch in range(cfg["lbfgs_epochs"]):
        loss = optimizer_lbfgs.step(closure)
        if epoch % 100 == 0:
            print(f"L-BFGS Epoch {epoch:3d} | Loss: {loss.item():.8f} "
                  f"(PDE: {loss.pde:.8f}, BC: {loss.bc:.8f}, R: {loss.range:.8f})")
            
    return model

def plot_results(model: torch.nn.Module, domain=None, filename='solution.png', resolution=300) -> None:
    """Generates and saves the solution plot with high resolution and masking."""
    device = next(model.parameters()).device
    
    # Determine bounds
    if domain:
        min_x, max_x = domain.min_x.cpu().item(), domain.max_x.cpu().item()
        min_y, max_y = domain.min_y.cpu().item(), domain.max_y.cpu().item()
        margin = 0.05
        dx, dy = max_x - min_x, max_y - min_y
        min_x, max_x = min_x - dx*margin, max_x + dx*margin
        min_y, max_y = min_y - dy*margin, max_y + dy*margin
    else:
        min_x, max_x, min_y, max_y = 0, 1, 0, 1

    # Create high-resolution meshgrid
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten and convert to torch
    coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), 
                         dtype=torch.float32, 
                         device=device)
    
    model.eval()
    with torch.no_grad():
        u_pred_flat = model(coords).cpu().numpy()
        
    u_pred = u_pred_flat.reshape(resolution, resolution)
    
    # Mask points outside the domain
    u_min, u_max = u_pred_flat.min(), u_pred_flat.max()
    
    if domain:
        px = torch.tensor(X.ravel(), dtype=torch.float32, device=domain.device)
        py = torch.tensor(Y.ravel(), dtype=torch.float32, device=domain.device)
        mask = domain.is_inside(px, py).reshape(resolution, resolution).cpu().numpy()
        
        # Diagnostics for values INSIDE the domain
        u_interior = u_pred[mask]
        u_min, u_max = u_interior.min(), u_interior.max()
        u_pred[~mask] = np.nan
        
    print(f"Diagnostics for {filename}:")
    print(f"  - Value Range in Domain: [{u_min:.4f}, {u_max:.4f}]")
    
    # Check for maximum principle violations (assuming BCs are in [0, 1])
    if u_min < -0.05 or u_max > 1.05:
        print(f"  - WARNING: Significant range violation detected (Maximum Principle violation).")
        
    plt.figure(figsize=(10, 8), dpi=200)
    # Using more levels for smoother contouring
    plt.contourf(X, Y, u_pred, levels=100, cmap='viridis')
    plt.colorbar(label='u(x, y)')
    
    # Overlay the domain boundary for reference
    if domain:
        # Plot outer boundary
        v = domain.vertices.cpu().numpy()
        plt.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), 'k-', lw=1.5, alpha=0.8)
        # Plot holes
        for hole in domain.holes:
            vh = hole.cpu().numpy()
            plt.plot(np.append(vh[:, 0], vh[0, 0]), np.append(vh[:, 1], vh[0, 1]), 'k-', lw=1.5, alpha=0.8)

    plt.title('High-Resolution PINN Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"High-resolution result saved to {filename}")

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
    
    def bc_nested(x, y, b_ids=None):
        u = torch.zeros((x.shape[0], 1), device=x.device)
        if b_ids is not None:
            # ID 0 is outer snowflake, ID 1 is the hole
            u[b_ids == 1] = 1.0
        else:
            # Fallback to distance check if IDs aren't available
            dist = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            u[dist < 0.4] = 1.0 
        return u

    print("Training on Nested Snowflake domain...")
    model = train(domain=domain, bc_fn=bc_nested, config=config)
    
    plot_results(model, domain=domain, filename='nested_snowflakes.png')
    print("Nested fractal solution saved to nested_snowflakes.png")

if __name__ == "__main__":
    # Multi-Frequency "Fourier" configuration:
    # Captures global heat flow (low omega) and fractal detail (high omega)
    config = {
        "num_layers": 6,             
        "hidden_dim": 128,           
        "activation": "sine",
        "omega": (2.0, 40.0),        # Multi-frequency range (min, max)
        "adam_epochs": 2000,         
        "lbfgs_epochs": 1000,        
        "adam_points_domain": 2000, 
        "adam_points_bc": 1000,      
        "lbfgs_points_domain": 4000, 
        "lbfgs_points_bc": 2000,
        "lambda_bc": 100.0,          
        "lambda_range": 20.0,        
    }

    solve_nested_snowflakes_example(config=config)
