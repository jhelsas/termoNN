import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
from src.model import PINN
from src.physics import laplace_loss, boundary_loss
from src.utils import generate_domain_data, generate_boundary_data, set_seed, get_device, PolygonDomain

# Suppress benign CUDA/cuBLAS context warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

def train(domain=None, bc_fn=None, adam_epochs: int = 2000, lbfgs_epochs: int = 500, lr: float = 0.001) -> torch.nn.Module:
    """
    Trains the PINN model using a two-stage approach.
    Now supports custom Polygon domains.
    """
    set_seed(42)
    device = get_device()
    print(f"Training on device: {device}")
    
    model = PINN().to(device)
    
    # Stage 1: Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=lr)
    
    # Hyperparameters
    lambda_bc = 10.0 # Weight for boundary condition loss
    
    print("--- Stage 1: Adam Optimization ---")
    for epoch in range(adam_epochs):
        x_domain, y_domain = generate_domain_data(2000, device=device, domain=domain)
        x_bc, y_bc, u_bc = generate_boundary_data(400, device=device, domain=domain, bc_fn=bc_fn)
        
        optimizer_adam.zero_grad()
        loss_pde = laplace_loss(model, x_domain, y_domain)
        loss_bc = boundary_loss(model, x_bc, y_bc, u_bc)
        total_loss = loss_pde + lambda_bc * loss_bc
        
        total_loss.backward()
        optimizer_adam.step()
        
        if epoch % 500 == 0:
            print(f"Adam Epoch {epoch:4d} | Loss: {total_loss.item():.6f} "
                  f"(PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f})")

    # Stage 2: L-BFGS (Fine-tuning)
    print("--- Stage 2: L-BFGS Fine-tuning ---")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1, 
        max_iter=20, 
        tolerance_grad=1e-7, 
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    # Static set for L-BFGS
    x_domain, y_domain = generate_domain_data(3000, device=device, domain=domain)
    x_bc, y_bc, u_bc = generate_boundary_data(600, device=device, domain=domain, bc_fn=bc_fn)

    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde = laplace_loss(model, x_domain, y_domain)
        loss_bc = boundary_loss(model, x_bc, y_bc, u_bc)
        total_loss = loss_pde + lambda_bc * loss_bc
        total_loss.backward()
        return total_loss

    for epoch in range(lbfgs_epochs):
        loss = optimizer_lbfgs.step(closure)
        if epoch % 100 == 0:
            print(f"L-BFGS Epoch {epoch:3d} | Loss: {loss.item():.8f}")
            
    return model

def plot_results(model: torch.nn.Module, domain=None) -> None:
    """Generates and saves the solution plot, masking areas outside the domain."""
    device = next(model.parameters()).device
    
    # Determine bounds
    if domain:
        min_x, max_x = domain.min_x, domain.max_x
        min_y, max_y = domain.min_y, domain.max_y
    else:
        min_x, max_x, min_y, max_y = 0, 1, 0, 1

    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    X, Y = np.meshgrid(x, y)
    
    # Flatten and convert to torch
    coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), 
                         dtype=torch.float32, 
                         device=device)
    
    model.eval()
    with torch.no_grad():
        u_pred_flat = model(coords).cpu().numpy()
        
    u_pred = u_pred_flat.reshape(100, 100)
    
    # Mask points outside the domain
    if domain:
        px = torch.tensor(X.ravel(), dtype=torch.float32, device=domain.device)
        py = torch.tensor(Y.ravel(), dtype=torch.float32, device=domain.device)
        mask = domain.is_inside(px, py).reshape(100, 100).cpu().numpy()
        u_pred[~mask] = np.nan
        
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, u_pred, levels=50, cmap='viridis')
    plt.colorbar(label='u(x, y)')
    plt.title('PINN Solution to Laplace Equation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('solution.png')
    print("Result saved to solution.png")

if __name__ == "__main__":
    # Example: A domain with a hole
    # Square [0, 1]x[0, 1] with a triangle hole
    outer = [(0, 0), (1, 0), (1, 1), (0, 1)]
    hole = [(0.4, 0.4), (0.6, 0.4), (0.5, 0.6)]
    
    domain = PolygonDomain(outer, holes=[hole])
    
    # BC Function: u(x, y) = sin(pi*x) on the bottom (y=0) and 0 elsewhere
    def my_bc_fn(x, y):
        u = torch.zeros((x.shape[0], 1), device=x.device)
        # Identify bottom edge points (y near 0)
        mask = (y < 1e-5) & (x > 0) & (x < 1)
        u[mask] = torch.sin(np.pi * x[mask]).unsqueeze(1)
        return u

    trained_model = train(domain=domain, bc_fn=my_bc_fn)
    plot_results(trained_model, domain=domain)

if __name__ == "__main__":
    trained_model = train()
    plot_results(trained_model)
