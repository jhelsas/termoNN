import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
from src.model import PINN
from src.physics import laplace_loss, boundary_loss
from src.utils import generate_domain_data, generate_boundary_data, set_seed, get_device

# Suppress benign CUDA/cuBLAS context warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

def train(adam_epochs: int = 2000, lbfgs_epochs: int = 500, lr: float = 0.001) -> torch.nn.Module:
    """
    Trains the PINN model using a two-stage approach:
    1. Adam for global exploration.
    2. L-BFGS for fine-tuning and 'snapping' to the physics solution.
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
        # Adaptive Resampling: Sampling new collocation points frequently 
        # improves generalization and prevents overfitting to a specific set of points.
        x_domain, y_domain = generate_domain_data(2000, device=device)
        x_bc, y_bc, u_bc = generate_boundary_data(400, device=device)
        
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
    # L-BFGS is a second-order optimizer that is much more efficient at finding 
    # the exact minimum in 'stiff' PINN landscapes.
    print("--- Stage 2: L-BFGS Fine-tuning ---")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1, 
        max_iter=20, 
        tolerance_grad=1e-7, 
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    # For L-BFGS, we typically use a static set of points for the closure
    x_domain, y_domain = generate_domain_data(3000, device=device)
    x_bc, y_bc, u_bc = generate_boundary_data(600, device=device)

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

def plot_results(model: torch.nn.Module) -> None:
    """Generates and saves the solution plot."""
    device = next(model.parameters()).device
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Flatten and convert to torch
    coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), 
                         dtype=torch.float32, 
                         device=device)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(coords).reshape(100, 100).cpu().numpy()
        
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, u_pred, levels=50, cmap='viridis')
    plt.colorbar(label='u(x, y)')
    plt.title('PINN Solution to Laplace Equation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('solution.png')
    print("Result saved to solution.png")

if __name__ == "__main__":
    trained_model = train()
    plot_results(trained_model)
