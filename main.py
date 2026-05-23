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

def train(epochs: int = 3000, lr: float = 0.001) -> torch.nn.Module:
    """Trains the PINN model."""
    set_seed(42)
    device = get_device()
    print(f"Training on device: {device}")
    
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Generate data
    x_domain, y_domain = generate_domain_data(2000, device=device)
    x_bc, y_bc, u_bc = generate_boundary_data(400, device=device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss_pde = laplace_loss(model, x_domain, y_domain)
        loss_bc = boundary_loss(model, x_bc, y_bc, u_bc)
        
        # Hyperparameter lambda=10 for BC importance
        total_loss = loss_pde + 10 * loss_bc 
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.6f} "
                  f"(PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f})")
            
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
