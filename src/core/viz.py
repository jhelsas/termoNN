import torch
import matplotlib.pyplot as plt
import numpy as np

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
        if len(u_interior) > 0:
            u_min, u_max = u_interior.min(), u_interior.max()
        u_pred[~mask] = np.nan
        
    print(f"Diagnostics for {filename}:")
    print(f"  - Value Range in Domain: [{u_min:.4f}, {u_max:.4f}]")
    
    # Check for maximum principle violations (assuming BCs are in [0, 1])
    if u_min < -0.05 or u_max > 1.05:
        print(f"  - WARNING: Significant range violation detected (Maximum Principle violation).")
        
    plt.figure(figsize=(10, 8), dpi=200)
    plt.contourf(X, Y, u_pred, levels=100, cmap='viridis')
    plt.colorbar(label='u(x, y)')
    
    # Overlay the domain boundary for reference
    if domain:
        v = domain.vertices.cpu().numpy()
        plt.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), 'k-', lw=1.5, alpha=0.8)
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
