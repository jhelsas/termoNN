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
    
    # Plot 1: Standard Contour
    plt.figure(figsize=(10, 8), dpi=200)
    if filename.startswith('nested_') or filename.startswith('fem_nested'):
        im = plt.contourf(X, Y, u_pred, levels=100, cmap='viridis', vmin=-0.1, vmax=1.1)
    else:
        im = plt.contourf(X, Y, u_pred, levels=100, cmap='viridis')
    plt.colorbar(im, label='u(x, y)')
    
    # Boundary overlays
    if domain:
        v = domain.vertices.cpu().numpy()
        plt.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), 'k-', lw=1.5, alpha=0.8)
        for hole in domain.holes:
            vh = hole.cpu().numpy()
            plt.plot(np.append(vh[:, 0], vh[0, 0]), np.append(vh[:, 1], vh[0, 1]), 'k-', lw=1.5, alpha=0.8)

    plt.title(f'Solution: {filename}')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    # Plot 2: Gradient Norm Plot (to inspect "undershooting" and sharp edges)
    # This helps see where the model is struggling with sharp spikes
    if domain:
        gx, gy = np.gradient(u_pred)
        grad_norm = np.sqrt(gx**2 + gy**2)
        plt.figure(figsize=(10, 8), dpi=200)
        im = plt.imshow(grad_norm, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='inferno')
        plt.colorbar(im, label='|grad u|')
        plt.title(f'Gradient Norm: {filename}')
        plt.savefig(filename.replace('.png', '_gradient.png'), bbox_inches='tight')
        plt.close()

    print(f"High-resolution results saved to {filename} and associated plots")

def plot_history(history: dict, filename: str = 'training_history.png') -> None:
    """Plots the training loss history for Adam and L-BFGS stages."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # 1. Adam Stage
    ax1 = axes[0]
    adam_hist = history.get("adam", {})
    if adam_hist.get("loss"):
        epochs = range(len(adam_hist["loss"]))
        ax1.semilogy(epochs, adam_hist["loss"], label='Total Loss', alpha=0.8)
        ax1.semilogy(epochs, adam_hist["loss_pde"], label='PDE Loss', alpha=0.6, linestyle='--')
        ax1.semilogy(epochs, adam_hist["loss_bc"], label='BC Loss', alpha=0.6, linestyle=':')
        
        # Plot adaptive weights on twin axis
        if adam_hist.get("lambda_bc"):
            ax1_twin = ax1.twinx()
            ax1_twin.plot(epochs, adam_hist["lambda_bc"], color='red', alpha=0.3, label='λ_bc (adaptive)')
            ax1_twin.set_ylabel('Adaptive Weight λ_bc', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            
        ax1.set_title('Stage 1: Adam Optimization')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Log Scale)')
        ax1.legend(loc='upper right')
        ax1.grid(True, which="both", ls="-", alpha=0.2)

    # 2. L-BFGS Stage
    ax2 = axes[1]
    lbfgs_hist = history.get("lbfgs", {})
    if lbfgs_hist.get("loss"):
        iters = range(len(lbfgs_hist["loss"]))
        ax2.semilogy(iters, lbfgs_hist["loss"], label='Total Loss', color='green')
        ax2.semilogy(iters, lbfgs_hist["loss_pde"], label='PDE Loss', alpha=0.6, linestyle='--')
        ax2.semilogy(iters, lbfgs_hist["loss_bc"], label='BC Loss', alpha=0.6, linestyle=':')
        
        ax2.set_title('Stage 2: L-BFGS Fine-tuning')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss (Log Scale)')
        ax2.legend()
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Training history saved to {filename}")

def plot_comparison(pinn_model, mesh, u_fem, domain, filename='comparison.png', resolution=300):
    """
    Plots a side-by-side comparison of FEM solution, PINN solution, and the Error.
    """
    from scipy.interpolate import griddata
    device = next(pinn_model.parameters()).device
    
    # 1. Generate grid for plotting
    min_x, max_x = domain.min_x.cpu().item(), domain.max_x.cpu().item()
    min_y, max_y = domain.min_y.cpu().item(), domain.max_y.cpu().item()
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 2. Get PINN solution
    coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), 
                         dtype=torch.float32, device=device)
    pinn_model.eval()
    with torch.no_grad():
        u_pinn = pinn_model(coords).cpu().numpy().reshape(resolution, resolution)
        
    # 3. Interpolate FEM solution to the same grid
    u_fem_interp = griddata(mesh.p.T, u_fem, (X, Y), method='linear')
    
    # 4. Apply Mask
    px = torch.tensor(X.ravel(), dtype=torch.float32, device=domain.device)
    py = torch.tensor(Y.ravel(), dtype=torch.float32, device=domain.device)
    mask = domain.is_inside(px, py).reshape(resolution, resolution).cpu().numpy()
    
    u_pinn[~mask] = np.nan
    u_fem_interp[~mask] = np.nan
    error = np.abs(u_pinn - u_fem_interp)
    
    # 5. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), dpi=150)
    
    # Determine common color scale for solutions
    v_min = min(np.nanmin(u_pinn), np.nanmin(u_fem_interp))
    v_max = max(np.nanmax(u_pinn), np.nanmax(u_fem_interp))
    
    # Subplot 1: FEM
    im0 = axes[0].contourf(X, Y, u_fem_interp, levels=100, cmap='viridis', vmin=v_min, vmax=v_max)
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title('FEM Solution (Reference)')
    
    # Subplot 2: PINN
    im1 = axes[1].contourf(X, Y, u_pinn, levels=100, cmap='viridis', vmin=v_min, vmax=v_max)
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title('PINN Solution')
    
    # Subplot 3: Error
    im2 = axes[2].contourf(X, Y, error, levels=100, cmap='inferno')
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title(f'Absolute Error (Max: {np.nanmax(error):.2e})')
    
    # Overlay boundaries on all plots
    for ax in axes:
        v = domain.vertices.cpu().numpy()
        ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), 'k-', lw=1)
        for hole in domain.holes:
            vh = hole.cpu().numpy()
            ax.plot(np.append(vh[:, 0], vh[0, 0]), np.append(vh[:, 1], vh[0, 1]), 'k-', lw=1)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Comparison plot saved to {filename}")

