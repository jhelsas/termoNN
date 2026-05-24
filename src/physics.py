import torch
from typing import Callable, Tuple

def poisson_loss(model: torch.nn.Module, 
                 x: torch.Tensor, 
                 y: torch.Tensor, 
                 f_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the Poisson equation loss: u_xx + u_yy = f(x, y).
    If f_fn is None, it defaults to the Laplace equation (f=0).
    
    Args:
        model: The PINN model.
        x: x-coordinates of domain points.
        y: y-coordinates of domain points.
        f_fn: A function that returns the source term f(x, y).
        
    Returns:
        Mean squared error of the Poisson residue.
    """
    coords = torch.stack([x, y], dim=1).requires_grad_(True)
    u = model(coords)
    
    # Calculate first derivatives
    grads = torch.autograd.grad(
        u, coords, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True
    )[0]
    u_x = grads[:, 0]
    u_y = grads[:, 1]
    
    # Calculate second derivatives
    u_xx = torch.zeros_like(u_x)
    if u_x.requires_grad:
        u_xx_grads = torch.autograd.grad(
            u_x, coords, 
            grad_outputs=torch.ones_like(u_x), 
            create_graph=True, 
            allow_unused=True
        )[0]
        u_xx = u_xx_grads[:, 0] if u_xx_grads is not None else torch.zeros_like(u_x)
    
    u_yy = torch.zeros_like(u_y)
    if u_y.requires_grad:
        u_yy_grads = torch.autograd.grad(
            u_y, coords, 
            grad_outputs=torch.ones_like(u_y), 
            create_graph=True, 
            allow_unused=True
        )[0]
        u_yy = u_yy_grads[:, 1] if u_yy_grads is not None else torch.zeros_like(u_y)
    
    # Source term
    f = f_fn(x, y) if f_fn is not None else torch.zeros_like(u_x)
    
    # Ensure f is the correct shape [N]
    if f.dim() > 1:
        f = f.squeeze()
        
    return torch.mean((u_xx + u_yy - f)**2)

def laplace_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Laplace equation loss: u_xx + u_yy = 0.
    Wrapped as a specific case of the Poisson loss.
    """
    return poisson_loss(model, x, y, f_fn=None)

def boundary_loss(model: Callable[[torch.Tensor], torch.Tensor], 
                  x_bc: torch.Tensor, 
                  y_bc: torch.Tensor, 
                  u_bc: torch.Tensor) -> torch.Tensor:
    """
    Computes the boundary condition loss (MSE).
    """
    coords = torch.stack([x_bc, y_bc], dim=1)
    u_pred = model(coords)
    return torch.mean((u_pred - u_bc)**2)
