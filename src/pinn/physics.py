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

def sobolev_laplace_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, h1_weight: float = 1e-4) -> torch.Tensor:
    r"""
    Computes the Sobolev-norm (Gradient-Enhanced) Laplace loss.
    L = ||\Delta u||^2 + h1_weight * ||\nabla(\Delta u)||^2
    
    Default h1_weight is very small because 3rd-order derivatives 
    scale with frequency^3.
    """
    coords = torch.stack([x, y], dim=1).requires_grad_(True)
    u = model(coords)
    
    # 1. First derivatives
    grads = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grads[:, 0], grads[:, 1]
    
    # 2. Second derivatives (Laplacian)
    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]
    laplacian = u_xx + u_yy
    
    # 3. Gradients of the Laplacian (Sobolev part)
    laplacian_grads = torch.autograd.grad(
        laplacian, coords, 
        grad_outputs=torch.ones_like(laplacian), 
        create_graph=True
    )[0]
    l_grad_x, l_grad_y = laplacian_grads[:, 0], laplacian_grads[:, 1]
    
    loss_l2 = torch.mean(laplacian**2)
    loss_h1 = torch.mean(l_grad_x**2 + l_grad_y**2)
    
    return loss_l2 + h1_weight * loss_h1

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

def boundary_gradient_loss(model: torch.nn.Module, 
                           x_bc: torch.Tensor, 
                           y_bc: torch.Tensor,
                           nx: torch.Tensor,
                           ny: torch.Tensor) -> torch.Tensor:
    """
    Penalizes the tangential derivative at the boundary.
    This forces the solution to be 'flat' along the boundary, reducing overshoot.
    
    Args:
        nx, ny: components of the normal vector at each boundary point.
    """
    coords = torch.stack([x_bc, y_bc], dim=1).requires_grad_(True)
    u = model(coords)
    
    grads = torch.autograd.grad(
        u, coords, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x, u_y = grads[:, 0], grads[:, 1]
    
    # Tangent vector is (-ny, nx)
    tx, ty = -ny, nx
    u_tangent = u_x * tx + u_y * ty
    
    return torch.mean(u_tangent**2)

def range_loss(u: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Penalizes values that violate the Maximum Principle (staying within [min, max]).
    """
    penalty = torch.relu(u - max_val)**2 + torch.relu(min_val - u)**2
    return torch.mean(penalty)
