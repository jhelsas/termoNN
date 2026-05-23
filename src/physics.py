import torch
from typing import Callable, Tuple

def laplace_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Laplace equation loss: u_xx + u_yy = 0.
    
    Args:
        model: The PINN model.
        x: x-coordinates of domain points.
        y: y-coordinates of domain points.
        
    Returns:
        Mean squared error of the Laplace operator.
    """
    coords = torch.stack([x, y], dim=1).requires_grad_(True)
    u = model(coords)
    
    # Calculate first derivatives (Jacobian)
    grads = torch.autograd.grad(
        u, coords, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True
    )[0]
    u_x = grads[:, 0]
    u_y = grads[:, 1]
    
    # Calculate second derivatives (Hessian diagonal)
    # We use allow_unused=True because for some model architectures (like linear models used in tests),
    # the second derivative is mathematically zero and may not have a gradient path in the autograd graph.
    u_xx_grads = torch.autograd.grad(
        u_x, coords, 
        grad_outputs=torch.ones_like(u_x), 
        create_graph=True, 
        allow_unused=True
    )[0]
    u_xx = u_xx_grads[:, 0] if u_xx_grads is not None else torch.zeros_like(u_x)
    
    u_yy_grads = torch.autograd.grad(
        u_y, coords, 
        grad_outputs=torch.ones_like(u_y), 
        create_graph=True, 
        allow_unused=True
    )[0]
    u_yy = u_yy_grads[:, 1] if u_yy_grads is not None else torch.zeros_like(u_y)
    
    return torch.mean((u_xx + u_yy)**2)

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
