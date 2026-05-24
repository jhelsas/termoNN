import torch
import numpy as np

def set_seed(seed=42):
    """Sets seeds for reproducibility."""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Returns the best available device and explicitly initializes CUDA."""
    if torch.cuda.is_available():
        torch.cuda.init()
        device = torch.device("cuda")
        torch.zeros(1, device=device)
        return device
    return torch.device("cpu")

def generate_domain_data(n_points=1000, device='cpu', domain=None):
    """
    Generates random points within the domain. 
    Defaults to Unit Square if no domain is provided.
    """
    if domain is None:
        x = torch.rand(n_points, device=device)
        y = torch.rand(n_points, device=device)
        return x, y
    
    return domain.sample_interior(n_points, device)

def generate_boundary_data(n_points=200, device='cpu', domain=None, bc_fn=None):
    """
    Generates boundary points and target values.
    """
    if domain is None:
        n_per_side = n_points // 4
        x_top = torch.rand(n_per_side, device=device)
        y_top = torch.ones(n_per_side, device=device)
        u_top = torch.zeros((n_per_side, 1), device=device)
        x_bot = torch.rand(n_per_side, device=device)
        y_bot = torch.zeros(n_per_side, device=device)
        u_bot = torch.sin(np.pi * x_bot).unsqueeze(1)
        x_left = torch.zeros(n_per_side, device=device)
        y_left = torch.rand(n_per_side, device=device)
        u_left = torch.zeros((n_per_side, 1), device=device)
        x_right = torch.ones(n_per_side, device=device)
        y_right = torch.rand(n_per_side, device=device)
        u_right = torch.zeros((n_per_side, 1), device=device)
        x_bc = torch.cat([x_top, x_bot, x_left, x_right])
        y_bc = torch.cat([y_top, y_bot, y_left, y_right])
        u_bc = torch.cat([u_top, u_bot, u_left, u_right])
        return x_bc, y_bc, u_bc

    # Use the custom domain logic
    x_bc, y_bc, b_ids = domain.sample_boundary(n_points, device)
    if bc_fn is not None:
        import inspect
        sig = inspect.signature(bc_fn)
        if 'b_ids' in sig.parameters:
            u_bc = bc_fn(x_bc, y_bc, b_ids=b_ids)
        else:
            u_bc = bc_fn(x_bc, y_bc)
    else:
        u_bc = torch.zeros((x_bc.shape[0], 1), device=device)
        
    return x_bc, y_bc, u_bc
