import torch
import numpy as np
import random

def set_seed(seed=42):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Returns the best available device and explicitly initializes CUDA."""
    if torch.cuda.is_available():
        # Explicitly initialize CUDA to prevent cuBLAS context warnings
        torch.cuda.init()
        device = torch.device("cuda")
        # Warm up the device
        torch.zeros(1, device=device)
        return device
    return torch.device("cpu")

def generate_domain_data(n_points=1000, device='cpu'):
    """Generates random points within [0, 1]x[0, 1]."""
    x = torch.rand(n_points, device=device)
    y = torch.rand(n_points, device=device)
    return x, y

def generate_boundary_data(n_points=200, device='cpu'):
    """
    Generates boundary points for [0, 1]x[0, 1] with Dirichlet BCs.
    u(x, 0) = sin(pi * x), and 0 elsewhere.
    """
    n_per_side = n_points // 4
    
    # Top: y = 1
    x_top = torch.rand(n_per_side, device=device)
    y_top = torch.ones(n_per_side, device=device)
    u_top = torch.zeros((n_per_side, 1), device=device)

    # Bottom: y = 0
    x_bot = torch.rand(n_per_side, device=device)
    y_bot = torch.zeros(n_per_side, device=device)
    u_bot = torch.sin(np.pi * x_bot).unsqueeze(1)

    # Left: x = 0
    x_left = torch.zeros(n_per_side, device=device)
    y_left = torch.rand(n_per_side, device=device)
    u_left = torch.zeros((n_per_side, 1), device=device)

    # Right: x = 1
    x_right = torch.ones(n_per_side, device=device)
    y_right = torch.rand(n_per_side, device=device)
    u_right = torch.zeros((n_per_side, 1), device=device)

    x_bc = torch.cat([x_top, x_bot, x_left, x_right])
    y_bc = torch.cat([y_top, y_bot, y_left, y_right])
    u_bc = torch.cat([u_top, u_bot, u_left, u_right])
    
    return x_bc, y_bc, u_bc
