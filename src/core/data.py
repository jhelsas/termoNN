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
    """Returns the best available device and explicitly initializes CUDA/MPS."""
    if torch.cuda.is_available():
        torch.cuda.init()
        device = torch.device("cuda")
        torch.zeros(1, device=device)
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
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

def generate_adaptive_domain_data(model, n_points, device='cpu', domain=None, f_fn=None, config=None):
    """
    Enhanced Residual-based Adaptive Refinement (RAR).
    Samples a large pool of candidates and selects points with the highest 
    combined PDE residue and Boundary Proximity error.
    """
    if domain is None:
        return generate_domain_data(n_points, device=device)
    
    # 1. Sample a larger candidate pool (10x for better resolution of sharp features)
    n_candidates = n_points * 10
    x_cand, y_cand = domain.sample_interior(n_candidates, device=device)
    
    model.eval()
    with torch.set_grad_enabled(True):
        coords = torch.stack([x_cand, y_cand], dim=1).requires_grad_(True)
        u = model(coords)
        
        if not u.requires_grad:
            residue = torch.zeros_like(x_cand)
        else:
            grads = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_x, u_y = grads[:, 0], grads[:, 1]
            
            u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]
            
            f = f_fn(x_cand, y_cand) if f_fn is not None else torch.zeros_like(u_xx)
            residue = torch.abs(u_xx + u_yy - f.squeeze())
        
        # 2. Add Range Violation Error to selection metric
        # Using a higher-order penalty (pow 4) to create a much stronger 
        # "attractor" for collocation points in undershooting/overshooting regions.
        u_val = u.detach()
        range_err = torch.pow(torch.relu(u_val - 1.0), 4) + torch.pow(torch.relu(0.0 - u_val), 4)
        
        # Selection Metric: PDE Residue + heavily weighted Range Error
        metric = residue + 200.0 * range_err.squeeze()

    # 3. Select Top-K points with highest combined metric
    _, indices = torch.topk(metric, n_points)
    
    return x_cand[indices].detach(), y_cand[indices].detach()

def generate_boundary_data(n_points=200, device='cpu', domain=None, bc_fn=None):
    """
    Generates boundary points, target values, and normals.
    """
    if domain is None:
        # Simplistic fallback for square (no normals returned)
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
        return x_bc, y_bc, u_bc, torch.zeros((len(x_bc), 2), device=device)

    # Use the custom domain logic
    x_bc, y_bc, b_ids, normals = domain.sample_boundary(n_points, device)
    if bc_fn is not None:
        import inspect
        sig = inspect.signature(bc_fn)
        u_bc = bc_fn(x_bc, y_bc, b_ids=b_ids) if 'b_ids' in sig.parameters else bc_fn(x_bc, y_bc)
    else:
        u_bc = torch.zeros((x_bc.shape[0], 1), device=device)
        
    return x_bc, y_bc, u_bc, normals
