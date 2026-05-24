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
    Defaults to the hardcoded sin(pi*x) square BC if no domain is provided.
    """
    if domain is None:
        # Default behavior for the unit square [0, 1]x[0, 1]
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

    # Use the custom domain logic
    x_bc, y_bc = domain.sample_boundary(n_points, device)
    if bc_fn is not None:
        u_bc = bc_fn(x_bc, y_bc)
    else:
        u_bc = torch.zeros((x_bc.shape[0], 1), device=device)
        
    return x_bc, y_bc, u_bc

class PolygonDomain:
    """
    Represents a 2D domain defined by an outer polygon and optional holes.
    Uses PyTorch tensors for efficient, potentially GPU-accelerated geometric operations.
    """
    def __init__(self, vertices, holes=None, device='cpu'):
        """
        Args:
            vertices: (N, 2) Tensor or array-like for the outer boundary.
            holes: List of (M, 2) Tensors or array-like for internal holes.
            device: Device to store the geometry on.
        """
        self.device = device
        self.vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
        self.holes = [torch.as_tensor(h, dtype=torch.float32, device=device) for h in holes] if holes else []
        
        # Precompute bounding box
        all_pts = [self.vertices] + self.holes
        combined = torch.cat(all_pts, dim=0)
        self.min_x, self.min_y = combined.min(dim=0).values
        self.max_x, self.max_y = combined.max(dim=0).values

    def is_inside(self, x, y):
        """Vectorized point-in-polygon check using ray-casting in PyTorch."""
        def check_poly(poly, px, py):
            # px, py: (M,) tensors
            # poly: (N, 2) tensor
            inside = torch.zeros_like(px, dtype=torch.bool)
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                
                # Ray casting logic
                intersect = ((p1[1] > py) != (p2[1] > py)) & \
                            (px < (p2[0] - p1[0]) * (py - p1[1]) / (p2[1] - p1[1] + 1e-10) + p1[0])
                inside ^= intersect
            return inside

        inside_outer = check_poly(self.vertices, x, y)
        for hole in self.holes:
            inside_outer &= ~check_poly(hole, x, y)
        return inside_outer

    def sample_interior(self, n_points, device=None):
        """Rejection sampling within the bounding box on the target device."""
        dev = device or self.device
        x_res, y_res = [], []
        
        # Iterative batch sampling
        while sum(len(b) for b in x_res) < n_points:
            # Sample on self.device first for the is_inside check
            x_cand = torch.empty(n_points * 2, device=self.device).uniform_(self.min_x, self.max_x)
            y_cand = torch.empty(n_points * 2, device=self.device).uniform_(self.min_y, self.max_y)
            
            mask = self.is_inside(x_cand, y_cand)
            x_res.append(x_cand[mask])
            y_res.append(y_cand[mask])
            
        x = torch.cat(x_res)[:n_points].to(dev)
        y = torch.cat(y_res)[:n_points].to(dev)
        return x, y

    def sample_boundary(self, n_points, device=None):
        """Samples points along the edges using length-weighted selection."""
        dev = device or self.device
        
        def get_edge_data(poly):
            p1 = poly
            p2 = torch.roll(poly, -1, dims=0)
            vecs = p2 - p1
            lengths = torch.norm(vecs, dim=1)
            return p1, vecs, lengths

        all_polys = [self.vertices] + self.holes
        edge_starts, edge_vecs, edge_lens = [], [], []
        
        for poly in all_polys:
            p1, v, l = get_edge_data(poly)
            edge_starts.append(p1)
            edge_vecs.append(v)
            edge_lens.append(l)
            
        starts = torch.cat(edge_starts)
        vecs = torch.cat(edge_vecs)
        lens = torch.cat(edge_lens)
        
        # Categorical sampling based on edge lengths
        probs = lens / lens.sum()
        edge_indices = torch.multinomial(probs, n_points, replacement=True)
        
        t = torch.rand(n_points, 1, device=self.device)
        sampled_pts = starts[edge_indices] + t * vecs[edge_indices]
        
        return sampled_pts[:, 0].to(dev), sampled_pts[:, 1].to(dev)

def generate_koch_snowflake(order=3, scale=1.0, center=(0.5, 0.5)):
    """
    Generates vertices for a Koch Snowflake fractal.
    
    Args:
        order: Fractal depth (recursion level).
        scale: Scale of the snowflake.
        center: (x, y) center of the snowflake.
        
    Returns:
        torch.Tensor: (N, 2) vertices.
    """
    def koch_recurse(p1, p2, order):
        if order == 0:
            return [p1]
        
        # Vector from p1 to p2
        v = p2 - p1
        
        # Compute the 3 intermediate points
        q = p1 + v / 3.0
        r = p1 + v * 2.0 / 3.0
        
        # s is the peak of the equilateral triangle
        # Rotation by 60 degrees: [cos -sin; sin cos]
        angle = -np.pi / 3.0
        rot = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ], dtype=torch.float32)
        
        s = q + torch.matmul(rot, (r - q))
        
        # Recursively get vertices
        return (koch_recurse(p1, q, order - 1) + 
                koch_recurse(q, s, order - 1) + 
                koch_recurse(s, r, order - 1) + 
                koch_recurse(r, p2, order - 1))

    # Initial equilateral triangle vertices
    # Radius of circumscribed circle
    r = scale / np.sqrt(3)
    angles = np.linspace(0, 2*np.pi, 4)[:-1] + np.pi/2
    p1 = torch.tensor([center[0] + r*np.cos(angles[0]), center[1] + r*np.sin(angles[0])], dtype=torch.float32)
    p2 = torch.tensor([center[0] + r*np.cos(angles[1]), center[1] + r*np.sin(angles[1])], dtype=torch.float32)
    p3 = torch.tensor([center[0] + r*np.cos(angles[2]), center[1] + r*np.sin(angles[2])], dtype=torch.float32)
    
    vertices = (koch_recurse(p1, p2, order) + 
                koch_recurse(p2, p3, order) + 
                koch_recurse(p3, p1, order))
    
    return torch.stack(vertices)
