import torch
import numpy as np

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
        # Ray-casting can be numerically unstable on some GPU backends (like MPS) 
        # due to precision or branching handling. We force it to CPU for robustness.
        orig_device = x.device
        px, py = x.cpu(), y.cpu()
        
        def check_poly(poly, px, py):
            poly = poly.cpu()
            inside = torch.zeros_like(px, dtype=torch.bool)
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                
                intersect = ((p1[1] > py) != (p2[1] > py)) & \
                            (px < (p2[0] - p1[0]) * (py - p1[1]) / (p2[1] - p1[1] + 1e-12) + p1[0])
                inside ^= intersect
            return inside

        inside_outer = check_poly(self.vertices, px, py)
        for hole in self.holes:
            inside_outer &= ~check_poly(hole, px, py)
            
        return inside_outer.to(orig_device)

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
        """Samples points along the edges and returns coordinates plus polygon IDs."""
        dev = device or self.device
        
        def get_edge_data(poly):
            p1 = poly
            p2 = torch.roll(poly, -1, dims=0)
            vecs = p2 - p1
            lengths = torch.norm(vecs, dim=1)
            return p1, vecs, lengths

        all_polys = [self.vertices] + self.holes
        edge_starts, edge_vecs, edge_lens, edge_ids = [], [], [], []
        
        for idx, poly in enumerate(all_polys):
            p1, v, l = get_edge_data(poly)
            edge_starts.append(p1)
            edge_vecs.append(v)
            edge_lens.append(l)
            # Assign poly ID to every edge of this polygon
            edge_ids.append(torch.full((len(l),), idx, dtype=torch.long, device=self.device))
            
        starts = torch.cat(edge_starts)
        vecs = torch.cat(edge_vecs)
        lens = torch.cat(edge_lens)
        ids = torch.cat(edge_ids)
        
        # Categorical sampling
        probs = lens / lens.sum()
        indices = torch.multinomial(probs, n_points, replacement=True)
        
        t = torch.rand(n_points, 1, device=self.device)
        sampled_pts = starts[indices] + t * vecs[indices]
        sampled_ids = ids[indices]
        
        return sampled_pts[:, 0].to(dev), sampled_pts[:, 1].to(dev), sampled_ids.to(dev)

def generate_koch_snowflake(order=3, scale=1.0, center=(0.5, 0.5)):
    """
    Generates vertices for a Koch Snowflake fractal.
    """
    def koch_recurse(p1, p2, order):
        if order == 0:
            return [p1]
        
        v = p2 - p1
        q = p1 + v / 3.0
        r = p1 + v * 2.0 / 3.0
        
        angle = -np.pi / 3.0
        rot = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ], dtype=torch.float32)
        
        s = q + torch.matmul(rot, (r - q))
        
        return (koch_recurse(p1, q, order - 1) + 
                koch_recurse(q, s, order - 1) + 
                koch_recurse(s, r, order - 1) + 
                koch_recurse(r, p2, order - 1))

    r = scale / np.sqrt(3)
    angles = np.linspace(0, 2*np.pi, 4)[:-1] + np.pi/2
    p1 = torch.tensor([center[0] + r*np.cos(angles[0]), center[1] + r*np.sin(angles[0])], dtype=torch.float32)
    p2 = torch.tensor([center[0] + r*np.cos(angles[1]), center[1] + r*np.sin(angles[1])], dtype=torch.float32)
    p3 = torch.tensor([center[0] + r*np.cos(angles[2]), center[1] + r*np.sin(angles[2])], dtype=torch.float32)
    
    vertices = (koch_recurse(p1, p2, order) + 
                koch_recurse(p2, p3, order) + 
                koch_recurse(p3, p1, order))
    
    return torch.stack(vertices)
