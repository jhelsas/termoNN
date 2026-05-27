import numpy as np
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm, solve, condense
from skfem.models.poisson import laplace
from scipy.spatial import Delaunay
import torch

def solve_fem(domain, bc_fn, f_fn=None, resolution=50):
    """
    Solves the Poisson equation on a PolygonDomain using Finite Element Method (scikit-fem).
    
    Args:
        domain: PolygonDomain instance.
        bc_fn: Boundary condition function (u_bc = bc_fn(x, y, b_ids)).
        f_fn: Source term function (f = f_fn(x, y)).
        resolution: approximate number of points along the bounding box side for mesh generation.
    """
    # 1. Generate points for triangulation
    # We sample boundary and interior points
    n_boundary = resolution * 4
    n_interior = resolution**2
    
    x_b, y_b, b_ids = domain.sample_boundary(n_boundary)
    x_i, y_i = domain.sample_interior(n_interior)
    
    pts = np.vstack([
        np.column_stack([x_b.cpu().numpy(), y_b.cpu().numpy()]),
        np.column_stack([x_i.cpu().numpy(), y_i.cpu().numpy()])
    ])
    
    # 2. Triangulate
    tri = Delaunay(pts)
    simplices = tri.simplices
    
    # 3. Filter triangles outside the domain
    centroids = pts[simplices].mean(axis=1)
    mask = domain.is_inside(
        torch.tensor(centroids[:, 0], device=domain.device),
        torch.tensor(centroids[:, 1], device=domain.device)
    ).cpu().numpy()
    simplices = simplices[mask]
    
    # Create skfem mesh
    # pts are (N, 2), skfem wants (2, N)
    mesh = MeshTri(pts.T, simplices.T)
    
    # 4. Define Basis
    element = ElementTriP1()
    basis = Basis(mesh, element)
    
    # 5. Assemble System
    # Note: PINN solves u_xx + u_yy = f
    # FEM standard form is -int(grad u . grad v) = int(f v)
    # which corresponds to -Delta u = f.
    # To match PINN's Delta u = f, we use the negative laplace form or negate f.
    
    @BilinearForm
    def laplacian(u, v, w):
        return (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

    A = laplacian.assemble(basis)
    
    @LinearForm
    def load(v, w):
        if f_fn is not None:
            # skfem provides w.x for coordinates
            f_val = f_fn(torch.tensor(w.x[0]), torch.tensor(w.x[1])).cpu().numpy()
            return f_val * v
        return 0.0 * v

    # We want Delta u = f, so -Delta u = -f
    # A * u = -load
    b = load.assemble(basis)
    
    # 6. Boundary Conditions
    # Identify boundary nodes
    boundary_dofs = basis.get_dofs().all()
    
    # Get boundary values
    node_coords = mesh.p[:, boundary_dofs]
    
    # Re-sample boundary IDs for these specific points
    x_nodes = torch.tensor(node_coords[0], dtype=torch.float32, device=domain.device)
    y_nodes = torch.tensor(node_coords[1], dtype=torch.float32, device=domain.device)
    
    import inspect
    sig = inspect.signature(bc_fn)
    if 'b_ids' in sig.parameters:
        # Re-sample boundary data to get b_ids for finding nearest
        x_b, y_b, b_ids_sampled = domain.sample_boundary(n_boundary)
        sampled_b_coords = torch.stack([x_b, y_b], dim=1)
        node_b_coords = torch.stack([x_nodes, y_nodes], dim=1)
        
        # Compute distances
        dist = torch.cdist(node_b_coords, sampled_b_coords)
        nearest_idx = dist.argmin(dim=1)
        node_b_ids = b_ids_sampled[nearest_idx]
        
        u_bc = bc_fn(x_nodes, y_nodes, b_ids=node_b_ids).cpu().numpy().flatten()
    else:
        u_bc = bc_fn(x_nodes, y_nodes).cpu().numpy().flatten()
        
    # Solve
    # x is the full solution vector, we need to initialize it with BCs for condense
    x_init = np.zeros(basis.N)
    x_init[boundary_dofs] = u_bc
    
    A_c, b_c, x_c, I = condense(A, -b, D=boundary_dofs, x=x_init)
    x_sol = solve(A_c, b_c, x_c, I)
    
    return mesh, x_sol
    
    return mesh, x

def interpolate_fem(mesh, u_fem, x_query, y_query):
    """Interpolates FEM solution at query points."""
    pts = np.column_stack([x_query.cpu().numpy(), y_query.cpu().numpy()])
    basis = Basis(mesh, ElementTriP1())
    try:
        # basis.interpolator(u_fem) returns a function that takes (2, N) pts
        return basis.interpolator(u_fem)(pts.T)
    except ValueError:
        # Some points might be outside the mesh due to boundary discretization
        # We can do a slower but more robust point-by-point interpolation or just return NaNs
        interp_func = basis.interpolator(u_fem)
        out = np.full(len(pts), np.nan)
        for i in range(len(pts)):
            try:
                out[i] = interp_func(pts[i:i+1].T)
            except ValueError:
                continue
        return out
