import numpy as np

import torch
from skfem import *
from skfem.models.poisson import laplace
from typing import Callable, Optional, Tuple
from scipy.spatial import Delaunay

def solve_fem(domain, 
              bc_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              f_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
              n_interior: Optional[int] = None,
              n_boundary: Optional[int] = None,
              resolution: Optional[int] = None) -> Tuple[MeshTri, np.ndarray]:
    """
    Solves the Laplace or Poisson equation using Finite Element Method.
    Uses Scipy Delaunay with domain masking for robust mesh generation on any Python version.
    
    Args:
        domain: PolygonDomain object.
        bc_fn: Function returning Dirichlet values (expects torch Tensors).
        f_fn: Function returning source term (expects torch Tensors).
        n_interior: Number of nodes to sample in the interior.
        n_boundary: Number of nodes to sample on the boundaries.
        resolution: Convenience to set n_interior=resolution**2 and n_boundary=resolution*4.
        
    Returns:
        mesh: The skfem Mesh object.
        u: The solution vector.
    """
    if resolution is not None:
        if n_interior is None: n_interior = resolution**2
        if n_boundary is None: n_boundary = resolution * 4
    
    if n_interior is None: n_interior = 2000
    if n_boundary is None: n_boundary = 500

    # 1. Mesh Generation using Scipy Delaunay + Polygon Masking
    # Sample points
    x_int, y_int = domain.sample_interior(n_interior, device='cpu')
    x_bd, y_bd, _ = domain.sample_boundary(n_boundary, device='cpu')
    
    # Combine and add vertices for boundary fidelity
    all_x = torch.cat([x_int, x_bd, domain.vertices[:, 0].cpu()])
    all_y = torch.cat([y_int, y_bd, domain.vertices[:, 1].cpu()])
    for hole in domain.holes:
        all_x = torch.cat([all_x, hole[:, 0].cpu()])
        all_y = torch.cat([all_y, hole[:, 1].cpu()])
        
    points = np.stack([all_x.numpy(), all_y.numpy()], axis=1)
    
    # Generate full Delaunay triangulation of the point cloud
    tri = Delaunay(points)
    
    # Mask triangles whose centroids are outside the domain or inside holes
    centers = np.mean(points[tri.simplices], axis=1)
    mask = domain.is_inside(torch.from_numpy(centers[:, 0]).float().to(domain.device), 
                            torch.from_numpy(centers[:, 1]).float().to(domain.device))
    mask = mask.cpu().numpy()
    
    # Create the filtered mesh
    mesh = MeshTri(points.T, tri.simplices[mask].T)
    
    # 2. FEM Assembly
    basis = Basis(mesh, ElementTriP1())
    
    # Stiffness matrix (Laplacian)
    A = asm(laplace, basis)
    
    # Load vector (Source term)
    if f_fn is not None:
        @LinearForm
        def source_form(v, w):
            # Convert skfem coordinates (D, Nq) to torch for the source function
            orig_shape = w.x.shape # (2, N_quad_pts, N_elements)
            pts_flat = w.x.reshape(2, -1)
            
            xt = torch.from_numpy(pts_flat[0]).float().to(domain.device)
            yt = torch.from_numpy(pts_flat[1]).float().to(domain.device)
            
            # PINN solves: u_xx + u_yy = f
            # skfem standard Poisson is: -Delta u = f
            # To match PINN, we solve: -Delta u = -f
            res_torch = -f_fn(xt, yt).cpu().numpy().reshape(orig_shape[1:])
            return res_torch * v
        b = asm(source_form, basis)
    else:
        b = np.zeros(basis.N)
        
    # 3. Boundary Conditions
    # dofs.all() gives ALL DOFs on the Dirichlet boundary
    boundary_dofs = basis.get_dofs().all()
    
    # Evaluate bc_fn at boundary node coordinates
    pb = mesh.p[:, boundary_dofs].T
    xt_b = torch.from_numpy(pb[:, 0]).float().to(domain.device)
    yt_b = torch.from_numpy(pb[:, 1]).float().to(domain.device)
    
    u_bc_vals = bc_fn(xt_b, yt_b).cpu().numpy().flatten()
    
    # Create full-size BC vector to avoid IndexError in condense
    x_bc = np.zeros(basis.N)
    x_bc[boundary_dofs] = u_bc_vals
        
    # Solve linear system using skfem.solve
    # 'D' expects the indices of the Dirichlet DOFs, and 'x' expects the full vector
    u = solve(*condense(A, b, x=x_bc, D=boundary_dofs))
    
    return mesh, u
