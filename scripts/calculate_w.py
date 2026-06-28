import sympy as sp

w, x, y = sp.symbols('w x y')
u = w * (1 - x**2) * (1 - y**2)

# Strong form equation is \Delta u = f
# Wait, the code says:
# PINN solves: u_xx + u_yy = f
# The corresponding energy functional adds + f * u
# Wait, if f is constant 1, we want \Delta u = 1.
# What is the true solution?

# Does the energy functional minimum approach the true solution?
# The true solution must satisfy \Delta u = f.
# The energy functional is J(u) = 1/2 \int |\nabla u|^2 d\Omega + \int f u d\Omega.
# Variation: \delta J = \int \nabla u \cdot \nabla(\delta u) + f \delta u = 0
# Integrating by parts (assuming \delta u = 0 on boundary):
# \int (-\Delta u + f) \delta u = 0  => -\Delta u + f = 0 => \Delta u = f.
# 
# Wait, let me check the physics code:
# energy = 0.5 * grad_norm_sq
# if f_fn is not None:
#     f = f_fn(x, y).squeeze()
#     energy = energy + f * u.squeeze()

# Let's consider the Dirichlet problem:
# \Delta u = f  in Omega
# u = g on \partial Omega
# Energy functional: J(u) = \int_\Omega ( \frac{1}{2} |\nabla u|^2 + f u ) dx
# Is it J(u) = \int_\Omega ( \frac{1}{2} |\nabla u|^2 + f u ) dx ?
# Let's see: \delta J(u)[v] = \int_\Omega ( \nabla u \cdot \nabla v + f v ) dx
# By Green's first identity:
# \int_\Omega \nabla u \cdot \nabla v dx = \int_{\partial \Omega} (\nabla u \cdot n) v dS - \int_\Omega (\Delta u) v dx
# Since v = 0 on boundary (Dirichlet BC), the boundary integral vanishes.
# So \delta J(u)[v] = \int_\Omega ( -\Delta u + f ) v dx = 0
# This implies -\Delta u + f = 0  => \Delta u = f.
# Yes, this is correct! 
# So why did the test in test_energy_issue.py show different w values?
# Because the subspace spanned by u = w(1-x^2)(1-y^2) does not contain the true solution!
# Galerkin method (Energy minimization) minimizes the energy over the subspace.
# Least squares (Poisson loss minimization) minimizes the L2 norm of the residual over the subspace.
# If the subspace doesn't contain the exact solution, the minimizer will be different.
