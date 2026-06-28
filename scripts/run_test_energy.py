import torch
import torch.nn as nn
from src.pinn.physics import energy_loss

# Let's test the formulation of energy loss for PINNs.
# The standard Deep Ritz formulation for \Delta u = f is
# J(u) = \int \frac{1}{2} |\nabla u|^2 + f u
# But wait! If the equation is \Delta u = f, the standard variation is:
# \int \nabla u \cdot \nabla v + f v
# integrating by parts gives: -\int (\Delta u) v + \int (\nabla u \cdot n) v + \int f v
# So \int (-\Delta u + f) v = 0 -> \Delta u = f.
# 
# Is the gradient of this loss function with respect to weights correct?
# loss = J(u) = \int \frac{1}{2} |\nabla u|^2 + f u
# grad_w = \int \nabla u \cdot \nabla(\partial u / \partial w) + f (\partial u / \partial w)
#        = \int (-\Delta u + f) (\partial u / \partial w) + \int (\nabla u \cdot n) (\partial u / \partial w)
#
# Let's check with an example. If \Delta u = 1 on [-1,1]^2 with u=0 on boundary.
# Is the minimum of J(u) actually the solution to \Delta u = 1 ?
# Actually J(u) can be negative!
# J(0) = 0.
# If u = w(1-x^2)(1-y^2). 
# J(u) = 1/2 \int |\nabla u|^2 + f u
# Let's re-run the previous script.

# Ah, the difference in `calculate_w.py` between `w for Poisson: -0.340909090909091` and `w for Energy: -0.312500000000000` is because `u = w(1-x^2)(1-y^2)` is just a single-parameter family. The exact solution to \Delta u = 1 is an infinite series!
# The Galerkin approximation (Energy) and Least Squares (Poisson) give different weights for a single parameter ansatz. That's entirely expected and correct!
# Let me check if there's any other issue with the energy formulation.
