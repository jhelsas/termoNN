# PINN Laplace Solver - Project Guide

## Project Overview
This project implements a **Physics-Informed Neural Network (PINN)** to solve the 2D **Poisson Equation** ($\nabla^2 u = f$) and the **Laplace Equation** ($\nabla^2 u = 0$) within arbitrary, non-convex, and multi-connected domains (polygons with holes).

PINNs represent a paradigm shift in scientific computing, where neural networks act as universal function approximators constrained by physical laws. This implementation goes beyond simple rectangular domains, supporting complex geometries through a robust polygon-based sampling engine.

### Key Technologies
- **Python 3.8+**: Core language.
- **PyTorch**: Used for the MLP architecture and its powerful `autograd` engine for computing PDE residues ($u_{xx}, u_{yy}$).
- **NumPy & Matplotlib**: Data handling and visualization of the heat distribution.
- **Unittest**: Comprehensive test suite for physics validation and geometric correctness.

### High-Level Architecture
- **Model (`src/model.py`)**: High-capacity architecture supporting both **Tanh** and **SIREN (Sine)** activations, with **Multi-frequency spectral decomposition** for capturing multi-scale details.
- **Physics (`src/physics.py`)**: Implements the Poisson/Laplace operators and Dirichlet boundary conditions.
- **Utilities (`src/utils.py`)**: 
    - `PolygonDomain`: Handles complex geometries using ray-casting for point-in-polygon checks and rejection sampling for interior points.
    - Data sampling and hardware abstraction.
- **Main (`main.py`)**: A production-ready training pipeline combining Adam (exploration) and L-BFGS (exploitation).

---

## Getting Started

### Prerequisites
- Python 3.8 or higher.
- `pip` and `venv`.

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run training and visualization
python main.py

# Run tests
python -m unittest discover tests
```

---

## Project Structure
- `src/`:
    - `model.py`: MLP architecture.
    - `physics.py`: PDE and BC loss functions.
    - `utils.py`: Sampling and hardware abstraction.
- `tests/`:
    - `base_test.py`: Shared testing logic and assertions.
    - `test_model.py`: Unit tests for the neural network.
    - `test_physics.py`: Validation of physical constraints.
    - `test_utils.py`: Verification of data generation.
    - `test_integration.py`: End-to-end workflow validation.

---

## Development Workflow

### Coding Standards (Staff+ Engineer Perspective)
1. **Differentiability**: Always use smooth activations (`Tanh`, `Sine`, `ELU`). Avoid `ReLU` for PINNs solving second-order PDEs.
2. **Deterministic Sampling**: Always use the `set_seed` utility to ensure experiments are reproducible.
3. **Device Awareness**: Always use `get_device()` to support both CPU and GPU transparently.
4. **Autograd Safety**: When computing higher-order gradients, use `create_graph=True` and `allow_unused=True` to handle edge cases in the computational graph.

### Testing Approach
Our test suite follows a "Physics-First" verification strategy:
- **Unit Tests**: Check individual components (shapes, initialization, sampling bounds).
- **Geometric Validation**: Verifies ray-casting (inside/outside) and boundary sampling for complex polygons and holes.
- **Physics Validation**: Verify the Laplace residue against analytical solutions (Linear, Quad, Harmonic).
- **Integration Tests**: Ensure the optimizer successfully reduces the loss on complex domains (e.g., L-shaped domains).

---

## Key Concepts

### Complex Domain Handling
The project supports domains defined by an outer polygon and multiple inner holes, using a Tensor-native geometry engine.
- **Interior Sampling**: Uses rejection sampling within the bounding box, executed entirely in PyTorch.
- **Boundary Sampling**: Uses length-weighted sampling across all polygon segments to ensure uniform point density.
- **Point-in-Polygon**: Vectorized ray-casting algorithm implemented in PyTorch for seamless GPU acceleration.

### Poisson/Laplace Operator
We solve $\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x, y)$, with an optional **Range Constraint** to enforce the Maximum Principle in high-gradient areas.
- If $f=0$, we are solving the **Laplace Equation**.
- If $f \neq 0$, we are solving the **Poisson Equation**.
In PyTorch, this is achieved by double-calling `torch.autograd.grad`.

### Hybrid Optimization
- **Adam**: Used initially to escape local minima and navigate the non-convex PINN loss landscape.
- **L-BFGS**: High-persistence second-order optimization with tight tolerances and adaptive history, used for final 'physics-snapping' and convergence.

---

## Common Tasks

### Defining a Custom Domain and Physics
To solve a problem:
1. Define the geometry using `PolygonDomain`.
2. Define a `bc_fn(x, y)` to set boundary values.
3. (Optional) Define an `f_fn(x, y)` source term for Poisson.
4. Pass them to `train()`.

```python
outer = [(0,0), (2,0), (2,1), (0,1)]
domain = PolygonDomain(outer)
# Solve Poisson: u_xx + u_yy = 1
model = train(domain=domain, 
              bc_fn=lambda x, y: torch.zeros_like(x),
              f_fn=lambda x, y: torch.ones_like(x))
```

### Adding a New PDE
1. Define the residue function in `src/physics.py`.
2. Add a verification test in `tests/test_physics.py` using a known analytical solution.
3. Update `main.py` to include the new loss term.

### Modifying Boundary Conditions
Update the `bc_fn` passed to `train()`. The default in `main.py` uses a mask-based approach to apply specific values (like `sin(pi*x)`) to specific edges based on coordinates.

---

## Troubleshooting
- **Vanishing Gradients**: Check if the network is too deep or if activations are saturating.
- **Loss Stagnation**: Try increasing the `lambda_bc` weight. Boundary conditions are often harder to satisfy than the PDE itself.
- **Shape Mismatches**: Ensure `u_bc` and `u_pred` have matching dimensions (typically `[N, 1]`).

---

## References
- [Raissi et al. (2019) PINN Paper](https://maziarraissi.github.io/PINNs/)
- [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
