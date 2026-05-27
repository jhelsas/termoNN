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
- **Model (`src/pinn/model.py`)**: High-capacity architecture supporting both **Tanh** and **SIREN (Sine)** activations, with **Multi-frequency spectral decomposition** and **Self-Adaptive** learnable scales.
- **Physics (`src/pinn/physics.py`)**: Implements the Poisson/Laplace operators, Dirichlet boundary conditions, and the Range/Maximum Principle penalty.
- **Solver (`src/pinn/solver.py`)**: Implements the two-stage (Adam + L-BFGS) optimization with self-adaptive loss weighting and residual-based refinement (RAR).
- **Core (`src/core/`)**: 
    - `geometry.py`: Handles complex geometries using ray-casting. Includes a conditional CPU fallback for stability on Apple Silicon.
    - `data.py`: Sampling, hardware abstraction (CUDA/MPS), and reproducibility.
    - `fem.py`: Finite Element Solver wrapper using `scikit-fem` for ground-truth verification.
    - `viz.py`: High-resolution plotting and masking.
- **Main (`main.py`)**: Production-ready training pipeline.
- **Comparison (`comparison_results.py`)**: End-to-end benchmarking tool for PINN vs FEM.

---

## Getting Started

### Prerequisites
- Python 3.8 or higher.
- `pip` and `venv`.

### Environment Setup (CRITICAL)
Always work within a virtual environment to ensure dependency isolation:
1. **Initialize/Check**:
   ```bash
   # Look for existing venv
   ls -d venv
   # Create if missing
   python3 -m venv venv
   ```
2. **Activate**:
   ```bash
   source venv/bin/activate
   ```
3. **Install/Sync Dependencies**:
   ```bash
   pip install -r requirements.txt
   # For FEM support:
   pip install scikit-fem
   ```

### Basic Usage
```bash
# Run training and visualization
python main.py

# Run PINN vs FEM Benchmarks
python comparison_results.py

# Run full test suite (70+ tests)
python -m unittest discover tests
```

---

## Project Structure
- `src/pinn/`: Neural network and PDE logic.
- `src/core/`: Geometric engine, FEM solver, and utilities.
- `tests/`:
    - `base_test.py`: Shared testing logic and assertions.
    - `test_model.py` / `test_model_advanced.py`: Model unit tests.
    - `test_physics.py` / `test_physics_advanced.py`: Physical constraint validation.
    - `test_geometry.py`: Point-in-polygon and sampling verification.
    - `test_fem.py`: PINN vs FEM verification tests.
    - `test_integration.py`: End-to-end workflow validation.

---

## Development Workflow

### Coding Standards (Staff+ Engineer Perspective)
1. **Differentiability**: Always use smooth activations (`Tanh`, `Sine`). `Sine` is preferred for second-order PDEs.
2. **Deterministic Sampling**: Always use the `set_seed` utility to ensure experiments are reproducible.
3. **Device Awareness**: Use `get_device()`. We support CUDA (NVIDIA) and MPS (Apple Silicon).
4. **Hardware Stability Workaround**: The ray-casting engine in `geometry.py` uses a **conditional CPU fallback** for the `is_inside` check when running on `mps` devices to avoid branching-related numerical instabilities. Keep this logic in place for cross-platform reliability.
5. **Autograd Safety**: When computing higher-order gradients, use `create_graph=True` and `allow_unused=True`.
6. **Code Navigation & Efficiency**: Before reading large source files, always consult `PROJECT_MAP.md`. This file contains an AST-derived summary of all classes, methods, and functions. Use it to identify the specific sections of code you need to modify or debug to minimize context overhead.

### Git Commit Style Guide
We follow a structured commit convention to maintain a clean and searchable history:
1. **Type Prefix**: Every commit must start with a lowercase type followed by a colon (e.g., `feat:`, `fix:`, `test:`, `chore:`, `docs:`).
   - `feat`: New feature or significant architecture change.
   - `fix`: Bug fix in logic, physics, or geometry.
   - `test`: Adding or updating tests.
   - `chore`: Maintenance tasks, dependency updates, or cleanup.
   - `docs`: Documentation changes.
2. **Imperative Mood**: Use the imperative mood in the subject line (e.g., "add test" instead of "added test").
3. **Detail Level**: For complex changes, include a bulleted list in the commit body explaining the "why" and "what" of the changes.

### Testing Approach
Our test suite follows a "Physics-First" verification strategy:
- **Unit Tests**: Check individual components (shapes, initialization, sampling bounds).
- **Geometric Validation**: Verifies ray-casting (inside/outside) and boundary sampling for complex polygons and holes.
- **Physics Validation**: Verify the Laplace residue against analytical solutions (Linear, Quad, Harmonic).
- **FEM Benchmarking**: Continuous verification of PINN solutions against a traditional Finite Element solver.
- **Integration Tests**: Ensure the optimizer successfully reduces the loss on complex domains.

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
