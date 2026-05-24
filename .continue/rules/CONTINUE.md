# PINN Laplace Solver - Project Guide

## Project Overview
This project implements a **Physics-Informed Neural Network (PINN)** to solve the 2D Laplace Equation ($\nabla^2 u = 0$) within a unit square domain $[0, 1] \times [0, 1]$. 

PINNs represent a paradigm shift in scientific computing, where neural networks act as universal function approximators constrained by physical laws. Instead of relying solely on data, we embed the differential equations directly into the loss function using automatic differentiation.

### Key Technologies
- **Python 3.8+**: Core language.
- **PyTorch**: Used for the MLP architecture and its powerful `autograd` engine for computing PDE residues ($u_{xx}, u_{yy}$).
- **NumPy & Matplotlib**: Data handling and visualization of the heat distribution.
- **Unittest**: Comprehensive test suite for physics validation and regression testing.

### High-Level Architecture
- **Model (`src/model.py`)**: A standard MLP using `Tanh` activations to ensure smooth higher-order derivatives.
- **Physics (`src/physics.py`)**: Implements the Laplace operator $\Delta u = 0$ and Dirichlet boundary conditions.
- **Utilities (`src/utils.py`)**: Robust data sampling and reproducibility helpers.
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
- **Physics Validation**: Verify the Laplace residue against analytical solutions (Linear, Quad, Harmonic).
- **Integration Tests**: Ensure the optimizer successfully reduces the loss and model weights are updated.

---

## Key Concepts

### Laplace Operator
We solve $\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$. In PyTorch, this is achieved by double-calling `torch.autograd.grad`.

### Hybrid Optimization
- **Adam**: Used initially to escape local minima and navigate the non-convex PINN loss landscape.
- **L-BFGS**: Switched to for final convergence, as it uses second-order curvature information to find the precise physics solution.

---

## Common Tasks

### Adding a New PDE
1. Define the residue function in `src/physics.py`.
2. Add a verification test in `tests/test_physics.py` using a known analytical solution.
3. Update `main.py` to include the new loss term.

### Modifying Boundary Conditions
Update `generate_boundary_data` in `src/utils.py`. The current implementation uses a $\sin(\pi x)$ profile on the bottom edge and zero elsewhere.

---

## Troubleshooting
- **Vanishing Gradients**: Check if the network is too deep or if activations are saturating.
- **Loss Stagnation**: Try increasing the `lambda_bc` weight. Boundary conditions are often harder to satisfy than the PDE itself.
- **Shape Mismatches**: Ensure `u_bc` and `u_pred` have matching dimensions (typically `[N, 1]`).

---

## References
- [Raissi et al. (2019) PINN Paper](https://maziarraissi.github.io/PINNs/)
- [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
