# PINN Laplace Solver

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

A modular and production-ready implementation of a **Physics-Informed Neural Network (PINN)** to solve the 2D **Poisson Equation** ($\nabla^2 u = f$) and **Laplace Equation** ($\nabla^2 u = 0$) on arbitrary, non-convex, and multi-connected domains.

## 🚀 Overview

This project demonstrates how to use deep learning and automatic differentiation to solve partial differential equations (PDEs) on complex geometries. Unlike traditional solvers restricted to simple grids, this PINN implementation utilizes a Tensor-native polygon engine and a flexible physics loss to handle diverse steady-state problems. It also includes a **Finite Element Method (FEM)** comparison suite for high-fidelity verification.

### Key Features
- **Complex Geometries**: Support for arbitrary polygons with multiple internal holes and fractal boundaries (e.g., Koch Snowflake).
- **High-Fidelity Representation**: Multi-frequency SIREN with **Fourier Feature Mapping** and **Residual Skip Connections** for capturing multiscale physics and fractal details.
- **Unified Constraints**: Combined PDE, Boundary, Range, and **Boundary Gradient** losses to strictly enforce the Maximum Principle and regularity.
- **Two-Stage Optimization**: Hybrid Adam and high-persistence L-BFGS for sub-millimetric convergence.
- **Self-Adaptive Loss Weighting**: Dynamically balances PDE and Boundary losses during training.
- **Adaptive Refinement**: **Residual-based Adaptive Refinement (RAR)** to focus sampling in high-residue regions automatically.
- **FEM Verification**: Built-in integration with `scikit-fem` to validate PINN results against traditional numerical methods.
- **Hardware Agnostic**: Full support for CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends.
- **Physics-Validated**: Comprehensive testing suite (70+ tests) verifying residues against analytical solutions.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd pinn-laplace-solver
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📉 Usage

### Training the Model
To start the training process and generate the solution plot:
```bash
python main.py
```
After training, the model will save a contour plot of the solution to `solution.png`.

### PINN vs FEM Comparison
To solve a complex geometry (like a nested snowflake annulus) with both PINN and FEM and compare the error:
```bash
python comparison_results.py
```

### Running Tests
We maintain a comprehensive suite of 95+ unit, geometric, physics, and integration tests:
```bash
python -m unittest discover tests
```

## 🏗️ Project Structure

- `src/`: Core logic modules.
  - `pinn/`: PINN specific modules (model, physics, solver).
  - `core/`: Shared core modules (data, geometry, viz, fem).
- `tests/`: Automated test suite for physical and architectural validation.
- `main.py`: Entry point for training and visualization.
- `comparison_results.py`: High-fidelity PINN vs FEM comparison tool.
- `CONTINUE.md`: Detailed project guide and developer context (located in `.continue/rules/`).

## 🧪 Key Concepts: The Physics Loss

The model solves the Poisson equation:
$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x, y)$$

It does so by minimizing the following objective:
$$L_{total} = \text{MSE}(\nabla^2 u, f) + \lambda \cdot \text{MSE}(u_{pred}, u_{bc})$$

Where:
- The first term ensures the network satisfies the governing equation (Laplace if $f=0$).
- The second term enforces the boundary conditions (e.g., $u(x, 0) = \sin(\pi x)$).

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 References
- [Raissi et al. (2019) - Physics-informed neural networks](https://maziarraissi.github.io/PINNs/)
- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
