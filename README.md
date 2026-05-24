# PINN Laplace Solver

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

A modular and production-ready implementation of a **Physics-Informed Neural Network (PINN)** to solve the 2D Laplace Equation ($\nabla^2 u = 0$) on arbitrary, non-convex, and multi-connected domains.

## 🚀 Overview

This project demonstrates how to use deep learning and automatic differentiation to solve partial differential equations (PDEs) on complex geometries. Unlike traditional solvers restricted to simple grids, this PINN implementation utilizes a Tensor-native polygon engine to handle domains with holes and non-convex boundaries.

### Key Features
- **Complex Geometries**: Support for arbitrary polygons with multiple internal holes.
- **Tensor-Native Geometry**: All geometric checks (Point-in-Polygon) and sampling are performed directly in PyTorch for maximum efficiency.
- **Two-Stage Optimization**: Combines Adam for exploration and L-BFGS for high-precision convergence.
- **Physics-Validated**: Robust testing suite (38+ tests) verifying residues against analytical solutions (Harmonic, Saddle, Linear).

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

## 📈 Usage

### Training the Model
To start the training process and generate the solution plot:
```bash
python main.py
```
After training, the model will save a contour plot of the solution to `solution.png`.

### Running Tests
We maintain a comprehensive suite of 38+ unit, geometric, and integration tests:
```bash
python -m unittest discover tests
```

## 🏗️ Project Structure

- `src/`: Core logic modules.
  - `model.py`: MLP architecture definition.
  - `physics.py`: Laplace operator and BC loss implementations using `torch.autograd`.
  - `utils.py`: Reproducibility settings and data sampling.
- `tests/`: Automated test suite for physical and architectural validation.
- `main.py`: Entry point for training and visualization.
- `CONTINUE.md`: Detailed project guide and developer context (located in `.continue/rules/`).

## 🧪 Key Concepts: The Physics Loss

The model solves the equation:
$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

It does so by minimizing the following objective:
$$L_{total} = \text{MSE}(\nabla^2 u, 0) + \lambda \cdot \text{MSE}(u_{pred}, u_{bc})$$

Where:
- The first term ensures the network satisfies the Laplace equation inside the domain.
- The second term enforces the boundary conditions (e.g., $u(x, 0) = \sin(\pi x)$).

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 References
- [Raissi et al. (2019) - Physics-informed neural networks](https://maziarraissi.github.io/PINNs/)
- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
