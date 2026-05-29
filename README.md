# PINN Laplace Solver

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen.svg)](#-running-tests--coverage)

A modular implementation of a **Physics-Informed Neural Network (PINN)** to solve the 2D **Poisson Equation** ($\nabla^2 u = f$) and **Laplace Equation** ($\nabla^2 u = 0$) on arbitrary, non-convex, and multi-connected domains.

## 🚀 Overview

This project is an **exploratory implementation** of PINNs for steady-state heat and potential problems on complex geometries. It currently supports 2D domains with Dirichlet and Neumann boundary conditions, utilizing a Tensor-native polygon engine.

### Key Features
- **Complex Geometries**: Support for arbitrary polygons with multiple internal holes and fractal boundaries (e.g., Koch Snowflake).
- **High-Fidelity Representation**: Multi-frequency SIREN with **Fourier Feature Mapping** and **Residual Skip Connections**.
- **Adaptive Training**: 
    - **Self-Adaptive Loss Weighting**: Dynamically balances PDE and Boundary losses based on gradient statistics.
    - **RAR Sampling**: Residual-based Adaptive Refinement that concentrates collocation points in regions with high PDE residue.
- **Verification**: Integrated FEM comparison suite using `scikit-fem`.

### Project Status (Exploratory)
This is an **early-stage research project**. Current limitations include:
- **Scope**: Limited to 2D steady-state problems (Laplace/Poisson).
- **Optimization**: No domain decomposition or large-scale multi-GPU parallelism.
- **Physics**: Does not yet support time-dependent equations or 3D geometries.

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

### Running Tests & Coverage
We maintain a comprehensive suite of 95+ tests with 95% code coverage. Our testing strategy follows a hierarchical approach:

- **Unit Tests**: Verification of individual components like `PolygonDomain` (geometry), `Sine` activation (model), and data samplers.
- **Physics Tests**: Validation of the PDE residues ($u_{xx} + u_{yy}$) against known analytical solutions (Linear, Quadratic, and Harmonic functions).
- **Integration Tests**: Verification of the full training loop (Adam + L-BFGS), plotting workflows, and cross-device (CPU/GPU) consistency.
- **Geometric Tests**: Rigorous checking of point-in-polygon logic, ray-casting robustness, and boundary normal calculations for complex/fractal shapes.
- **FEM Benchmarking**: End-to-end verification of the PINN solution against a ground-truth Finite Element solver.

```bash
# Run all tests
pytest tests/

# Run tests with coverage report
pytest --cov=src tests/
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
