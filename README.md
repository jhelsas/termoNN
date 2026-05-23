# PINN Laplace Solver

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

A modular and production-ready implementation of a **Physics-Informed Neural Network (PINN)** to solve the 2D Laplace Equation ($\nabla^2 u = 0$) on a unit square domain.

## 🚀 Overview

This project demonstrates how to use deep learning and automatic differentiation to solve partial differential equations (PDEs). The network is trained by minimizing a multi-part loss function that enforces both the governing physical laws and the Dirichlet boundary conditions.

### Key Features
- **Device Agnostic**: Automatically detects and utilizes CUDA if available.
- **Physics-Validated**: Robust testing suite verifying the Laplace operator against analytical solutions.
- **Reproducible**: Fixed random seeds for consistent training results.
- **Modular Design**: Separated concerns for model architecture, physics logic, and data utilities.

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
We maintain a comprehensive suite of 12 unit and integration tests:
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
