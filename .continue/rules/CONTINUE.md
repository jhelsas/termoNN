# PINN Laplace Solver Project Guide

## Project Overview
This project implements a **Physics-Informed Neural Network (PINN)** to solve the 2D Laplace Equation ($\nabla^2 u = 0$) within a unit square domain $[0, 1] \times [0, 1]$. 

### Key Technologies
- **Python 3.14** (Virtual Environment managed via `venv`)
- **PyTorch**: Deep learning framework used for building the neural network and performing automatic differentiation.
- **NumPy & Matplotlib**: For data manipulation and visualization.
- **Unittest**: For ensuring code reliability.

### High-Level Architecture
The project is modularized into four main components:
1.  **Model (`src/model.py`)**: Defines the MLP (Multi-Layer Perceptron) architecture.
2.  **Physics (`src/physics.py`)**: Contains logic for the PDE loss (Laplace operator) and boundary condition enforcement using automatic differentiation.
3.  **Utilities (`src/utils.py`)**: Handles data generation for the domain and boundaries.
4.  **Main Loop (`main.py`)**: Orchestrates the training process and generates visualizations.

---

## Getting Started

### Prerequisites
- Python 3.8+ (Tested on 3.14)
- Virtual environment support

### Installation
1. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage
To train the model and generate the solution plot:
```bash
python main.py
```
This produces `solution.png` showing the heat distribution.

### Running Tests
To run the unit tests:
```bash
python -m unittest discover tests
```

---

## Project Structure
- `src/`: Core logic
    - `model.py`: PINN architecture definition.
    - `physics.py`: Laplace and BC loss functions using `torch.autograd`.
    - `utils.py`: Domain and boundary data sampling.
- `tests/`: Unit tests for components.
- `main.py`: Entry point for training and plotting.
- `requirements.txt`: Project dependencies.
- `solution.png`: Generated result after running the script.

---

## Development Workflow

### Coding Standards
- Follow PEP 8 guidelines.
- Use descriptive docstrings for all functions and classes.
- Ensure all physical constraints are implemented using `torch.autograd`.
- **Important**: When computing higher-order derivatives, use `allow_unused=True` in `torch.autograd.grad` to handle cases where certain inputs don't contribute to the gradient (e.g., in linear models during testing).

### Testing Approach
- **Unit Tests**: Every component (Model, Physics, Utils) has corresponding tests in `tests/test_pinn.py`.
- **Physics Validation**: We verify the Laplace loss against known analytical solutions:
    - Linear functions ($u=ax+by+c$) must yield 0 loss.
    - Quadratic functions ($u=x^2+y^2$) must yield a calculated loss of 16.
- **Integration Tests**: We confirm gradient flow by ensuring model weights update after a backward pass.
- **Data Integrity**: Sampling utilities are tested for correct output shapes and physical boundary ranges.

---

## Key Concepts

### Physics-Informed Neural Networks (PINNs)
PINNs integrate physical laws (PDEs) into the loss function of a neural network. The total loss is typically:
$$L = L_{PDE} + \lambda L_{BC}$$
where $L_{PDE}$ ensures the network satisfies the differential equation across the domain, and $L_{BC}$ enforces boundary conditions.

### Laplace Equation
The project solves:
$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

---

## Common Tasks

### Changing Boundary Conditions
Modify `src/utils.py` in the `generate_boundary_data` function to define different boundary values or geometries.

### Adjusting Hyperparameters
Training parameters like learning rate, epochs, and network depth can be adjusted in `main.py` and `src/model.py`.

---

## Troubleshooting
- **Loss not Kids converging**: Try increasing the weight $\lambda$ of the boundary loss (currently 10 in `main.py`) or decreasing the learning rate.
- **Cuda Errors**: The current implementation defaults to CPU. For GPU support, move the model and tensors to `cuda`.
- **Autograd Errors**: If you encounter "One of the differentiated Tensors appears to not have been used in the graph", ensure you are using `allow_unused=True` for secondary gradients.

---

## References
- [Raissi et al. (2019) - Physics-informed neural networks](https://maziarraissi.github.io/PINNs/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
