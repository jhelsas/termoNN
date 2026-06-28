import torch
import torch.nn as nn
from src.core.geometry import PolygonDomain
from src.core.data import get_device
from src.pinn.solver import train
import matplotlib.pyplot as plt

device = get_device()
outer = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
domain = PolygonDomain(outer)

def f_fn(x, y):
    return torch.zeros_like(x)

def bc_fn(x, y):
    return x**2 - y**2

config_poisson = {
    "adam_epochs": 1000,
    "lbfgs_epochs": 500,
    "use_energy": False,
    "lambda_bc": 100.0,
    "num_layers": 3,
    "hidden_dim": 16,
    "activation": "sine",
}

print("Training with Poisson Loss (Sine)")
model_p = train(
    domain=domain,
    bc_fn=bc_fn,
    f_fn=f_fn,
    config=config_poisson
)[0]

x_test = torch.tensor([[0.5, 0.25], [-0.3, 0.8]], device=device)
print("Exact: ", x_test[:,0]**2 - x_test[:,1]**2)
print("Poisson: ", model_p(x_test).flatten().detach())

