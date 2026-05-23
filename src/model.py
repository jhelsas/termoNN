import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Simple Physics-Informed Neural Network for 2D problems.
    """
    def __init__(self, input_dim=2, hidden_dim=20, output_dim=1, num_layers=4):
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
