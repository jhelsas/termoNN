import torch
import torch.nn as nn
import numpy as np

class Sine(nn.Module):
    """Sine activation function with scaling for SIREN-like behavior."""
    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega
    def forward(self, x):
        return torch.sin(self.omega * x)

class PINN(nn.Module):
    """
    Enhanced PINN with support for SIREN (Sine) or Tanh activations.
    SIREN is preferred for complex geometries and higher-order derivatives.
    """
    def __init__(self, input_dim=2, hidden_dim=20, output_dim=1, num_layers=4, activation='sine'):
        super(PINN, self).__init__()
        
        layers = []
        
        # First Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if activation == 'sine':
            layers.append(Sine(omega=30.0))
        else:
            layers.append(nn.Tanh())
            
        # Hidden Layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == 'sine':
                layers.append(Sine(omega=30.0))
            else:
                layers.append(nn.Tanh())
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # SIREN specific initialization
        if activation == 'sine':
            self._init_siren()

    def _init_siren(self):
        """Standard SIREN initialization scheme."""
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                if isinstance(layer, nn.Linear):
                    num_input = layer.weight.size(-1)
                    if i == 0:
                        # First layer special init
                        layer.weight.uniform_(-1 / num_input, 1 / num_input)
                    else:
                        # Hidden layers
                        layer.weight.uniform_(-np.sqrt(6 / num_input) / 30.0, 
                                            np.sqrt(6 / num_input) / 30.0)
                    layer.bias.uniform_(-1e-6, 1e-6)

    def forward(self, x):
        return self.net(x)
