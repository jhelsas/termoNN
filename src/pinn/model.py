import torch


import torch.nn as nn
import numpy as np

class Sine(nn.Module):
    """Sine activation function with support for per-channel frequencies and learnable scaling."""
    def __init__(self, omega=1.0, adaptive=False):
        super().__init__()
        # omega can be a scalar or a (channels,) tensor
        self.register_buffer("omega", torch.as_tensor(omega))
        
        # Learnable scaling factor (a in sin(a * omega * x))
        if adaptive:
            self.a = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("a", torch.ones(1))
        
    def forward(self, x):
        return torch.sin(self.a * self.omega * x)

class PINN(nn.Module):
    """
    Enhanced PINN with support for SIREN (Sine) or Tanh activations.
    Supports Multi-frequency SIREN and Self-Adaptive Activations (learnable scaling).
    """
    def __init__(self, input_dim=2, hidden_dim=20, output_dim=1, num_layers=4, 
                 activation='sine', omega=30.0, adaptive_activations=False):
        super(PINN, self).__init__()
        self.activation = activation
        
        # Determine omega for each channel (multi-frequency support)
        if isinstance(omega, (list, tuple)):
            # If omega is (min, max), distribute frequencies across hidden units
            omega_min, omega_max = omega
            # Create a distribution of frequencies
            freqs = torch.linspace(omega_min, omega_max, hidden_dim)
            self.omega_val = freqs
        else:
            self.omega_val = torch.full((hidden_dim,), float(omega))
            
        layers = []
        
        # First Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if activation == 'sine':
            layers.append(Sine(omega=self.omega_val, adaptive=adaptive_activations))
        else:
            layers.append(nn.Tanh())
            
        # Hidden Layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == 'sine':
                # Use same multi-frequency distribution for all hidden layers
                layers.append(Sine(omega=self.omega_val, adaptive=adaptive_activations))
            else:
                layers.append(nn.Tanh())
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        if activation == 'sine':
            self._init_siren()

    def _init_siren(self):
        """Standard SIREN initialization scheme adapted for multi-frequency."""
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                if isinstance(layer, nn.Linear):
                    num_input = layer.weight.size(-1)
                    # For multi-frequency, we use the average omega for scaling the init
                    avg_omega = self.omega_val.mean().item()
                    
                    if i == 0:
                        layer.weight.uniform_(-1 / num_input, 1 / num_input)
                    else:
                        layer.weight.uniform_(-np.sqrt(6 / num_input) / avg_omega, 
                                            np.sqrt(6 / num_input) / avg_omega)
                    layer.bias.uniform_(-1e-6, 1e-6)

    def forward(self, x):
        return self.net(x)
