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
    Advanced PINN with Fourier Feature Mapping and Residual Skip Connections.
    Designed for high-accuracy field reconstruction on fractal domains.
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_layers=4, 
                 activation='sine', omega=30.0, adaptive_activations=False,
                 use_fourier_features=True, fourier_scale=10.0,
                 output_transform=None):
        super(PINN, self).__init__()
        self.activation = activation
        self.use_fourier_features = use_fourier_features
        self.output_transform = output_transform
        
        # 1. Fourier Feature Mapping (Input Embedding)
        # This allows the network to capture high-frequency fractal details
        mapping_dim = hidden_dim // 2
        if use_fourier_features:
            self.register_buffer("B", torch.randn(input_dim, mapping_dim) * fourier_scale)
            current_input_dim = mapping_dim * 2
        else:
            current_input_dim = input_dim

        # 2. Determine omega distribution
        if isinstance(omega, (list, tuple)):
            omega_min, omega_max = omega
            self.omega_val = torch.linspace(omega_min, omega_max, hidden_dim)
        else:
            self.omega_val = torch.full((hidden_dim,), float(omega))
            
        # 3. Layer Construction
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(current_input_dim, hidden_dim))
        
        # Hidden layers with Residual support
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Activation modules
        self.act = nn.ModuleList()
        for _ in range(num_layers):
            if activation == 'sine':
                self.act.append(Sine(omega=self.omega_val, adaptive=adaptive_activations))
            else:
                self.act.append(nn.Tanh())
        
        if activation == 'sine':
            self._init_siren()

    def _init_siren(self):
        with torch.no_grad():
            for layer in self.layers:
                num_input = layer.weight.size(-1)
                avg_omega = self.omega_val.mean().item()
                layer.weight.uniform_(-np.sqrt(6 / num_input) / avg_omega, 
                                    np.sqrt(6 / num_input) / avg_omega)
                layer.bias.uniform_(-1e-6, 1e-6)

    def forward(self, x):
        # Apply Fourier Mapping
        if self.use_fourier_features:
            x_proj = 2 * np.pi * x @ self.B
            x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            
        # Forward through layers with Skip Connections
        h = self.act[0](self.layers[0](x))
        for i in range(1, len(self.layers)):
            # Residual connection: H_new = Act(Linear(H)) + H
            h = self.act[i](self.layers[i](h)) + h
            
        out = self.output_layer(h)
        
        # Apply strict output transformation if defined
        if self.output_transform == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.output_transform == 'tanh':
            out = torch.tanh(out)
            
        return out

    @property
    def device(self):
        return next(self.parameters()).device
