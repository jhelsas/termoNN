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
    Advanced PINN with Multi-Scale Fourier Feature Mapping and Residual Skip Connections.
    Designed for high-accuracy field reconstruction on fractal domains with sharp corners.
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_layers=4, 
                 activation='sine', omega=30.0, adaptive_activations=False,
                 use_fourier_features=True, fourier_scale=10.0,
                 output_transform=None):
        super(PINN, self).__init__()
        self.activation = activation
        self.use_fourier_features = use_fourier_features
        self.output_transform = output_transform
        
        # 1. Multi-Scale Fourier Feature Mapping (Input Embedding)
        # We use multiple frequency banks to capture both global field trends 
        # and local high-curvature fractal details.
        if use_fourier_features:
            # scales: [1.0, 5.0, 10.0, 50.0]
            scales = [1.0, 5.0, 10.0, 50.0]
            
            # Ensure mapping_dim is at least 1 per scale
            mapping_dim_per_scale = max(1, hidden_dim // (2 * len(scales)))
            
            self.B_list = nn.ParameterList()
            for s in scales:
                B = torch.randn(input_dim, mapping_dim_per_scale) * s
                self.B_list.append(nn.Parameter(B, requires_grad=False))
                
            current_input_dim = mapping_dim_per_scale * 2 * len(scales)
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
        # We use a slight learnable scaling for the residual to prevent 
        # high-frequency destabilization in SIREN.
        self.res_scales = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers - 1)])
        
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
        
        # Pre-cache transform modes to avoid runtime string comparisons
        self._transform_mode = 0
        if output_transform == 'sigmoid':
            self._transform_mode = 1
        elif output_transform == 'tanh':
            self._transform_mode = 2

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
            # Also initialize the output layer with SIREN-appropriate scaling
            # to prevent activation scaling collapse at the final projection
            num_input_out = self.output_layer.weight.size(-1)
            self.output_layer.weight.uniform_(-np.sqrt(6 / num_input_out) / avg_omega,
                                              np.sqrt(6 / num_input_out) / avg_omega)
            self.output_layer.bias.uniform_(-1e-6, 1e-6)

    def forward(self, x):
        # Apply Multi-Scale Fourier Mapping
        if self.use_fourier_features:
            features = []
            for B in self.B_list:
                x_proj = 2 * torch.pi * x @ B
                features.append(torch.sin(x_proj))
                features.append(torch.cos(x_proj))
            x = torch.cat(features, dim=-1)
            
        # Forward through layers with Skip Connections
        h = self.act[0](self.layers[0](x))
        for i in range(1, len(self.layers)):
            # Residual connection with learnable scaling on the residual branch
            # H_new = H_old + scale * Act(Linear(H_old))
            h = h + self.res_scales[i-1] * self.act[i](self.layers[i](h))
            
        out = self.output_layer(h)
        
        # Apply strict output transformation if defined using cached modes
        if self._transform_mode == 1:
            out = torch.sigmoid(out)
        elif self._transform_mode == 2:
            out = torch.tanh(out)
            
        return out

    @property
    def device(self):
        return next(self.parameters()).device

class ExactBoundaryAnsatz(nn.Module):
    """
    Wraps a core network to strictly enforce boundary conditions using an Ansatz.
    u(x) = G(x) + D(x) * N_theta(x)

    Currently only supports bc_mode='nested', which assumes exactly two boundaries:
        - poly_idx=0 (outer boundary), where u=0
        - poly_idx=1 (inner hole), where u=1
    This is a limitation — generalizing to arbitrary numbers of boundaries
    would require constructing G(x) as a weighted sum of multiple distance
    functions.

    Args:
        core_model: The PINN network to wrap.
        domain: A PolygonDomain with exactly one hole.
        bc_mode: Must be 'nested'.
    """
    def __init__(self, core_model: PINN, domain, bc_mode='nested'):
        super().__init__()
        self.core = core_model
        self.domain = domain
        self.bc_mode = bc_mode

    def forward(self, coords):
        x, y = coords[:, 0], coords[:, 1]
        n_out = self.core(coords)
        
        if self.bc_mode == 'nested':
            # Calculate distance to outer boundary (id 0) and inner boundary (id 1)
            # Using exact distance with epsilon for safe gradients
            d_out = self.domain.exact_distance(x, y, poly_idx=0)
            d_in = self.domain.exact_distance(x, y, poly_idx=1)
            
            # G(x) = d_out / (d_in + d_out)  => 1 at inner (d_in=0), 0 at outer (d_out=0)
            # D(x) = d_in * d_out => 0 at both boundaries
            
            # Smooth interpolation function to avoid singularities
            # We add a tiny epsilon to the denominator to prevent division by zero
            d_sum = d_in + d_out + 1e-12
            
            # Target BC: 1 at inner boundary, 0 at outer boundary
            G = (d_out / d_sum).unsqueeze(1)
            
            # Distance field: 0 at both boundaries, smooth inside
            # tanh is used to bound the distance multiplier so it doesn't 
            # magnify the network outputs too much in the deep interior
            D = torch.tanh(d_in * d_out).unsqueeze(1)
            
            return G + D * n_out
        else:
            raise NotImplementedError(f"Ansatz mode {self.bc_mode} not implemented.")

    @property
    def device(self):
        return self.core.device
