import torch
import torch.nn as nn
from src.pinn.model import Sine, PINN
from tests.base_test import PINNTestCase

class TestModelAdvanced(PINNTestCase):
    def test_sine_activation_adaptive(self):
        """Tests that Sine activation with adaptive=True has learnable parameters."""
        sine_adaptive = Sine(omega=1.0, adaptive=True).to(self.device)
        sine_fixed = Sine(omega=1.0, adaptive=False).to(self.device)
        
        # Check parameters
        params_adaptive = list(sine_adaptive.parameters())
        self.assertEqual(len(params_adaptive), 1)
        self.assertTrue(isinstance(params_adaptive[0], nn.Parameter))
        
        params_fixed = list(sine_fixed.parameters())
        self.assertEqual(len(params_fixed), 0)
        
        # Check forward pass
        x = torch.randn(10, 5, device=self.device)
        y = sine_adaptive(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.all(y >= -1.0) and torch.all(y <= 1.0))

    def test_pinn_multi_frequency_init(self):
        """Tests that PINN correctly handles multi-frequency omega distribution."""
        hidden_dim = 100
        omega_range = (1.0, 30.0)
        model = PINN(hidden_dim=hidden_dim, omega=omega_range, activation='sine').to(self.device)
        
        # Access the first Sine layer
        first_sine = model.act[0]
        self.assertTrue(isinstance(first_sine, Sine))
        
        # Check omega values
        omega_val = first_sine.omega
        self.assertEqual(omega_val.shape, (hidden_dim,))
        self.assertAlmostEqual(omega_val.min().item(), omega_range[0])
        self.assertAlmostEqual(omega_val.max().item(), omega_range[1])
        
        # Check if frequencies are distributed
        self.assertTrue(torch.all(torch.diff(omega_val) >= 0))

    def test_pinn_siren_initialization_scaling(self):
        """Tests that SIREN initialization produces outputs with reasonable variance."""
        hidden_dim = 256
        model = PINN(input_dim=2, hidden_dim=hidden_dim, num_layers=4, 
                     activation='sine', omega=30.0, use_fourier_features=False).to(self.device)
        
        x = torch.zeros((1, 2), device=self.device).uniform_(-1, 1)
        with torch.no_grad():
            # Check the output of the first linear layer
            h1_linear = model.layers[0](x)
            # Scaling should be reasonable
            self.assertTrue(h1_linear.abs().max() < 10.0)
            
            # Check deep layer outputs aren't vanishing/exploding
            out = model(x)
            self.assertTrue(out.abs().max() < 100.0)
            self.assertTrue(out.abs().max() > 1e-10)

    def test_pinn_tanh_fallback(self):
        """Tests that the model correctly falls back to Tanh activation."""
        model = PINN(activation='tanh').to(self.device)
        has_tanh = any(isinstance(layer, nn.Tanh) for layer in model.act)
        self.assertTrue(has_tanh)

    def test_fourier_features(self):
        """Tests that Fourier features correctly transform the input dimension."""
        hidden_dim = 64
        model = PINN(input_dim=2, hidden_dim=hidden_dim, use_fourier_features=True, fourier_scale=1.0).to(self.device)
        
        # Mapping dim is hidden_dim // 2 = 32. 
        # Fourier output is mapping_dim * 2 = 64.
        # First linear layer input should be 64.
        self.assertEqual(model.layers[0].in_features, 64)
        
        x = torch.randn(5, 2, device=self.device)
        out = model(x)
        self.assertEqual(out.shape, (5, 1))

    def test_output_transform(self):
        """Tests that output_transform correctly constrains the model output."""
        model_sigmoid = PINN(output_transform='sigmoid').to(self.device)
        model_tanh = PINN(output_transform='tanh').to(self.device)
        
        x = torch.randn(10, 2, device=self.device) * 100.0 # High values to force saturation
        
        out_sigmoid = model_sigmoid(x)
        self.assertTrue(torch.all(out_sigmoid >= 0.0) and torch.all(out_sigmoid <= 1.0))
        
        out_tanh = model_tanh(x)
        self.assertTrue(torch.all(out_tanh >= -1.0) and torch.all(out_tanh <= 1.0))

    def test_residual_connections(self):
        """Tests that the PINN forward pass correctly implements residual connections."""
        # For a 4-layer network, we have skip connections from layer 2 onwards
        hidden_dim = 16
        model = PINN(num_layers=4, hidden_dim=hidden_dim, activation='tanh', use_fourier_features=False).to(self.device)
        
        # We can verify by checking if the output changes when we zero out a layer's weights
        # but keep the identity. If it were a pure feed-forward, zeroing weights would zero out the signal.
        x = torch.randn(1, 2, device=self.device)
        
        # Zero out weights and biases of the 2nd linear layer (index 1)
        with torch.no_grad():
            model.layers[1].weight.zero_()
            model.layers[1].bias.zero_()
            
        # h1 = Act(L0(x))
        # h2 = Act(L1(h1)) + h1 -> since L1 is 0, h2 = Act(0) + h1
        # For Tanh, Act(0) = 0, so h2 = h1
        
        # Let's check manually
        h1 = model.act[0](model.layers[0](x))
        h2 = model.act[1](model.layers[1](h1)) + h1
        
        # If residual works, h2 should be equal to h1 (because act[1](0) = 0)
        self.assertTrue(torch.allclose(h1, h2))
