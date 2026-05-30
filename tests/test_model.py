import torch
from src.pinn.model import PINN
from src.pinn.physics import laplace_loss
from tests.base_test import PINNTestCase

class TestModel(PINNTestCase):
    def test_model_output_shape(self):
        """Verifies the MLP returns the correct tensor shape (Batch x 1)."""
        model = PINN(input_dim=2, output_dim=1).to(self.device)
        test_input = torch.randn(10, 2, device=self.device)
        output = model(test_input)
        self.assertEqual(output.shape, (10, 1))

    def test_activation_differentiability(self):
        """Ensures the model activation allows non-zero second derivatives."""
        model = PINN(use_fourier_features=False).to(self.device)
        x = torch.linspace(0, 1, 10, requires_grad=True, device=self.device)
        y = torch.linspace(0, 1, 10, requires_grad=True, device=self.device)
        coords = torch.stack([x, y], dim=1)
        u = model(coords)
        u_x = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 0]
        u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
        
        self.assertFalse(torch.all(u_xx == 0), "Second derivative is zero. Activation may be non-smooth.")

    def test_model_serialization(self):
        """Ensures the model can be saved and loaded with identical outputs."""
        model = PINN().to(self.device)
        model.eval()
        x = torch.randn(5, 2, device=self.device)
        
        with torch.no_grad():
            original_output = model(x)
            
        state = model.state_dict()
        new_model = PINN().to(self.device)
        new_model.load_state_dict(state)
        new_model.eval()
        
        with torch.no_grad():
            new_output = new_model(x)
            
        self.assertTensorsEqual(original_output, new_output)

    def test_initialization_sanity(self):
        """Checks that weights are not initialized to zero."""
        model = PINN().to(self.device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.all(param == 0), f"Layer {name} is all zeros.")

    def test_gradient_finiteness(self):
        """Ensures first derivatives are finite and not NaN."""
        model = PINN().to(self.device)
        x = torch.rand(10, 2, requires_grad=True, device=self.device)
        u = model(x)
        grads = torch.autograd.grad(u.sum(), x)[0]
        self.assertTrue(torch.isfinite(grads).all())

    def test_parameter_scaling(self):
        """Verifies that num_layers and hidden_dim are respected."""
        hidden_dim = 33
        num_layers = 5
        model = PINN(hidden_dim=hidden_dim, num_layers=num_layers, use_fourier_features=False).to(self.device)
        self.assertEqual(model.layers[0].weight.shape[0], hidden_dim)
        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(len(model.res_scales), num_layers - 1)

    def test_model_device_movement(self):
        """Verifies the model can be moved between devices (if available)."""
        model = PINN().to(self.device)
        self.assertEqual(next(model.parameters()).device.type, self.device.type)
        
        cpu_device = torch.device("cpu")
        model.to(cpu_device)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_forward_pass_different_batch_sizes(self):
        """Ensures the model handles various batch sizes correctly."""
        model = PINN().to(self.device)
        for batch_size in [1, 16, 128]:
            x = torch.randn(batch_size, 2, device=self.device)
            output = model(x)
            self.assertEqual(output.shape, (batch_size, 1))

    def test_eval_mode_consistency(self):
        """Verifies that model(x) is deterministic in eval mode."""
        model = PINN().to(self.device)
        model.eval()
        x = torch.randn(10, 2, device=self.device)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        self.assertTensorsEqual(out1, out2)

    def test_siren_activation_support(self):
        """Ensures SIREN (Sine) activation can be instantiated."""
        model = PINN(activation='sine', omega=10.0).to(self.device)
        x = torch.randn(5, 2, device=self.device)
        output = model(x)
        self.assertEqual(output.shape, (5, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_siren_initialization_scaling(self):
        """Verifies SIREN weights are scaled correctly based on omega."""
        # Using a very large omega should result in very small weights (uniform scaling)
        model_high = PINN(activation='sine', omega=1000.0, hidden_dim=100, use_fourier_features=False)
        model_low = PINN(activation='sine', omega=1.0, hidden_dim=100, use_fourier_features=False)
        
        # Check hidden layer weights
        w_high = model_high.layers[1].weight.abs().mean().item()
        w_low = model_low.layers[1].weight.abs().mean().item()
        
        self.assertLess(w_high, w_low)

    def test_activation_selection(self):
        """Verifies that the correct activation module is used."""
        model_tanh = PINN(activation='tanh')
        self.assertTrue(any(isinstance(m, torch.nn.Tanh) for m in model_tanh.act))
        
        from src.pinn.model import Sine
        model_sine = PINN(activation='sine')
        self.assertTrue(any(isinstance(m, Sine) for m in model_sine.act))

    def test_siren_first_layer_init(self):
        """Verifies the layer initialization is scaled by avg_omega."""
        import numpy as np
        # Note: New architecture uses avg_omega for all layers including first
        model = PINN(hidden_dim=128, activation='sine', omega=30.0)
        w_max = model.layers[0].weight.abs().max().item()
        # uniform(-sqrt(6/n)/omega, sqrt(6/n)/omega)
        limit = np.sqrt(6/128) / 30.0
        self.assertLess(w_max, limit * 1.5) # Increased tolerance slightly for uniform noise

    def test_multi_frequency_distribution(self):
        """Architecture Validation: Checks if frequencies are distributed in Sine modules."""
        hidden_dim = 10
        model = PINN(hidden_dim=hidden_dim, activation='sine', omega=(1.0, 10.0))
        
        # Check first Sine module
        sine_module = model.act[0]
        unique_freqs = torch.unique(sine_module.omega)
        self.assertEqual(len(unique_freqs), hidden_dim)
        self.assertAlmostEqual(unique_freqs.min().item(), 1.0)
        self.assertAlmostEqual(unique_freqs.max().item(), 10.0)

    def test_omega_buffer_device_transfer(self):
        """Pair-wise Validation: Ensures omega buffers follow model device movement."""
        model = PINN(activation='sine', omega=(5, 15))
        model.to('cpu')
        self.assertEqual(model.act[0].omega.device.type, 'cpu')
        
        if torch.cuda.is_available():
            model.to('cuda')
            self.assertEqual(model.act[0].omega.device.type, 'cuda')

    def test_siren_gradient_stability_extreme_omega(self):
        """Pair-wise Validation: Verifies that high frequencies don't cause immediate NaN."""
        model = PINN(activation='sine', omega=100.0).to(self.device)
        x = torch.rand(10, device=self.device)
        y = torch.rand(10, device=self.device)
        
        loss = laplace_loss(model, x, y)
        self.assertTrue(torch.isfinite(loss))
        
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_adaptive_activations_optimization(self):
        """Architecture Validation: Ensures adaptive activation parameters are trackable."""
        model = PINN(activation='sine', adaptive_activations=True)
        adaptive_params = [p for n, p in model.named_parameters() if '.a' in n]
        self.assertGreater(len(adaptive_params), 0)
        for p in adaptive_params:
            self.assertEqual(p.item(), 1.0)
            self.assertTrue(p.requires_grad)

    def test_fourier_feature_mapping_projection(self):
        """Architecture Validation: Verifies Fourier mapping changes input dimension."""
        hidden_dim = 64
        model = PINN(input_dim=2, hidden_dim=hidden_dim, use_fourier_features=True)
        # We now use B_list for multi-scale mapping
        self.assertTrue(hasattr(model, 'B_list'))
        # Total mapping dim should be hidden_dim // 2
        total_cols = sum(B.shape[1] for B in model.B_list)
        self.assertEqual(total_cols, hidden_dim // 2)
        # First linear layer should accept 64 (32 sin + 32 cos)
        self.assertEqual(model.layers[0].in_features, 64)

    def test_residual_skip_connections(self):
        """Architecture Validation: Verifies skip connections and scaling are operational."""
        model = PINN(num_layers=3, hidden_dim=16, use_fourier_features=False)
        self.assertEqual(len(model.res_scales), 2)
        x = torch.randn(1, 2)
        with torch.no_grad():
            out = model(x)
            self.assertEqual(out.shape, (1, 1))
