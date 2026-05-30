import torch
import torch.nn as nn
from src.pinn.model import PINN
from tests.base_test import PINNTestCase

class TestMultiScaleFourier(PINNTestCase):
    def test_multiscale_initialization(self):
        """Verifies that multi-scale B_list is created with expected scales."""
        hidden_dim = 64
        model = PINN(hidden_dim=hidden_dim, use_fourier_features=True)
        
        self.assertTrue(hasattr(model, 'B_list'))
        self.assertEqual(len(model.B_list), 4) # Default scales: [1, 5, 10, 50]
        
        # Check that scales are different
        scales = [torch.std(B).item() for B in model.B_list]
        self.assertAlmostEqual(scales[0], 1.0, delta=0.5)
        self.assertAlmostEqual(scales[3], 50.0, delta=10.0)

    def test_mapping_dimension_logic(self):
        """Verifies that current_input_dim is calculated correctly based on hidden_dim."""
        # 64 hidden dim / (2 * 4 scales) = 8 dims per scale
        # 8 dims * 2 (sin/cos) * 4 scales = 64 total input dim
        model = PINN(hidden_dim=64, use_fourier_features=True)
        first_layer = model.layers[0]
        self.assertEqual(first_layer.in_features, 64)

    def test_forward_pass_multiscale(self):
        """Verifies forward pass with multi-scale features produces correct shapes."""
        model = PINN(hidden_dim=32, use_fourier_features=True)
        x = torch.randn(10, 2)
        out = model(x)
        self.assertEqual(out.shape, (10, 1))

    def test_fourier_features_disabled(self):
        """Ensures model still works without Fourier features."""
        model = PINN(hidden_dim=32, use_fourier_features=False)
        self.assertFalse(hasattr(model, 'B_list'))
        first_layer = model.layers[0]
        self.assertEqual(first_layer.in_features, 2)
