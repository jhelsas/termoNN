import torch
import unittest
import random
import numpy as np
from src.core.data import get_device, set_seed

class PINNTestCase(unittest.TestCase):
    """
    Base class for all PINN tests to ensure consistent device 
    handling and reproducibility.
    """
    def setUp(self):
        self.device = get_device()
        # Use a specific seed for tests to ensure they are deterministic
        set_seed(42)
        
    def assertTensorsEqual(self, t1: torch.Tensor, t2: torch.Tensor, msg: str = None, atol: float = 1e-7, rtol: float = 1e-5):
        """Custom assertion for Tensors with shape validation and descriptive errors."""
        self.assertEqual(t1.shape, t2.shape, msg=f"Shape mismatch: {t1.shape} vs {t2.shape} {msg or ''}")
        self.assertTrue(torch.allclose(t1, t2, atol=atol, rtol=rtol), msg=f"Tensor values not close enough. {msg or ''}")

    def assertTensorFinite(self, t: torch.Tensor, msg: str = None):
        """Ensures all elements in a tensor are finite (no NaN/Inf)."""
        self.assertTrue(torch.isfinite(t).all(), msg=f"Tensor contains non-finite values (NaN/Inf). {msg or ''}")
