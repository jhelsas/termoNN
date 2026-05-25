import torch
import os
import numpy as np
from src.core.viz import plot_results
from src.core.geometry import PolygonDomain
from src.pinn.model import PINN
from tests.base_test import PINNTestCase

class TestViz(PINNTestCase):
    def test_plot_results_smoke(self):
        """Viz Validation: Ensures plotting runs and saves a file."""
        model = PINN(hidden_dim=5).to(self.device)
        filename = "test_plot.png"
        if os.path.exists(filename):
            os.remove(filename)
            
        plot_results(model, filename=filename, resolution=50)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_plot_results_masking_robustness(self):
        """Viz Validation: Ensures masking logic handles domains with holes."""
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        hole = torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
        domain = PolygonDomain(outer, holes=[hole])
        model = PINN(hidden_dim=5).to(self.device)
        
        # This checks for crashes during meshgrid generation and point-in-poly check
        try:
            plot_results(model, domain=domain, filename="test_mask.png", resolution=30)
        except Exception as e:
            self.fail(f"plot_results crashed with hole domain: {e}")
        finally:
            if os.path.exists("test_mask.png"):
                os.remove("test_mask.png")
