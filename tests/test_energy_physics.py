import torch
from src.pinn.physics import energy_loss
from src.pinn.model import PINN
from src.core.geometry import PolygonDomain
from tests.base_test import PINNTestCase

class TestEnergyFormulation(PINNTestCase):
    def test_energy_loss_const(self):
        """Verifies energy loss is zero for a constant function (grad=0)."""
        model = PINN(hidden_dim=16)
        # Force weights to zero and bias to 5.0
        for p in model.parameters():
            p.data.zero_()
        model.output_layer.bias.data.fill_(5.0)
        
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        
        loss = energy_loss(model, x, y)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_energy_loss_linear(self):
        """Verifies energy loss for u = x => grad = (1, 0) => |grad|^2 = 1."""
        model = PINN(hidden_dim=16, num_layers=1, use_fourier_features=False)
        
        # Construct a simple u = x network
        # Linear layer: y = xA^T + b
        # To get y = x, we need A = [[1, 0], [0, 0], ...] and b = 0
        model.layers[0].weight.data.zero_()
        model.layers[0].weight.data[0, 0] = 1.0 # map x to first hidden unit
        model.layers[0].bias.data.zero_()
        
        model.output_layer.weight.data.zero_()
        model.output_layer.weight.data[0, 0] = 1.0 # map first hidden unit to output
        model.output_layer.bias.data.zero_()
        
        # Identity activation
        model.act[0] = torch.nn.Identity()

        x = torch.linspace(0, 1, 100, requires_grad=True)
        y = torch.linspace(0, 1, 100, requires_grad=True)
        
        # Energy = 0.5 * integral(|grad u|^2) dOmega
        # For u=x, |grad u|^2 = 1.
        # Integral(1) dOmega = Area.
        # Energy = 0.5 * Area.
        area = 2.0
        loss = energy_loss(model, x, y, area=area)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_area_calculation(self):
        """Verifies shoelace area calculation for square and hole."""
        # Square area = 1.0
        outer = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        domain = PolygonDomain(outer)
        self.assertAlmostEqual(domain.area, 1.0)
        
        # Square with hole: 1.0 - 0.25 = 0.75
        hole = torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
        domain_with_hole = PolygonDomain(outer, holes=[hole])
        self.assertAlmostEqual(domain_with_hole.area, 0.75)
