import torch
from src.pinn.physics import energy_loss, poisson_loss

def f_fn(x, y):
    return torch.ones_like(x)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))
        
    def forward(self, coords):
        return self.w * (coords[:, 0:1]**2 + coords[:, 1:2]**2)

model = Model()
x = torch.tensor([0.5, 0.5])
y = torch.tensor([0.5, 0.5])
print("u", model(torch.stack([x,y], dim=1)).squeeze())
print("u_x", 2*model.w*x)
print("u_y", 2*model.w*y)
print("u_xx+u_yy", 4*model.w)

loss = energy_loss(model, x, y, area=1.0, f_fn=f_fn)
print("Energy loss:", loss.item())

# 0.5 * (u_x**2 + u_y**2) + f * u
# 0.5 * ((2*w*x)**2 + (2*w*y)**2) + 1 * w * (x^2 + y^2)
# 0.5 * (4*w^2*x^2 + 4*w^2*y^2) + w * (x^2 + y^2)
# 2*w^2*(x^2 + y^2) + w * (x^2 + y^2)
# at x=0.5, y=0.5: x^2+y^2 = 0.5
# 2*1*0.5 + 1*0.5 = 1.0 + 0.5 = 1.5
print("Expected energy loss:", 1.5)

loss = poisson_loss(model, x, y, f_fn=f_fn)
print("Poisson loss:", loss.item())
print("Expected Poisson loss:", (4 - 1)**2)

