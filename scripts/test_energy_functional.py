import torch
import torch.nn as nn
from src.pinn.physics import energy_loss, poisson_loss

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.0]))
    def forward(self, coords):
        # satisfy BC u=0 on x=+-1, y=+-1
        return self.w * (1 - coords[:, 0:1]**2) * (1 - coords[:, 1:2]**2)

# Minimize energy loss
x = torch.linspace(-1, 1, 50)
y = torch.linspace(-1, 1, 50)
X, Y = torch.meshgrid(x, y, indexing='ij')
x_flat = X.flatten()
y_flat = Y.flatten()

model1 = Model()
opt1 = torch.optim.Adam(model1.parameters(), lr=0.1)
f_fn = lambda x, y: torch.ones_like(x)

for _ in range(500):
    opt1.zero_grad()
    # area is 4
    loss = energy_loss(model1, x_flat, y_flat, area=4.0, f_fn=f_fn)
    loss.backward()
    opt1.step()

print("Energy min w:", model1.w.item())

# Minimize poisson loss
model2 = Model()
opt2 = torch.optim.Adam(model2.parameters(), lr=0.1)
for _ in range(500):
    opt2.zero_grad()
    loss = poisson_loss(model2, x_flat, y_flat, f_fn=f_fn)
    loss.backward()
    opt2.step()
    
print("Poisson min w:", model2.w.item())
