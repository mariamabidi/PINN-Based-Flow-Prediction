import torch
import torch.nn as nn
import numpy as np
import pyvista as pv

# Model class
class PINN(nn.Module):
    def init(self):
        super(PINN, self).init()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

# Load trained model and normalization
model = PINN()
model.load_state_dict(torch.load("model_epoch_600.pt", map_location=torch.device("cpu")))
model.eval()

norm = torch.load("normalization.pt")
X_mean = norm["mean"]
X_std = norm["std"]

# prediction function
def predict_velocity(x, y, z, p):
    input_tensor = torch.tensor([[x, y, z, p]], dtype=torch.float32)
    input_tensor = (input_tensor - X_mean) / X_std
    with torch.no_grad():
        velocity = model(input_tensor)
    return velocity.numpy().flatten()

# Load VTK and data preparation
input_vtk = r"data/raw/E_S_WWC_WM_005.vtk"
output_vtk = "updated.vtk"

print(f"Loading VTK file: {input_vtk}")
mesh = pv.read(input_vtk)
points = mesh.points
n_points = mesh.n_points
print(f"Loaded {n_points} points")

if "p" not in mesh.point_data:
    raise ValueError("Pressure field 'P' not found.")
pressures = mesh.point_data["p"]

# Run predictions
print("Predicting velocity at each point...")
predicted_U = np.array([
    predict_velocity(x, y, z, p)
    for (x, y, z), p in zip(points, pressures)
])
mesh.point_data["U"] = predicted_U
print("Velocity prediction complete")

mesh.save(output_vtk)
print(f" Saved updated VTK file: {output_vtk}")

mesh.plot(
    scalars="U",
    cmap="viridis",
    show_scalar_bar=True,
    lighting=True,
    notebook=False
)