# File 1: extract_features.py
import pyvista as pv
import numpy as np
import pandas as pd

# Load VTK mesh with PINN-predicted flow
mesh = pv.read("../data/processed/updated.vtk")

# Extract features
features = {}
features["vel_mag"] = np.linalg.norm(mesh["U"], axis=1)

if "p" in mesh.array_names:
    features["p"] = mesh["p"]
for shear in ["wallShearStress_x", "wallShearStress_y", "wallShearStress_z"]:
    if shear in mesh.array_names:
        features[shear] = mesh[shear]

# Create feature matrix, keep valid rows
df_all = pd.DataFrame(features)
valid_mask = ~df_all.isna().any(axis=1)
df_valid = df_all[valid_mask]
X = df_valid.values.astype(np.float32)
valid_indices = np.where(valid_mask)[0]

# Outputs
np.save("flow_features.npy", X)
np.save("valid_point_indices.npy", valid_indices)
print("✅ Features and indices saved.")