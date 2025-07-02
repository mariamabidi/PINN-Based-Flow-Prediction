import pyvista as pv
import numpy as np
from collections import Counter

# Load mesh and labels
mesh = pv.read("/data/processed/updated.vtk")
labels = np.load("../feature_extraction/latent_clusters.npy")
indices = np.load("../feature_extraction/valid_point_indices.npy")

# Trim to only valid indices (if needed)
valid = indices < mesh.n_points
indices = indices[valid]
labels = labels[valid]

# Build full label array
full_labels = np.full(mesh.n_points, -1)
full_labels[indices] = labels

# Count cluster sizes and get top 3
label_counts = Counter(labels)
top3 = [label for label, _ in label_counts.most_common(3)]
print("Top 3 clusters:", top3)

# Filter: keep only top 3 clusters, set others to -1
filtered_labels = np.where(np.isin(full_labels, top3), full_labels, -1)

# Plot
mesh["top3_clusters"] = filtered_labels
print("Visualizing")
mesh.plot(scalars="top3_clusters", cmap="Set2", nan_color="gray")
