# File 4: analyze_clusters.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.utils import resample

X = np.load("flow_features.npy")
labels = np.load("latent_clusters.npy")
latent = np.load("latent.npy")

# Plot cluster sizes
counts = Counter(labels)
plt.figure(figsize=(8, 4))
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.xlabel("Cluster ID")
plt.ylabel("Number of Points")
plt.title("Cluster Size Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature averages per cluster
df = pd.DataFrame(X, columns=["vel_mag", "p", "shear_x", "shear_y", "shear_z"][:X.shape[1]])
df["cluster"] = labels
cluster_means = df.groupby("cluster").mean()

cluster_means.plot(kind="bar", figsize=(10, 6), colormap="viridis")
plt.title("Average Feature Values Per Cluster")
plt.ylabel("Feature Value")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# PCA of latent space
pca = PCA(n_components=2)
X_pca = pca.fit_transform(latent)

plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set2", s=5)
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.title("Latent Space PCA Projection by Cluster")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()