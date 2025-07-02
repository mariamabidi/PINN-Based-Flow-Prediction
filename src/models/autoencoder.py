# File 2: train_cluster.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

# Load features
X = np.load("../feature_extraction/flow_features.npy")
X_tensor = torch.tensor(X)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Train model
model = Autoencoder(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(200):
    total_loss = 0
    for batch in loader:
        x = batch[0]
        optimizer.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss = {total_loss:.4f}")

# Extract latent space
with torch.no_grad():
    latent = model.encoder(X_tensor).numpy()

np.save("../feature_extraction/latent.npy", latent)

# Cluster in latent space
labels = KMeans(n_clusters=5).fit_predict(latent)
np.save("../feature_extraction/latent_clusters.npy", labels)
print("✅ Clustering complete and labels saved.")