import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load feature data
X = np.load("../feature_extraction/flow_features.npy")
X_tensor = torch.tensor(X)

# Define the same autoencoder structure
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

# Load trained weights (if available)
model = Autoencoder(input_dim=X.shape[1])
# Optional: model.load_state_dict(torch.load("autoencoder.pt"))

# Just re-run encoding
model.eval()
with torch.no_grad():
    latent = model.encoder(X_tensor).numpy()

# Save latent
np.save("../feature_extraction/latent.npy", latent)
print("✅ Latent space saved as latent.npy")
