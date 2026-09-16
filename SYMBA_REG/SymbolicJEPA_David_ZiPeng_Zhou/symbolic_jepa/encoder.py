"""
T-Net point-cloud encoder.

Order-invariant: shared MLP per point → global max-pool → FC projection.
"""

import torch
import torch.nn as nn


class TNet(nn.Module):
    def __init__(self, d_input: int, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * d_model),
            nn.BatchNorm1d(2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, 4 * d_model),
            nn.BatchNorm1d(4 * d_model),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch, n_points, d_input)
        Returns:
            (batch, d_model) — one embedding per point cloud.
        """
        batch, n, d = X.shape
        X_flat = X.reshape(batch * n, d)
        X_flat = self.shared_mlp(X_flat)
        X_enc = X_flat.view(batch, n, -1)
        X_pool = X_enc.max(dim=1).values
        return self.fc(X_pool)
