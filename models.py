


import torch
import torch.nn as nn

# ─── 1) Context encoder ────────────────────────────────────────────────────────
class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 6) raw physical parameters (grid size, etc.)
        return self.net(x)  # → (batch, emb_dim)


