import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, emb_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MixtureHead(nn.Module):
    def __init__(self, emb_dim: int, num_components: int = 3):
        super().__init__()
        self.num_components = num_components
        self.output_layer = nn.Linear(emb_dim, num_components * 3)

    def forward(self, context: torch.Tensor):
        out = self.output_layer(context)  # shape (B, 3K)
        means, log_stds, logits = out.chunk(3, dim=-1)
        stds = log_stds.exp()
        weights = F.softmax(logits, dim=-1)
        return means, stds, weights


class ConditionalMixtureModel(nn.Module):
    def __init__(self, context_dim: int, emb_dim: int = 32, num_components: int = 3):
        super().__init__()
        self.encoder = ContextEncoder(context_dim, emb_dim)
        self.head = MixtureHead(emb_dim, num_components)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        context = self.encoder(x)
        means, stds, weights = self.head(context)  # (B, K)
        y = y.unsqueeze(-1)  # (B, 1)
        comp_log_probs = -0.5 * (((y - means) / stds) ** 2 + 2 * torch.log(stds) + torch.log(torch.tensor(2 * torch.pi)))
        total_log_prob = torch.log((weights * comp_log_probs.exp()).sum(dim=-1) + 1e-9)  # (B,)
        return total_log_prob

    def sample(self, x: torch.Tensor, n_samples: int = 100):
        context = self.encoder(x)  # (B, emb_dim)
        means, stds, weights = self.head(context)  # (B, K)
        B, K = means.shape

        mix = Categorical(weights)  # (B,)
        indices = mix.sample((n_samples,))  # (n_samples, B)

        normal = Normal(means.unsqueeze(0), stds.unsqueeze(0))  # (1, B, K)
        samples = normal.sample()  # (n_samples, B, K)

        idx_expanded = indices.unsqueeze(-1)  # (n_samples, B, 1)
        selected_samples = samples.gather(2, idx_expanded).squeeze(-1)  # (n_samples, B)
        return selected_samples.transpose(0, 1)  # (B, n_samples)


# Example usage (plug into your training pipeline)
if __name__ == "__main__":
    x = torch.randn(1024, 6)  # 6 context features
    y = torch.rand(1024, 1)   # scalar target in [0, 1]

    model = ConditionalMixtureModel(context_dim=6, emb_dim=32, num_components=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        log_probs = model(x, y)
        loss = -log_probs.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Sampling example
    model.eval()
    with torch.no_grad():
        samples = model.sample(x[:4], n_samples=100)  # (4 setups, 100 samples each)
        print("Samples shape:", samples.shape)  # (4, 100)
