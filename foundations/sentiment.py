import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.network = nn.Sequential(
            nn.Embedding(vocabulary_size, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        embeddings = self.network[0](x)

        mean_embed = embeddings.mean(dim=1)

        y_hat = self.network[1:](mean_embed)

        return y_hat.round(decimals=4)
