import numpy as np
from typing import List
import torch


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        
        x = torch.tensor(x, dtype=torch.float64)
        gamma = torch.tensor(gamma, dtype=torch.float64)

        rms_x = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)

        x_hat = x / rms_x

        y = gamma * x_hat

        return y.round(decimals=4).tolist()