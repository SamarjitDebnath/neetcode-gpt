import numpy as np
from typing import Tuple, List
import torch


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        x = torch.tensor(x, dtype=torch.float64)
        gamma = torch.tensor(gamma, dtype=torch.float64)
        beta = torch.tensor(beta, dtype=torch.float64)
        running_mean = torch.tensor(running_mean, dtype=torch.float64)
        running_var = torch.tensor(running_var, dtype=torch.float64)
        
        if training:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, correction=0)
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + eps)
            running_mean = ((1 - momentum) * running_mean) + (momentum * batch_mean)
            running_var = ((1 - momentum) * running_var) + (momentum * batch_var)
        else:
            x_hat = (x - running_mean) / torch.sqrt(running_var + eps)
        
        y = gamma * x_hat + beta

        return (
            y.round(decimals=4).tolist(),
            running_mean.round(decimals=4).tolist(),
            running_var.round(decimals=4).tolist()
        )
