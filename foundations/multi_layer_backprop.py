import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        def _relu(z: NDArray[np.float64]):
            return np.maximum(0, z)

        def _relu_deriv(z: NDArray[np.float64]):
            return (z > 0).astype(float)

        # convert to numpy format
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)


        # forward pass
        z1 = x @ W1.T + b1
        a1 = _relu(z1) # activation
        z2 = a1 @ W2.T + b2
        loss = np.mean((z2 - y_true) ** 2)

        # back propagation
        n = len(y_true) if y_true.ndim > 0 else 1
        dz2 = 2 * (z2 - y_true) / n # dL/dz2
        dw2 = np.outer(dz2, a1) # dL/dw2 -> dz2.reshape(-1, 1) @ a1.reshape(1, -1)
        db2 = dz2

        da1 = dz2 @ W2          # dL/da1 -> (n_out,) @ (n_out, n_hidden) -> (n_hidden,)
        da1 = da1.flatten()
        dz1 = da1 * _relu_deriv(z1)
        dw1 = np.outer(dz1, x) # dL/dw1
        db1 = dz1

        return {
            'loss': np.round(loss, 4),
            'dW1': np.round(dw1, 4),
            'db1': np.round(db1, 4),
            'dW2': np.round(dw2, 4),
            'db2': np.round(db2, 4)
        }

