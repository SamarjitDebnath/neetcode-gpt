import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        def _sigmoid(z: NDArray[np.float64]):
            sig = np.where(
                z >= 0,
                (1 / (1 + np.exp(-z))),
                (np.exp(z) / (1 + np.exp(z)))
            )

            return sig

        def _relu(z: NDArray[np.float64]):
            return np.where(z > 0, z, 0)

        z = np.dot(x, w) + b

        if activation == "sigmoid":
            y = _sigmoid(z)
            return np.round(y, 5)

        if activation == "relu":
            y = _relu(z)
            return np.round(y, 5)
