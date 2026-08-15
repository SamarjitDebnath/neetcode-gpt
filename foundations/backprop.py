import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        def _sigmoid(z: NDArray[np.float64]):
            return np.where(
                z >= 0,
                (1 / (1 + np.exp(-z))),
                np.exp(z) / (1 + np.exp(z))
            )

        # forward step
        z = np.dot(x, w) + b
        y_hat = _sigmoid(z)

        # backward step
        error = y_hat - y_true
        sig_deriv = y_hat * (1 - y_hat)
        delta = error * sig_deriv

        dl_dw = np.round(np.dot(delta, x), 5)
        dl_db = np.round(delta, 5)

        return (dl_dw, dl_db)





