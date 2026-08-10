class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        for _ in range(iterations):
            derivative = 2 * x
            x -= learning_rate * derivative

        return round(x, 5)