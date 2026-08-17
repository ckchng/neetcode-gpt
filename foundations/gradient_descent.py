class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        curr_x = init
        curr_d = 2 * curr_x
        i = 0
        while i < iterations:
            curr_x = curr_x - learning_rate * curr_d
            curr_d = 2 * curr_x
            i +=1 


        return round(curr_x, 5)
