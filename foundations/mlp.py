import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        x = np.asarray(x, dtype=float)
        # weights = np.asarray(weights, dtype=float)
        # biases = np.asarray(biases, dtype=float)

        def relu(x):
            return np.maximum(0, x)

        def linear_forward(x, W, b):
            return x @ W + b

        if x.ndim == 1:
            x = x[None, :]

        curr_in = x
        for w_id, weight in enumerate(weights):
            z = linear_forward(curr_in, np.asarray(weight), np.asarray(biases[w_id]))
            if w_id < len(weights) - 1:
                a = relu(z)
            else:
                a = z
            
            curr_in = a
            
        
        return np.round(a, 5)[0]