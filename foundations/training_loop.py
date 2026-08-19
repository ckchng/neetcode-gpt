import numpy as np
from numpy.typing import NDArray
from typing import Tuple



class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        if X.ndim == 1:
            X = X[None, :]

        # w = np.zeros((1, X.shape[1]))
        # their w convention is [num_in, num_out]
        w = np.zeros(X.shape[1])
        b = 0
        for _ in range(epochs):
            # forward
            y_hat = X @ w + b
            diff = y_hat - y
            # loss = np.mean(diff**2)
            # print(loss)
            dl_dy = (2 * diff) / diff.size

            # backward
            # dl_dw = dl_dy.T @ X
            dl_dw = X.T @ dl_dy
            dl_db = np.sum(dl_dy, axis=0)
            # gradient update
            w = w - lr * dl_dw
            b = b - lr * dl_db



        return (np.round(w, 5), np.round(b, 5))
