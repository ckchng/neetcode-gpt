
class Solution:
    def linear_forward(self, x, weight, bias):
        """
        x:      (batch, in_features)
        weight: (out_features, in_features)
        bias:   (out_features,)
        """
        output = x @ weight.T + bias
        cache = (x, weight)
        return output, cache


    def linear_backward(self, doutput, cache):
        """
        doutput: (batch, out_features)
        """
        x, weight = cache

        dx = doutput @ weight
        dweight = doutput.T @ x
        dbias = np.sum(doutput, axis=0)

        return dx, dweight, dbias


    def relu_forward(self, x):
        output = np.maximum(0, x)
        return output, x


    def relu_backward(self, doutput, x):
        return doutput * (x > 0)


    def mse_forward_backward(self, prediction, target):
        difference = prediction - target

        loss = np.mean(difference**2)
        dprediction = 2 * difference / difference.size

        return loss, dprediction
    

    def forward_and_backward(self, x, W1, b1, W2, b2, y_true):
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        # Add a batch dimension for a single example.
        if x.ndim == 1:
            x = x[None, :]

        if y_true.ndim == 1:
            y_true = y_true[None, :]

        # Forward pass
        z1, linear1_cache = self.linear_forward(x, W1, b1)
        a1, relu_cache = self.relu_forward(z1)
        y_hat, linear2_cache = self.linear_forward(a1, W2, b2)

        loss, dy_hat = self.mse_forward_backward(y_hat, y_true)

        # Backward pass
        da1, dW2, db2 = self.linear_backward(dy_hat, linear2_cache)
        dz1 = self.relu_backward(da1, relu_cache)
        _, dW1, db1 = self.linear_backward(dz1, linear1_cache)

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }