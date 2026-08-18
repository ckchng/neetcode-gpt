
class Solution:
    def linear_forward(self, x, weight, bias):
        """
        x:      (batch, in_features)
        weight: (out_features, in_features)
        bias:   (out_features,)
        """
        cache = x, weight
        return np.round(x @ weight.T + bias, 4), cache


    def linear_backward(self, doutput, cache):
        """
        doutput: (batch, out_features)
        """
        x, weight = cache
        dl_dw = doutput.T @ x
        dl_dx = doutput @ weight
        dl_db = np.sum(doutput, axis=0)
        return np.round(dl_dw, 4), np.round(dl_dx, 4), np.round(dl_db, 4)


    def relu_forward(self, x):
        return np.maximum(0, x)


    def relu_backward(self, doutput, x):
        return doutput * (x > 0)


    def mse_forward_backward(self, prediction, target):
        difference = prediction -  target

        loss = np.mean(difference**2)
        dl_dpred = (2 * difference) / difference.size
        return np.round(loss, 4), np.round(dl_dpred, 4)
    

    def forward_and_backward(self, x, W1, b1, W2, b2, y_true):
        # make them array
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        if x.ndim == 1:
            x = x[None, :]

        if y_true.ndim == 1:
            y_true = y_true[None, :]



        # forward pass two times
        z1, layer1_cache = self.linear_forward(x, W1, b1)
        a1 = self.relu_forward(z1)

        y_hat, layer2_cache = self.linear_forward(a1, W2, b2)

        # mse loss
        loss, dl_dy = self.mse_forward_backward(y_hat, y_true)

        # compute backward
        dl_dW2, dl_da1, dl_db2  = self.linear_backward(dl_dy, layer2_cache)
        dl_dz1 = self.relu_backward(dl_da1, z1)

        dl_dW1, _, dl_db1 = self.linear_backward(dl_dz1, layer1_cache)

        return {'loss': loss,
                'dW1': dl_dW1,
                'db1': dl_db1,
                'dW2': dl_dW2,
                'db2': dl_db2}