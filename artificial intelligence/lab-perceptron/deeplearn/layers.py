import numpy as np
from .activators import Linear, Tanh, Sigmoid

def create_activator(activator_type):
        if activator_type == "linear":
            return Linear()
        elif activator_type == "tanh":
            return Tanh()
        elif activator_type == "sigmoid":
            return Sigmoid()
        else:
            raise ValueError(f"{activator_type} does not name a type")

class Layer:
    def __init__(self, in_count, out_count, *, activator_type="linear"):
        self.activator = create_activator(activator_type)
        self.weights   = np.random.normal(0, 1.0/np.sqrt(in_count), (out_count, in_count))
        self.bias      = np.zeros((out_count, ))
        self.d_weights = np.zeros((out_count, in_count))
        self.d_bias    = np.zeros((out_count, ))

    def calc_forward(self, x):
        self.x = x
        p = np.dot(x, self.weights.T) + self.bias
        return self.activator.calc_forward(p)

    def calc_backward(self, dy):
        dz = self.activator.calc_backward(dy)
        dx = np.dot(dz, self.weights)
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis=0)

        self.d_weights = dW
        self.d_bias    = db

        return dx

    def update(self, learn_rate):
        self.weights -= learn_rate * self.d_weights
        self.bias    -= learn_rate * self.d_bias

class Softmax:
    def calc_forward(self, z):
        exp_z = np.exp(z)
        row_sums = np.sum(exp_z, axis=1)
        self.p = exp_z / row_sums[:, None]
        return np.copy(self.p)

    def calc_backward(self, dp):
        pdp = self.p * dp
        return pdp - self.p * pdp.sum(axis=1, keepdims=True)

    def update(self, learn_rate):
        # Nothing to update
        pass

class CrossEntropyLoss:
    def calc_forward(self, p, y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()

    def calc_backward(self, loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
        return dlog_softmax / self.p

    def update(self, learn_rate):
        # Nothing to update
        pass