import numpy as np

class Linear:
    def calc_forward(self, x):
        return x

    def calc_backward(self, dy):
        return dy

class Sigmoid:
    def calc_forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def calc_backward(self, dy):
        return self.y * (1 - self.y) * dy

class Tanh:
    def calc_forward(self, x):
        y = np.tanh(x)
        self.y = y
        return y

    def calc_backward(self, dy):
        return (1 - self.y ** 2) * dy