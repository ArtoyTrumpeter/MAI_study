import numpy as np
from .layers import CrossEntropyLoss

class NeuralNetwork:
    def __init__(self):
        self._epoch = 5
        self._learning_rate = 0.1
        self._batch_size = 5
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def configure(self, *, epoch=5, learning_rate=0.1, batch_size=5):
        if epoch < 1:
            raise ValueError("Epoch cannot be lower than 1")
        if batch_size < 1:
            raise ValueError("Batch size cannot be lower than 1")
        self._epoch         = epoch
        self._learning_rate = learning_rate
        self._batch_size    = batch_size

    def train(self, train_data, train_targets):
        esteemer = CrossEntropyLoss()

        for _ in range(self._epoch):
            for i in range(0, len(train_data), self._batch_size):
                frame_end = i + self._batch_size
                data = train_data[i:frame_end]
                targets = train_targets[i:frame_end]

                y_probs = self._forward_calc(data)
                loss = esteemer.calc_forward(y_probs, targets)

                self._backward_calc(loss, esteemer)
                self._update_all()

    def predict(self, input_data):
        return self._forward_calc(input_data)

    def _forward_calc(self, train_data):
        current_data = train_data
        
        for l in self._layers:
            current_data = l.calc_forward(current_data)

        return current_data

    def _backward_calc(self, loss, esteemer):
        curr_derivative = esteemer.calc_backward(loss)
        
        for l in self._layers[::-1]:
            curr_derivative = l.calc_backward(curr_derivative)

    def _update_all(self):
        for l in self._layers:
            l.update(self._learning_rate)
