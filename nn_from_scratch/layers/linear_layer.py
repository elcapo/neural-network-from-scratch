from enum import Enum
import numpy as np
from nn_from_scratch.layers.abstract_layer import AbstractLayer


class LinearLayer(AbstractLayer):
    def __init__(self, neurons: int, prev_neurons: int):
        self.neurons = neurons
        self.prev_neurons = prev_neurons
        self.initialize()

    def initialize(self):
        self.W = np.random.rand(self.neurons, self.prev_neurons) * 2 - 1
        self.b = np.random.rand(self.neurons, 1) * 2 - 1

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        assert isinstance(X, np.ndarray), "X must be a Numpy array"
        assert X.shape[0] == self.prev_neurons

        if training:
            self.X = X

        return self.W.dot(X) + self.b

    def backward(self, prev: np.ndarray, dL_prev: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        m = dL_prev.shape[1]

        dW = dL_prev.dot(self.X.T) / m
        db = np.sum(dL_prev, axis=1, keepdims=True) / m
        dX = self.W.T.dot(dL_prev)

        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db

        return dX
