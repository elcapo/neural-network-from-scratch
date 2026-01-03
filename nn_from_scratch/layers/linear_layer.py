from enum import Enum
import numpy as np
from nn_from_scratch.layers.abstract_layer import AbstractLayer


class ActivationType(Enum):
    RELU = 1
    SOFTMAX = 2


class LinearLayer(AbstractLayer):
    def __init__(self, neurons: int, prev_neurons: int):
        self.neurons = neurons
        self.prev_neurons = prev_neurons
        self.initialize()

    def initialize(self):
        self.W = np.random.rand(self.neurons, self.prev_neurons) * 2 - 1
        self.b = np.random.rand(self.neurons, 1) * 2 - 1

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray), "X must be a Numpy array"
        assert X.shape[0] == self.prev_neurons

        return self.W.dot(X) + self.b
