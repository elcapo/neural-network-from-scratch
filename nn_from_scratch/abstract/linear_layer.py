from enum import Enum
import numpy as np
from nn_from_scratch.abstract.abstract_layer import AbstractLayer

class ActivationType(Enum):
    RELU = 1
    SOFTMAX = 2

class LinearLayer(AbstractLayer):
    def __init__(self, neurons: int, prev_neurons: int, activation: ActivationType = ActivationType.RELU):
        self.neurons = neurons
        self.prev_neurons = prev_neurons
        self.activation = activation
        self.initialize()

    def initialize(self):
        self.W = np.random.rand(self.neurons, self.prev_neurons) * 2 - 1
        self.b = np.random.rand(self.neurons, 1) * 2 - 1
        self.clear_cache()

    def clear_cache(self):
        self.Z = None
        self.A = None

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        assert isinstance(X, np.ndarray), "X must be a Numpy array"
        assert X.shape[0] == self.prev_neurons

        Z = self.W.dot(X) + self.b

        if training:
            self.Z = Z

        A = self.activate(Z)

        if training:
            self.A = A

        return A

    def activate(self, Z: np.ndarray) -> np.ndarray:
        assert isinstance(Z, np.ndarray), "Z must be a Numpy array"
        assert Z.shape[0] == self.neurons

        if (self.activation == ActivationType.RELU):
            return np.maximum(0, Z)
        
        if (self.activation == ActivationType.SOFTMAX):
            # Subtract max for stability
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        return Z