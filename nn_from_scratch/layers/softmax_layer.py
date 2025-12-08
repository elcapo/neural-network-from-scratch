import numpy as np
from nn_from_scratch.layers.abstract_layer import AbstractLayer

class SoftmaxLayer(AbstractLayer):
    def __init__(self, prev_neurons: int):
        self.prev_neurons = prev_neurons

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        assert isinstance(X, np.ndarray), "X must be a Numpy array"
        assert X.shape[0] == self.prev_neurons

        return np.exp(X) / np.sum(np.exp(X), axis=0, keepdims=True)
