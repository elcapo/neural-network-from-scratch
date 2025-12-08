import numpy as np
from nn_from_scratch.layers.abstract_layer import AbstractLayer

class ReluLayer(AbstractLayer):
    def __init__(self, prev_neurons: int):
        self.prev_neurons = prev_neurons

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray), "X must be a Numpy array"
        assert X.shape[0] == self.prev_neurons

        return np.maximum(0, X)
