import numpy as np
from nn_from_scratch.layers.abstract_layer import AbstractLayer

class IdentityLayer(AbstractLayer):
    def __init__(self, neurons: int):
        self.neurons = neurons

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray), "X must be a Numpy array"
        assert X.shape[0] == self.neurons

        return X
