import numpy as np
from nn_from_scratch.layers.abstract_layer import AbstractLayer
from nn_from_scratch.encoders.one_hot_encoder import one_hot_encode


class SoftmaxLayer(AbstractLayer):
    def __init__(self, prev_neurons: int):
        self.prev_neurons = prev_neurons

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        assert isinstance(X, np.ndarray), "X must be a Numpy array"
        assert X.shape[0] == self.prev_neurons

        if training:
            self.Y_pred = X

        return np.exp(X) / np.sum(np.exp(X), axis=0, keepdims=True)

    def backward(self, prev: np.ndarray, Y: np.ndarray, dL_prev: np.ndarray) -> np.ndarray:
        return self.Y_pred - one_hot_encode(Y, self.prev_neurons)
