import numpy as np
from nn_from_scratch.network import Network

class Evaluation:
    def __init__(self, network: Network):
        self.network = network

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y_pred = self.network.forward(X, training=True)

        predictions = np.argmax(Y_pred, axis=0)
        accuracy = round(np.sum(predictions == Y) / Y.size, 2)

        return accuracy