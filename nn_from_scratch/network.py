import numpy as np
from nn_from_scratch.layers.identity_layer import IdentityLayer
from nn_from_scratch.layers.linear_layer import LinearLayer
from nn_from_scratch.layers.relu_layer import ReluLayer
from nn_from_scratch.layers.softmax_layer import SoftmaxLayer


class Network:
    def __init__(self, input_size=28 * 28, hidden_size=10, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initialize()

    def initialize(self):
        self.input_layer = IdentityLayer(self.input_size)
        self.hidden_layer = LinearLayer(self.hidden_size, self.input_size)
        self.relu_layer = ReluLayer(self.hidden_size)
        self.output_layer = LinearLayer(self.output_size, self.hidden_size)
        self.softmax_layer = SoftmaxLayer(self.output_size)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        X = self.input_layer.forward(X, training)
        X = self.hidden_layer.forward(X, training)
        X = self.relu_layer.forward(X, training)
        X = self.output_layer.forward(X, training)
        X = self.softmax_layer.forward(X, training)

        return X

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 100, learning_rate: float = 0.01):
        for i in range(iterations):
            Y_pred = self.forward(X, training=True)
            self.backward(Y, learning_rate)

            predictions = np.argmax(Y_pred, axis=0)
            accuracy = round(np.sum(predictions == Y) / Y.size, 2)

            yield (
                i + 1,
                accuracy,
            )

    def backward(self, Y: np.ndarray, learning_rate: float = 0.01):
        dZ = self.softmax_layer.backward(Y)
        dZ = self.output_layer.backward(dZ, learning_rate)
        dZ = self.relu_layer.backward(dZ)
        self.hidden_layer.backward(dZ, learning_rate)
