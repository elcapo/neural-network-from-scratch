import numpy as np
from nn_from_scratch.layers.identity_layer import IdentityLayer
from nn_from_scratch.layers.linear_layer import LinearLayer
from nn_from_scratch.layers.relu_layer import ReluLayer
from nn_from_scratch.layers.softmax_layer import SoftmaxLayer
from nn_from_scratch.encoders.one_hot_encoder import one_hot_encode


class Steps:
    input_layer: np.ndarray
    hidden_layer: np.ndarray
    relu_layer: np.ndarray
    output_layer: np.ndarray
    softmax_layer: np.ndarray


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

    def forward(self, X: np.ndarray, training: bool = False) -> tuple[np.ndarray, Steps]:
        steps = Steps()

        X = self.input_layer.forward(X, training)

        if training:
            steps.input_layer = X

        X = self.hidden_layer.forward(X, training)

        if training:
            steps.hidden_layer = X

        X = self.relu_layer.forward(X, training)

        if training:
            steps.relu_layer = X

        X = self.output_layer.forward(X, training)

        if training:
            steps.output_layer = X

        X = self.softmax_layer.forward(X, training)

        if training:
            steps.softmax_layer = X

        return X, steps

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 100, learning_rate: float = 0.01):
        for i in range(iterations):
            Y_pred, steps = self.forward(X, training=True)
            self.backward(X, Y, Y_pred, steps, learning_rate)

            predictions = np.argmax(Y_pred, axis=0)
            accuracy = round(np.sum(predictions == Y) / Y.size, 2)

            yield (
                i + 1,
                accuracy,
            )

    def backward(self, X: np.ndarray, Y: np.ndarray, Y_pred: np.ndarray, steps: Steps, learning_rate: float = 0.01):
        dZ_o = self.softmax_layer.backward(steps.softmax_layer, Y, np.ones(steps.softmax_layer.shape))

        dZ_h = self.output_layer.backward(steps.softmax_layer, dZ_o, learning_rate)

        dZ_h = self.relu_layer.backward(steps.output_layer, dZ_h)

        self.hidden_layer.backward(steps.relu_layer, dZ_h, learning_rate)

    
