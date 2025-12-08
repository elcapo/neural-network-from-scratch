import numpy as np
from nn_from_scratch.layers.identity_layer import IdentityLayer
from nn_from_scratch.layers.linear_layer import LinearLayer, ActivationType
from nn_from_scratch.layers.relu_layer import ReluLayer
from nn_from_scratch.layers.softmax_layer import SoftmaxLayer
from nn_from_scratch.encoders.one_hot_encoder import one_hot_encode

class Network:
    def __init__(self, input_size = 28*28, hidden_size = 10, output_size = 10):
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

    def forward(self, X: np.ndarray, training: bool = False) -> tuple[np.ndarray]:
        steps = {}

        X = self.input_layer.forward(X)

        if training:
            steps["input_layer"] = X

        X = self.hidden_layer.forward(X)

        if training:
            steps["hidden_layer"] = X

        X = self.relu_layer.forward(X)

        if training:
            steps["relu_layer"] = X

        X = self.output_layer.forward(X)

        if training:
            steps["output_layer"] = X

        X = self.softmax_layer.forward(X)

        if training:
            steps["softmax_layer"] = X

        return X, steps

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 100, learning_rate: float = 0.01):
        for i in range(iterations):
            Y_pred, steps = self.forward(X, training=True)
            self.backward(X, Y, Y_pred, steps, learning_rate)

            predictions = np.argmax(Y_pred, axis=0)
            accuracy = round(np.sum(predictions == Y) / Y.size, 2)

            yield (i + 1, accuracy,)

    def backward(self, X: np.ndarray, Y: np.ndarray, Y_pred: np.ndarray, steps: dict, learning_rate: float = 0.01):
        m = Y.shape[0]

        dZ_o = Y_pred - one_hot_encode(Y)
        dW_o = dZ_o.dot(steps["softmax_layer"].T) / m
        db_o = np.sum(dZ_o, axis=1, keepdims=True) / m

        W_o = self.output_layer.W
        Z_h = steps["hidden_layer"]

        dZ_h = W_o.T.dot(dZ_o) * (Z_h > 0)
        dW_h = dZ_h.dot(X.T) / m
        db_h = np.sum(dZ_h, axis=1, keepdims=True) / m

        self.update_params(dW_o, db_o, dW_h, db_h, learning_rate)

    def update_params(self, dW_o: np.ndarray, db_o: np.ndarray, dW_h: np.ndarray, db_h: np.ndarray, learning_rate: float = 0.01):
        self.output_layer.W = self.output_layer.W - learning_rate * dW_o
        self.output_layer.b = self.output_layer.b - learning_rate * db_o
        self.hidden_layer.W = self.hidden_layer.W - learning_rate * dW_h
        self.hidden_layer.b = self.hidden_layer.b - learning_rate * db_h
