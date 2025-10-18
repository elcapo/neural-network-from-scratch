import numpy as np
from nn_from_scratch.layers.identity_layer import IdentityLayer
from nn_from_scratch.layers.linear_layer import LinearLayer, ActivationType
from nn_from_scratch.encoders.one_hot_encoder import one_hot_encode

class Network:
    def __init__(self, input_size = 28*28, hidden_size = 10, output_size = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initialize()

    def initialize(self):
        self.input_layer = IdentityLayer(self.input_size)
        self.hidden_layer = LinearLayer(self.hidden_size, self.input_size, ActivationType.RELU)
        self.output_layer = LinearLayer(self.output_size, self.hidden_size, ActivationType.SOFTMAX)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        X = self.input_layer.forward(X)
        X = self.hidden_layer.forward(X, training)
        return self.output_layer.forward(X, training)

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 100, learning_rate: float = 0.01):
        for i in range(iterations):
            Y_pred = self.forward(X, training=True)
            self.backward(X, Y, Y_pred, learning_rate)

            predictions = np.argmax(Y_pred, axis=0)
            accuracy = round(np.sum(predictions == Y) / Y.size, 2)

            yield (i + 1, accuracy,)

    def backward(self, X: np.ndarray, Y: np.ndarray, Y_pred: np.ndarray, learning_rate: float = 0.01):
        m = Y.shape[0]

        dZ_o = Y_pred - one_hot_encode(Y)
        dW_o = dZ_o.dot(self.hidden_layer.A.T) / m
        db_o = np.sum(dZ_o, axis=1, keepdims=True) / m

        W_o = self.output_layer.W
        Z_h = self.hidden_layer.Z

        dZ_h = W_o.T.dot(dZ_o) * (Z_h > 0)
        dW_h = dZ_h.dot(X.T) / m
        db_h = np.sum(dZ_h, axis=1, keepdims=True) / m

        self.update_params(dW_o, db_o, dW_h, db_h, learning_rate)

    def update_params(self, dW_o: np.ndarray, db_o: np.ndarray, dW_h: np.ndarray, db_h: np.ndarray, learning_rate: float = 0.01):
        self.output_layer.W = self.output_layer.W - learning_rate * dW_o
        self.output_layer.b = self.output_layer.b - learning_rate * db_o
        self.hidden_layer.W = self.hidden_layer.W - learning_rate * dW_h
        self.hidden_layer.b = self.hidden_layer.b - learning_rate * db_h
