import pytest
import numpy as np
from nn_from_scratch.layers.linear_layer import LinearLayer, ActivationType

def test_linear_layer():
    linear_layer = LinearLayer(2, 3, activation=ActivationType.SOFTMAX)

    assert linear_layer.neurons == 2, "The linear layer constructor sets the number of neurons"
    assert linear_layer.prev_neurons == 3, "The linear layer constructor sets the number of neurons of the previous layer"
    assert linear_layer.activation == ActivationType.SOFTMAX, "The linear layer constructor sets the activation type"
    assert linear_layer.W.shape == (2, 3,), "The linear layer constructor initializes the weights"
    assert linear_layer.b.shape == (2, 1,), "The linear layer constructor initializes the biases"

def test_forward_expects_numpy_arrays():
    linear_layer = LinearLayer(2, 3)

    with pytest.raises(AssertionError):
        X = [1, 2, 3, 4, 5]
        linear_layer.forward(X)

def test_forward_expects_correct_dimensions():
    linear_layer = LinearLayer(2, 3)

    with pytest.raises(AssertionError):
        X = np.array([1, 2, 3, 4, 5, 6])
        linear_layer.forward(X)

def test_forward_processes_input():
    linear_layer = LinearLayer(2, 3)

    X = np.random.rand(3, 5)
    Y = linear_layer.forward(X)

    assert Y.shape == (2, 5,), "The linear layer's forward output must have the correct dimensions"

def test_forward_uses_relu_activation():
    linear_layer = LinearLayer(2, 3, activation=ActivationType.RELU)

    X = np.random.rand(3, 5)
    Y = linear_layer.forward(X)

    assert (Y >= 0).all(), "The RELU activation must kill all negative values"

def test_forward_uses_softmax_activation():
    linear_layer = LinearLayer(2, 3, activation=ActivationType.SOFTMAX)

    X = np.random.rand(3, 5) * 2 - 1
    Y = linear_layer.forward(X)

    assert (Y >= 0).all(), "The Softmax activation must return positive values"
    assert np.allclose(np.sum(Y, axis=0, keepdims=True), 1), "The Softmax activation returns values that add up to one"

def test_relu_activation_kills_negative_values():
    linear_layer = LinearLayer(2, 3, activation=ActivationType.RELU)

    Z = np.random.rand(2, 5) * 2 - 1
    A = linear_layer.activate(Z)

    assert (A >= 0).all(), "The RELU activation must kill all negative values"

def test_forward_uses_softmax_activation():
    linear_layer = LinearLayer(2, 3, activation=ActivationType.SOFTMAX)

    Z = np.random.rand(2, 5) * 2 - 1
    A = linear_layer.activate(Z)

    assert (A >= 0).all(), "The Softmax activation must return positive values"
    assert np.allclose(np.sum(A, axis=0, keepdims=True), 1), "The Softmax activation returns values that add up to one"
