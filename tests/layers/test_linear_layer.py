import pytest
import numpy as np
from nn_from_scratch.layers.linear_layer import LinearLayer


def test_linear_layer():
    linear_layer = LinearLayer(2, 3)

    assert linear_layer.neurons == 2, "The linear layer constructor sets the number of neurons"
    assert linear_layer.prev_neurons == 3, "The linear layer constructor sets the number of neurons of the previous layer"
    assert linear_layer.W.shape == (
        2,
        3,
    ), "The linear layer constructor initializes the weights"
    assert linear_layer.b.shape == (
        2,
        1,
    ), "The linear layer constructor initializes the biases"


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

    assert Y.shape == (
        2,
        5,
    ), "The linear layer's forward output must have the correct dimensions"
