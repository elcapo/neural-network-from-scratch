import pytest
import numpy as np
from nn_from_scratch.layers.identity_layer import IdentityLayer


def test_identity_layer():
    identity_layer = IdentityLayer(5)

    assert identity_layer.neurons == 5, "The constructor of the identity layer should set its number of neurons"


def test_forward_expects_numpy_arrays():
    identity_layer = IdentityLayer(5)

    with pytest.raises(AssertionError):
        X = [1, 2, 3, 4, 5]
        identity_layer.forward(X)


def test_forward_expects_correct_dimensions():
    identity_layer = IdentityLayer(5)

    with pytest.raises(AssertionError):
        X = np.array([1, 2, 3, 4, 5, 6])
        identity_layer.forward(X)


def test_returns_the_unmodified_input():
    identity_layer = IdentityLayer(5)
    X = np.random.rand(5)
    assert (identity_layer.forward(X) == X).all(), "The identity layer should return its unmodified input"
