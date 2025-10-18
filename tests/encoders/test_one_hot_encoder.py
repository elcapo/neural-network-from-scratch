import numpy as np
from nn_from_scratch.encoders.one_hot_encoder import one_hot_encode

def test_one_hot_encoder():
    assert (one_hot_encode(np.array([1])) == [[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]).all()
    assert (one_hot_encode(np.array([0, 2])) == [[1, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]).all()