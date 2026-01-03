import numpy as np


def one_hot_encode(y: np.ndarray) -> np.ndarray:
    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y] = 1

    return one_hot.T
