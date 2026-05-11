import numpy as np


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1

    return one_hot.T
