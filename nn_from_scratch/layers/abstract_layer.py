from abc import ABC
import numpy as np


class AbstractLayer(ABC):
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass # pragma: no cover

    def backward(self, prev: np.ndarray, dL_prev: np.ndarray) -> np.ndarray:
        pass # pragma: no cover
