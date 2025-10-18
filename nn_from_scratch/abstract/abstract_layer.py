from abc import ABC
import numpy as np

class AbstractLayer(ABC):
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def activate(self, Z: np.ndarray) -> np.ndarray:
        pass