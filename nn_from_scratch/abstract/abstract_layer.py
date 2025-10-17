import numpy as np

class AbstractLayer:
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def activate(self, Z: np.ndarray) -> np.ndarray:
        pass