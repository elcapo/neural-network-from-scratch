import numpy as np
from nn_from_scratch.network import Network


class Evaluation:
    def __init__(self, network: Network):
        self.network = network

    def accuracy(self, Y: np.ndarray, Y_pred: np.ndarray) -> float:
        return round(np.sum(Y == Y_pred) / Y.size, 2)

    def confusion_matrix(self, Y: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        empty_predictions = {}
        for predicted_value in range(0, 10):
            empty_predictions[predicted_value] = 0

        prediction_count = {}
        for real_value in range(0, 10):
            prediction_count[real_value] = empty_predictions.copy()

        for index, real_value in enumerate(Y):
            predicted_value = Y_pred[index]
            prediction_count[real_value][predicted_value] += 1

        confusion_matrix = []
        for category in prediction_count.values():
            confusion_matrix.append(list(category.values()))

        return np.array(confusion_matrix)

    def binary_confusion_matrix_per_category(self, Y: np.ndarray, Y_pred: np.ndarray, category: int) -> np.ndarray:
        confusion_matrix = self.confusion_matrix(Y, Y_pred)

        TP = confusion_matrix[category][category]
        FN = np.sum(confusion_matrix[category]) - TP
        FP = np.sum(confusion_matrix[:, category]) - TP
        TN = np.sum(confusion_matrix) - (TP + FN + FP)

        return np.array([[TP, FN], [FP, TN]])

    def binary_confusion_matrix(self, Y: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        binary_confusion_matrix = np.zeros((2, 2))

        for category in range(0, 10):
            binary_confusion_matrix += self.binary_confusion_matrix_per_category(Y, Y_pred, category)

        return binary_confusion_matrix
