import numpy as np
from nn_from_scratch.evaluation import Evaluation
from nn_from_scratch.network import Network


def test_accuracy_perfect():
    evaluation = Evaluation(Network())

    Y = np.array([0, 1, 2, 3])

    assert evaluation.accuracy(Y, Y) == 1.0, "Accuracy must be 1.0 when all predictions match"


def test_accuracy_partial():
    evaluation = Evaluation(Network())

    Y = np.array([0, 1, 2, 3])
    Y_pred = np.array([0, 1, 9, 9])

    assert evaluation.accuracy(Y, Y_pred) == 0.5, "Accuracy must reflect the proportion of correct predictions"


def test_confusion_matrix_shape_and_counts():
    evaluation = Evaluation(Network())

    Y = np.array([0, 1, 2, 0])
    Y_pred = np.array([0, 1, 2, 1])

    matrix = evaluation.confusion_matrix(Y, Y_pred)

    assert matrix.shape == (10, 10), "Confusion matrix must be 10x10 for the MNIST classes"
    assert matrix[0][0] == 1, "Diagonal must count correctly classified samples"
    assert matrix[0][1] == 1, "Off-diagonal cells must count misclassifications by actual/predicted class"
    assert matrix[1][1] == 1
    assert matrix[2][2] == 1
    assert matrix.sum() == Y.size, "Total counts must equal the number of samples"


def test_binary_confusion_matrix_per_category():
    evaluation = Evaluation(Network())

    Y = np.array([0, 1, 2, 0])
    Y_pred = np.array([0, 1, 2, 1])

    matrix = evaluation.binary_confusion_matrix_per_category(Y, Y_pred, 0)

    assert matrix[0][0] == 1, "TP for class 0 must count samples correctly predicted as 0"
    assert matrix[0][1] == 1, "FN for class 0 must count actual 0s predicted as other classes"
    assert matrix[1][0] == 0, "FP for class 0 must count other classes predicted as 0"
    assert matrix[1][1] == 2, "TN for class 0 must count samples neither actual nor predicted as 0"


def test_binary_confusion_matrix_aggregates_all_categories():
    evaluation = Evaluation(Network())

    Y = np.array([0, 1, 2, 0])
    Y_pred = np.array([0, 1, 2, 1])

    matrix = evaluation.binary_confusion_matrix(Y, Y_pred)

    assert matrix.shape == (2, 2), "Aggregated binary confusion matrix must be 2x2"
    assert matrix.sum() == 10 * Y.size, "Each sample contributes to one cell per class (10 classes)"
