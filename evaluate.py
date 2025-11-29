import sys
import numpy as np
import matplotlib.pyplot as plt
from nn_from_scratch.dataset import Dataset
from nn_from_scratch.evaluation import Evaluation
from nn_from_scratch.network import Network
from nn_from_scratch.repository import Repository

def main():
    sys.stdout.write("Loading the test dataset. This may take a minute...")
    sys.stdout.flush()

    dataset = Dataset(split='test')
    X, Y = dataset.get_features_and_labels()

    sys.stdout.write("\r")

    nn = Network()
    repository = Repository()
    repository.load(nn, "resources/weights_and_biases.json")
    evaluation = Evaluation(nn)

    predicted_probabilities = nn.forward(X, training=True)
    Y_pred = np.argmax(predicted_probabilities, axis=0)

    accuracy = evaluation.accuracy(Y, Y_pred)
    confusion_matrix = evaluation.confusion_matrix(Y, Y_pred)

    print(f"The mean accuracy of the model in the test set is: {accuracy}\n")
    print("The confusion matrix of the model is:\n")
    print(confusion_matrix)
    print("\n")
    print("The binary confusion matrix of the model is:\n")
    print(evaluation.binary_confusion_matrix(Y, Y_pred))

    plot(confusion_matrix)

def plot(confusion_matrix: np.ndarray):
    fig, ax = plt.subplots(1 ,1)

    image = ax.imshow(confusion_matrix, interpolation=None)
    colorbar = fig.colorbar(image, ax=ax, orientation='horizontal', fraction=.1)

    colorbar.ax.set_xlabel('Sample count')

    ax.set_xticks(range(0, 10))
    ax.set_yticks(range(0, 10))

    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.title("Confusion matrix")
    plt.show()

if __name__ == "__main__":
    main()