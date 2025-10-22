import numpy as np
from nn_from_scratch.dataset import Dataset
from nn_from_scratch.evaluation import Evaluation
from nn_from_scratch.network import Network
from nn_from_scratch.repository import Repository

def main():
    print("Loading the test dataset. This may take a minute...")

    dataset = Dataset(split='test')
    X, Y = dataset.get_features_and_labels()

    nn = Network()
    repository = Repository()
    repository.load(nn, "resources/weights_and_biases.npy")
    evaluation = Evaluation(nn)

    predicted_probabilities = nn.forward(X, training=True)
    Y_pred = np.argmax(predicted_probabilities, axis=0)

    accuracy = evaluation.accuracy(Y, Y_pred)
    print(f"The mean accuracy of the model in the test set is: {accuracy}\n")

    confusion_matrix = evaluation.confusion_matrix(Y, Y_pred)

    print("The confusion matrix of the model is:\n")
    print(confusion_matrix)
    print("\n")

    print("The binary confusion matrix of the model is:\n")
    print(evaluation.binary_confusion_matrix(Y, Y_pred))

if __name__ == "__main__":
    main()