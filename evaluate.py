from nn_from_scratch.dataset import Dataset
from nn_from_scratch.network import Network
from nn_from_scratch.repository import Repository

def main():
    print("Loading the test dataset. This may take a minute...")

    dataset = Dataset(split='test')
    X, Y = dataset.get_features_and_labels()

    nn = Network()
    repository = Repository()
    repository.load(nn, "resources/weights_and_biases.npy")

    accuracy = nn.evaluate(X, Y)
    print(f"The mean accuracy of the model in the test set is: {accuracy}")

if __name__ == "__main__":
    main()