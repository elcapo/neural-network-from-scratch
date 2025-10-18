import json
from nn_from_scratch.dataset import Dataset
from nn_from_scratch.network import Network

def main():
    dataset = Dataset(split='train')
    X, Y = dataset.get_features_and_labels()

    iterations = 1000

    train(X, Y, iterations, 0.25)
    train(X, Y, iterations, 0.1)
    train(X, Y, iterations, 0.01)
    train(X, Y, iterations, 0.001)

def train(X, Y, iterations: int, learning_rate: float):
    nn = Network()

    for i, accuracy in nn.train(X, Y, iterations=iterations, learning_rate=learning_rate):
        report = {
            "learning_rate": learning_rate,
            "iteration": i,
            "accuracy": accuracy,
        }

        print(json.dumps(report))

if __name__ == "__main__":
    main()