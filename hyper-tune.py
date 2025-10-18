import matplotlib.pyplot as plt
from nn_from_scratch.dataset import Dataset
from nn_from_scratch.network import Network

def main():
    dataset = Dataset(split='train')
    X, Y = dataset.get_features_and_labels()

    iterations = 50

    accuracies_report = {}

    accuracies_report[0.5] = train(X, Y, iterations, 0.5)
    accuracies_report[0.25] = train(X, Y, iterations, 0.25)
    accuracies_report[0.1] = train(X, Y, iterations, 0.1)
    accuracies_report[0.01] = train(X, Y, iterations, 0.01)
    accuracies_report[0.001] = train(X, Y, iterations, 0.001)

    plot(accuracies_report)

def train(X, Y, iterations: int, learning_rate: float):
    nn = Network()

    accuracies = []

    for i, accuracy in nn.train(X, Y, iterations=iterations, learning_rate=learning_rate):
        accuracies.append(accuracy)

    return accuracies

def plot(accuracies_report):
    plt.figure(figsize=(10, 6))

    for learning_rate in sorted(accuracies_report.keys()):
        accuracies = accuracies_report[learning_rate]
        iterations = range(len(accuracies))

        plt.plot(
            iterations,
            accuracies,
            marker='o',
            label=f"Learning rate = {learning_rate}",
            linewidth=2,
            markersize=4
        )

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy and Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()