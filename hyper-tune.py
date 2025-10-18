import sys
import matplotlib.pyplot as plt
from nn_from_scratch.dataset import Dataset
from nn_from_scratch.network import Network
from nn_from_scratch.ui.progress import show_progress

def main():
    sys.stdout.write("Loading the dataset. This may take a minute...")
    sys.stdout.flush()

    dataset = Dataset(split='train')
    X, Y = dataset.get_features_and_labels()

    iterations = 1500
    learning_rates = [0.5, 0.25, 0.1, 0.01, 0.001]

    total_iterations = iterations * len(learning_rates)
    completed_iterations = 0
    show_progress(0, 100, f"{completed_iterations} completed iterations from a total of {total_iterations}")

    accuracies_report = {}
    for learning_rate in learning_rates:
        accuracies_report[learning_rate] = {"iterations": [], "accuracies": []}

        nn = Network()
        for i, accuracy in nn.train(X, Y, iterations=iterations, learning_rate=learning_rate):
            if i % 30 == 1:
                accuracies_report[learning_rate]["iterations"].append(i)
                accuracies_report[learning_rate]["accuracies"].append(accuracy)

            completed_iterations += 1
            show_progress(100 * completed_iterations / total_iterations, 100, f"{completed_iterations} completed iterations from a total of {total_iterations}")

    plot(accuracies_report)

def plot(accuracies_report):
    plt.figure(figsize=(10, 6))

    for learning_rate in sorted(accuracies_report.keys()):
        accuracies = accuracies_report[learning_rate]["accuracies"]
        iterations = accuracies_report[learning_rate]["iterations"]

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