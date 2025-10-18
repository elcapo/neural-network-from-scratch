import sys
from nn_from_scratch.dataset import Dataset
from nn_from_scratch.network import Network
from nn_from_scratch.ui.progress import show_progress

def main():
    sys.stdout.write("Loading the dataset. This may take a minute...")
    sys.stdout.flush()

    dataset = Dataset(split='train')
    X, Y = dataset.get_features_and_labels()

    nn = Network()

    total_iterations = 1500
    show_progress(0, 100, f"0 completed iterations from a total of {total_iterations}")

    for iteration, accuracy in nn.train(X, Y, iterations=total_iterations, learning_rate=0.1):
        show_progress(100 * iteration / total_iterations, 100, f"{iteration} completed iterations from a total of {total_iterations}")

if __name__ == "__main__":
    main()