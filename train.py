import sys
from nn_from_scratch.dataset import Dataset
from nn_from_scratch.network import Network
from nn_from_scratch.repository import Repository
from nn_from_scratch.ui.progress import show_progress

def main():
    sys.stdout.write("Loading the training dataset. This may take a minute...")
    sys.stdout.flush()

    dataset = Dataset(split='train')
    dataset.extract()
    X, Y = dataset.get_features_and_labels()

    sys.stdout.write("\r")

    nn = Network()

    total_iterations = 2500
    show_progress(0, 100, f"0 completed iterations from a total of {total_iterations}")

    for iteration, accuracy in nn.train(X, Y, iterations=total_iterations, learning_rate=0.25):
        show_progress(100 * iteration / total_iterations, 100, f"{iteration} completed iterations from a total of {total_iterations}")
    
    weights_and_biases_path = "resources/weights_and_biases.json"
    repository = Repository()
    repository.store(nn, weights_and_biases_path)

    print(f"\nThe weights and biases of the model were saved to {weights_and_biases_path}")

if __name__ == "__main__":
    main()