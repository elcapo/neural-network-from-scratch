import numpy as np
import matplotlib.pyplot as plt
from nn_from_scratch.network import Network
from nn_from_scratch.repository import Repository

def main():
    nn = Network()
    repository = Repository()
    repository.load(nn, "resources/weights_and_biases.npy")

    plot_hidden_layer(nn.hidden_layer.W + nn.hidden_layer.b)

def plot_hidden_layer(hidden_layer: np.ndarray):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))

    for n in range(0, 10):
        weights = hidden_layer[n].reshape(28, 28)
        ax = axes[n // 5, n % 5]
        ax.imshow(weights, cmap='gray')
        ax.set_title(f'Hidden neuron {n + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
