import sys
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from nn_from_scratch.dataset import Dataset


def plot(images: np.ndarray, labels: np.ndarray, indices: list):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))

    for n in range(0, len(indices)):
        image = images[:, n].reshape(28, 28)
        label = labels[n]
        index = indices[n]

        ax = axes[n // 5, n % 5]
        ax.imshow(image, cmap="gray")
        ax.set_title(f"{label}")
        ax.set_xlabel(f"Id: {index}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    print("Loading the training dataset. This may take a minute...")

    dataset = Dataset(split="train")
    dataset.extract()
    X, Y = dataset.get_features_and_labels()

    samples = len(Y)

    print(f"{samples} images were found")

    random_indices = [randint(0, samples) for i in range(0, 10)]
    plot(X[:, random_indices], Y[random_indices], random_indices)

if __name__ == "__main__":
    main()
