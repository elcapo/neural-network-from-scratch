import numpy as np
import matplotlib.pyplot as plt
from nn_from_scratch.dataset import Dataset


def plot(images: np.ndarray, labels: np.ndarray, indices: list):
    _, axes = plt.subplots(1, 5, figsize=(10, 6))

    for n in range(0, len(indices)):
        image = images[:, n].reshape(28, 28)
        label = labels[n]
        index = indices[n]

        ax = axes[n % 5]
        ax.imshow(image, cmap="gray")
        ax.set_title(f"{label}")
        ax.set_xlabel(f"Id {index}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    dataset = Dataset(split="train")
    dataset.extract()
    X, Y, indices = dataset.sample(5)

    plot(X, Y, indices)


if __name__ == "__main__":
    main()
