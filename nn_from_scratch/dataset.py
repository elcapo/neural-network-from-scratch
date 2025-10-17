from pathlib import Path
import pandas as pd

class Dataset:
    def __init__(self, split: str = 'train'):
        self.split = split

    def get_path(self):
        return Path(__file__).parent.parent.resolve().joinpath(f"resources/data/mnist_{self.split}.csv")

    def read_dataframe(self, shuffle: bool = True):
        path = self.get_path()

        print("Reading the dataset")

        dataframe = pd.read_csv(path, header=None, names=["label"] + [f"pixel_{n}" for n in range(784)])

        if shuffle:
            print("Shuffling the dataset")
            dataframe = dataframe.sample(frac=1)

        return dataframe

    def get_features_and_labels(self):
        X = self.read_dataframe()
        Y = X.label.copy()
        del X["label"]
        return X.T.to_numpy() / 255., Y.to_numpy()