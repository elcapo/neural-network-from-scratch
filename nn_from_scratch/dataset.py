from pathlib import Path
from zipfile import ZipFile
import pandas as pd


class Dataset:
    def __init__(self, split: str = "train"):
        self.split = split

    def get_path(self):
        return Path(__file__).parent.parent.resolve().joinpath("resources/data/")

    def get_compressed_filename(self) -> str:
        return str(self.get_path().joinpath(f"mnist_{self.split}.zip"))

    def get_uncompressed_filename(self) -> str:
        return str(self.get_path().joinpath(f"mnist_{self.split}.csv"))

    def extract(self):
        if Path(self.get_uncompressed_filename()).exists():
            return

        with ZipFile(self.get_compressed_filename(), "r") as zf:
            zf.extract(f"mnist_{self.split}.csv", self.get_path())

    def read_dataframe(self, shuffle: bool = True):
        uncompressed_filename = self.get_uncompressed_filename()

        dataframe = pd.read_csv(uncompressed_filename, header=None, names=["label"] + [f"pixel_{n}" for n in range(784)])

        if shuffle:
            dataframe = dataframe.sample(frac=1)

        return dataframe

    def get_features_and_labels(self):
        X = self.read_dataframe()
        Y = X.label.copy()
        del X["label"]
        return X.T.to_numpy() / 255.0, Y.to_numpy()
