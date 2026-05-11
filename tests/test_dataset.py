from pathlib import Path
from zipfile import ZipFile
from nn_from_scratch.dataset import Dataset


def test_default_split_is_train():
    dataset = Dataset()

    assert dataset.split == "train", "The default split must be the training set"


def test_custom_split():
    dataset = Dataset(split="test")

    assert dataset.split == "test", "The constructor must respect the requested split"


def test_get_path_points_to_the_resources_directory():
    path = Dataset().get_path()

    assert path.name == "data", "Datasets must live under the data directory"
    assert path.parent.name == "resources", "The data directory must be inside resources"


def test_filenames_follow_the_split_naming_convention(monkeypatch, tmp_path):
    dataset = Dataset(split="train")
    monkeypatch.setattr(dataset, "get_path", lambda: tmp_path)

    assert dataset.get_compressed_filename().endswith("mnist_train.zip")
    assert dataset.get_uncompressed_filename().endswith("mnist_train.csv")


def test_extract_is_a_noop_when_csv_already_exists(monkeypatch, tmp_path):
    dataset = Dataset(split="train")
    monkeypatch.setattr(dataset, "get_path", lambda: tmp_path)
    (tmp_path / "mnist_train.csv").write_text("already-extracted")

    dataset.extract()

    assert (tmp_path / "mnist_train.csv").read_text() == "already-extracted", "extract must not overwrite an existing csv"


def test_extract_unzips_the_csv(monkeypatch, tmp_path):
    dataset = Dataset(split="train")
    monkeypatch.setattr(dataset, "get_path", lambda: tmp_path)
    zip_path = tmp_path / "mnist_train.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("mnist_train.csv", "label,pixel\n0,255\n")

    dataset.extract()

    assert (tmp_path / "mnist_train.csv").exists(), "extract must produce the uncompressed csv from the zip archive"


def test_get_features_and_labels_normalizes_pixels_and_splits_label(monkeypatch, tmp_path):
    dataset = Dataset(split="train")
    monkeypatch.setattr(dataset, "get_path", lambda: tmp_path)
    csv_path = tmp_path / "mnist_train.csv"
    rows = [
        ",".join(["3"] + ["255"] * 784),
        ",".join(["7"] + ["0"] * 784),
    ]
    csv_path.write_text("\n".join(rows) + "\n")

    X, Y = dataset.get_features_and_labels()

    assert X.shape == (784, 2), "Features must be returned with shape (pixels, samples)"
    assert ((X == 1.0) | (X == 0.0)).all(), "Pixel values must be normalized to the [0, 1] range"
    assert set(Y.tolist()) == {3, 7}, "Labels must be returned as a 1D array of class indices"
