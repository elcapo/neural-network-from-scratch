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


def test_sample_returns_n_random_rows_with_indices(monkeypatch, tmp_path):
    dataset = Dataset(split="train")
    monkeypatch.setattr(dataset, "get_path", lambda: tmp_path)
    csv_path = tmp_path / "mnist_train.csv"
    rows = [",".join([str(label)] + ["255"] * 784) for label in range(20)]
    csv_path.write_text("\n".join(rows) + "\n")

    X, Y, indices = dataset.sample(5, seed=42)

    assert X.shape == (784, 5), "Features must be returned with shape (pixels, samples)"
    assert Y.shape == (5,), "Labels must be returned as a 1D array of size n_samples"
    assert len(indices) == 5, "Indices must have length n_samples"
    assert len(set(indices)) == 5, "Each chosen row must appear at most once"
    assert all(0 <= idx < 20 for idx in indices), "Indices must point inside the CSV"
    assert (X == 1.0).all(), "Pixel values must be normalized to the [0, 1] range"
    for k, idx in enumerate(indices):
        assert Y[k] == idx, "Each returned label must correspond to its source row"


def test_sample_returns_every_row_when_n_exceeds_dataset(monkeypatch, tmp_path):
    dataset = Dataset(split="train")
    monkeypatch.setattr(dataset, "get_path", lambda: tmp_path)
    csv_path = tmp_path / "mnist_train.csv"
    rows = [",".join([str(label)] + ["0"] * 784) for label in range(3)]
    csv_path.write_text("\n".join(rows) + "\n")

    X, Y, indices = dataset.sample(10, seed=0)

    assert X.shape == (784, 3), "sample must clamp to the available number of rows"
    assert sorted(indices) == [0, 1, 2], "All rows must be returned when n exceeds the dataset"
