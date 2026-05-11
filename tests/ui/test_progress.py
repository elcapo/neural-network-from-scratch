import pytest
from nn_from_scratch.ui.progress import show_progress


def test_show_progress_renders_filled_and_empty_segments(capsys):
    show_progress(2, 5, "hello")

    captured = capsys.readouterr()

    assert captured.out.startswith("\rProgress: |"), "Progress must start with a carriage return and the prefix"
    assert captured.out.endswith("| hello"), "Progress must end with the trailing bar and message"
    assert captured.out.count("█") == 3, "Filled segments must include all positions up to and including the current step"
    assert captured.out.count("·") == 2, "Remaining positions must be rendered as dots"


def test_show_progress_complete(capsys):
    show_progress(4, 5, "done")

    captured = capsys.readouterr()

    assert captured.out.count("█") == 5, "When step reaches max - 1 every segment must be filled"
    assert captured.out.count("·") == 0


def test_show_progress_rejects_step_above_max():
    with pytest.raises(AssertionError):
        show_progress(10, 5, "overflow")
