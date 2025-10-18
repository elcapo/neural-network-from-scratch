import sys

def show_progress(step: int, max: int, message: str):
    assert max >= step

    sys.stdout.write("\rProgress: |")

    for i in range(max):
        if i <= step:
            sys.stdout.write("█")
        else:
            sys.stdout.write("·")

    sys.stdout.write(f"| {message}")

    sys.stdout.flush()