import os
import pathlib


DIR_EXCEPTIONS = [
    "__pycache__",
    "circuit_blocks",  # functions already tested in surface_sim.experiments
]
FILE_EXCEPTIONS = [
    "__init__.py",
    "surface_sim/models/util.py",  # functions already tested in surface_sim.models
]


def test_tests():
    test_dir = pathlib.Path("tests")
    mod_dir = pathlib.Path("surface_sim")
    if not mod_dir.exists():
        raise ValueError("module directory does not exist.")

    for path, _, files in os.walk(mod_dir):
        for file in files:
            if file[-3:] != ".py" or file[0] == "_":
                continue
            if os.path.basename(os.path.normpath(path)) in DIR_EXCEPTIONS:
                continue
            if (file in FILE_EXCEPTIONS) or (
                os.path.join(path, file) in FILE_EXCEPTIONS
            ):
                continue

            # change root dir to test_dir
            relpath = os.path.relpath(path, mod_dir)
            testpath = os.path.join(test_dir, relpath)
            if not os.path.exists(os.path.join(testpath, "test_" + file)):
                raise ValueError(
                    f"test file for {os.path.join(mod_dir, relpath, file)}"
                    " does not exist"
                )
    return
