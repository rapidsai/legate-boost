import importlib
import subprocess
import sys
from pathlib import Path

dirname = Path(__file__).parent
example_dir = dirname / "../../examples"
noteboook_dir = example_dir / "../../examples/notebook"


def test_examples():
    sys.path.append(str(example_dir))
    for path in example_dir.glob("*.py"):
        importlib.import_module(path.stem)


def test_notebooks():
    sys.path.append(str(noteboook_dir))
    for path in noteboook_dir.glob("*.ipynb"):
        # use nbconvert to convert notebook to python script
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "script",
            "--RegexRemovePreprocessor.patterns='^%'",
            str(path),
        ]
        subprocess.check_call(cmd)
        # import the script to run it in the existing python process
        print("Running notebook: " + path + "\n")
        importlib.import_module(path.stem)
