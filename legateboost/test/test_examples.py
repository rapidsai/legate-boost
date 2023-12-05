import importlib
import subprocess
import sys
from pathlib import Path

dirname = Path(__file__).parent
example_dir = dirname / "../../examples"
noteboook_dir = example_dir / "notebook"


def test_examples():
    sys.path.append(str(example_dir))
    examples = list(example_dir.glob("**/*.py"))
    # dont test the notebook again
    examples = list(filter(lambda x: "notebook" not in str(x), examples))
    assert examples, "No examples found"
    for path in examples:
        rel = path.relative_to(example_dir).with_suffix("")
        print("Running example: " + str(rel) + "\n")
        rel = str(rel).replace("/", ".")
        importlib.import_module(rel)


def test_notebooks():
    sys.path.append(str(noteboook_dir))
    notebooks = list(noteboook_dir.glob("*.ipynb"))
    assert notebooks, "No notebooks found"
    for path in notebooks:
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
        print("Running notebook: " + str(path) + "\n")
        importlib.import_module(path.stem)
