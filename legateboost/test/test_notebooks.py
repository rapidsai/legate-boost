import glob
import importlib
import subprocess
import sys
from pathlib import Path

noteboook_dir = str(Path(__file__).parent / "../../examples/notebook")


def test_notebooks():
    sys.path.append(noteboook_dir)
    for nb in glob.glob(noteboook_dir + "/*.ipynb"):
        # use nbconvert to convert notebook to python script
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "script",
            "--RegexRemovePreprocessor.patterns='^%'",
            nb,
        ]
        subprocess.check_call(cmd)
        # import the script to run it in the existing python process
        print("Running notebook: " + nb + "\n")
        importlib.import_module(Path(nb).stem)
