import os
import subprocess
from pathlib import Path

dirname = os.path.dirname(__file__)
example_dir = os.path.join(dirname, "../../examples")


def test_examples():
    for path in Path(example_dir).glob("*.py"):
        cmd = ["legate", path]
        subprocess.check_call(cmd)
