import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from legate.core import TaskTarget, get_legate_runtime

dirname = Path(__file__).parent
benchmark_dir = dirname / "../../benchmark"
example_dir = dirname / "../../examples"
sys.path.append(str(example_dir))
noteboook_dir = example_dir / "notebook"
sys.path.append(str(noteboook_dir))

examples = list(filter(lambda x: "notebook" not in str(x), example_dir.glob("**/*.py")))


@pytest.mark.parametrize("path", examples, ids=[str(e.name) for e in examples])
def test_examples(path):
    os.environ["CI"] = "1"
    rel = path.relative_to(example_dir).with_suffix("")
    rel = str(rel).replace("/", ".")
    importlib.import_module(rel)


notebooks = list(noteboook_dir.glob("*.ipynb"))


using_gpu = get_legate_runtime().machine.count(TaskTarget.GPU) > 0


@pytest.mark.parametrize("path", notebooks, ids=[str(e) for e in notebooks])
@pytest.mark.skipif(not using_gpu, reason="Notebooks too slow without GPU")
def test_notebooks(path):
    os.environ["CI"] = "1"
    # use nbconvert to convert notebook to python script
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "script",
        "--RegexRemovePreprocessor.patterns='^%'",
        str(path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode())
        raise e
    # import the script to run it in the existing python process
    importlib.import_module(path.stem)


def test_benchmark():
    subprocess.run(
        "legate --cpus 2 scaling.py --nrows 100 --ncols 5 --niter 2",
        shell=True,
        check=True,
        cwd=benchmark_dir,
    )
