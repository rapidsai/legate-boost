import doctest
import importlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

from legate.core import TaskTarget, get_machine

dirname = Path(__file__).parent
example_dir = dirname / "../../examples"
legateboost_dir = example_dir.parent
sys.path.append(str(example_dir))
noteboook_dir = example_dir / "notebook"
sys.path.append(str(noteboook_dir))

examples = list(filter(lambda x: "notebook" not in str(x), example_dir.glob("**/*.py")))


@pytest.fixture(scope="module")
def benchmark_dir():
    return Path(__file__).parent / "../../benchmark"


@pytest.mark.parametrize("path", examples, ids=[str(e.name) for e in examples])
def test_examples(path):
    os.environ["CI"] = "1"
    rel = path.relative_to(example_dir).with_suffix("")
    rel = str(rel).replace("/", ".")
    # temporary openml issue, this try/catch could be removed once fixed
    # https://github.com/openml/OpenML/issues/1232
    try:
        importlib.import_module(rel)
    except HTTPError as e:
        print(f"HTTP error in {path}: {e}")


notebooks = list(noteboook_dir.glob("*.ipynb"))


@pytest.mark.parametrize("path", notebooks, ids=[str(e) for e in notebooks])
@pytest.mark.skipif(
    get_machine().count(TaskTarget.GPU) == 0, reason="Notebooks too slow without GPU"
)
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


def test_benchmark(benchmark_dir):
    # Current legate 24.09.dev has issues with overriding legate config
    # from the legate call.  So have to unset `LEGATE_CONFIG`.
    env = os.environ.copy()
    del env["LEGATE_CONFIG"]

    cmd = (
        "legate --cpus=2 --gpus=0 --omps=0"
        " legateboost_scaling.py --nrows 100 --ncols 5 --niter 2"
        " --model_types tree,linear,krr,nn"
    )
    res = subprocess.run(
        cmd, shell=True, capture_output=True, cwd=benchmark_dir, env=env
    )
    assert res.returncode == 0, res.stderr.decode("utf-8")


def assert_docstrings_run(files):
    for f in files:
        print(f)
        result = doctest.testfile(str(f), verbose=True, module_relative=False)
        assert result.failed == 0


def test_markdown():
    markdown = list(legateboost_dir.glob("*.md"))
    markdown += list(example_dir.glob("*.md"))
    assert_docstrings_run(markdown)


def test_docstrings():
    python_files = list((legateboost_dir / "legateboost").glob("**/*.py"))
    # remove tests
    test_dir = legateboost_dir / "legateboost" / "test"
    python_files = list(filter(lambda x: test_dir not in x.parents, python_files))
    assert_docstrings_run(python_files)
