import argparse
import time

import pandas as pd

import cunumeric as cn
import legateboost as lb
from legate.core import TaskTarget, get_legate_runtime

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x):
        return x


from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

sns.set()
plt.rcParams["font.family"] = "serif"


def train_model(X, y, model_type):
    if model_type == "tree":
        base_models = (lb.models.Tree(max_depth=12),)
    elif model_type == "linear":
        base_models = (lb.models.Linear(),)
    elif model_type == "krr":
        base_models = (lb.models.KRR(sigma=1.0),)

    model = lb.LBClassifier(base_models=base_models).fit(X, y)
    # force legate to realise result
    x = model.predict(X[0:2])[0]  # noqa


def benchmark(args):
    m = get_legate_runtime().machine

    if m.count(TaskTarget.GPU) == 0:
        processors = m.only(TaskTarget.CPU)
        print("No GPUs found. Running on {} CPUs.".format(len(processors)))
        processor_kind = "CPU"
    else:
        processors = m.only(TaskTarget.GPU)
        print("Running on {} GPUs.".format(len(processors)))
        processor_kind = "GPU"

    df = pd.DataFrame(
        {
            "n_processors": pd.Series([], dtype="int"),
            "time": pd.Series([], dtype="float"),
            "iteration": pd.Series([], dtype="int"),
        }
    )

    if args.logscale:
        n_processors = [2**i for i in range(0, len(processors).bit_length() - 1)]
        if n_processors[-1] != len(processors):
            n_processors.append(len(processors))
    else:
        n_processors = range(1, len(processors) + 1)

    model_types = args.model_types.split(",")

    for n in tqdm(n_processors):
        with processors[:n]:
            gen = cn.random.Generator(cn.random.XORWOW(seed=42))
            rows = args.nrows if args.strong_scaling else args.nrows * n
            X = gen.normal(size=(rows, args.ncols), dtype=cn.float32)
            y = gen.integers(0, args.nclasses, size=X.shape[0], dtype=cn.int32)
            # dry run
            n_dry_run = X.shape[0] // 10
            train_model(X[0:n_dry_run], y[0:n_dry_run], "tree")
            for model_type in model_types:
                for j in range(args.repeats):
                    start = time.time()
                    train_model(X, y, model_type)
                    elapsed = time.time() - start
                    df = df.append(
                        {
                            "n_processors": n,
                            "time": elapsed,
                            "iteration": j,
                            "Model type": model_type,
                        },
                        ignore_index=True,
                    )
    plot(df, args, processor_kind)


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def plot(df, args, processor_kind):
    sns.lineplot(x="n_processors", y="time", data=df, hue="Model type")
    plt.title(
        "{} scaling {} rows x {} cols".format(
            "Strong" if args.strong_scaling else "Weak",
            human_format(args.nrows),
            human_format(args.ncols),
        )
    )
    if args.logscale:
        plt.xscale("log", base=2)
    plt.xlabel("Number of {}s".format(processor_kind))
    plt.ylabel("Time (s)")
    plt.ylim(ymin=0)
    if args.output == "":
        image_dir = Path(__file__).parent
        plt.savefig(image_dir / "{}_scaling.png".format(processor_kind.lower()))
    else:
        plt.savefig(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nrows", type=int, default=100000, help="Number of dataset rows"
    )
    parser.add_argument(
        "--ncols", type=int, default=100, help="Number of dataset columns"
    )
    parser.add_argument(
        "--nclasses",
        type=int,
        default=2,
        help="Number of classes for classificationd dataset.",
    )
    parser.add_argument(
        "--logscale",
        default=False,
        action="store_true",
        help="Plot the number of processors on a log scale e.g. 1, 2, 4, 8",
    )
    parser.add_argument(
        "--strong_scaling",
        default=False,
        action="store_true",
        help="Plot strong scaling instead of weak scaling. "
        "The dataset size remains constant with the number of processors.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat the experiment."
        " Generates error bars on output plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file name. Automatically generated if not specified.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default="tree,linear,krr",
        help="Comma separated list of base model types."
        " Can be 'tree', 'linear', 'krr'.",
    )
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
