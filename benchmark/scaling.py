import argparse
import time

import pandas as pd

import cunumeric as cn
import legateboost as lb
from legate.core import get_legate_runtime


def train_model(X, y, model_type, args):
    if model_type == "tree":
        base_models = (lb.models.Tree(max_depth=8),)
    elif model_type == "linear":
        base_models = (lb.models.Linear(solver="lbfgs"),)
    elif model_type == "krr":
        base_models = (lb.models.KRR(sigma=1.0, n_components=50),)
    model = lb.LBClassifier(base_models=base_models, n_estimators=args.niters).fit(X, y)
    # force legate to realise result
    x = model.predict(X[0:2])[0]  # noqa
    del model


def benchmark(args):
    model_types = args.model_types.split(",")

    m = get_legate_runtime().machine
    n_processors = len(m)
    gen = cn.random.Generator(cn.random.XORWOW(seed=42))
    rows = args.nrows if args.strong_scaling else args.nrows * n_processors
    X = gen.normal(size=(rows, args.ncols), dtype=cn.float32)
    y = gen.integers(0, args.nclasses, size=X.shape[0], dtype=cn.int32)
    # dry run
    n_dry_run = X.shape[0] // 10
    for model_type in model_types:
        train_model(X[0:n_dry_run], y[0:n_dry_run], model_type, args)
    dfs = []
    for model_type in model_types:
        for j in range(args.repeats):
            start = time.time()
            train_model(X, y, model_type, args)
            elapsed = time.time() - start
            dfs.append(
                pd.DataFrame(
                    {
                        "n_processors": n_processors,
                        "time": elapsed,
                        "iteration": j,
                        "Model type": model_type,
                        "nrows": args.nrows,
                        "ncols": args.ncols,
                    },
                    index=[0],
                )
            )
    df = pd.concat(dfs, ignore_index=True)
    print(df)
    df.to_csv(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nrows", type=int, default=100000, help="Number of dataset rows"
    )
    parser.add_argument(
        "--ncols", type=int, default=100, help="Number of dataset columns"
    )
    parser.add_argument(
        "--niters", type=int, default=100, help="Number of boosting iterations"
    )
    parser.add_argument(
        "--nclasses",
        type=int,
        default=2,
        help="Number of classes for classificationd dataset.",
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
