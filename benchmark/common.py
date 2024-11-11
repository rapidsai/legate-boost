import argparse
import time

import pandas as pd


def argparser():
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
        default="scaling.csv",
        help="Output file name. Automatically generated if not specified.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default="tree",
        help="Comma separated list of base model types."
        " Can be 'tree', 'linear', 'krr', 'nn'.",
    )
    parser.add_argument(
        "--proportion_numerical",
        type=float,
        default=0.5,
        help="Proportion of numerical features"
        + " where the remaining features will be binary.",
    )
    return parser


def run_experiment(X, y, args, train_function, n_processors, name):
    model_types = args.model_types.split(",")
    # dry run / limit models instead of data - prevent data shuffeling
    for model_type in model_types:
        train_function(X, y, model_type, args, True)

    dfs = []
    for model_type in model_types:
        for j in range(args.repeats):
            start = time.time()
            logloss = train_function(X, y, model_type, args, False)
            elapsed = time.time() - start
            dfs.append(
                pd.DataFrame(
                    {
                        "n_processors": n_processors,
                        "time": elapsed,
                        "iteration": j,
                        "Model type": name + "-" + model_type,
                        "nrows": X.shape[0],
                        "ncols": args.ncols,
                        "train_logloss": logloss,
                    },
                    index=[0],
                )
            )
    df = pd.concat(dfs, ignore_index=True)
    print(df)
    df.to_csv(args.output)
