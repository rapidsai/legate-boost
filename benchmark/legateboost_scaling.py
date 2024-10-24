import common

import cunumeric as cn
import legateboost as lb
from legate.core import get_legate_runtime


def train_model(X, y, model_type, args, dry_run=False):
    if model_type == "tree":
        base_models = (lb.models.Tree(max_depth=2 if dry_run else 8),)
    elif model_type == "linear":
        base_models = (lb.models.Linear(solver="lbfgs"),)
    elif model_type == "krr":
        base_models = (lb.models.KRR(sigma=1.0, n_components=2 if dry_run else 50),)
    elif model_type == "nn":
        base_models = (lb.models.NN(alpha=0.0, verbose=1),)
    eval_result = {}
    model = lb.LBClassifier(
        base_models=base_models,
        n_estimators=2 if dry_run else args.niters,
        verbose=True,
    ).fit(X, y, eval_result=eval_result)
    train_logloss = eval_result["train"]["log_loss"][-1]
    get_legate_runtime().issue_execution_fence(block=True)
    del model
    return train_logloss


def create_dataset(args):
    n_processors = len(get_legate_runtime().machine)
    gen = cn.random.Generator(cn.random.XORWOW(seed=42))
    rows = args.nrows if args.strong_scaling else args.nrows * n_processors
    X = gen.normal(size=(rows, args.ncols), dtype=cn.float32)
    X[:, args.ncols // 2 :] = gen.binomial(1, 0.5, size=(rows, args.ncols // 2)).astype(
        cn.float32
    )
    y = gen.integers(0, args.nclasses, size=X.shape[0], dtype=cn.int32)
    get_legate_runtime().issue_execution_fence(block=True)
    return X, y


def benchmark(args):
    n_processors = len(get_legate_runtime().machine)

    X, y = create_dataset(args)
    common.run_experiment(X, y, args, train_model, n_processors, "legate-boost")


if __name__ == "__main__":
    parser = common.argparser()
    benchmark(parser.parse_args())
