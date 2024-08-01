import common
import numpy as np
from dask import array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from xgboost import dask as dxgb


def train_model(X: da.Array, y: da.Array, model_type, args, dry_run=False):
    n_estimators = 2 if dry_run else args.niters
    if model_type != "tree":
        raise ValueError("Only trees are supported for dask-xgb")
    max_depth = 2 if dry_run else 8
    clf = dxgb.DaskXGBClassifier(
        learning_rate=0.1,
        tree_method="hist",
        device="cuda",
        max_depth=max_depth,
        n_estimators=n_estimators,
    )
    clf.fit(X, y, eval_set=[(X, y)])
    train_logloss = clf.evals_result()["validation_0"]["logloss"][-1]
    return train_logloss


def create_dataset(args):
    n_processors = args.gpus
    rng = da.random.default_rng(42)
    rows = args.nrows if args.strong_scaling else args.nrows * n_processors
    chunk_size = np.ceil(rows / n_processors)
    X = rng.normal(size=(rows, args.ncols), chunks=(chunk_size, -1))
    y = rng.integers(
        0, args.nclasses, size=X.shape[0], chunks=(chunk_size), dtype=np.int32
    )
    return X, y


def benchmark(args):
    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker process.
    with LocalCUDACluster(n_workers=args.gpus, threads_per_worker=4) as cluster:
        # Create client from cluster, set the backend to GPU array (cupy).
        with Client(cluster):
            # Generate some random data for demonstration
            X, y = create_dataset(args)
            common.run_experiment(X, y, args, train_model, args.gpus, "dask-xgb")


if __name__ == "__main__":
    parser = common.argparser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    benchmark(parser.parse_args())
