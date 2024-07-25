"""
Example of training with Dask on GPU
====================================
"""

from dask import array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from xgboost import dask as dxgb


def using_quantile_device_dmatrix(client: Client, X: da.Array, y: da.Array) -> da.Array:
    clf = dxgb.DaskXGBClassifier(tree_method="hist", max_depth=8)
    clf.client = client
    clf.fit(X, y, eval_set=[(X, y)])


if __name__ == "__main__":
    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker process.
    with LocalCUDACluster(n_workers=8, threads_per_worker=4) as cluster:
        # Create client from cluster, set the backend to GPU array (cupy).
        with Client(cluster) as client:
            # Generate some random data for demonstration
            rng = da.random.default_rng(1)
            X = rng.normal(size=(1000000, 100), chunks=(128**2, -1))
            y = rng.normal(size=(1000000, 1), chunks=(128**2, 1))
            using_quantile_device_dmatrix(client, X, y)
