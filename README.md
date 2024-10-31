# legate-boost

GBM implementation on Legate. The primary goals of `legate-boost` is to provide a state-of-the-art distributed GBM implementation on Legate, capable of running on CPUs or GPUs at supercomputer scale.

[API Documentation](https://rapidsai.github.io/legate-boost)

For developers - see [contributing](contributing.md)

## Example

Run with the legate launcher
```bash
legate example_script.py
```

```python
import cunumeric as cn
import legateboost as lb

X = cn.random.random((1000, 10))
y = cn.random.random(X.shape[0])
model = lb.LBRegressor(verbose=1, n_estimators=100, random_state=0, max_depth=2).fit(
    X, y
)
```

## Features

### Probabilistic regression

`legate-boost` can learn distributions for continuous data. This is useful in cases where simply predicting the mean does not carry enough information about the training data:

<img src="examples/probabalistic_regression/probabilistic_regression.gif" alt="drawing" width="800"/>

The above example can be found here: [examples/probabilistic_regression](https://github.com/rapidsai/legate-boost/tree/main/examples/probabalistic_regression).

### Batch training

`legate-boost` can train on datasets that do not fit into memory by splitting the dataset into batches and training the model with `partial_fit`.

```python
total_estimators = 100
model = lb.LBRegressor(n_estimators=estimators_per_batch)
for i in range(total_estimators // estimators_per_batch):
    X_batch, y_batch = train_batches[i % n_batches]
    model.partial_fit(
        X_batch,
        y_batch,
    )
```

<img src="examples/batch_training/batch_training.png" alt="drawing" width="600"/>

The above example can be found here: [examples/batch_training](https://github.com/rapidsai/legate-boost/tree/main/examples/batch_training).

### Different model types

`legate-boost` supports tree models, linear models, kernel ridge regression models, custom user models and any combinations of these models.

The following example shows a model combining linear and decision tree base learners on a synthetic dataset.

```python
model = lb.LBRegressor(base_models=(lb.models.Linear(), lb.models.Tree(max_depth=1),), **params).fit(X, y)
```

<img src="examples/linear_model/linear_model.png" alt="drawing" width="800"/>

The second example shows a model combining kernel ridge regression and decision tree base learners on the wine quality dataset.

```python
model = lb.LBRegressor(base_models=(lb.models.KRR(sigma=0.5), lb.models.Tree(max_depth=5),), **params).fit(X, y)
```

<img src="examples/kernel_ridge_regression/kernel_ridge_regression.png" alt="drawing" width="400"/>

## Installation

Install using `conda`.

```shell
# stable release
conda install -c legate -c conda-forge -c nvidia legate-boost

# nightly release
conda install -c legate/label/experimental -c legate -c conda-forge -c nvidia legate-boost
```

On systems without a GPU, the CPU-only package should automatically be installed.
On systems with a GPU and compatible CUDA version, the GPU package should automatically be installed.

To force `conda` to prefer one, pass the build strings `*_cpu*` or `*_gpu*`, for example:

```shell
# nightly release (CPU-only)
conda install --dry-run -c legate/label/experimental -c legate -c conda-forge -c nvidia \
    'legate-boost=*=*_cpu*'
```

For more details on building from source and setting up a development environment, see [`contributing.md`](./contributing.md).
