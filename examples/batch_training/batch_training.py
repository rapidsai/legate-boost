from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_even_slices

import cunumeric as cn
import legateboost as lb

sns.set()
plt.rcParams["font.family"] = "serif"

total_estimators = 100
n_batches = 10
estimators_per_batch = 5

random_state = 23
training_params = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "verbose": True,
    "init": "average",
    "random_state": random_state,
}

# Fetch the data, create a train test split and convert to cunumeric arrays
X, y = fetch_openml(name="year", version=1, return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
X_train, X_test, y_train, y_test = (
    cn.array(X_train),
    cn.array(X_test),
    cn.array(y_train),
    cn.array(y_test),
)

# Single batch training
eval_result = {}
single_batch_model = lb.LBRegressor(
    **training_params, n_estimators=total_estimators
).fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_result=eval_result)
single_batch_train_error = eval_result["train"]["mse"]
single_batch_test_error = eval_result["eval-0"]["mse"]


# Multi batch training
train_batches = [
    (cn.array(X_train[i]), cn.array(y_train[i]))
    for i in gen_even_slices(X_train.shape[0], n_batches)
]
train_error = []
test_error = []
multi_batch_model = lb.LBRegressor(**training_params, n_estimators=estimators_per_batch)
for i in range(total_estimators // estimators_per_batch):
    X_batch, y_batch = train_batches[i % n_batches]
    eval_result = {}
    multi_batch_model.partial_fit(
        X_batch,
        y_batch,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_result=eval_result,
    )
    train_error += eval_result["eval-0"]["mse"]
    test_error += eval_result["eval-1"]["mse"]

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
sns.lineplot(
    x=range(total_estimators),
    y=cn.sqrt(single_batch_train_error),
    ax=ax[0],
    errorbar=None,
    label="1-batch",
)
sns.lineplot(
    x=range(total_estimators),
    y=cn.sqrt(train_error),
    ax=ax[0],
    errorbar=None,
    label="{}-batch".format(n_batches),
)
ax[0].set_title("Train Error")
ax[0].set_xlabel("Iterations")
ax[0].set_ylabel("RMSE")
sns.lineplot(
    x=range(total_estimators),
    y=cn.sqrt(single_batch_test_error),
    ax=ax[1],
    errorbar=None,
    label="1-batch",
)
sns.lineplot(
    x=range(total_estimators),
    y=cn.sqrt(test_error),
    ax=ax[1],
    errorbar=None,
    label="{}-batch".format(n_batches),
)
ax[1].set_title("Test Error")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("RMSE")

plt.suptitle("Year Prediction MSD Dataset - Batch Training")
plt.tight_layout()
image_dir = Path(__file__).parent
plt.savefig(image_dir / "batch_training.png")
