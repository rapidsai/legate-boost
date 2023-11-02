from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import legateboost as lb

sns.set()
plt.rcParams["font.family"] = "serif"

X, y = fetch_openml("wine-quality-white", return_X_y=True, as_frame=False)
y = y.astype(np.float64)
X = StandardScaler().fit_transform(X.astype(np.float64))
n_folds = 5

params = {
    "n_estimators": 100,
    "learning_rate": 0.3,
    "random_state": 98,
    "verbose": True,
}
krr = lb.models.KRR(n_components=100, sigma=0.5)
tree = lb.models.Tree(max_depth=5)
models = {
    "KRR": lb.LBRegressor(base_models=(krr,), **params),
    "Tree": lb.LBRegressor(base_models=(tree,), **params),
    "KRR + Tree": lb.LBRegressor(base_models=(krr, tree), **params),
}
error = {m: [] for m in models}
for fold in range(n_folds):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=fold
    )
    for name, model in models.items():
        eval_result = {}
        model.fit(
            X_train, y_train, eval_set=[(X_test, y_test)], eval_result=eval_result
        )
        error[name].append(eval_result["eval-0"]["mse"])

df = pd.DataFrame(columns=["Algorithm", "Iteration", "Fold", "Test MSE"])
for name, model in models.items():
    for i in range(n_folds):
        n = len(error[name][i])
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "Algorithm": pd.Series(name, index=range(n)),
                        "Iteration": pd.Series(range(n)),
                        "Fold": pd.Series(i, index=range(n)),
                        "Test MSE": pd.Series(error[name][i]),
                    }
                ),
            ],
        )
sns.lineplot(data=df, x="Iteration", y="Test MSE", hue="Algorithm")
plt.suptitle("KRR Models + Tree Models")
plt.tight_layout()
image_dir = Path(__file__).parent
plt.savefig(image_dir / "kernel_ridge_regression.png")
