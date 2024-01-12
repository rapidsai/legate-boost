from __future__ import annotations

import os
from pathlib import Path
import cunumeric as cn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation
from matplotlib.figure import Figure
from scipy.stats import gamma, norm
from sklearn.datasets import fetch_california_housing

import legateboost as lb

sns.set()
plt.rcParams["font.family"] = "serif"

rs = np.random.RandomState(0)
X, y = fetch_california_housing(return_X_y=True, as_frame=False)

feature_name = "Latitude"
X = X[:, 6:7]
sample = rs.choice(X.shape[0], 1000, replace=False)
X_test = X[sample]
y_test = y[sample]
n_estimators = 10
n_frames = 2 if os.environ.get("CI") else 40


def fit_normal_distribution() -> tuple[lb.LBRegressor, list[cn.ndarray]]:
    obj = lb.NormalObjective()
    model = lb.LBRegressor(
        verbose=True,
        init="average",
        base_models=(lb.models.Tree(max_depth=2),),
        n_estimators=n_estimators,
        learning_rate=0.1,
        random_state=rs,
        objective=obj,
    )
    return model, [model.partial_fit(X, y).predict(X_test) for _ in range(n_frames)]


def fit_gamma_distribution() -> tuple[lb.LBRegressor, list[cn.ndarray]]:
    obj = lb.GammaObjective()
    model = lb.LBRegressor(
        verbose=True,
        init="average",
        base_models=(lb.models.Tree(max_depth=2),),
        n_estimators=n_estimators,
        learning_rate=0.1,
        random_state=rs,
        objective=obj,
    )
    return model, [model.partial_fit(X, y).predict(X_test) for _ in range(n_frames)]


quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def fit_quantile_regression() -> tuple[lb.LBRegressor, list]:
    model = lb.LBRegressor(
        verbose=True,
        base_models=(lb.models.Tree(max_depth=2),),
        n_estimators=n_estimators,
        learning_rate=0.1,
        random_state=rs,
        objective=lb.QuantileObjective(quantiles),
    )
    return model, [model.partial_fit(X, y).predict(X_test) for _ in range(n_frames)]


normal_model, normal_preds = fit_normal_distribution()
gamma_mode, gamma_preds = fit_gamma_distribution()
quantile_model, quantile_preds = fit_quantile_regression()

fig, ax = plt.subplots(1, 3, figsize=(12, 6))


def animate(i: int) -> tuple[Figure]:
    lower, upper = -0.5, 6.5

    fig.suptitle(
        "Distribution of House Values: Boosting iterations {}".format(
            (i + 1) * n_estimators
        )
    )

    # Plot the normal distribution
    ax[0].cla()
    ax[0].set_title("Normal Distribution - 95% Confidence Interval")
    norm_obj = lb.NormalObjective()
    data = pd.DataFrame(
        {
            feature_name: X_test[:, 0],
            "y": y_test,
            "Predicted house value": norm_obj.mean(normal_preds[i]),
            "sigma": norm_obj.var(normal_preds[i]),
        }
    ).sort_values(by=feature_name)
    sns.lineplot(
        x=feature_name,
        y="Predicted house value",
        data=data[[feature_name, "Predicted house value"]],
        ax=ax[0],
        errorbar=("sd", 0),
    )
    interval = norm.interval(
        0.95, loc=data["Predicted house value"], scale=data["sigma"]
    )
    ax[0].fill_between(data[feature_name], interval[0], interval[1], alpha=0.2)
    ax[0].set_ylim(lower, upper)

    sns.scatterplot(
        x=feature_name, y="y", data=data, ax=ax[0], s=15, color=".2", alpha=0.2
    )

    # Plot the gamma distribution
    ax[1].cla()
    ax[1].set_title("Gamma Distribution - 95% Confidence Interval")
    gamma_obj = lb.GammaObjective()
    data = pd.DataFrame(
        {
            feature_name: X_test[:, 0],
            "y": y_test,
            "Predicted house value": gamma_obj.mean(gamma_preds[i]),
            "shape": gamma_obj.shape(gamma_preds[i]),
            "scale": gamma_obj.scale(gamma_preds[i]),
        }
    ).sort_values(by=feature_name)
    sns.lineplot(
        x=feature_name,
        y="Predicted house value",
        data=data[[feature_name, "Predicted house value"]],
        ax=ax[1],
        errorbar=("sd", 0),
    )
    interval = gamma.interval(0.95, data["shape"], scale=data["scale"])
    ax[1].fill_between(data[feature_name], interval[0], interval[1], alpha=0.2)
    ax[1].set_ylim(lower, upper)
    sns.scatterplot(
        x=feature_name, y="y", data=data, ax=ax[1], s=15, color=".2", alpha=0.2
    )

    # Plot the quantile regression
    ax[2].cla()
    ax[2].set_title("Quantile Regression")

    data = {
        feature_name: X_test[:, 0],
        "y": y_test,
    }
    data.update({str(q): quantile_preds[i][:, j] for j, q in enumerate(quantiles)})
    data = pd.DataFrame(data).sort_values(by=feature_name)
    lines = data[[feature_name] + [str(q) for q in quantiles]].melt(
        feature_name, var_name="quantile", value_name="Predicted house value"
    )

    dashes = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 0), (4, 4), (3, 3), (2, 2), (1, 1)]
    sns.lineplot(
        x=feature_name,
        y="Predicted house value",
        data=lines,
        style="quantile",
        ax=ax[2],
        dashes=dashes,
        errorbar=("sd", 0),
    )
    ax[2].set_ylim(lower, upper)

    sns.scatterplot(
        x=feature_name, y="y", data=data, ax=ax[2], s=15, color=".2", alpha=0.2
    )

    plt.tight_layout()
    print("Drawing frame {}".format(i + 1))
    return (fig,)


anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=500)

image_dir = Path(__file__).parent
anim.save(image_dir / "probabilistic_regression.gif", dpi=80)
