from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation
from scipy.stats import norm
from sklearn.datasets import fetch_california_housing

import legateboost as lb

sns.set()
plt.rcParams["font.family"] = "serif"

rs = np.random.RandomState(0)
X, y = fetch_california_housing(return_X_y=True, as_frame=False)

feature_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
feature = 6
X = X[:, feature : feature + 1]
n_estimators = 10
n_frames = 40
model = lb.LBRegressor(
    verbose=True,
    init="average",
    max_depth=2,
    n_estimators=n_estimators,
    learning_rate=0.1,
    random_state=rs,
    objective="normal",
)
preds = [model.partial_fit(X, y).predict(X) for _ in range(n_frames)]

sample = rs.choice(X.shape[0], 1000, replace=False)
X_test = X[sample]
y_test = y[sample]
feature_name = feature_names[feature]
fig, ax = plt.subplots()


def animate(i):
    ax = fig.axes[0]
    ax.cla()
    ax.set_title(
        "Distribution of House Values: Boosting iterations {}".format(
            (i + 1) * n_estimators
        )
    )
    data = pd.DataFrame(
        {
            feature_name: X_test[:, 0],
            "y": y_test,
            "Predicted house value": preds[i][sample, 0],
            "var": preds[i][sample, 1],
        }
    ).sort_values(by=feature_name)
    ax = sns.lineplot(x=feature_name, y="Predicted house value", data=data, ax=ax)
    interval = norm.interval(
        0.95, loc=data["Predicted house value"], scale=data["var"] ** 0.5
    )
    ax.fill_between(data[feature_name], interval[0], interval[1], alpha=0.2)
    ax.set_ylim(-0.5, 5.5)

    sns.scatterplot(
        x=feature_name, y="y", data=data, ax=ax, s=15, color=".2", alpha=0.2
    )
    plt.tight_layout()
    if i == 0:
        plt.savefig("probabilistic_regression.png")
    return (fig,)


anim = animation.FuncAnimation(fig, animate, frames=len(preds), interval=500, blit=True)

image_dir = Path(__file__).parent / "images"
anim.save(image_dir / "probabilistic_regression.gif")
