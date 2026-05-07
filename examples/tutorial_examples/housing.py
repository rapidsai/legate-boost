# [Import libraries]
import os

from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import legateboost as lb
from legate.timing import time

# [Import data]
if os.environ.get("CI"):
    X, y = make_regression(n_samples=100, n_features=5, n_targets=1, random_state=42)
    total_estimators = 10
else:
    data = fetch_california_housing()
    X, y = data.data, data.target
    total_estimators = 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# [Create and fit Legate Boost regressor]
model = lb.LBRegressor(
    n_estimators=total_estimators,
    base_models=(lb.models.Tree(max_depth=8),),
    objective="squared_error",
    learning_rate=0.1,
    verbose=True,
)

start = time()
model.fit(X_train, y_train)
end = time()

# [Prediction]
istart = time()
y_pred = model.predict(X_test)
iend = time()

# [Evaluation]
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
print(f"\nThe training time for housing exp is: {(end - start)/1000:.6f} ms")
print(f"\nThe inference time for housing exp is {(iend - istart)/1000:.6f} ms")
