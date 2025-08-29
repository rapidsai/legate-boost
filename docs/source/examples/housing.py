# [Import libraries]
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import cupynumeric as cn
import legateboost as lb
from legate.timing import time

# [Import data]
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# [Create and fit Legate Boost regressor]
model = lb.LBRegressor(
    n_estimators=100,
    base_models=(lb.models.Tree(max_depth=5),),
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
