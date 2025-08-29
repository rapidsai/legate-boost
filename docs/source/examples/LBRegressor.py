import legateboost as lb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# creating synthetic dataset
X, y = make_regression(n_samples=100, n_features=4, noise=8, random_state=42)

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# regression model with 100 estimators
regression_model = lb.LBRegressor(n_estimators=100)

# fit the model
regression_model.fit(X_train, y_train)

# predict
y_pred = regression_model.predict(X_test)

print(y_pred)
