from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import legateboost as lb

# creating synthetic dataset
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# classification model with 50 estimators
classification_model = lb.LBClassifier(n_estimators=50)

# train the model
classification_model.fit(X_train, y_train)

# predictions
y_pred = classification_model.predict(X_test)
print(y_pred)
