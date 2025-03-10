import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import legateboost as lb

wine_reviews = fetch_openml(data_id=42074, as_frame=True)

df = wine_reviews.frame
numerical_features = ["price"]
categorical_features = [
    "country",
    "province",
    "region_1",
    "region_2",
    "variety",
    "winery",
]
target_name = "points"
df = df.fillna(-1)
df[categorical_features] = df[categorical_features].apply(
    lambda x: x.astype("category").cat.codes
)

X = df[numerical_features + categorical_features].to_numpy()
y = df[target_name].to_numpy()

encoded_categoricals = lb.encoder.TargetEncoder(target_type="continuous").fit_transform(
    df[categorical_features].to_numpy(), y
)
X_encoded = df[numerical_features].to_numpy()
X_encoded = np.concatenate((X_encoded, encoded_categoricals), axis=1)

# test train splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_encoded_train, X_encoded_test, y_encoded_train, y_encoded_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

unencoded = lb.LBRegressor().fit(X_train, y_train)
encoded = lb.LBRegressor().fit(X_encoded_train, y_encoded_train)

print("Unencoded score:", unencoded.score(X_test, y_test))
print("Encoded score:", encoded.score(X_encoded_test, y_encoded_test))
