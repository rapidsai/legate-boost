import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import legateboost as lb

rng = np.random.RandomState(2)
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=2,
    n_redundant=10,
    random_state=rng,
    flip_y=0.5,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

# example of out of core training as an alternative to the batch training example
# train an entire model on each batch of data and combine them
pivot = X_train.shape[0] // 2
X_train_a, X_train_b = X_train[:pivot], X_train[pivot:]
y_train_a, y_train_b = y_train[:pivot], y_train[pivot:]

model_a = lb.LBClassifier(random_state=rng).fit(X_train_a, y_train_a)
model_b = lb.LBClassifier(random_state=rng).fit(X_train_b, y_train_b)

# combine models
model_c = (model_a + model_b) * 0.5

# gradient boosting models are additive but only
# before the nonlinear link function is applied
assert np.allclose(
    model_c.predict_raw(X_test),
    (model_a.predict_raw(X_test) + model_b.predict_raw(X_test)) * 0.5,
)
assert not np.allclose(
    model_c.predict_proba(X_test),
    (model_a.predict_proba(X_test) + model_b.predict_proba(X_test)) * 0.5,
)

# the number of estimators in the combined model is the sum
# of the number of estimators in the two models
assert len(model_c) == len(model_a) + len(model_b)

# the model initialisation has also been added together
assert np.allclose(
    model_c.model_init_, (model_a.model_init_ + model_b.model_init_) * 0.5
)

# it is illegal to combine models when:
# - the number of features in the two models is different
# - the models are of different types (regressor vs classifier)
# - the number of classes in the two models is different
# - models are trained on different objective functions
model_d = lb.LBClassifier(random_state=rng).fit(X_train_a[:, :-1], y_train_a)
try:
    model_c + model_d
except ValueError as e:
    print(e)
    pass
