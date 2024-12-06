import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import legateboost as lb

# Alberto, T. & Lochter, J. (2015). YouTube Spam Collection [Dataset].
# UCI Machine Learning Repository. https://doi.org/10.24432/C58885.
dataset_names = [
    "youtube-spam-psy",
    "youtube-spam-shakira",
    "youtube-spam-lmfao",
    "youtube-spam-eminem",
    "youtube-spam-katyperry",
]
X = []
for dataset_name in dataset_names:
    dataset = fetch_openml(name=dataset_name, as_frame=True)
    X.append(dataset.data)

X = pd.concat(X)
y = X["CLASS"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train["CONTENT"])
X_test_vectorized = vectorizer.transform(X_test["CONTENT"])

model = lb.LBClassifier().fit(
    X_train_vectorized, y_train, eval_set=[(X_test_vectorized, y_test)]
)


def evaluate_comment(comment):
    print("Comment: {}".format(comment))
    print(
        "Probability of spam: {}".format(
            model.predict_proba(vectorizer.transform([comment]))[0, 1]
        )
    )


evaluate_comment(X_test.iloc[15]["CONTENT"])
evaluate_comment(X_test.iloc[3]["CONTENT"])
evaluate_comment("Your text here")
