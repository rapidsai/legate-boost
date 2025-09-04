# [import libraries]
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from joblib import dump
from legate_dataframe.lib.core.column import LogicalColumn
from legate_dataframe.lib.core.table import LogicalTable
from legate_dataframe.lib.replace import replace_nulls
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

import legate.core as lg
import legateboost as lb
from legate.timing import time

rt = lg.get_legate_runtime()

# [import data]
data = fetch_openml(data_id=46929, as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

if os.environ.get("CI"):
    df = df.sample(n=100, random_state=42).reset_index(drop=True)

# [convert to LogicalTable]
df = pa.Table.from_pandas(df)
ldf = LogicalTable.from_arrow(df)
# [covert to LogicalTable end]

# [Replace nulls]
median_salary = df["MonthlyIncome"].median()
median_dependents = df["NumberOfDependents"].median()

mmi = LogicalColumn(replace_nulls(LogicalColumn(ldf["MonthlyIncome"]), median_salary))
mnd = LogicalColumn(
    replace_nulls(LogicalColumn(ldf["NumberOfDependents"]), median_dependents)
)

# [Create new LogicalTable with updated columns]

features = ldf.get_column_names()
nldf = LogicalTable(
    [ldf[0], ldf[1], ldf[2], ldf[3], mmi, ldf[5], ldf[6], ldf[7], ldf[8], mnd, ldf[10]],
    features,
)

# [Convert to cupynumeric array]

data_arr = nldf.to_array()

# [convert to cupynumeric array end]

# [preparing data for training and testing]
x = data_arr[:, :-1]  # all columns except last
y = data_arr[:, -1]

# [Splitting the data into training and testing]
num_samples = x.shape[0]
split_ratio = 0.8
split_index = int(num_samples * split_ratio)

x_train = x[:split_index]
y_train = y[:split_index]
x_test = x[split_index:]
y_test = y[split_index:]

# [training]
rt.issue_execution_fence()
start = time()
nn_iter = 2 if os.environ.get("CI") else 10
hidden_layers = (2, 2) if os.environ.get("CI") else (10, 10)

model = lb.LBClassifier(
    base_models=(
        lb.models.Tree(max_depth=5),
        lb.models.NN(max_iter=nn_iter, hidden_layer_sizes=hidden_layers, verbose=True),
    )
).fit(x_train, y_train)

rt.issue_execution_fence()
end = time()
# [training end]

# [Prediction]
predictions = model.predict(x_test)
print(type(predictions))

# [Evaluation]
acc = accuracy_score(y_test, predictions)
print("Accuracy:", acc)
print(f"\nThe training time for creditscore exp is: {(end - start)/1000:.6f} ms")

# [Save model]
dump(model, "legate_boost_model.joblib")

# [Save test data]
x_test_cpu = x_test.get() if hasattr(x_test, "get") else np.array(x_test)
y_test_cpu = y_test.get() if hasattr(y_test, "get") else np.array(y_test)

pd.DataFrame(x_test_cpu).to_csv("x_test.csv", index=False)
pd.DataFrame(y_test_cpu, columns=["Target"]).to_csv("y_test.csv", index=False)
