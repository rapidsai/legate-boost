import cudf
import pandas
import cupy as cp
import pyarrow as pa
import legate_dataframe
import legateboost as lb
import cupynumeric as cpn
from legate.timing import time
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from legate_dataframe.lib.replace import replace_nulls
from legate_dataframe.lib.core.table import LogicalTable
from legate_dataframe.lib.core.column import LogicalColumn

rt = lg.get_legate_runtime()

# import data
data = fetch_openml(data_id= 46929, as_frame=True)
xd = cudf if cp.cuda.runtime.getDeviceCount() > 0 else pandas
df = xd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# covert to logicalTable
if cp.cuda.runtime.getDeviceCount() > 0:
	ldf = LogicalTable.from_cudf(df)
else:
	df = pa.Table.from_pandas(df)
	ldf =  LogicalTable.from_arrow(df)

#replace nulls
median_salary = df["MonthlyIncome"].median()
median_dependents =  df["NumberOfDependents"].median()
mmi = LogicalColumn(replace_nulls(LogicalColumn(ldf["MonthlyIncome"]),median_salary))
mnd = LogicalColumn(replace_nulls(LogicalColumn(ldf["NumberOfDependents"]),median_dependents))

#create a new logical Table with updated columns
features = ldf.get_column_names()

nldf = LogicalTable( [ ldf[0], ldf[1], ldf[2], ldf[3], mmi, ldf[5], ldf[6], ldf[7], ldf[8], mnd, ldf[10]], features)

# covert to cupynumeric array
data_arr = nldf.to_array()
print(type(data_arr))
#print(f"\n The dataprocessing time for creditscore exp is: {(e - st)/1000:.6f}ms")

x = data_arr[:, :-1]   # all columns except last
y = data_arr[:, -1]

#splitting the data into training and testing
num_samples = x.shape[0]
split_ratio = 0.8
split_index = int(num_samples * split_ratio)

x_train = x[:split_index]
y_train = y[:split_index]
x_test = x[split_index:]
y_test = y[split_index:]

rt.issue_execution_fence()
start=time()
# Create model and trian it
model = lb.LBClassifier(base_models=(lb.models.Tree(max_depth=2),lb.models.NN(max_iter=2,hidden_layer_sizes=(10,),verbose=True))).fit(x_train,y_train)
rt.issue_execution_fence()
end=time()

#predict
predictions = model.predict(x_test)
print(type(predictions))

#evalution
acc = accuracy_score(y_test, predictions)
print("Accuracy:", acc)

print(f"\n The training time for creditscore exp is: {(end - start)/1000:.6f}ms")

#model save
dump(model, "legate_boost_model.joblib")

x_test_cpu = x_test.get() if hasattr(x_test, "get") else np.array(x_test)
y_test_cpu = y_test.get() if hasattr(y_test, "get") else np.array(y_test)

# Save as CSV for easy inspection
pd.DataFrame(x_test_cpu).to_csv("x_test.csv", index=False)
pd.DataFrame(y_test_cpu, columns=["Target"]).to_csv("y_test.csv", index=False)
