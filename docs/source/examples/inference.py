import cupynumeric as cp
from joblib import load
from legate.timing import time
import pandas as pd
import legate.core as lg

rt = lg.get_legate_runtime()

timings = []
model =  load("legate_boost_housing.joblib")
X = pd.read_csv("x_test_housing.csv")

for _ in range(10):
    rt.issue_execution_fence()
    start = time()
    model.predict(X)
    rt.issue_execution_fence()
    end = time()
    timings.append(end - start)

timings = timings[1:]
timings_gpu = cp.array(timings)

mean_time = cp.mean(timings_gpu)
median_time = cp.median(timings_gpu)
min_time = cp.min(timings_gpu)
max_time = cp.max(timings_gpu)
var_time = cp.var(timings_gpu)
std = cp.sqrt(var_time)

print(f"Mean: {float(mean_time)/1000:.2f} ms")
print(f"Median: {float(median_time)/1000:.2f} ms")
print(f"Min: {float(min_time)/1000:.2f} ms")
print(f"Max: {float(max_time)/1000:.2f} ms")
print(f"Variance: {float(var_time)/1000:.2f} ms")
print(f"standard deviation: {float(std)/1000:.2f} ms")
