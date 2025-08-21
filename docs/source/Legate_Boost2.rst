This article assumes familiarity with the basic usage of gradient
boosting libraries such as XGBoost or LightGBM, as well as cuPyNumeric
for GPU-accelerated array computations.

**What is legate boost?**
-------------------------

In scenarios where high-performance training is needed across large
datasets or distributed hardware, Legate Boost offers a scalable
alternative. Legate Boost is an advanced gradient boosting library built
on the Legate and Legion parallel programming frameworks. Unlike
traditional boosting libraries such as XGBoost or LightGBM, Legate Boost
provides a unified infrastructure that seamlessly scales across CPUs and
GPUs, supporting both single-node and distributed training while
integrating naturally with cuPyNumeric workflows for efficient
end-to-end data processing. It enables users to define not only
conventional boosted decision trees but also hybrid ensembles combining
trees, kernel ridge regression, linear models, or neural networks, all
written in Python with minimal code changes.

These models are automatically parallelized and executed efficiently
without manual data movement or partitioning. Legate Boost emphasizes
architectural simplicity, extensibility, and performance, delivering
state-of-the-art results on tabular data while leveraging the full
computational power of modern heterogeneous hardware.

Please refer to `Distributed computing
cuPyNumeric <https://docs.nvidia.com/cupynumeric/latest/user/tutorial.html>`__
and `Legate
boost <https://github.com/rapidsai/legate-boost/tree/main>`__ for more
information and detailed instructions on installation.

**Quick installation and setup**
--------------------------------

+--------------------------------------------------------------------------------------------+
| # create a new env and install legate boost and dependencies                               |
|                                                                                            |
| $ conda create -n legate-boost -c legate -c conda-forge -c nvidia legate-boost             |
|                                                                                            |
| # activate env                                                                             |
|                                                                                            |
| $ conda activate legate-boost                                                              |
|                                                                                            |
| # install wrappers                                                                         |
|                                                                                            |
| | $ conda install -c legate/label/gex-experimental realm-gex-wrapper legate-mpi-wrapper    |
| | # install cmake                                                                          |
|                                                                                            |
| $ conda install -c conda-forge cmake>=3.26.4                                               |
|                                                                                            |
| # build wrappers                                                                           |
|                                                                                            |
| $                                                                                          |
| /global/homes/**n/ngraddon**/.conda/envs/legate-boost-new/mpi-wrapper/build-mpi-wrapper.sh |
|                                                                                            |
| $                                                                                          |
| /global/homes/**n/ngraddon**/.conda/envs/legate-boost-new/gex-wrapper/build-gex-wrapper.sh |
|                                                                                            |
| # reactivate env                                                                           |
|                                                                                            |
| $ conda activate legate-boost                                                              |
|                                                                                            |
| # install legate-dataframe                                                                 |
|                                                                                            |
| $ conda install -c legate -c rapidsai -c conda-forge legate-dataframe                      |
+============================================================================================+

**Usage**
---------

Legate Boost offers two main estimator types:

- LBRegressor for regression tasks

- LBClassifier for classification tasks

These estimators follow a similar interface to those in **XGboost**,
making them easy to integrate into existing machine learning pipelines.

**Regression with LBRegressor**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LBRegressor estimator is used to predict continuous values such as
house prices, temperature, or sales forecasting. The following code
demonstrates how to create an instance of the LBRegressor model, use the
fit() function to train it on a dataset, and then apply the predict()
function to generate predictions on new data. Here’s how to set it up:

import legateboost as lb

from sklearn.datasets import make_regression

#creating synthetic dataset

x,y = make_regression(n_samples=100, n_features=4, noise=8,
random_state=42)

#splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(

data.data, data.target, test_size=0.2, random_state=42)

#regression model with 100 estimators

regresion_model = lb.LBRegressor(n_estimators=100)

#fit the model

regression_model.fit(X_train, y_train)

#predict using the trained model

y_pred = regression_model.predict(X_test)



In this example:

- LBRegressor is initialized with 100 boosting estimators.

- The fit() method trains the model using the input features (X_train)
  and target values (y_train).

- After training, the predict() method is used to make predictions on
  the test set (X_test).

This represents a typical workflow for applying a regression model using
Legate Boost. The LBRegressor estimator offers several configurable
options, such as base_model and learning_rate, to help optimize model
performance. For a comprehensive list of features and parameters, refer
to the official
`documentation <https://rapidsai.github.io/legate-boost/api/estimators.html>`__.

**Classification with LBClassifier**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LBClassifier is designed for predicting categorical outcomes and
supports both binary and multi-class classification tasks. It is ideal
for a wide range of applications, including spam detection, image
classification, and sentiment analysis.

The example below demonstrates how to implement a classification model
using the LBClassifier estimator from Legate Boost:

import legateboost as lb

from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split

#creating synthetic dataset

x,y = make_regression(n_samples=100, n_features=4, n_informative=3,
n_redundant=0, n_classes=2, random_state=42)

#splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(

data.data, data.target, test_size=0.2, random_state=42)

# Create a classification model with 50 boosting estimators

classification_model = lb.LBClassifier(n_estimators=50)

# Train the model on labeled data

classification_model.fit(X_train, y_train)

# Predict class labels on new data

y_pred = classification_model.predict(X_test)

In this example:

- LBClassifier(n_estimators=50) sets up a classifier with 50 boosting
  rounds.

- fit(X_train, y_train) learns the patterns from your training dataset.

- predict(X_test) outputs predicted class labels for the test dataset.

Just like the regressor, the LBClassifier follows a clean and intuitive
workflow. It provides additional options and advanced configurations to
optimize model performance. For more detailed information, refer to the
Legate Boost estimators
`documentation <https://rapidsai.github.io/legate-boost/api/estimators.html#legateboost.LBClassifier>`__.

**Example 1**
-------------

Here is an example of using Legate Boost to build a regression model on
the California housing dataset.

It showcases key features like scalable training across GPUs/nodes,
customizable base models, and adjustable learning rates.

**About dataset** 
~~~~~~~~~~~~~~~~~~

The California housing dataset is a classic benchmark dataset containing
information collected from California districts in the 1990 census. Each
record describes a block group (a neighborhood-level area), including
predictors such as:

- Median income of residents

- Average house age

- Average number of rooms and bedrooms

- Population and household count

- Latitude and longitude

The target variable is the **median house value** in that block group.
This dataset is often used to illustrate regression techniques and
assess predictive performance on real-world tabular data.

**About this implementation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code creates a Legate Boost regression model using
LBRegressor, which trains a gradient boosting model optimized for
multi-GPU and multi-node environments. The model is configured to use
100 boosting rounds (n_estimators=100), with each round adding a
decision tree (lb.models.Tree) limited to a maximum depth of 5. The loss
function is set to "squared_error", suitable for regression tasks as it
minimizes mean squared error. A learning_rate of 0.1 controls how much
each tree contributes to the final prediction, balancing speed and
stability. The verbose=True flag enables logging during training,
allowing to monitor progress and internal operations.

**Code module** 
~~~~~~~~~~~~~~~~

import cupynumeric as cn

import legateboost as lb

from legate.timing import time

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing

#load and split data

data = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(

data.data, data.target, test_size=0.2, random_state=42)

#create Legate Boost regressor

model = lb.LBRegressor(n_estimators=100,
base_models=(lb.models.Tree(max_depth=5),),

objective = **"squared_error"**, learning_rate = 0.1, verbose=\ **True**

)

#training

start = time()

model.fit(X_train, y_train)

end = time()

#prediction

istart = time()

y_pred = model.predict(X_test)

iend = time()

#evaluate

mse = mean_squared_error(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")

print(f"\\n The training time for housing exp is: {(end -
start)/1000:.6f}ms")

print(f"\\n The inference time for housing exp is {(iend -
istart)/1000:.6f}")


-

This simple example demonstrates how to train a regression model on the
California Housing dataset using Legate Boost. Although the code looks
similar to standard XGBoost, Legate Boost automatically enables
multi-GPU and multi-node computation. Legate Boost achieves multi-GPU
and multi-node scaling through its integration with cupynumeric and the
Legion runtime. Unlike traditional GPU libraries that allocate data to a
single device, cupynumeric creates Logical Arrays and abstract
representations of the data that are not bound to one GPU. The Legate
automatically partitions these logical arrays into physical chunks and
maps them across all available GPUs and nodes.

During training, operations such as histogram building, gradient
computation, and tree construction are expressed as parallel tasks.
Legate schedules these tasks close to where the data resides, minimizing
communication overhead. When synchronization is needed (e.g., to combine
histograms from multiple GPUs), it is handled by legate-mpi-wrapper and
realm-gex-wrapper, so we never have to write MPI or manage explicit GPU
memory transfers.

**Running on CPU and GPU**
~~~~~~~~~~~~~~~~~~~~~~~~~~

**CPU** - To run with CPU, use the following command.

+-----------------------------------------------------------------------+
| **legate --cpus 1 --gpus 0 ./housing.py**                             |
+=======================================================================+

Output:

+-----------------------------------------------------------------------+
| **The training time for housing exp is: 7846.303000 milliseconds**    |
+=======================================================================+

**GPU** - To run with GPU, use the following command.

+-----------------------------------------------------------------------+
| **legate --gpus 2 ./housing.py**                                      |
+=======================================================================+

Output:

+-----------------------------------------------------------------------+
| **The training time for housing exp is : 846.949000 milliseconds**    |
+=======================================================================+

**To Do: Multi Node and Multi GPU**

**Example 2**
-------------

This example demonstrates how Legate Boost can be applied to the *“Give
Me Some Credit”* dataset (OpenML data_id: 46929) to build a
classification model using ensemble learning by combining different
model types. It also highlights the integration of Legate DataFrame with
Legate Boost to enable distributed training across multi-GPU and
multi-node environments, showcasing scalable machine learning on the
Credit Score dataset.

**About the dataset**
~~~~~~~~~~~~~~~~~~~~~

The Give Me Some Credit dataset is a financial risk prediction dataset
originally introduced in a Kaggle competition. It includes anonymized
credit and demographic data for individuals, with the goal of predicting
whether a person is likely to experience serious financial distress
within the next two years.

Each record represents an individual and includes features such as:

- Revolving utilization of unsecured credit lines

- Age

- Number of late payments (30–59, 60–89, and 90+ days past due)

- Debt ratio

- Monthly income

- Number of open credit lines and loans

- Number of dependents\ **
  **

The target variable is binary (0 = no distress, 1 = distress),
indicating the likelihood of future financial trouble.

.. _about-this-implementation-1:

**About this implementation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This implementation will focus on demonstrating Legate Boost’s flexible
model ensembling capabilities, specifically:

- Tree-based gradient boosting models, ideal for structured/tabular
  data.

- Neural network-based classifiers, allowing hybrid or deep learning
  approaches.

By leveraging Legate Boost, we can ensemble these two models and
efficiently train and evaluate both model types on GPUs or CPUs,
showcasing scalable performance for large tabular datasets in financial
risk prediction.

The pipeline begins with importing required libraries and its functions
and also loading the dataset using fetch_openml. Depending on hardware
availability, the data is initially handled either with cuDF (for GPU
execution) or pandas (for CPU execution). The dataset is then wrapped
into a LogicalTable, the distributed data representation used by Legate
DataFrame. LogicalTables internally break data into logical columns,
enabling Legate’s runtime to partition, distribute, and schedule
computations across multiple GPUs and nodes.



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

from sklearn.model_selection import train_test_split

from legate_dataframe.lib.replace import replace_nulls

from legate_dataframe.lib.core.table import LogicalTable

from legate_dataframe.lib.core.column import LogicalColumn

#import data

data = fetch_openml(data_id= 46929, as_frame=True)

xd = cudf if cp.cuda.runtime.getDeviceCount() > 0 else pandas # based on
CPU or GPU

df = xd.DataFrame(data.data, columns=data.feature_names)

df['Target'] = data.target

#covert to logicalTable

if cp.cuda.runtime.getDeviceCount() > 0:

ldf = LogicalTable.from_cudf(df)

else:

df = pa.Table.from_pandas(df)

ldf = LogicalTable.from_arrow(df)



Let’s see how data preprocessing is performed directly on the
LogicalTable. Missing values in key columns (MonthlyIncome and
NumberOfDependents) are filled using median imputation through the
replace_nulls operation. These operations are executed in parallel
across distributed partitions of the LogicalTable, avoiding centralized
bottlenecks. Because LogicalTables are immutable, a new LogicalTable
with updated LogicalColumn’s is created after preprocessing. The cleaned
data is then converted into a cuPyNumeric array, Legate’s
GPU-accelerated array type that leverages logical partitioning for
distributed computation. This enables the subsequent machine learning
tasks to execute efficiently across multiple GPUs or nodes.

#replace nulls

median_salary = df["MonthlyIncome"].median()

median_dependents = df["NumberOfDependents"].median()

mni =
LogicalColumn(replace_nulls(LogicalColumn(ldf["MonthlyIncome"]),median_salary))

mnd =
LogicalColumn(replace_nulls(LogicalColumn(ldf["NumberOfDependents"]),median_dependents))

#create a new logical Table with updated columns

features = ldf.get_column_names()

nldf = LogicalTable( [ ldf[0], ldf[1], ldf[2], ldf[3], mmi, ldf[5],
ldf[6], ldf[7], ldf[8], mnd, ldf[10]], features)

#coverting LogicalTable to cupyNumeric array

data_arr = nldf.to_array()



As we have a data_arr backed by cuPyNumeric, we first split the dataset
into training and testing subsets, which are then passed to Legate Boost
for efficient training across available hardware resources. The model is
built using Legate Boost’s ensemble framework (LBClassifier), which
allows combining multiple types of base learners into a single unified
model.

In this example, the ensemble consists of a Decision Tree
(lb.models.Tree) with max_depth=8, enabling the capture of complex
non-linear decision boundaries by splitting the feature space
hierarchically up to 8 levels deep, and a Neural Network (lb.models.NN)
with two hidden layers of 10 neurons each (hidden_layer_sizes=(10,10)),
trained for max_iter=10 epochs with verbose=True to monitor progress. By
combining a tree-based model with a neural network, Legate Boost
leverages the interpretability and rule-based decision-making of trees
together with the ability of neural networks to model intricate,
high-dimensional relationships. This ensemble design results in a more
accurate and robust classifier than either model could achieve
individually.

#preparing data for training and testing

x = data_arr[:, :-1]

y = data_arr[:, -1]

num_samples = x.shape[0]

split_ratio = 0.8

split_index = int(num_samples \* split_ratio)

x_train = x[:split_index]

y_train = y[:split_index]

x_test = x[split_index:]

y_test = y[split_index:]

start=time()

#create model and trian it

model =
lb.LBClassifier(base_models=(lb.models.Tree(max_depth=8),lb.models.NN(max_iter=10,hidden_layer_sizes=(10,10),verbose=True)))

model.fit(x_train,y_train)

end=time()

The trained ensemble model is used to generate predictions on the test
set, and its accuracy is evaluated using accuracy_score. Finally, the
model is saved with Joblib for future inference without retraining.

#predict

predictions = model.predict(x_test)

#evalution

acc = accuracy_score(y_test, predictions)

print("Accuracy:", acc)

print(f"\\n The training time for creditscore exp is: {(end -
start)/1000:.6f}ms")

#model save

dump(model, "legate_boost_model.joblib")

#save the test sets to CSV for easy inference performance calculation

x_test_cpu = x_test.get() if hasattr(x_test, "get") else
np.array(x_test)

y_test_cpu = y_test.get() if hasattr(y_test, "get") else
np.array(y_test)

pd.DataFrame(x_test_cpu).to_csv("x_test.csv", index=False)

pd.DataFrame(y_test_cpu, columns=["Target"]).to_csv("y_test.csv",
index=False)



This workflow illustrates how Legate DataFrame provides a scalable
preprocessing layer, cupynumeric arrays enable distributed GPU
computation, and Legate Boost delivers a flexible ensemble learning
framework capable of leveraging multi-node, multi-GPU infrastructure
efficiently.

.. _running-on-cpu-and-gpu-1:

**Running on CPU and GPU**
~~~~~~~~~~~~~~~~~~~~~~~~~~

**CPU** - To run with CPU, use the following command.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------------------------------------------+
| **legate --cpus 1 --gpus 0 ./creditscore.py**                         |
+=======================================================================+

Output:

+-----------------------------------------------------------------------+
| **Accuracy: 0.9343**                                                  |
|                                                                       |
| **The training time for creditscore exp is: 45337.714000 ms**         |
+=======================================================================+

**GPU** - To run with GPU, use the following command.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------------------------------------------+
| **legate --gpus 2 ./creditscore.py**                                  |
+=======================================================================+

Output:

+-----------------------------------------------------------------------+
| **Accuracy: 0.9353**                                                  |
|                                                                       |
| **The training time for creditscore exp is: 2688.233000 ms**          |
+=======================================================================+

**To Do: Multi Node and Multi GPU**

**Inference performance**
-------------------------

Let’s explore how cuPyNumeric can be leveraged to measure inference
performance statistics seamlessly across both CPU and GPU all without
modifying the code. In this example, we evaluate a pre-trained machine
learning model by calculating key metrics such as mean, median, minimum,
maximum, variance, and standard deviation of inference times. The model
is loaded using joblib, and predictions are executed multiple times on
the test dataset. By utilizing cuPyNumeric arrays, the timing results
are efficiently processed while ensuring compatibility with both CPU and
GPU environments. This approach provides a simple yet powerful way to
compare inference performance across hardware, offering clear insights
into the speedup and variability achieved with GPU acceleration.

import cupynumeric as cp

from joblib import load

from legate.timing import time

import pandas as pd

import legate.core as lg

timings = []

#load pre-trained model

model = load("legate_boost_model.joblib")

X = pd.read_csv("x_test.csv")

rt = lg.get_legate_runtime()

for \_ in range(10):

rt.issue_execution_fence()

start = time()

model.predict(X)

rt.issue_execution_fence()

end = time()

timings.append(end - start)

#first iteration is not considered here because, it includes env setup
time for gpu usage

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



.. _running-on-cpu-and-gpu-2:

**Running on CPU and GPU**
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _cpu---to-run-with-cpu-use-the-following-command.-1:

**CPU** - To run with CPU, use the following command.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------------------------------------------+
| **legate --cpus 1 --gpus 0 ./inference.py**                           |
+=======================================================================+

Output:

+-----------------------------------------------------------------------+
| **Mean: 265.66 ms**                                                   |
|                                                                       |
| **Median: 262.97 ms**                                                 |
|                                                                       |
| **Min: 249.78 ms**                                                    |
|                                                                       |
| **Max: 284.44 ms**                                                    |
|                                                                       |
| **Variance: 117319.15 ms**                                            |
|                                                                       |
| **standard deviation: 10.83 ms**                                      |
+=======================================================================+

.. _gpu---to-run-with-gpu-use-the-following-command.-1:

**GPU** - To run with GPU, use the following command.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------------------------------------------+
| **legate --gpus 1 ./inference.py**                                    |
+=======================================================================+

Output:

+-----------------------------------------------------------------------+
| **Mean: 122.35 ms**                                                   |
|                                                                       |
| **Median: 122.11 ms**                                                 |
|                                                                       |
| **Min: 121.28 ms**                                                    |
|                                                                       |
| **Max: 125.97 ms**                                                    |
|                                                                       |
| **Variance: 1793.76 ms**                                              |
|                                                                       |
| **standard deviation: 1.34 ms**                                       |
+=======================================================================+

These results clearly show the performance benefits of running inference
on a GPU compared to a CPU using cuPyNumeric arrays. On the CPU, the
model achieved a mean inference time of approximately **265.66 ms**,
with relatively low variability (standard deviation ~\ **10.83 ms**). In
contrast, the GPU significantly reduced the mean inference time to
around **122.35 ms**, representing more than a **2x speedup**, with even
lower variability (standard deviation ~\ **1.34 ms**). This highlights
how cuPyNumeric enables the same code to seamlessly scale across CPU and
GPU, allowing both accurate performance benchmarking and efficient model
deployment across heterogeneous hardware.

**To Do: Multi Node and Multi GPU**
