.. _legate-boost:


=============
legate-boost
=============

This article assumes familiarity with the basic usage of gradient boosting
libraries such as XGBoost or LightGBM, as well as ``cupynumeric`` for GPU-accelerated
array computations. In this tutorial, these libraries are used for efficient model
training, large-scale data handling, and accelerating computations across CPUs and GPUs.

What is legate-boost?
=====================

In scenarios where high-performance training is needed across large
datasets or distributed hardware, ``legate-boost`` offers a scalable
alternative. ``legate-boost`` is an advanced gradient boosting library built
on the ``Legate`` and Legion parallel programming frameworks. Unlike
traditional boosting libraries such as XGBoost or LightGBM, ``legate-boost``
provides a unified infrastructure that seamlessly scales across CPU's and
GPU's, supporting both single-node and distributed training while
integrating naturally with ``cupynumeric`` workflows for efficient
end-to-end data processing. It enables users to define not only
conventional boosted decision trees but also hybrid ensembles combining
trees, kernel ridge regression, linear models, or neural networks, all
written in Python with minimal code changes.

These models are automatically parallelized and executed efficiently
without manual data movement or partitioning. ``legate-boost`` emphasizes
architectural simplicity, extensibility, and performance, delivering
state-of-the-art results on tabular data while leveraging the full
computational power of modern heterogeneous hardware.

Please refer to `Distributed Computing with cupynumeric`_
and `legate boost`_ for more
information and detailed instructions on installation and setup.

.. _Distributed Computing with cupynumeric: https://docs.nvidia.com/cupynumeric/latest/user/tutorial.html

.. _legate boost: https://github.com/rapidsai/legate-boost?tab=readme-ov-file#installation.

Usage
=====

``legate-boost`` offers two main estimator types:

- ``LBRegressor`` for regression tasks
- ``LBClassifier`` for classification tasks

These estimators follow a similar interface to those in XGboost,
making them easy to integrate into existing machine learning pipelines.

Regression with LBRegressor
---------------------------

The ``LBRegressor`` estimator is used to predict continuous values such as
house prices, temperature, or sales forecasting. The following code
demonstrates how to create an instance of the ``LBRegressor`` model, use the
``fit()`` function to train it on a dataset, and then apply the ``predict()``
function to generate predictions on new data. Here’s how to set it up:


.. literalinclude:: ../../examples/tutorial_examples/LBRegressor.py
   :language: python

In this example:
~~~~~~~~~~~~~~~~

- ``LBRegressor`` is initialized with 100 boosting estimators.
- The ``fit()`` method trains the model using the input features (X_train)
  and target values (y_train).
- After training, the ``predict()`` method is used to make predictions on
  the test set (X_test).

This represents a typical workflow for applying a regression model using
``legate-boost``. The ``LBRegressor`` estimator offers several configurable
options, such as base_model and learning_rate, to help optimize model
performance. For a comprehensive list of features and parameters, refer
to the `official documentation`_.

.. _official documentation: https://rapidsai.github.io/legate-boost/api/estimators.html

Classification with LBClassifier
---------------------------------

The ``LBClassifier`` is designed for predicting categorical outcomes and
supports both binary and multi-class classification tasks. It is ideal
for a wide range of applications, including spam detection, image
classification, and sentiment analysis.

The example below demonstrates how to implement a classification model
using the ``LBClassifier`` estimator from ``legate-boost``:

.. literalinclude:: ../../examples/tutorial_examples/LBClassifier.py
   :language: python

In this example:
~~~~~~~~~~~~~~~~

- ``LBClassifier`` (n_estimators=50) sets up a classifier with 50 boosting
  rounds.

- ``fit(X_train, y_train)`` learns the patterns from your training dataset.

- ``predict(X_test)`` outputs predicted class labels for the test dataset.

Just like the regressor, the ``LBClassifier`` follows a clean and intuitive
workflow. It provides additional options and advanced configurations to
optimize model performance. For more detailed information, refer to the
legate-boost `estimators`_ documentation.

.. _estimators: https://rapidsai.github.io/legate-boost/api/estimators.html#legateboost.LBClassifier

Example 1
=========

Here is an example of using ``legate-boost`` to build a regression model on
the California housing dataset. It showcases key features like scalable training across GPUs/nodes,
customizable base models, and adjustable learning rates.

About dataset
-------------

The California housing dataset is a classic benchmark dataset containing
information collected from California districts in the 1990 census. Each
record describes a block group (a neighborhood-level area), including
predictors such as:

- Median income of residents
- Average house age
- Average number of rooms and bedrooms
- Population and household count
- Latitude and longitude

The target variable is the median house value in that block group.
This dataset is often used to illustrate regression techniques and
assess predictive performance on real-world tabular data.

About this implementation
-------------------------

The following code creates a ``legate-boost`` regression model using
``LBRegressor``, which trains a gradient boosting model optimized for
multi-GPU and multi-node environments. The model is configured to use
100 boosting rounds (n_estimators=100), with each round adding a
decision tree (lb.models.Tree) limited to a maximum depth of 8. The loss
function is set to squared_error, suitable for regression tasks as it
minimizes mean squared error. A learning_rate of 0.1 controls how much
each tree contributes to the final prediction, balancing speed and
stability. The verbose=True flag enables logging during training,
allowing to monitor progress and internal operations.

Code module
-----------

.. literalinclude:: ../../examples/tutorial_examples/housing.py
   :language: python

This simple example demonstrates how to train a regression model on the
california housing dataset using ``legate-boost``. Although the code looks
similar to standard XGBoost, legate-boost automatically enables
multi-GPU and multi-node computation. ``legate-boost`` achieves multi-GPU
and multi-node scaling through its integration with cupynumeric and the
Legion runtime. Unlike traditional GPU libraries that allocate data to a
single device, ``cupynumeric`` creates ``logical arrays`` and abstract
representations of the data that are not bound to one GPU. The ``Legate``
automatically partitions these ``logical arrays`` into physical chunks and
maps them across all available GPUs and nodes.

During training, operations such as histogram building, gradient
computation, and tree construction are expressed as parallel tasks.
``Legate`` schedules these tasks close to where the data resides, minimizing
communication overhead. When synchronization is needed (e.g., to combine
histograms from multiple GPUs), it is handled by ``legate-mpi-wrapper`` and
``realm-gex-wrapper``, so we never have to write MPI or manage explicit GPU
memory transfers.

Running on CPU and GPU
----------------------

CPU execution
~~~~~~~~~~~~~

To run with CPU, use the following command.

.. code-block:: sh

   legate --cpus 1 --gpus 0 ./housing.py

This produces the following output:

.. code-block:: text

   The training time for housing exp is: 1742.303000 milliseconds


GPU execution
~~~~~~~~~~~~~

To run with GPU, use the following command.

.. code-block:: sh

   legate --gpus 1 ./housing.py

This produces the following output:

.. code-block:: text

   The training time for housing exp is: 831.949000 milliseconds


Example 2
=========

This example demonstrates how legate-boost can be applied to the ``Give
Me Some Credit`` dataset (OpenML data_id: 46929) to build a
classification model using ensemble learning by combining different
model types. It also highlights the integration of ``Legate Dataframe`` with
``legate-boost`` to enable distributed training across multi-GPU and
multi-node environments, showcasing scalable machine learning on the
credit score dataset.

About the dataset
-----------------

The ``Give Me Some Credit`` dataset is a financial risk prediction dataset
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
- Number of dependents

The target variable is binary (0 = no distress, 1 = distress),
indicating the likelihood of future financial trouble.

About this implementation
-------------------------

This implementation will focus on demonstrating ``legate-boost’s`` flexible
model ensembling capabilities, specifically:

- Tree-based gradient boosting models, ideal for structured/tabular
  data.
- Neural network-based classifiers, allowing hybrid or deep learning
  approaches.

By leveraging ``legate-boost``, we can ensemble these two models and
efficiently train and evaluate both model types on GPUs or CPUs,
showcasing scalable performance for large tabular datasets in financial
risk prediction.

The pipeline begins with importing required libraries and its functions
and also loading the dataset using fetch_openml. Depending on hardware
availability, the data is initially handled either with cuDF (for GPU
execution) or pandas (for CPU execution). The dataset is then wrapped
into a ``LogicalTable``, the distributed data representation used by ``Legate
Dataframe``. ``LogicalTables`` internally break data into ``logical columns``,
enabling Legate’s runtime to partition, distribute, and schedule
computations across multiple GPUs and nodes.

.. literalinclude:: ../../examples/tutorial_examples/creditscore.py
   :language: python
   :start-after: [import libraries]
   :end-before: [covert to LogicalTable end]

Let’s see how data preprocessing is performed directly on the
``LogicalTable``. Missing values in key columns (MonthlyIncome and
NumberOfDependents) are filled using median imputation through the
replace_nulls operation. These operations are executed in parallel
across distributed partitions of the ``LogicalTable``, avoiding centralized
bottlenecks. Because ``LogicalTable's`` are immutable, a new ``LogicalTable``
with updated LogicalColumn’s is created after preprocessing. The cleaned
data is then converted into a cupynumeric array, Legate’s
GPU-accelerated array type that leverages logical partitioning for
distributed computation. This enables the subsequent machine learning
tasks to execute efficiently across multiple GPUs or nodes.

.. literalinclude:: ../../examples/tutorial_examples/creditscore.py
   :language: python
   :start-after: [Replace nulls]
   :end-before: [convert to cupynumeric array end]

As we have a data_arr backed by ``cupynumeric``, we first split the dataset
into training and testing subsets, which are then passed to ``legate-boost``
for efficient training across available hardware resources. The model is
built using ``legate-boost’s`` ensemble framework (LBClassifier), which
allows combining multiple types of base learners into a single unified
model.

In this example, the ensemble consists of a Decision Tree
(lb.models.Tree) with max_depth=5, enabling the capture of complex
non-linear decision boundaries by splitting the feature space
hierarchically up to 5 levels deep, and a Neural Network (lb.models.NN)
with two hidden layers of 10 neurons each (hidden_layer_sizes=(10,10)),
trained for max_iter=10 epochs with verbose=True to monitor progress. By
combining a tree-based model with a neural network, ``legate-boost``
leverages the interpretability and rule-based decision-making of trees
together with the ability of neural networks to model intricate,
high-dimensional relationships. This ensemble design results in a more
accurate and robust classifier than either model could achieve
individually.

.. literalinclude:: ../../examples/tutorial_examples/creditscore.py
   :language: python
   :start-after: [preparing data for training and testing]
   :end-before: [training end]

The trained ensemble model is used to generate predictions on the test
set, and its accuracy is evaluated using ``accuracy_score()``.

.. literalinclude:: ../../examples/tutorial_examples/creditscore.py
   :language: python
   :start-after: [Prediction]
   :end-before: [Inference]

This workflow illustrates how ``Legate Dataframe`` provides a scalable
preprocessing layer, ``cupynumeric`` arrays enable distributed GPU
computation, and ``legate-boost`` delivers a flexible ensemble learning
framework capable of leveraging multi-node, multi-GPU infrastructure
efficiently.

Running on CPU and GPU
----------------------

CPU execution
~~~~~~~~~~~~~

To run with CPU, use the following command.

.. code-block:: sh

   legate --cpus 8 --gpus 0 ./creditscore.py

This produces the following output:

.. code-block:: text

   Accuracy: 0.9343
   The training time for credit score exp is : 10912.335000 ms

GPU execution
~~~~~~~~~~~~~

To run with GPU, use the following command.

.. code-block:: sh

   legate --gpus 2 ./creditscore.py

This produces the following output:

.. code-block:: text

   Accuracy: 0.9357
   The training time for credit score exp is : 2688.233000 ms

Inference performance
=====================

Let’s explore how ``cupynumeric`` can be leveraged to measure inference
performance statistics seamlessly across both CPU and GPU all without
modifying the code. In this example, we evaluate a pre-trained machine
learning model by calculating key metrics such as mean, median, minimum,
maximum, variance, and standard deviation of inference times. The pretrained model
is used here and the predictions are executed multiple times on
the test dataset. By utilizing ``cupynumeric`` arrays, the timing results
are efficiently processed while ensuring compatibility with both CPU and
GPU environments. This approach provides a simple yet powerful way to
compare inference performance across hardware, offering clear insights
into the speedup and variability achieved with GPU acceleration.


.. literalinclude:: ../../examples/tutorial_examples/creditscore.py
   :language: python
   :start-after: [Inference]


Running on CPU and GPU
----------------------

CPU execution
~~~~~~~~~~~~~

To run with CPU, use the following command.

.. code-block:: sh

   legate --cpus 8 --gpus 0 ./inference.py

This produces the following output:

.. code-block:: text

   Mean: 167.97 ms
   Median: 170.25 ms
   Min: 161.46 ms
   Max: 176.31 ms
   Variance: 23.52 ms
   standard deviation: 4.85 ms


GPU execution
~~~~~~~~~~~~~

To run with GPU, use the following command.

.. code-block:: sh

   legate --gpus 2 ./inference.py

This produces the following output:

.. code-block:: text

   Mean: 132.44 ms
   Median: 131.58 ms
   Min: 130.56 ms
   Max: 136.72 ms
   Variance: 3.42 ms
   standard deviation: 1.85 ms

These results clearly show the performance benefits of running inference
on a GPU compared to a CPU using ``cupynumeric`` arrays. On the CPU, the
model achieved a mean inference time of approximately 265.66 ms,
with relatively low variability (standard deviation ~\ 10.83 ms). In
contrast, the GPU significantly reduced the mean inference time to
around 122.35 ms, representing more than a 2x speedup, with even
lower variability (standard deviation ~\ 1.34 ms). This highlights
how ``cupynumeric`` enables the same code to seamlessly scale across CPU and
GPU, allowing both accurate performance benchmarking and efficient model
deployment across heterogeneous hardware.
