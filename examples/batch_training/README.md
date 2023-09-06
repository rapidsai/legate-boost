# Batch training
This example shows how to split the dataset into batches and train the model in multiple steps. This is useful for large datasets that do not fit into memory.

In this example we use the million song dataset, where the task is to predict the year a song was released given a set of audio features.

The training set is split into 10 batches and we use calls to the partial fit method to train 5 estimators on each batch, until we have 100 estimators in total.

This is a toy example in that we do not load the batches each time from memory. In a real world example with too much data to fit in memory, the batches would be loaded from disk or a database in each iteration.

The example outputs the following chart, showing how difference in train and test error between the model trained on the entire dataset and the model trained in batches.

<img src="batch_training.png" alt="drawing" width="800"/>
