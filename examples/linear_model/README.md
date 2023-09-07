# Linear model
This example shows how to train a mixed model with linear and tree components. The dataset is a linear function with some added noise, then a step in the middle of the function. This is challenging for a linear model due to the step, and challenging for a tree model due to the sloped function (see the characteristic axis aligned step function of the tree model). We create a combined model by first boosting 5 iterations of a linear model and then 15 iterations of the tree model. The result is a model that is better able to fit the linear function and the step function.

<img src="linear_model.png" alt="drawing" width="800"/>
