Note for Developers
===================

This document is for people who wants to look inside the implementation and is not
necessary for using LegateBoost for common tasks.

The names of the objectives don't necessarily describe various parameterization
accurately. This is due to the fact that probabilistic distributions usually have
constraints on their parameters and some extra transformations are employed to satisfy
these constraints. For instance, the raw prediction output of Normal distribution
forecasting consists of the mean and the log-variance. We later transform the log-variance
via :math:`exp` to ensure that it's a positive value. The :math:`\Gamma`-distribution
follows a similar pattern for constrained optimization.

Due to the use of Newton's method for optimization, the Hessian returned by the objective
needs to be strictly positive. However, the Hessian value derived from the negative
log-likelihood might not be as nice as one would like. As a workaround, we sometimes use
the Fisher information as a proxy for Hessian, which is defined as the expected value of
Hessian.

Fisher information is not invariant to re-parameterization. Luckily, once we have the
transformation function for the re-parameterization and the Fisher info for the original
parameterization, the new Fisher information can be easily calculated. Let :math:`\theta`
be the original parameter, and :math:`\mu = g(\theta)` be the new parameter.

.. math::

   \theta = g^{-1}(\mu)

   I_{new}(\mu) = I_{old}(\theta) \cdot (\frac{d{g^{-1}(\mu)}}{d\mu})^2
