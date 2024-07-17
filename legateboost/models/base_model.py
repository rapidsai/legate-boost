from abc import ABC, abstractmethod

import numpy as np

import cunumeric as cn

from ..utils import PickleCunumericMixin


class BaseModel(PickleCunumericMixin, ABC):
    """Base class for all models in LegateBoost.

    Defines the interface for fitting, updating, and predicting a model,
    as well as string representation and equality comparison. Implement
    these methods to create a custom model.
    """

    def set_random_state(self, random_state: np.random.RandomState) -> "BaseModel":
        self.random_state = random_state
        return self

    @abstractmethod
    def fit(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "BaseModel":
        """Fit the model to a second order Taylor expansion of the loss
        function.

        Parameters
        ----------
        X :
            The training data.
        g :
            The first derivative of the loss function with
            respect to the predicted values.
        h :
            The second derivative of the loss function with
             respect to the predicted values.

        Returns
        -------
        BaseModel
            The fitted model.
        """
        pass

    @abstractmethod
    def update(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "BaseModel":
        """Update the model with new training data.

        Parameters
        ----------
        X :
            The training data to update the model with.
        g :
            The first derivative of the loss function with
            respect to the model's predictions.
        h :
            The second derivative of the loss function with
            respect to the model's predictions.

        Returns
        -------
        BaseModel
            The updated model.
        """
        pass

    @abstractmethod
    def predict(self, X: cn.ndarray) -> cn.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None: ...  # noqa: E704
