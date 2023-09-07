from abc import ABC, abstractmethod

import numpy as np

import cunumeric as cn

from ..utils import PickleCunumericMixin


class BaseModel(PickleCunumericMixin, ABC):
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
        pass

    @abstractmethod
    def update(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, X: cn.ndarray) -> cn.ndarray:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass
