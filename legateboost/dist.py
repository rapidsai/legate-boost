# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from abc import ABC, abstractmethod

import cunumeric as cn


class Dist(ABC):
    """Abstract base class for distributions."""

    @abstractmethod
    def mean(self) -> cn.ndarray:
        """Return the mean value."""
        raise NotImplementedError("abstractmethod")

    @abstractmethod
    def var(self) -> cn.ndarray:
        """Return the variance."""
        raise NotImplementedError("abstractmethod")


class Normal(Dist):
    """The normal distribution.

    Parameters
    ----------
    param :
        Prediction from the boosting model, containing the parameters for the
        distribution.
    """

    def __init__(self, param: cn.ndarray) -> None:
        self._mean = param[:, 0]
        self._var = cn.exp(param[:, 1])

    def mean(self) -> cn.ndarray:
        return self._mean

    def var(self) -> cn.ndarray:
        return self._var
