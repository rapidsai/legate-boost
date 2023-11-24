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


class Gamma(Dist):
    """The :math:`\\Gamma` distribution.

    Parameters
    ----------
    param :
        Prediction from the boosting model, containing the parameters for the
        distribution.

    parameterization :
        How to parameterize the :math:`\\Gamma`-distribution. Available options are:

        - shape-scale:

          :math:`f(x; k, \\theta) =
          \\frac{x^{k-1}\\exp^{-x/\\theta}}{\\theta^k\\Gamma(k)}`

        - shape-rate:

          :math:`f(x; \\alpha, \\beta) =
          \\frac{x^{\\alpha-1}\\exp^{-x\\beta}{\\beta^{\\alpha}}}{\\Gamma(\\alpha)}`

        - canonical:

          :math:`f(x; n_0, n_1) = \\exp^{[n_0, n_1] \\cdot [\\ln(x), x]^T -
          [\\ln{\\Gamma(n_0 + 1) - (n_1 + 1)\\ln(-n_1)}]}`
    """

    def __init__(self, param: cn.ndarray, parameterization: str) -> None:
        self.check_parameterization(parameterization)
        # parameters are after transformation
        self._n0 = param[:, 0]
        self._n1 = param[:, 1]
        self._p = parameterization

    def shape(self) -> cn.ndarray:
        if self._p != "canonical":
            return self._n0
        else:
            return self._n0 + 1

    def scale(self) -> cn.ndarray:
        if self._p == "shape-scale":
            return self._n1
        elif self._p == "shape-rate":
            return 1.0 / self._n1
        else:
            return 1.0 / -self._n1

    def mean(self) -> cn.ndarray:
        if self._p == "shape-scale":
            return self._n0 * self._n1
        elif self._p == "shape-rate":
            return self._n0 / self._n1
        else:
            return (self._n0 + 1) / -self._n1

    def var(self) -> cn.ndarray:
        if self._p == "shape-scale":
            return self.mean() * self._n1
        elif self._p == "shape-rate":
            return self._n0 / (self._n1**2)
        else:
            return (self._n0 + 1) / (self._n1**2)

    @staticmethod
    def check_parameterization(p: str) -> None:
        if p not in {
            "shape-scale",
            "shape-rate",
            "canonical",
        }:
            raise ValueError(f"Unknown parametrization: {p}")
