from typing import Callable, Tuple, Type, Union

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes

import cunumeric as cn
from legateboost import special
from scipy.special import digamma, polygamma


def run_erf(
    dtype: Union[Type[np.float32], Type[np.float64]], shape: Tuple[int, ...]
) -> None:
    from scipy.special import erf as scipy_erf

    rng = cn.random.default_rng(1)
    x = rng.normal(size=shape).astype(dtype)
    y0 = special.erf(x)
    y1 = scipy_erf(x)
    assert y0.shape == x.shape
    assert y0.dtype == x.dtype
    np.testing.assert_allclose(y0, y1, rtol=1e-6)


@given(
    dtype=st.sampled_from([np.float32, np.float64]),
    shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=256),
)
@settings(deadline=None, max_examples=100)
def test_erf_basic(
    dtype: Union[Type[np.float32], Type[np.float64]], shape: Tuple[int, ...]
) -> None:
    run_erf(dtype, shape)


def run_special(fn: Callable) -> None:
    # Special tests for maximum dimension without exploding memory usage.
    run_erf(np.float32, (256, 1024, 10, 3))
    run_erf(np.float32, (256, 10, 3, 1024))
    # error tests.
    with pytest.raises(TypeError, match="not supported"):
        rng = cn.random.default_rng(1)
        x = rng.integers(3, size=10)
        fn(x)


def test_erf_special() -> None:
    run_special(special.erf)


def run_lgamma(
    dtype: Union[Type[np.float32], Type[np.float64]], shape: Tuple[int, ...]
) -> None:
    from scipy.special import loggamma as scipy_lgamma

    rng = cn.random.default_rng(1)
    x = rng.uniform(size=shape, low=0.1, high=3.0).astype(dtype)
    y0 = special.loggamma(x)
    y1 = scipy_lgamma(x)
    assert y0.shape == x.shape
    assert y0.dtype == x.dtype
    np.testing.assert_allclose(y0, y1, rtol=1e-6)


@given(
    dtype=st.sampled_from([np.float32, np.float64]),
    shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=256),
)
@settings(deadline=None, max_examples=100)
def test_lgamma_basic(
    dtype: Union[Type[np.float32], Type[np.float64]], shape: Tuple[int, ...]
) -> None:
    run_lgamma(dtype, shape)


def test_lgamma_special() -> None:
    run_special(special.loggamma)


def test_digamma() -> None:
    pass


def test_trigamma() -> None:
    pass


def test_polygamma() -> None:
    pass
