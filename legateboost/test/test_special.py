from typing import Callable, Tuple, Type, Union

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes

import cunumeric as cn
from legateboost import special


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


@given(
    dtype=st.sampled_from([np.float32, np.float64]),
    shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=128),
)
@settings(deadline=None, max_examples=64)
def test_tgamma(
    dtype: Union[Type[np.float32], Type[np.float64]], shape: Tuple[int, ...]
) -> None:
    from scipy.special import gamma as scipy_gamma

    rng = cn.random.default_rng(1)
    x = rng.normal(size=shape).astype(dtype)
    y0 = special.gamma(x)
    y1 = scipy_gamma(x)
    assert y0.shape == x.shape
    assert y0.dtype == x.dtype
    np.testing.assert_allclose(y0, y1, rtol=1e-6)


@given(
    dtype=st.sampled_from([np.float32, np.float64]),
    shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=128),
)
@settings(deadline=None, max_examples=64)
def test_digamma(
    dtype: Union[Type[np.float32], Type[np.float64]], shape: Tuple[int, ...]
) -> None:
    from scipy.special import digamma as scipy_digamma

    rng = cn.random.default_rng(1)
    x = rng.uniform(size=shape, low=0.1, high=3.0).astype(dtype)

    y0 = special.digamma(x)
    y1 = scipy_digamma(x)

    np.testing.assert_allclose(y0, y1, atol=1e-3)


@given(
    dtype=st.sampled_from([np.float32, np.float64]),
    shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=128),
)
@settings(deadline=None, max_examples=64)
def test_trigamma(
    dtype: Union[Type[np.float32], Type[np.float64]], shape: Tuple[int, ...]
) -> None:
    from scipy.special import polygamma as scipy_polygamma

    rng = cn.random.default_rng(1)
    x = rng.uniform(size=shape, low=0.1, high=3.0).astype(dtype)

    y0 = special.polygamma(1, x)
    y1 = scipy_polygamma(1, x)

    np.testing.assert_allclose(y0, y1, rtol=1e-6)
