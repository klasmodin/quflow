"""Shared pytest fixtures for the quflow test suite."""
import numpy as np
import pytest


def _cupy_available():
    try:
        import cupy as cp
    except ImportError:
        return False
    return bool(cp.is_available())


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``gpu`` when no CUDA device is available."""
    if _cupy_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires cupy and a CUDA device")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)


@pytest.fixture
def rng():
    """Seeded random generator, so that tests are reproducible."""
    return np.random.default_rng(1234)


@pytest.fixture
def random_skewh(rng):
    """
    Factory for random skew-Hermitian matrices.

    Usage: ``W = random_skewh(N, tracefree=True, dtype=np.complex128)``.
    """
    def make(N, tracefree=True, dtype=np.complex128):
        A = rng.standard_normal((N, N)) + 1j*rng.standard_normal((N, N))
        W = (A - A.conj().T)/2
        if tracefree:
            W -= np.eye(N)*np.trace(W)/N
        return W.astype(dtype)
    return make


@pytest.fixture
def random_shr(rng):
    """
    Factory for random real spherical-harmonic coefficient vectors of length ``N**2``.

    Usage: ``omega = random_shr(N, zero_mean=True)``. With ``zero_mean`` the
    ``el = 0`` coefficient is zero, giving a trace-free matrix under ``shr2mat``.
    """
    def make(N, zero_mean=True, dtype=np.float64):
        omega = rng.standard_normal(N**2)
        if zero_mean:
            omega[0] = 0.0
        return omega.astype(dtype)
    return make
