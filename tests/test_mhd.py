"""Tests for quflow.integrators.mhd."""
import numpy as np
import pytest
import quflow as qf
from quflow.integrators import isospectral, mhd


N = 8


@pytest.fixture
def state(random_shr):
    """MHD state [W, Theta] of two skew-Hermitian matrices."""
    return np.array([qf.shr2mat(random_shr(N), N=N), qf.shr2mat(random_shr(N), N=N)])


@pytest.fixture
def nonskewh_mode():
    """Switch to general (non skew-Hermitian) mode, and restore the default afterwards."""
    isospectral.select_skewherm(False)
    yield
    isospectral.select_skewherm(True)


@pytest.mark.parametrize("integrator, dt", [
    (mhd.magmp_fixedpoint, 0.1*qf.hbar(N)),
    (mhd.magmp_fixedpoint_old, 0.1),
])
def test_magmp_requires_skewherm_mode(integrator, dt, state, nonskewh_mode):
    # The setting must be read at call time, not at import time
    with pytest.raises(NotImplementedError):
        integrator(state.copy(), dt, 1)


def test_magmp_runs_in_skewherm_mode(state):
    out = mhd.magmp(state.copy(), 0.1*qf.hbar(N), 2)
    assert out.shape == state.shape
