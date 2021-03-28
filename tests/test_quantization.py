import numpy as np
import pytest
import quflow as qf


def get_random_omega_real(N=5):
    return np.random.randn(N**2)


def get_random_mat(N=5):
    W = np.random.randn(N,N)
    W -= W.conj().T
    return W


@pytest.mark.parametrize("omega", [get_random_omega_real(), get_random_omega_real(17)])
def test_shr2mat_(omega):
    N = round(np.sqrt(omega.shape[0]))
    basis = qf.get_basis(N)
    omega_complex = qf.shr2shc(omega)
    W = np.zeros((N, N), dtype=complex)
    qf.shc2mat_(omega_complex, basis, W)

    W2 = np.zeros((N, N), dtype=complex)
    qf.shr2mat_(omega, basis, W2)

    assert W == pytest.approx(W2)


@pytest.mark.parametrize("W", [get_random_mat(), get_random_mat(17)])
def test_mat2shr_(W):
    N = W.shape[0]
    basis = qf.get_basis(N)

    omega_complex = np.zeros(N**2, dtype=complex)
    qf.mat2shc_(W, basis, omega_complex)
    omega = qf.shc2shr(omega_complex)

    omega2 = np.zeros(N**2, dtype=float)
    qf.mat2shr_(W, basis, omega2)

    assert omega == pytest.approx(omega2)
