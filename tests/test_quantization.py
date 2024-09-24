import numpy as np
import pytest
import quflow as qf


def get_random_omega_real(N=5):
    return np.random.randn(N**2)


def get_random_mat(N=5):
    W = np.random.randn(N, N) + 1j*np.random.randn(N, N)
    W -= W.conj().T
    return W


@pytest.mark.parametrize("N", [33, 65, 513])
def test_compute_basis(N):

    basis_computed = qf.compute_basis(N)
    basis = qf.get_basis(N, allow_compute=True)

    np.testing.assert_allclose(basis, basis_computed)
    # assert np.abs(basis-basis_computed).max() < 1e-10


@pytest.mark.parametrize("omega", [get_random_omega_real(), get_random_omega_real(17)])
def test_shr2mat_(omega):
    N = round(np.sqrt(omega.shape[0]))
    basis = qf.get_basis(N)
    omega_complex = qf.shr2shc(omega)
    W = np.zeros((N, N), dtype=complex)
    qf.shc2mat_(omega_complex, basis, W)

    W2 = np.zeros((N, N), dtype=complex)
    qf.shr2mat_(omega, basis, W2)

    np.testing.assert_allclose(W, W2)
    # assert W == pytest.approx(W2)


@pytest.mark.parametrize("W", [get_random_mat(), get_random_mat(17)])
def test_mat2shr_(W):
    N = W.shape[0]
    basis = qf.get_basis(N)

    omega_complex = np.zeros(N**2, dtype=complex)
    qf.mat2shc_(W, basis, omega_complex)
    omega = qf.shc2shr(omega_complex)

    omega2 = np.zeros(N**2, dtype=float)
    qf.mat2shr_(W, basis, omega2)

    np.testing.assert_allclose(omega, omega2)
    # assert omega == pytest.approx(omega2)


@pytest.mark.parametrize("m", [0, -4, 4, -9, 9])
@pytest.mark.parametrize("el", [9, 15])
@pytest.mark.parametrize("N", [16, 19, 63])
def test_elmr2mat(el, m, N):
    
    i = qf.elm2ind(el, m)
    omegar = np.zeros(N**2, dtype=np.float64)
    omegar[i] = 1.0

    Tref = qf.shr2mat(omegar, N=N)

    T = qf.elmr2mat(el, m, N)
    
    np.testing.assert_allclose(T.toarray(), Tref)


@pytest.mark.parametrize("m", [0, -4, 4, -9, 9])
@pytest.mark.parametrize("el", [6, 15])
@pytest.mark.parametrize("N", [16, 19])
def test_elmr2mat_norm(el, m, N):
    
    T = qf.elmr2mat(el, m, N)
    
    np.testing.assert_allclose(qf.geometry.norm_L2(T.toarray()), 1.0)


@pytest.mark.parametrize("m", [0, -4, 4, -9, 9])
@pytest.mark.parametrize("el", [9, 15])
@pytest.mark.parametrize("N", [16, 19, 63])
def test_elmc2mat(el, m, N):
    
    i = qf.elm2ind(el, m)
    omegac = np.zeros(N**2, dtype=np.complex128)
    omegac[i] = 1.0

    Tref = qf.shc2mat(omegac, N=N)

    T = qf.elmc2mat(el, m, N)
    
    np.testing.assert_allclose(T.toarray(), Tref)


@pytest.mark.parametrize("m", [0, -4, 4, -9, 9])
@pytest.mark.parametrize("el", [6, 15])
@pytest.mark.parametrize("N", [16, 19])
def test_elmc2mat_norm(el, m, N):
    
    T = qf.elmc2mat(el, m, N)
    
    np.testing.assert_allclose(qf.geometry.norm_L2(T.toarray()), 1.0)
