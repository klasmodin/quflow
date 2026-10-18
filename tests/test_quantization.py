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


@pytest.mark.parametrize("N", [33, 65, 513])
def test_basis_break_index(N):
    basis_break_indices = np.hstack((0, (np.arange(N, 0, -1)**2).cumsum()))
    for m in range(N+1):
        np.testing.assert_equal(qf.basis_break_index(m, N), basis_break_indices[m])


@pytest.mark.parametrize("N", [33, 65, 513])
def test_basis_break_index_vectorized(N):
    basis_break_indices = np.hstack((0, (np.arange(N, 0, -1)**2).cumsum()))
    np.testing.assert_equal(qf.basis_break_index(np.arange(N+1), N), basis_break_indices)


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


@pytest.mark.parametrize("omega", [get_random_omega_real(10)])
@pytest.mark.parametrize("N", [33, 64, 128])
def test_shr2mat_short_omega(omega, N):
    basis = qf.get_basis(N)
    W = np.zeros((N, N), dtype=complex)
    qf.shr2mat_(omega, basis, W)

    omega2 = np.hstack((omega, np.zeros(N**2-omega.shape[0], dtype=omega.dtype)))
    W2 = np.zeros((N, N), dtype=complex)
    qf.shr2mat_(omega2, basis, W2)

    np.testing.assert_allclose(W, W2)


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


@pytest.mark.parametrize("omega", [get_random_omega_real(10)])
@pytest.mark.parametrize("N", [33, 64, 128])
def test_mat2shr_short_omega(omega, N):
    basis = qf.get_basis(N)

    W = qf.shr2mat(omega, N=N)

    omega2 = omega.copy()
    qf.mat2shr_(W, basis, omega2)

    np.testing.assert_allclose(omega, omega2)


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


# --- Documented properties of the quantization ------------------------------

@pytest.mark.parametrize("N", [3, 10])
def test_shr2mat_isometry(N, random_shr):
    # shr2mat is an isometry onto u(N) with respect to inner_L2
    a, b = random_shr(N, zero_mean=False), random_shr(N, zero_mean=False)
    A, B = qf.shr2mat(a), qf.shr2mat(b)
    np.testing.assert_allclose(A, -A.conj().T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(qf.geometry.inner_L2(A, B), a@b, rtol=1e-13)
    np.testing.assert_allclose(qf.mat2shr(A), a, rtol=0, atol=1e-13)


@pytest.mark.parametrize("N", [3, 8])
def test_elmr2mat_elmc2mat_are_unit_vectors(N):
    for el in range(N):
        for m in range(-el, el+1):
            e = np.zeros(N**2)
            e[qf.elm2ind(el, m)] = 1.0
            Tr = qf.elmr2mat(el, m, N).toarray()
            Tc = qf.elmc2mat(el, m, N).toarray()
            np.testing.assert_allclose(Tr, qf.shr2mat(e), rtol=0, atol=1e-14)
            np.testing.assert_allclose(Tc, qf.shc2mat(e.astype(complex)), rtol=0, atol=1e-14)
            assert abs(qf.geometry.norm_L2(Tr) - 1) < 1e-13
            assert abs(qf.geometry.norm_L2(Tc) - 1) < 1e-13


def test_T00_is_i_times_identity():
    N = 6
    np.testing.assert_allclose(qf.elmr2mat(0, 0, N).toarray(), 1j*np.eye(N), rtol=0, atol=1e-14)


@pytest.mark.parametrize("N", [3, 10])
def test_shc2mat_round_trip_and_relation_to_real(N, rng, random_shr):
    omega_c = rng.standard_normal(N**2) + 1j*rng.standard_normal(N**2)
    np.testing.assert_allclose(qf.mat2shc(qf.shc2mat(omega_c)), omega_c, rtol=0, atol=1e-13)
    omega_r = random_shr(N, zero_mean=False)
    np.testing.assert_allclose(qf.shc2mat(qf.shr2shc(omega_r)), qf.shr2mat(omega_r), rtol=0, atol=1e-13)


def test_mat2shr_reads_lower_triangle_only(random_shr):
    N = 7
    W = qf.shr2mat(random_shr(N))
    np.testing.assert_array_equal(qf.mat2shr(np.tril(W)), qf.mat2shr(W))


def test_shr2mat_band_limited_input(random_shr):
    N = 9
    omega = random_shr(4)  # el < 4
    np.testing.assert_array_equal(qf.shr2mat(omega, N=N), qf.shr2mat(np.r_[omega, np.zeros(N**2 - 16)]))


def test_basis_layout():
    N = 7
    basis = qf.get_basis(N)
    assert basis.shape == (N*(N+1)*(2*N+1)//6,)
    for m in range(N):
        start = qf.basis_break_index(m, N)
        B = basis[start:start + (N-m)**2].reshape(N-m, N-m)
        np.testing.assert_allclose(B.T @ B, N*np.eye(N-m), rtol=0, atol=1e-12)
