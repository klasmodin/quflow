import numpy as np
import pytest
import quflow as qf


def get_random_omega_real(N=5):
    return np.random.randn(N**2)


def get_random_omega_complex(N=5):
    return np.random.randn(N**2) + 1j*np.random.randn(N**2)


def get_random_mat(N=5):
    W = np.random.randn(N, N) + 1j*np.random.randn(N, N)
    W -= W.conj().T
    return W


@pytest.mark.parametrize("omega", [get_random_omega_real(), get_random_omega_real(17)])
def test_norm_L2(omega):
    N = round(np.sqrt(omega.shape[0]))

    W = qf.shr2mat(omega, N=N)

    norm_omega = np.linalg.norm(omega)
    norm_W = qf.geometry.norm_L2(W)

    np.testing.assert_allclose(norm_omega, norm_W)


@pytest.mark.parametrize("N", [5, 17, 64])
def test_inner_L2_real(N):
    omega1 = get_random_omega_real(N)
    omega2 = get_random_omega_real(N)

    W1 = qf.shr2mat(omega1, N=N)
    W2 = qf.shr2mat(omega2, N=N)

    inner_omega = (omega1*omega2).sum()
    inner_W = qf.geometry.inner_L2(W1, W2)

    np.testing.assert_allclose(inner_omega, inner_W)



@pytest.mark.parametrize("N", [17, 64])
def test_inner_L2_complex(N):
    omega1 = get_random_omega_complex(N)
    omega2 = get_random_omega_complex(N)

    W1 = qf.shc2mat(omega1, N=N)
    W2 = qf.shc2mat(omega2, N=N)

    inner_omega = (omega1*omega2.conj()).sum().real
    inner_W = qf.geometry.inner_L2(W1, W2)

    np.testing.assert_allclose(inner_omega, inner_W)


@pytest.mark.parametrize("N", [17, 64])
def test_inner_vs_norm_L2(N):
    W = get_random_mat(N)

    norm1 = qf.geometry.norm_L2(W)
    norm2 = np.sqrt(qf.geometry.inner_L2(W,W))

    np.testing.assert_allclose(norm1, norm2)


@pytest.mark.parametrize("N", [17, 64])
def test_norm_Linf(N):
    W = get_random_mat(N)

    norm1 = qf.geometry.norm_Linf(W)
    norm2 = np.linalg.norm(W, ord=2)

    np.testing.assert_allclose(norm1, norm2)


@pytest.mark.parametrize("N", [15, 16, 64])
def test_hoppe_yau_laplacian(N):
    P = get_random_mat(N)
    X = qf.geometry.cartesian_generators(N)

    def Delta_N(P):
        W = np.zeros_like(P)
        for k in range(3):
            W += qf.geometry.bracket(X[k], qf.geometry.bracket(X[k], P))
        return W

    Wtilde = Delta_N(P)
    W = qf.laplacian.laplace(P)

    np.testing.assert_allclose(Wtilde, W)


@pytest.mark.parametrize("N", [15, 16, 64, 128])
def test_so3_generators(N):
    S1, S2, S3 = qf.geometry.so3_generators(N)

    np.testing.assert_allclose(S1@S2-S2@S1, S3, atol=1e-14)
    np.testing.assert_allclose(S2@S3-S3@S2, S1, atol=1e-14)
    np.testing.assert_allclose(S3@S1-S1@S3, S2, atol=1e-14)


@pytest.mark.parametrize("N", [15, 16, 64, 128])
def test_cartesian_generators(N):
    X1, X2, X3 = qf.geometry.cartesian_generators(N)

    np.testing.assert_allclose(qf.geometry.bracket(X1, X2), X3, atol=1e-14)
    np.testing.assert_allclose(qf.geometry.bracket(X2, X3), X1, atol=1e-14)
    np.testing.assert_allclose(qf.geometry.bracket(X3, X1), X2, atol=1e-14)


@pytest.mark.parametrize("N", [15, 16, 64])
def test_cartesian_generators_scale(N):
    X1, X2, X3 = qf.geometry.cartesian_generators(N)

    T1m1 = qf.shr2mat(np.array([0, 1, 0, 0], dtype=np.float64), N=N)
    T10 = qf.shr2mat(np.array([0, 0, 1, 0], dtype=np.float64), N=N)
    T1p1 = qf.shr2mat(np.array([0, 0, 0, 1], dtype=np.float64), N=N)

    scale = np.sqrt(3)

    np.testing.assert_allclose(scale*X1, T1p1, atol=1e-14)
    np.testing.assert_allclose(scale*X2, T1m1, atol=1e-14)
    np.testing.assert_allclose(scale*X3, T10, atol=1e-14)


@pytest.mark.parametrize("N, ref", [(64, 0.98449518), (45, 0.97801929), (128, 0.99221778)])
def test_cartesian_generators_spectrum(N, ref):
    X = qf.geometry.cartesian_generators(N)
    for Xi in X:
        np.testing.assert_allclose(qf.geometry.norm_Linf(Xi), ref, atol=1e-8)


@pytest.mark.parametrize("N", [256])
def test_bracket_convergence(N):
    np.random.seed(42)
    omega = np.random.randn(16)
    omega[0] = 0.0

    ell = np.floor(np.sqrt(np.arange(1,omega.shape[0]))).astype(int)
    psi = np.zeros_like(omega)
    psi[1:] = -omega[1:]/(ell*(ell+1))

    laplace_omega = np.zeros_like(omega)
    laplace_omega[1:] = omega[1:]*ell*(ell+1)

    f = qf.shr2fun(omega, N=512, berezin=False)
    g = qf.shr2fun(psi, N=512, berezin=False)
    fg = qf.poisson_finite_differences(f, g)
    omegapsi = qf.fun2shr(fg)

    W = qf.shr2mat(omega, N=N)
    P = qf.shr2mat(psi, N=N)
    WP = qf.shr2mat(omegapsi, N=N)
    WPprime = qf.geometry.bracket(W, P)

    np.testing.assert_allclose(WP, WPprime, atol=0.05)



# --- Documented properties --------------------------------------------------------

def _commutator(A, B):
    return A@B - B@A


@pytest.mark.parametrize("N", [2, 5, 8])
def test_so3_generators(N):
    S1, S2, S3 = qf.geometry.so3_generators(N)
    s = (N - 1)/2
    for S in (S1, S2, S3):
        np.testing.assert_allclose(S, -S.conj().T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(_commutator(S1, S2), S3, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_commutator(S2, S3), S1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_commutator(S3, S1), S2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(S1@S1 + S2@S2 + S3@S3, -s*(s + 1)*np.eye(N), rtol=0, atol=1e-12)


@pytest.mark.parametrize("N", [3, 8])
def test_cartesian_generators(N):
    X1, X2, X3 = qf.geometry.cartesian_generators(N)
    np.testing.assert_allclose(X1@X1 + X2@X2 + X3@X3, -np.eye(N), rtol=0, atol=1e-13)
    T = lambda m: qf.elmr2mat(1, m, N).toarray()
    np.testing.assert_allclose(T(-1), np.sqrt(3)*X2, rtol=0, atol=1e-13)
    np.testing.assert_allclose(T(0), np.sqrt(3)*X3, rtol=0, atol=1e-13)
    np.testing.assert_allclose(T(1), np.sqrt(3)*X1, rtol=0, atol=1e-13)


def test_rotate_direction():
    N = 6
    X1, X2, X3 = qf.geometry.cartesian_generators(N)
    theta = 0.3
    R = qf.geometry.rotate(np.array([0.0, 0.0, theta]), X1)
    np.testing.assert_allclose(R, np.cos(theta)*X1 + np.sin(theta)*X2, rtol=0, atol=1e-13)


def test_sgrad_of_coordinate_function():
    N = 6
    X1, X2, X3 = qf.geometry.cartesian_generators(N)
    dP = qf.geometry.sgrad(X3)
    np.testing.assert_allclose(dP[0], -X2, rtol=0, atol=1e-13)
    np.testing.assert_allclose(dP[1], X1, rtol=0, atol=1e-13)
    np.testing.assert_allclose(dP[2], 0, rtol=0, atol=1e-13)


def test_integral_is_el0_coefficient(random_shr):
    omega = random_shr(7, zero_mean=False)
    assert abs(qf.geometry.integral(qf.shr2mat(omega)) - omega[0]) < 1e-13


def test_norms_of_skew_hermitian(random_skewh):
    N = 7
    W = random_skewh(N)
    sv = np.linalg.svd(W, compute_uv=False)
    assert abs(qf.geometry.norm_Linf(W) - sv.max()) < 1e-12
    assert abs(qf.geometry.norm_L1(W) - sv.sum()/N) < 1e-12
    assert abs(qf.geometry.norm_L2(W) - np.sqrt((sv**2).sum()/N)) < 1e-12


def test_bracket_sparse_matches_dense():
    N = 7
    A, B = qf.elmr2mat(3, 2, N), qf.elmr2mat(4, -1, N)
    np.testing.assert_allclose(qf.geometry.bracket(A, B).toarray(),
                               qf.geometry.bracket(A.toarray(), B.toarray()), rtol=0, atol=1e-13)


def test_inner_L2_sparse_same_offsets():
    N = 7
    A, B = qf.elmr2mat(3, 2, N), qf.elmr2mat(5, 2, N)
    assert abs(qf.geometry.inner_L2(A, A) - 1) < 1e-13
    assert abs(qf.geometry.inner_L2(A, B) - qf.geometry.inner_L2(A.toarray(), B.toarray())) < 1e-13


@pytest.mark.xfail(strict=True, reason="inner_L2 uses `*`, which is the matrix product for scipy sparse matrices")
def test_inner_L2_sparse_and_dense():
    N = 7
    A = qf.elmr2mat(3, 2, N)
    assert abs(qf.geometry.inner_L2(A, A.toarray()) - 1) < 1e-13


# --- physics -----------------------------------------------------------------------

def test_energy_and_enstrophy_identities(random_shr):
    N = 9
    omega = random_shr(N)  # trace free
    W = qf.shr2mat(omega)
    P = qf.solve_poisson(W).copy()
    E = qf.physics.energy_euler(W)
    ells = qf.ind2elm(np.arange(1, N**2))[0]
    np.testing.assert_allclose(E, 0.5*np.sum(omega[1:]**2/(ells*(ells + 1))), rtol=1e-12)
    np.testing.assert_allclose(E, 0.5*qf.physics.norm_Hm1(W)**2, rtol=1e-12)
    np.testing.assert_allclose(E, 0.5*qf.physics.norm_H1(P)**2, rtol=1e-12)
    np.testing.assert_allclose(qf.physics.enstrophy(W), 0.5*np.sum(omega**2), rtol=1e-12)
    # The trace does not contribute to the energy
    np.testing.assert_allclose(qf.physics.energy_euler(W + 1j*np.eye(N)), E, rtol=1e-12)
