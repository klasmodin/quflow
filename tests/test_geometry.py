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

    f = qf.shr2fun(omega, N=512)
    g = qf.shr2fun(psi, N=512)
    fg = qf.poisson_finite_differences(f, g)
    omegapsi = qf.fun2shr(fg)

    W = qf.shr2mat(omega, N=N)
    P = qf.shr2mat(psi, N=N)
    WP = qf.shr2mat(omegapsi, N=N)
    WPprime = qf.geometry.bracket(W, P)

    np.testing.assert_allclose(WP, WPprime, atol=0.05)

