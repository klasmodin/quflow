import numpy as np
import pytest
import quflow.laplacian.sparse as qusparse
import quflow.laplacian.direct as qudirect


def get_random_omega_real(N=5):
    return np.random.randn(N**2)


def get_random_mat(N=5):
    W = np.random.randn(N, N) + 1j*np.random.randn(N, N)
    W -= W.conj().T
    W -= np.eye(N)*np.trace(W)/N
    return W


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_laplace(N):

    P = get_random_mat(N)

    W_sparse = qusparse.laplace(P)
    W_direct = qudirect.laplace(P)

    assert np.abs(W_sparse-W_direct).max() < 1e-10


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_solve_poisson(N):

    W = get_random_mat(N)

    P_sparse = qusparse.solve_poisson(W)
    P_direct = qudirect.solve_poisson(W)

    assert np.abs(P_sparse-P_direct).max() < 1e-10


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_solve_poisson_sparse(N):

    W = get_random_mat(N)

    P = qusparse.solve_poisson(W)
    W2 = qusparse.laplace(P)

    assert np.abs(W-W2).max() < 1e-10


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_solve_poisson_direct(N):

    W = get_random_mat(N)

    P = qudirect.solve_poisson(W)
    W2 = qudirect.laplace(P)

    assert np.abs(W-W2).max() < 1e-10
