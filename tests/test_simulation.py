import numpy as np
import pytest
import quflow as qf
from quflow.simulation import QuSimulation


def get_random_omega_real(N=5):
    return np.random.randn(N**2)


def get_random_mat(N=5):
    W = np.random.randn(N, N) + 1j*np.random.randn(N, N)
    W -= W.conj().T
    return W


@pytest.mark.parametrize("W", [get_random_mat(4), get_random_mat(128)])
@pytest.mark.parametrize("t", [0.0, 0.34543])
def test_init_sim_(W, t, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    # filename = "/Users/moklas/Downloads/testsim.hdf5"

    sim = QuSimulation(filename, overwrite=True, W=W, time=t, energy=0.0, enstrophy=0.0)
    sim['hamiltonian'] = qf.solve_poisson
    sim2 = QuSimulation(filename)

    assert sim.qutypes == sim2.qutypes
    assert sim['hamiltonian'] == qf.solve_poisson
    assert sim2['hamiltonian'] == qf.solve_poisson
    for name in ["W", "time", "energy", "enstrophy"]:
        np.testing.assert_equal(sim[name], sim2[name])

    # np.testing.assert_allclose(omega, omega2)


@pytest.mark.parametrize("W", [get_random_mat(35)])
def test_callback(W, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    N = W.shape[-1]
    sim = QuSimulation(filename, overwrite=True, W=W, energy=0.7)

    Wlist = np.zeros((10, N, N), dtype=np.complex128)
    Wlist[0, ...] = W
    for W in Wlist[1:, ...]:
        W[:, :] = get_random_mat(W.shape[-1])
        sim(W=W, delta_time=0.1, delta_steps=4, energy=0.7)

    np.testing.assert_allclose(sim['time'], np.arange(0.0, 1.0, 0.1))
    np.testing.assert_equal(sim['step'], np.arange(0, 10*4, 4))
    np.testing.assert_equal(sim['W', :], Wlist)
    np.testing.assert_equal(sim['energy', :], np.full(10, 0.7))
    np.testing.assert_equal(qf.shr2fun(qf.mat2shr(Wlist[-1])).astype(np.float32), sim['omegav', -1])


@pytest.mark.parametrize("W", [get_random_mat(35), get_random_mat(35).astype(np.complex64)])
@pytest.mark.parametrize("qutypes", [{'mat': np.complex64, 'shc': None, 'shr': np.float16}, {'shr': np.float32, 'fun': np.float32}])
def test_qutypes(W, qutypes, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    N = W.shape[-1]
    sim = QuSimulation(filename, overwrite=True, qutypes=qutypes, W=W)

    Wlist = np.zeros((10, N, N), dtype=np.complex128)
    Wlist[0, ...] = W
    for W in Wlist[1:, ...]:
        W[:, :] = get_random_mat(W.shape[-1])
        sim(W=W, delta_time=0.1)

    if 'W' in sim.fieldnames:
        assert sim['W', -1].dtype == qutypes['mat']
    if 'omegar' in sim.fieldnames:
        assert sim['omegar', -1].dtype == qutypes['shr']
    if 'omegac' in sim.fieldnames:
        assert sim['omegac', -1].dtype == qutypes['shc'] if qutypes['shc'] is not None else W.dtype
    if 'value' in sim.fieldnames:
        assert sim['value', -1].dtype == qutypes['fun']


@pytest.mark.parametrize("W", [get_random_mat(22)])
def test_prerun(W, tmpdir):
    filename = tmpdir.join("testsim.hdf5")

    def createsim(W):
        sim = QuSimulation(filename, overwrite=True, W=W)

        def myham(W):
            return 0.5*qf.solve_poisson(W)

        prerun = """
import quflow as qf
def myham(W):
    return 0.5*qf.solve_poisson(W)
"""
        sim['hamiltonian'] = myham
        sim['prerun'] = prerun

        return myham(W)

    P = createsim(W)

    sim2 = QuSimulation(filename)
    P2 = sim2['hamiltonian'](W)

    np.testing.assert_equal(P, P2)
