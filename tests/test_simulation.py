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
@pytest.mark.parametrize("datapath", ["/", "mypath/"])
def test_init_sim_(W, t, datapath, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    # filename = "/Users/moklas/Downloads/testsim.hdf5"

    sim = QuSimulation(filename, overwrite=True, datapath=datapath, W=W, time=t, energy=0.0, enstrophy=0.0)
    sim['hamiltonian'] = qf.solve_poisson
    sim2 = QuSimulation(filename, datapath=datapath)

    assert sim.qutypes == sim2.qutypes
    assert sim['hamiltonian'] == qf.solve_poisson
    assert sim2['hamiltonian'] == qf.solve_poisson
    for name in ["mat", "time", "energy", "enstrophy"]:
        np.testing.assert_equal(sim[name], sim2[name])


@pytest.mark.parametrize("W", [get_random_mat(35)])
@pytest.mark.parametrize("datapath", ["/", "/mypath/"])
def test_callback(W, datapath, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    N = W.shape[-1]
    sim = QuSimulation(filename, overwrite=True, datapath=datapath, W=W, energy=0.7)

    Wlist = np.zeros((10, N, N), dtype=np.complex128)
    Wlist[0, ...] = W
    for W in Wlist[1:, ...]:
        W[:, :] = get_random_mat(W.shape[-1])
        sim(W=W, delta_time=0.1, delta_steps=4, energy=0.7)

    np.testing.assert_allclose(sim['time'], np.arange(0.0, 1.0, 0.1))
    np.testing.assert_equal(sim['step'], np.arange(0, 10*4, 4))
    np.testing.assert_equal(sim['mat', :], Wlist)
    np.testing.assert_equal(sim['energy', :], np.full(10, 0.7))
    np.testing.assert_equal(qf.shr2fun(qf.mat2shr(Wlist[-1])).astype(np.float32), sim['fun', -1])


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

    varname = qf.simulation._default_qutype2varname['mat']
    if varname in sim.fieldnames:
        assert sim[varname, -1].dtype == qutypes['mat']
    varname = qf.simulation._default_qutype2varname['shr']
    if varname in sim.fieldnames:
        assert sim[varname, -1].dtype == qutypes['shr']
    varname = qf.simulation._default_qutype2varname['shc']
    if varname in sim.fieldnames:
        assert sim[varname, -1].dtype == qutypes['shc'] if qutypes['shc'] is not None else W.dtype
    varname = qf.simulation._default_qutype2varname['fun']
    if varname in sim.fieldnames:
        assert sim[varname, -1].dtype == qutypes['fun']


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


@pytest.mark.parametrize("W", [get_random_mat(35)])
def test_logger(W, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    N = W.shape[-1]

    def vector_output(W):
        return W[:, 0]

    sim = QuSimulation(filename, overwrite=True, W=W, loggers={'normL2': qf.geometry.norm_L2, 'vector': vector_output})

    Wlist = np.zeros((10, N, N), dtype=np.complex128)
    Wlist[0, ...] = W
    for W in Wlist[1:, ...]:
        W[:, :] = get_random_mat(W.shape[-1])
        sim(W=W, delta_time=0.1, delta_steps=4)

    np.testing.assert_equal(sim['normL2', -1], qf.geometry.norm_L2(Wlist[-1]))
    np.testing.assert_equal(sim['vector', 3], vector_output(Wlist[3]))


@pytest.mark.parametrize("W", [get_random_mat(35)])
def test_solve(W, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    N = W.shape[-1]

    sim = QuSimulation(filename, overwrite=True,  W=W, loggers={'normL2': qf.geometry.norm_L2})

    qf.simulation.solve(W, stepsize=0.1, steps=100, inner_steps=10, progress_bar=False, callback=sim)

    # print(sim['time'])
    # print(sim['step'])

    np.testing.assert_allclose(qf.qtime2seconds(1.0, N=N)*np.arange(11), sim['time'])
    np.testing.assert_equal(10*np.arange(11), sim['step'])
    np.testing.assert_equal(qf.geometry.norm_L2(sim['mat', -1]), sim['normL2', -1])


@pytest.mark.parametrize("W", [get_random_mat(35)])
def test_solve_restart(W, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    N = W.shape[-1]

    sim = QuSimulation(filename, overwrite=True,  W=W)

    qf.simulation.solve(W.copy(), stepsize=0.1, steps=50, inner_steps=10, progress_bar=False, callback=sim)

    sim2 = QuSimulation(filename)

    qf.simulation.solve(sim2['mat', -1], stepsize=0.1, steps=50, inner_steps=10, progress_bar=False, callback=sim)

    filename3 = tmpdir.join("testsim3.hdf5")
    sim3 = QuSimulation(filename3, overwrite=True,  W=W)

    qf.simulation.solve(W.copy(), stepsize=0.1, steps=100, inner_steps=10, progress_bar=False, callback=sim3)

    np.testing.assert_allclose(qf.qtime2seconds(1.0, N=N)*np.arange(11), sim['time'])
    np.testing.assert_equal(10*np.arange(11), sim['step'])

    np.testing.assert_equal(sim3['mat', -1], sim['mat', -1])



