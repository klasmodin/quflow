import h5py
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

    sim = QuSimulation(filename, overwrite=True, datapath=datapath, state=W, time=t, energy=0.0, enstrophy=0.0)
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
    sim = QuSimulation(filename, overwrite=True, datapath=datapath, state=W, energy=0.7)

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
    sim = QuSimulation(filename, overwrite=True, qutypes=qutypes, state=W)

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


@pytest.mark.parametrize("W", [get_random_mat(35)])
def test_logger(W, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    N = W.shape[-1]

    def vector_output(W):
        return W[:, 0]

    sim = QuSimulation(filename, overwrite=True, state=W, loggers={'normL2': qf.geometry.norm_L2, 'vector': vector_output})

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

    sim = QuSimulation(filename, overwrite=True,  state=W, loggers={'normL2': qf.geometry.norm_L2})

    qf.simulation.solve(W, stepsize=0.1, steps=100, steps_out=10, progress_bar=False, callback=sim)

    # print(sim['time'])
    # print(sim['step'])

    np.testing.assert_allclose(qf.qtime2seconds(1.0, N=N)*np.arange(11), sim['time'])
    np.testing.assert_equal(10*np.arange(11), sim['step'])
    np.testing.assert_equal(qf.geometry.norm_L2(sim['mat', -1]), sim['normL2', -1])


@pytest.mark.parametrize("W", [get_random_mat(35)])
def test_solve_restart(W, tmpdir):
    filename = tmpdir.join("testsim.hdf5")
    N = W.shape[-1]

    sim = QuSimulation(filename, overwrite=True,  state=W)

    qf.simulation.solve(W.copy(), stepsize=0.1, steps=50, steps_out=10, progress_bar=False, callback=sim)

    sim2 = QuSimulation(filename)

    qf.simulation.solve(sim2['mat', -1], stepsize=0.1, steps=50, steps_out=10, progress_bar=False, callback=sim)

    filename3 = tmpdir.join("testsim3.hdf5")
    sim3 = QuSimulation(filename3, overwrite=True,  state=W)

    qf.simulation.solve(W.copy(), stepsize=0.1, steps=100, steps_out=10, progress_bar=False, callback=sim3)

    np.testing.assert_allclose(qf.qtime2seconds(1.0, N=N)*np.arange(11), sim['time'])
    np.testing.assert_equal(10*np.arange(11), sim['step'])

    np.testing.assert_equal(sim3['mat', -1], sim['mat', -1])





def _seeded_skewh(N, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) + 1j*rng.standard_normal((N, N))
    return A - A.conj().T


def test_unpicklable_logger_warns_and_reopens(tmp_path):
    # A logger defined locally cannot be pickled (AttributeError up to Python 3.13,
    # PicklingError from 3.14). The simulation should warn, keep logging in memory,
    # and be possible to reopen with the logger supplied again.
    filename = tmp_path / "testsim.hdf5"
    W = _seeded_skewh(16)

    def first_column(W):
        return W[:, 0]

    with pytest.warns(UserWarning, match="could not be pickled"):
        sim = QuSimulation(filename, overwrite=True, state=W, loggers={'col': first_column})
    sim(W=W, delta_time=0.1)
    assert sim['col'].shape == (2, 16)

    # Reopening without loggers must not crash, but warn that none are stored
    with pytest.warns(UserWarning, match="No loggers are stored"):
        sim2 = QuSimulation(filename)
    assert sim2.loggers == {}

    # Reopening with the logger supplied again continues logging
    sim3 = QuSimulation(filename, loggers={'col': first_column})
    sim3(W=2*W, delta_time=0.1)
    np.testing.assert_equal(sim3['col', -1], 2*W[:, 0])
    assert sim3['col'].shape == (3, 16)


def test_picklable_logger_restored_on_reopen(tmp_path):
    filename = tmp_path / "testsim.hdf5"
    W = _seeded_skewh(16)
    QuSimulation(filename, overwrite=True, state=W, loggers={'normL2': qf.geometry.norm_L2})

    sim = QuSimulation(filename)
    assert sim.loggers == {'normL2': qf.geometry.norm_L2}


def test_unpicklable_hamiltonian_stored_by_name(tmp_path):
    # Functions that cannot be pickled are stored by name instead
    filename = tmp_path / "testsim.hdf5"
    sim = QuSimulation(filename, overwrite=True, state=_seeded_skewh(8))

    def myham(W):
        return qf.solve_poisson(W)

    sim['hamiltonian'] = myham
    with h5py.File(filename, "r") as f:
        assert f[sim.args_datapath].attrs['hamiltonian'] == 'myham'


# --- Documented behaviour ---------------------------------------------------------

def _state(N=8, seed=0):
    rng = np.random.default_rng(seed)
    return qf.shr2mat(np.r_[0.0, rng.standard_normal(N**2 - 1)], N=N)


def test_default_qutypes(tmp_path):
    W0 = _state()
    sim = QuSimulation(tmp_path/"sim.hdf5", overwrite=True, state=W0)
    omega = qf.mat2shr(W0)
    np.testing.assert_allclose(sim['fun', 0], qf.shr2fun(omega).astype(np.float32), rtol=0, atol=1e-6)
    np.testing.assert_allclose(sim['funL2', 0], qf.shr2fun(omega, berezin=False).astype(np.float32), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sim['mat', 0], W0)
    assert sim['fun'].shape == (1, 8, 15)


def test_funhalf_qutype(tmp_path):
    sim = QuSimulation(tmp_path/"sim.hdf5", overwrite=True, state=_state(), qutypes={'funhalf': np.float32})
    assert sim['fun'].shape == (1, 4, 7)


def test_solve_length_precedence():
    W0 = _state()
    counts = []

    def count(W, delta_time, delta_steps, **kwargs):
        counts.append(delta_steps)

    with pytest.warns(UserWarning, match="One, and only one"):
        qf.solve(W0.copy(), dt=0.1, steps=5, simtime=1.0, steps_out=3, callback=count, progress_bar=False)
    assert counts == [3, 3, 3, 1]   # simtime wins: round(1.0/0.1) = 10 steps


def test_solve_continues_simulation(tmp_path):
    W0 = _state()
    N = W0.shape[-1]
    sim = QuSimulation(tmp_path/"sim.hdf5", overwrite=True, state=W0)
    sim['dt'] = 0.1*qf.hbar(N)
    sim['steps'] = 10
    sim['dt_out'] = 0.5*qf.hbar(N)
    qf.solve(sim, progress_bar=False)
    np.testing.assert_allclose(sim['time'], [0.0, 0.5*qf.hbar(N), qf.hbar(N)], rtol=1e-12)
    np.testing.assert_array_equal(sim['step'], [0, 5, 10])
    # Same result as a direct integration, up to the fixed-point iteration
    # tolerance (solve calls isomp twice with 5 steps; each call restarts the
    # iteration from scratch)
    W = qf.isomp(W0.copy(), 0.1*qf.hbar(N), 10)
    np.testing.assert_allclose(sim['mat', -1], W, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(sim[-1], sim['mat', -1])


@pytest.mark.xfail(strict=True, reason="qutypes_iterator uses mat2shr instead of mat2shc for a complex 'fun'")
def test_complex_fun_qutype(tmp_path):
    W0 = _state()
    sim = QuSimulation(tmp_path/"sim.hdf5", overwrite=True, state=W0, qutypes={'fun': np.complex128})
    np.testing.assert_allclose(sim['fun', 0], qf.shc2fun(qf.mat2shc(W0)), rtol=0, atol=1e-12)


@pytest.mark.xfail(strict=True, reason="initialize_field creates but does not raise ValueError")
def test_invalid_field_value_rejected(tmp_path):
    with pytest.raises(ValueError):
        QuSimulation(tmp_path/"sim.hdf5", overwrite=True, state=_state(), mylist=[1, 2])


@pytest.mark.xfail(strict=True, reason="create_runfile writes the default filename without quotes")
def test_runfile_runs(tmp_path):
    import subprocess, sys
    sim = QuSimulation(tmp_path/"sim.hdf5", overwrite=True, state=_state())
    sim['prerun'] = "import quflow as qf"
    qf.simulation.create_runfile(sim)
    result = subprocess.run([sys.executable, str(tmp_path/"sim_runfile.py"), "--help"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@pytest.mark.xfail(strict=True, reason="solve fails with TypeError when no length is given")
def test_solve_without_length_raises_valueerror():
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            qf.solve(_state(), dt=0.1, progress_bar=False)
