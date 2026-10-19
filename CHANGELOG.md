# Changelog

## [0.2.0] - 2026-10-20

### Changed
Changes that can affect results or existing scripts:
- `solve` now returns the final state. Note that in Jupyter, a cell ending in
  `qf.solve(sim)` now displays the returned matrix; end the line with `;` or assign the result.
- `solve` uses a `progress_bar` setting stored in a `QuSimulation` (the parameter now
  defaults to `None`).
- Renamed `geometry.grad` to `geometry.sgrad` (it computes the symplectic gradient).
- `physics.sectional_curvature` now uses the quantized Poisson bracket
  `geometry.bracket` instead of the plain commutator, so its values are divided by
  `hbar**2` compared with before. It equals the sectional curvature of Modin and
  Perrot (Comm. Math. Phys., 2024) up to the factor `1/(4 pi)`.
- `solve_heat`, `solve_helmholtz` and `solve_viscdamp` (all Laplacian backends) keep the
  trace (mean) of the solution; only `solve_poisson` removes it. `solve_globalqg` now
  satisfies its equation also on the diagonal.
- The integrator statistics of `magmp` are reported as `'tol_auto'` and
  `'number_of_maxit'` (previously `'tol'` and `'maxit'`), as for `isomp`, so that
  `QuSimulation` records them.
- `isomp_quasinewton` and `isomp_simple` raise `NotImplementedError` for `forcing`
  (previously ignored silently), as does `isomp_quasinewton` in non-skew-Hermitian mode.
  The MHD integrators raise `NotImplementedError` in non-skew-Hermitian mode.
- `mat2shr(W, elmax=...)` returns the `(elmax+1)**2` coefficients with `el <= elmax`.
- Single-precision input to `shr2mat` uses the single-precision basis, as `mat2shr`
  already did.
- `pip install .` installs `h5py` and `matplotlib`, which `import quflow` requires.
  New optional extras: `test`, `docs` and `maps`.

### Added
- API documentation for the main modules (integrators, Laplacian backends,
  quantization, transforms, utilities, geometry, physics and simulation), built with
  Sphinx (`python -m sphinx -b html docs docs/_build/html`).
- Many new tests, including properties of the mathematics (conservation laws,
  convergence orders, consistency between backends).
- New README with installation instructions, a quick start and references.
- Random test data is now reproducible, and failed tests are listed in the
  pytest summary.

### Fixed
- `QuSimulation` with local loggers or functions on Python 3.14 and later
  (`PicklingError`); reopening a simulation whose loggers could not be stored.
- `isomp` and `magmp` modified arrays returned by the Hamiltonian or the forcing, which
  gave wrong results for a forcing cached per time step, and did not fill in empty
  `stats` dicts. In `isomp`, compensated summation with forcing failed only after the
  first step had modified the state; in the legacy `magmp_fixedpoint_old`, the time did
  not advance from 0.
- `select_skewherm` did not affect copies of `commutator` imported in other modules,
  nor the check in the MHD integrators.
- Laplacian: the sparse path of `solve_poisson` and the `gpu` backend did not remove the
  trace of `W`; `solve_viscdamp` cached its operator without `theta`; operator caches
  ignored the dtype; two unused CPU kernels were wrong.
- `inner_L2` and `norm_L2` gave wrong results for sparse input that was mixed with dense
  input or had different diagonals.
- `get_basis` cached the same basis twice; `basis_break_index` modified array input.
- `as_fun` lost the imaginary part of complex data; `cart2sph` failed for scalars.
- `QuSimulation`: complex `'fun'` output was wrong; invalid field values were stored
  as wrong data instead of raising `ValueError`.
- `create_runfile` produced scripts that always failed, and did not accept
  `pathlib.Path` file names.
- `solve` without `steps`, `simtime` or `endtime` raised an unclear `TypeError`
  (now `ValueError`).
- The isospectral reference test, which used data generated before the trace fix of
  `solve_poisson`.

### Removed
- `utils.run_cluster` (broken and unused; use `quflow.cluster`).
- The dependency on `pytest-runner`.

## [0.0.1] - 2021-02-24

### Added
- First version of the package.
