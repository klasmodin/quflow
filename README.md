# quflow

A Python module for quantized vorticity flows.

## Spherical coordinates

We use the following convention for spherical coordinates:

![](https://upload.wikimedia.org/wikipedia/commons/4/4f/3D_Spherical.svg)

Here, $\theta \in [0,\pi]$ is the *inclination* and $\phi \in [0,2\pi)$ is the *azimuth*.

## TODOs

- If I use HDF5, this is how to [partially copy files with `rsync`](https://fedoramagazine.org/copying-large-files-with-rsync-and-some-misconceptions/).

## OLD BELOW

![](https://github.com/aelanman/pyspherical/workflows/Tests/badge.svg?branch=master)
![](https://codecov.io/gh/aelanman/pyspherical/branch/master/graph/badge.svg)


An implementation of the fast spin-weighted spherical harmonic transform methods of McEwan and Wiaux (2011) [1], using
the recursion relations of Trapani and Navaza (2006) [2] to calculate Wigner-d functions. Transforms are
supported for any spherical sampling pattern with equally-spaced samples of azimuth at each latitude (iso-latitude sampling).
Additional functions are provided to evaluate spin-weighted spherical harmonics at arbitrary positions.

These methods are written entirely in Python, taking advantage of numba jit compilation and numpy vector operations
for speed.

This README, a tutorial, and all function docstrings may be found on [ReadTheDocs](https://pyspherical.readthedocs.io).

## Dependencies

Required:

* `numpy`
* `numba`
* `scipy`

Optional:

* `sympy`
* `pytest`

## Installation

The latest release of `pyspherical` is available on PyPi:
```
pip install pyspherical
```

The bleeding-edge version may be installed directly from the repository:
```
> git clone https://github.com/aelanman/pyspherical.git
> python setup.py install
# or
> pip install .
```

## Quick Start

Tests can be run using `pytest` to confirm that the installation was successful.

An example script `scripts/example_1.py` demonstrates how to use some of the available evaluation and transform functions. Another script `scripts/example_2.py` plots the spherical harmonics for el < 4. Further documentation is under development.


## References

[1] McEwen, J. D., and Y. Wiaux. “A Novel Sampling Theorem on the Sphere.” IEEE Transactions on Signal Processing, vol. 59, no. 12, Dec. 2011, pp. 5876–87. arXiv.org, doi:10.1109/TSP.2011.2166394.

[2] S, Trapani, and Navaza J. “Calculation of Spherical Harmonics and Wigner d Functions by FFT. Applications to Fast Rotational Matching in Molecular Replacement and Implementation into AMoRe.” Acta Crystallographica Section A, vol. 62, no. 4, 2006, pp. 262–69. Wiley Online Library, doi:10.1107/S0108767306017478.
