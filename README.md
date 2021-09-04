# quflow

A Python module for quantized vorticity flows. 
The code is based on a paper by Modin and Viviani (2020) [1] 
where a quantized Euler equation on the sphere is presented.

## Spherical coordinates

We use the following convention for spherical coordinates:

![](https://upload.wikimedia.org/wikipedia/commons/4/4f/3D_Spherical.svg)

Here, $\theta \in [0,\pi]$ is the *inclination* and $\phi \in [0,2\pi)$ is the *azimuth*.

## Dependencies

Required:

* `numpy`
* `numba`
* `scipy`
* `pyssht`

Optional:

* `matplotlib`
* `pytest`

## Installation

The module may be installed directly from the repository:
```
> git clone https://github.com/kmodin/quflow.git
> python setup.py install
# or
> pip install .
```

## Quick Start

Tests can be run using `pytest` to confirm that the installation was successful.

An example notebook `notebooks/basic-example.ipynb` demonstrates the basic functionality. 

## TODOs

- If I use HDF5, this is how to [partially copy files with `rsync`](https://fedoramagazine.org/copying-large-files-with-rsync-and-some-misconceptions/).

## References

[1] K. Modin and M. Viviani. *A Casimir preserving scheme for long-time simulation of spherical ideal hydrodynamics*, J. Fluid Mech., 884:A22, 2020.
