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
* `h5py`
* `appdirs`

Optional:

* `matplotlib`
* `cartopy`
* `ffmpeg`
* `pytest`
* `tqdm`
* `jupyter`
* `ipywidgets`
* `prettytable`

## Installation

The module may be installed directly from the repository:
```
> git clone https://github.com/kmodin/quflow.git
> cd quflow
> pip install .
```

Alternatively, it can be installed via the included `conda` environment 
description file `quflow-env.yaml`. A new conda environment named `quflow` 
can be installed via
```
> conda env create -f quflow-env.yaml
```
If you're running a Mac with the Apple Silicon hardware, 
you might want to run the following command after:
```
> conda install "libblas=*=*accelerate"
```
On the M1 Pro hardware, it gives a speedup of about a factor 5. 

## Quick Start

Tests can be run using `pytest` to confirm that the installation was successful.

An example notebook `notebooks/basic-example.ipynb` demonstrates the basic functionality. 

## Remarks

- For large HDF5 file, this is how to [partially copy files with `rsync`](https://fedoramagazine.org/copying-large-files-with-rsync-and-some-misconceptions/).

## References

[1] K. Modin and M. Viviani. *A Casimir preserving scheme for long-time simulation of spherical ideal hydrodynamics*, J. Fluid Mech., 884:A22, 2020.
