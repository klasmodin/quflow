quflow
======

*quflow* simulates the incompressible Euler equations on the sphere using
matrix hydrodynamics: the vorticity field is quantized into an
:math:`N\times N` skew-Hermitian matrix :math:`W`, and the dynamics becomes the
isospectral flow

.. math::

   \dot W = \frac{1}{\hbar}[P, W], \qquad \Delta_N P = W,
   \qquad \hbar = \frac{2}{\sqrt{N^2-1}} .

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   conventions
   api/index

References
----------

* K. Modin and M. Viviani. *A Casimir preserving scheme for long-time
  simulation of spherical ideal hydrodynamics*, J. Fluid Mech., 884:A22, 2020.
* K. Modin and M. Viviani. *A brief introduction to matrix
  hydrodynamics*, J. Comput. Dyn. 14, 17–35, 2026.
