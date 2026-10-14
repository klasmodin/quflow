Installation
============

From the repository::

   git clone https://github.com/kmodin/quflow.git
   cd quflow
   pip install .

or via the conda environment file::

   conda env create -f quflow-env.yaml

The environment uses the default BLAS/LAPACK from conda-forge (OpenBLAS). For a
significant speed-up, switch to a hardware-optimized BLAS afterwards. On a Mac
with Apple Silicon (Accelerate)::

   conda install -n quflow "libblas=*=*_newaccelerate"

On x86_64 hardware (Intel MKL)::

   conda install -n quflow "libblas=*=*mkl"

Run the test suite from the repository root to check the installation::

   python -m pytest
