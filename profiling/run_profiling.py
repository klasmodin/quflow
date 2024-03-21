import numpy as np
import quflow as qf
import time
import prettytable
import io
from contextlib import redirect_stdout, redirect_stderr
from scipy.linalg.blas import zgemm, cgemm

import argparse
from datetime import datetime
import platform

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--single", help="Use single precision (default is double).", action="store_true")
parser.add_argument("-b", "--basename", help="Base for output textfile.", type=str, default="profile")
args = parser.parse_args()


############ PRERUN CODE BEGIN #############
#_PRERUNCODE_
############ PRERUN CODE END ###############

def profile_mat2shr(W, repeats):
    omega = qf.mat2shr(W)
    start_time = time.time()
    for k in range(repeats):
        omega = qf.mat2shr(W)
    return (time.time()-start_time)/repeats


def profile_shr2mat(W, repeats):
    omega = np.random.randn(W.shape[-1]**2)
    P = qf.shr2mat(omega, N=W.shape[-1])
    start_time = time.time()
    for k in range(repeats):
        P = qf.shr2mat(omega, N=W.shape[-1])
    return (time.time()-start_time)/repeats


def profile_commutator(W, repeats, skewherm=True, repeat_correction=100):
    repeats *= repeat_correction
    P = W.copy()
    if skewherm:
        comm = qf.commutator_skewherm
    else:
        comm = qf.commutator_generic
    X = comm(P, W)
    start_time = time.time()
    for k in range(repeats):
        X = comm(P, W)
    return (time.time()-start_time)/repeats


def profile_matmul(W, repeats, repeat_correction=100):
    repeats *= repeat_correction
    P = W.copy()
    X = W.copy()
    start_time = time.time()
    for k in range(repeats):
        np.matmul(W, P, out=X)
    return (time.time()-start_time)/repeats


def profile_gemm(W, repeats, repeat_correction=100):
    repeats *= repeat_correction
    P = W.copy()
    X = W.copy()
    if W.dtype == np.complex128:
        gemm = zgemm
        complexunit = 1.0 + 0.0j
    elif W.dtype == np.complex64:
        gemm = cgemm
        complexunit = np.complex64(1.0 + 0.0j)
    start_time = time.time()
    for k in range(repeats):
        gemm(complexunit, W, P, c=X)
    return (time.time()-start_time)/repeats


def profile_poisson_cpu(W, repeats, repeat_correction=10):
    repeats *= repeat_correction
    P = qf.laplacian.cpu.solve_poisson(W)
    start_time = time.time()
    for k in range(repeats):
        P = qf.laplacian.cpu.solve_poisson(W)
    return (time.time()-start_time)/repeats


def profile_poisson_gpu(W, repeats, repeat_correction=10):
    repeats *= repeat_correction
    P = qf.laplacian.gpu.solve_poisson(W)
    start_time = time.time()
    for k in range(repeats):
        P = qf.laplacian.gpu.solve_poisson(W)
    return (time.time()-start_time)/repeats


def profile_isomp(W, repeats, compsum=True):
    start_time = time.time()
    qf.isomp_fixedpoint2(W, stepsize=0.01, steps=repeats, minit=10, maxit=10, compsum=compsum)
    return (time.time()-start_time)/repeats


# Get some initial conditions
lmax = 10  # How many spherical harmonics (SH) coefficients to include
np.random.seed(42)  # For reproducability
omega0 = np.random.randn(lmax**2)  # Array with SH coefficients
omega0[0] = 0.0  # Set vanishing total circulation
omega0[1:4] = 0.0  # Set vanishing total angular momentum

# Create pretty table
tab = prettytable.PrettyTable()
tab.field_names = ["N", 
                   "matmul", 
                   "gemm", 
                   "commutator", 
                   "shr2mat", 
                   "mat2shr", 
                   "poisson_cpu", 
                   "poisson_gpu", 
                   "isomp"]

# Create N list
N_list = [2**k for k in range(5,11)]

# Platform
args.basename += "_" + platform.machine()

# dtype
if args.single:
    dtype = np.complex64
    args.basename += "_c"
else:
    dtype = np.complex128
    args.basename += "_z"

# Date
args.basename += "_" + datetime.today().strftime('%Y-%m-%d') 

filename = args.basename + ".txt"

# Print system info
fs = io.StringIO()
with redirect_stdout(fs):
    np.show_config()
    np.show_runtime()
with open(filename, 'w') as f:
    with redirect_stdout(f):
        print("\n----------------- System info ------------------")
        print("Platform: {}\n".format(platform.platform()))
        print("Numpy show_config and show_runtime:\n{}".format(fs.getvalue()))
        try:
            import mkl
        except ImportError:
            print("MKL module: False")
        else:
            print("MKL module: True")
            print("MKL cores (get_max_threads): {}".format(mkl.get_max_threads()))
        print("\ndtype: {}".format(dtype))

        # Run profiling
        for N in N_list:
            W0 = qf.shr2mat(omega0, N=N).astype(dtype)
            P0 = qf.solve_poisson(W0)

            results = [N]
            repeats = 2**11//N

            for funstr in tab.field_names[1:]:
                t = eval("profile_"+funstr)(W0.copy(), repeats=repeats)
                results.append("{:.1E}".format(t))
            
            tab.add_row(results)

        # Print results to output
        print("\n------------ Results from profiling ------------")
        print("(average time in seconds per evaluation or step)")
        print(tab)
