import numpy as np
import quflow as qf
import time
import prettytable
import io
from contextlib import redirect_stdout, redirect_stderr


############ PRERUN CODE BEGIN #############
#_PRERUNCODE_
############ PRERUN CODE END ###############

def profile_mat2shr(W, repeats):
    start_time = time.time()
    for k in range(repeats):
        omega = qf.mat2shr(W)
    return (time.time()-start_time)/repeats


def profile_shr2mat(W, repeats):
    start_time = time.time()
    omega = np.random.randn(W.shape[-1])
    for k in range(repeats):
        P = qf.shr2mat(omega, N=W.shape[-1])
    return (time.time()-start_time)/repeats


def profile_matmul(W, repeats, skewherm=True, repeat_correction=100):
    repeats *= repeat_correction
    P = W.copy()
    if skewherm:
        comm = qf.commutator_skewherm
    else:
        comm = qf.commutator_generic
    start_time = time.time()
    for k in range(repeats):
        X = comm(P, W)
    return (time.time()-start_time)/repeats


def profile_laplacian(W, repeats, repeat_correction=10):
    repeats *= repeat_correction
    start_time = time.time()
    for k in range(repeats):
        P = qf.solve_poisson(W)
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
tab.field_names = ["N", "matmul", "shr2mat", "mat2shr", "laplacian", "isomp"]

# Create N list
N_list = [2**k for k in range(5,11)]

# Print system info
print("\n----------------- System info ------------------")
f = io.StringIO()
with redirect_stdout(f):
    np.show_config()
for line in f.getvalue().split('\n'):
    if "libraries = " in line:
        print("BLAS libraries: {}".format(line.split("libraries = ")[-1]))
    if "library_dirs = " in line:
        print("BLAS library dirs: {}".format(line.split("library_dirs = ")[-1]))
        break
try:
    import mkl
except ImportError:
    print("MKL BLAS: False")
else:
    print("MKL BLAS: True")
    print("MKL cores (get_max_threads): {}".format(mkl.get_max_threads()))


# Run profiling
for N in N_list:
    W0 = qf.shr2mat(omega0, N=N)
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
