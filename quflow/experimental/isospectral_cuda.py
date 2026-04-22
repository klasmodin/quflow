import math
import numpy as np
import cupy as cp
from nvmath.linalg.advanced import Matmul

from .cuda import DiagTriDiagOp
from ..geometry import hbar

# performs PWcomm -= PWcomm.conj().T
kernel_src = r'''
#include <thrust/complex.h>

template <typename T>
__device__ inline void conj_subtract_body(thrust::complex<T>* __restrict__ A,
                                          int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ( (i>=N) || (j>i) ) return;
    
    thrust::complex<T> a_ij = A[i*N + j];
    thrust::complex<T> a_ji = A[i + j*N];
    
    A[i*N + j] -= thrust::conj(a_ji);
    
    if (i != j)
    {
      A[i + j*N] -= thrust::conj(a_ij);
    }
    
}

extern "C" __global__
void conj_subtract_f(thrust::complex<float>* __restrict__ A,
                     int N)
{
    conj_subtract_body<float>(A, N);
}

extern "C" __global__
void conj_subtract_d(thrust::complex<double>* __restrict__ A,
                     int N)
{
    conj_subtract_body<double>(A, N);
}
''';


class IsompCUDA(object):

    def __init__(self, N, dtype):
        
        self.N = N
        
        prop = cp.cuda.runtime.getDeviceProperties(0)
        cc = f"{prop['major']}{prop['minor']}"
        self.c_mod = cp.RawModule(code=kernel_src,
                                  backend='nvcc',
                                  options=(f'-arch=sm_{cc}','-std=c++17','-O3',),)
        
        dtype = cp.dtype(dtype)
        
        if dtype == cp.complex64:
            self.conj_subtract_ker = self.c_mod.get_function("conj_subtract_f")
        elif dtype == cp.complex128:
            self.conj_subtract_ker = self.c_mod.get_function("conj_subtract_d")
        else:
            raise TypeError("W0 has to be of dtype complex64 or complex128.")
          
        self.dW     = cp.zeros((N, N), dtype=dtype)
        self.dW_old = cp.zeros((N, N), dtype=dtype)
        self.Whalf  = cp.zeros((N, N), dtype=dtype)
        self.Phalf  = cp.zeros((N, N), dtype=dtype)
        self.PWcomm = cp.zeros((N, N), dtype=dtype)

        self.set_matmul_state()

# -------------------------------------------------------------------------------------- #
    def set_matmul_state(self):
      
        # C --> alpha * AxB + beta * C
        if self.Phalf.dtype == cp.complex128:
            alpha = 1.0
            beta  = 0.0
        elif self.Phalf.dtype == cp.complex64:
            alpha = cp.float64(1.0)
            beta = cp.float64(0.0)
    
        self.matmul = Matmul(self.Phalf,
                             self.Whalf,
                             c=self.PWcomm,
                             alpha=alpha,
                             beta=beta)
        self.matmul.plan()
        self.matmul.autotune()
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
    def conj_subtract(self,X):
        
        threads = (16, 16)
        blocks  = (math.ceil(self.N / threads[0]),
                   math.ceil(self.N / threads[1]))
        
        self.conj_subtract_ker(blocks,threads,(X,self.N))
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
    def cuda_mul(self,A,B,C):
        
        self.matmul.reset_operands(a=A, b=B, c=C)
        return self.matmul.execute()
# -------------------------------------------------------------------------------------- #
        

    def __call__(self,
                 W: cp.ndarray|np.ndarray,
                 dt: np.float64|np.float32, 
                 steps: int=100, 
                 hamiltonian: DiagTriDiagOp=None, 
                 time: np.float64|np.float32=None, 
                 forcing=None, 
                 strang_splitting=None,
                 stats=None,
                 callback=None,
                 tol='auto', 
                 maxit: int=10, 
                 minit: int=1, 
                 verbatim: bool=False, 
                 compsum=False, 
                 reinitialize: bool=True
                ):
        """
        Time-stepping by isospectral midpoint second order method for skew-Hermitian W
        using fixed-point iterations. This implementation uses a CUDA-device to speed
        up the computations.

        Parameters
        ----------
        W: ndarray (cupy array or numpy array)
            Initial skew-Hermitian vorticity matrix (overwritten and returned).
        dt: float
            Time step length.
        steps: int
            Number of steps to take.
        hamiltonian: function(W) or function(W, time)
            The Hamiltonian returning a stream matrix.
        time: float or None (default)
            Time at the initial state. If `None` the system is assumed to
            be autonomous, and the time parameter is not passed to the hamiltonian and forcing.
        forcing: function(P, W) or function(P, W, time) or None (default)
            Extra force function (to allow non-isospectral perturbations).
        strang_splitting: function(dt, W) or None (default)
            Strang splitting updates applied before and after each basic step.
        stats: dict or None
            Dictionary to be filled in with integration statistics.
        callback: function(W, dW, stats) or None (default)
            Callback function evaluated at the end of every step, 
            just before W is updated W += dW.
        tol: float or 'auto' (default)
            Tolerance for iterations. Negative value or "auto" means automatic choice.
        maxit: int
            Maximum number of iterations.
        minit: int
            Minimum number of iterations.
        verbatim: bool
            Print extra information if True. Default is False.
        reinitialize: bool
            Whether to re-initiate the iteration vector at every step.

        Returns
        -------
        W: ndarray
        """
        
        # check if W is cupy array
        if not isinstance(W, cp.ndarray):
            # print("Warning: passed numpy array W at cuda integrator")
            W = cp.asarray(W, dtype=self.dW.dtype)

        # Check input
        assert minit >= 1, "minit must be at least 1."
        assert maxit >= minit, "maxit must be at minit."

        # Check if force function is autonomous
        if forcing is not None:
            raise NotImplementedError("Forcing is not yet implemented.")
            autonomous_force = True
            if time is not None:
                try:
                    FW = forcing(W, W, time=time)
                except TypeError:
                    pass
                else:
                    autonomous_force = False
            FW = cp.zeros_like(W)
        
        # Check if autonomous
        autonomous = True
        if time is not None:
            autonomous = False
        
        # Stats variables
        total_iterations = 0
        number_of_maxit = 0

        # Define constants
        hb = hbar(N=W.shape[-1])
        vareps = dt/(2*hb)
        zero_real = 0.0

        # Convert constants to single precision if needed
        if self.Phalf.dtype == cp.complex64:
            hb = cp.float32(hb)
            vareps = cp.float32(vareps)
            zero_real = cp.float32(zero_real)
            dt = cp.float32(dt)
            tol = cp.float32(tol)

        # Specify tolerance if needed
        if (tol == 'auto') or (tol < 0):
            mach_eps = cp.finfo(W.dtype).eps
            if not compsum:
                mach_eps = np.sqrt(mach_eps)
            if W.ndim > 2:
                zeroind = (0,)*(W.ndim-2) + (Ellipsis,)
                tol = (mach_eps*dt/hb)*cp.linalg.norm(W[zeroind], np.inf)
            else:
                tol = (mach_eps*dt/hb)*cp.linalg.norm(W, np.inf)
                
            if verbatim:
                print("Tolerance set to {}.".format(tol))
            if stats:
                stats['tol'] = tol.get()

        # Convert tolerance to single precision if needed
        if self.Phalf.dtype == cp.complex64:
            tol = cp.float32(tol)

        # --- Beginning of step loop ---
        for k in range(steps):

            # Apply half a Strang step
            if strang_splitting:
                print("WARNING: is strang splitting implemented on cuda?")
                W = strang_splitting(dt/2, W)

            # Per step updates
            resnorm = np.inf
            if reinitialize:
                self.dW.fill(zero_real)

            # --- Beginning of iterations ---
            for i in range(maxit):

                # Update iterations
                total_iterations += 1

                # Compute Wtilde
                cp.copyto(self.Whalf, W)
                self.Whalf += self.dW

                # Save dW from previous step
                cp.copyto(self.dW_old, self.dW)

                # Update Ptilde
                hamiltonian(self.Phalf, self.Whalf)
                
                self.Phalf *= vareps
                
                # Compute commutator + correction
                self.PWcomm = self.cuda_mul(self.Phalf, self.Whalf, self.PWcomm)
                
                self.dW = self.cuda_mul(self.PWcomm, self.Phalf, self.dW)
                
                self.conj_subtract(self.PWcomm)

                # Update dW
                self.dW += self.PWcomm

                # Add forcing if needed
                if forcing:
                    raise NotImplementedError("Forcing is not yet implemented.")
                    # Compute forcing if needed
                    Phalf /= vareps
                    if autonomous_force:
                        FW = forcing(self.Phalf, self.Whalf)
                    else:
                        FW = forcing(self.Phalf, self.Whalf, time=time + dt/2)
                    FW *= dt/2
                    self.dW += FW

                # Check if time to break
                if i+1 >= minit:
                    # Compute error
                    resnorm_old = resnorm
                    self.dW_old -= self.dW
                    # if dW_old.ndim > 2:
                    #     resnormvec = scipy.linalg.norm(dW_old, ord=np.inf, axis=(-1, -2))
                    #     if Phalf.ndim == 2:
                    #         resnorm = resnormvec[0]
                    #     else:
                    #         resnorm = resnormvec.max()
                    # else:
                    resnorm = cp.linalg.norm(self.dW_old, ord=np.inf)
                    if resnorm <= tol or resnorm >= resnorm_old:
                        break

            else:
                # We used maxit iterations
                number_of_maxit += 1
                if verbatim:
                    print("Max iterations {} reached at step {}.".format(maxit, k))
                # if stats:
                #     stats['maxit_reached'] = True

            # Update W
            self.PWcomm *= 2

            # Evaluate callback function
            if callback is not None:
                raise NotImplementedError("callback is not yet implemented.")
                callback(W, self.PWcomm)

            W += self.PWcomm

            if forcing:
                FW *= 2
                W += FW
        
            if time is not None:
                time += dt.get()

            # Apply half a Strang step
            if strang_splitting:
                W = strang_splitting(dt/2, W)

            # --- End of step ---
        
        # print("Iterations per step: ",total_iterations/steps)
        
        if verbatim:
            print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))
        if stats:
            stats["iterations"] = total_iterations/steps
            stats["maxit"] = number_of_maxit/steps
        
        return W.get()

# -------------------------------------------------------------------------------------- #
    def timing(self,
                 W: cp.ndarray,
                 dt: np.float64|np.float32, 
                 steps: int=100, 
                 hamiltonian: DiagTriDiagOp=None, 
                 time: np.float64|np.float32=None, 
                 forcing=None, 
                 strang_splitting=None,
                 stats=None,
                 callback=None,
                 tol='auto', 
                 maxit: int=10, 
                 minit: int=1, 
                 verbatim: bool=False, 
                 compsum=False, 
                 reinitialize: bool=False
              ):
              
              
        stream = cp.cuda.get_current_stream()
    
        start = cp.cuda.Event()
        end   = cp.cuda.Event()

        # Stats variables
        total_iterations = 0
        number_of_maxit = 0

        # Define constants
        hb = hbar(N=W.shape[-1])
        vareps = dt/(2*hb)

        # Specify tolerance if needed
        stream.synchronize()
        start.record(stream)
        
        if (tol == 'auto') or (tol < 0):
            mach_eps = cp.finfo(W.dtype).eps
            if not compsum:
                mach_eps = np.sqrt(mach_eps)
            if W.ndim > 2:
                zeroind = (0,)*(W.ndim-2) + (Ellipsis,)
                tol = (mach_eps*dt/hb)*cp.linalg.norm(W[zeroind], np.inf)
            else:
                tol = (mach_eps*dt/hb)*cp.linalg.norm(W, np.inf)
            if verbatim:
                print("Tolerance set to {}.".format(tol))
            if stats:
                stats['tol'] = tol
        
        end.record(stream)
        end.synchronize()

        ms = cp.cuda.get_elapsed_time(start, end)
        print(" setting tolerance ",ms / 1000.0)  

        # --- Beginning of step loop ---
        for k in range(steps):

            # Per step updates
            resnorm = np.inf
            if reinitialize:
                self.dW.fill(0.0)

            # --- Beginning of iterations ---
            for i in range(maxit):

                # Update iterations
                total_iterations += 1

                # Compute Wtilde
                stream.synchronize()
                start.record(stream)
                
                cp.copyto(self.Whalf, W)
                self.Whalf += self.dW

                # Save dW from previous step
                cp.copyto(self.dW_old, self.dW)
                
                end.record(stream)
                end.synchronize()
                ms = cp.cuda.get_elapsed_time(start, end)
                print(" copy and += ",ms / 1000.0)  

                # Update Ptilde
                stream.synchronize()
                start.record(stream)
                
                hamiltonian(self.Phalf, self.Whalf)
                self.Phalf *= vareps

                end.record(stream)
                end.synchronize()
                ms = cp.cuda.get_elapsed_time(start, end)
                print(" Stream function ",ms / 1000.0)  
                
                # Compute commutator + correction
                stream.synchronize()
                start.record(stream)
                self.PWcomm = self.cuda_mul(self.Phalf, self.Whalf, self.PWcomm)
                
                self.dW = self.cuda_mul(self.PWcomm, self.Phalf, self.dW)
                
                end.record(stream)
                end.synchronize()
                ms = cp.cuda.get_elapsed_time(start, end)
                print(" Two multiplies ",ms / 1000.0)                  
                
                stream.synchronize()
                start.record(stream)
                
                self.conj_subtract(self.PWcomm)

                end.record(stream)
                end.synchronize()
                ms = cp.cuda.get_elapsed_time(start, end)
                print(" Conjugate subtract ",ms / 1000.0)   

                # Update dW
                self.dW += self.PWcomm

                # Check if time to break
                if i+1 >= minit:
                    # Compute error
                    stream.synchronize()
                    start.record(stream)
                    
                    resnorm_old = resnorm
                    self.dW_old -= self.dW
                    resnorm = cp.linalg.norm(self.dW_old, ord=np.inf)
                    if resnorm <= tol or resnorm >= resnorm_old:
                        break
                
                    end.record(stream)
                    end.synchronize()
                    ms = cp.cuda.get_elapsed_time(start, end)
                    print(" resnorm computation ",ms / 1000.0)   

            else:
                # We used maxit iterations
                number_of_maxit += 1
                if verbatim:
                    print("Max iterations {} reached at step {}.".format(maxit, k))

            # Update W
            self.PWcomm *= 2

            W += self.PWcomm
        
            if time is not None:
                time += dt

            # --- End of step ---
        
        return W
# -------------------------------------------------------------------------------------- #