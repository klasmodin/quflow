import math
import time
import numpy as np
import cupy as cp
from nvmath.bindings import cusparse as cusp

kernel_src = r'''
#include <thrust/complex.h>

template <typename T>
__device__ inline void extract_body(const thrust::complex<T>* __restrict__ A,
                                    int N1,
                                    int N2,
                                    T* __restrict__ Xr,
                                    T* __restrict__ Xi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ( (i>=N1) || (j>=N2) ) return;
    
    int N = N1;
    
    int k = i + j*N;
    
    int row,col;
    
    thrust::complex<T> z;
    
    if (i <= N-j-1)
    {
      row = i+j;
      col = i;
    }
    else
    {
      row = i;
      col = i-(N-j);
    }

    z = A[row*N+col];
    Xr[k] = z.real();
    Xi[k] = z.imag();
}

template <typename T>
__device__ inline void reorder_body(const T* __restrict__ Xr,
                                    const T* __restrict__ Xi,
                                    int N1,
                                    int N2,
                                    thrust::complex<T>* __restrict__ A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ( (i>=N1) || (j>=N2) ) return;
    
    int N = N1;
    
    int k = i + j*N;
    
    int row,col;
    
    thrust::complex<T> z(Xr[k], Xi[k]);
    
    if (i <= N-j-1)
    {
      row = i+j;
      col = i;
    }
    else
    {
      row = i;
      col = i-(N-j);
    }

    A[row*N+col] = z;
    A[col*N+row] = -thrust::conj(z);
}

extern "C" __global__
void extract_diag_f(const thrust::complex<float>* __restrict__ A,
                    int N1,
                    int N2,
                    float* __restrict__ Xr,
                    float* __restrict__ Xi)
{
    extract_body<float>(A, N1, N2, Xr, Xi);
}

extern "C" __global__
void extract_diag_d(const thrust::complex<double>* __restrict__ A,
                    int N1,
                    int N2,
                    double* __restrict__ Xr,
                    double* __restrict__ Xi)
{
    extract_body<double>(A, N1, N2, Xr, Xi);
}

extern "C" __global__
void reorder_diag_f(const float* __restrict__ Xr,
                    const float* __restrict__ Xi,
                    int N1,
                    int N2,
                    thrust::complex<float>* __restrict__ A)
{
    reorder_body<float>(Xr, Xi, N1, N2, A);
}

extern "C" __global__
void reorder_diag_d(const double* __restrict__ Xr,
                    const double* __restrict__ Xi,
                    int N1,
                    int N2,
                    thrust::complex<double>* __restrict__ A)
{
    reorder_body<double>(Xr, Xi, N1, N2, A);
}
''';


class DiagTriDiagOp(object):

# -------------------------------------------------------------------------------------- # 
  def __init__(self, N, dtype, cuda_debug=False):
    
    self.N = N
    
    self.dtype = cp.dtype(dtype)

    allowed_dtypes = (cp.float32,cp.float64,cp.complex64,cp.complex128)
    
    if self.dtype not in allowed_dtypes:
      raise TypeError(f"Unsupported dtype {dtype}. Must be one of {allowed_dtypes}")
    
    if self.dtype == cp.float32 or self.dtype == cp.complex64:
      self.dtype = cp.float32
      self.single_precision = True
    else:
      self.dtype = cp.float64
      self.single_precision = False
    
    self.set_system()
    
    self.set_state()
    
    prop = cp.cuda.runtime.getDeviceProperties(0)
    cc = f"{prop['major']}{prop['minor']}"
    options = (f'-arch=sm_{cc}','-std=c++17','-O3',)
    if cuda_debug:
        options = (f'-arch=sm_{cc}','-std=c++17','-g','-G')
    self.c_mod = cp.RawModule(code=kernel_src,
                              backend='nvcc',
                              options=options,)

    if self.single_precision:
      self.extract_diag = self.c_mod.get_function("extract_diag_f")
      self.reorder_diag = self.c_mod.get_function("reorder_diag_f")
    else:
      self.extract_diag = self.c_mod.get_function("extract_diag_d")
      self.reorder_diag = self.c_mod.get_function("reorder_diag_d")
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
  def __call__(self, P: cp.ndarray, W: cp.ndarray):
    """
      Solve the Poisson equation
      
      ΔP = W
      i.e. return the stream matrix `P` for `W`.
      
      Parameters
      ----------
      P: ndarray
      W: ndarray(shape=(N, N) or (k, N, N), dtype=complex)
      
      Returns
      -------
      None
    """

    self.fill_rhs(W)

    self.solve_system()

    self.scale_main_diag()

    self.reorder_sol(P)
    
    # cp.cuda.Stream.null.synchronize()
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
  def timing(self, P: cp.ndarray, W: cp.ndarray):
    
    stream = cp.cuda.get_current_stream()
    
    start = cp.cuda.Event()
    end   = cp.cuda.Event()
    
    stream.synchronize()
    start.record(stream)
    self.fill_rhs(W)
    end.record(stream)
    end.synchronize()

    ms = cp.cuda.get_elapsed_time(start, end)
    print(" fill_rhs ",ms / 1000.0)  
    
    stream.synchronize()
    start.record(stream)
    self.solve_system()
    end.record(stream)
    end.synchronize()
    
    ms = cp.cuda.get_elapsed_time(start, end)
    print(" solve_system ",ms / 1000.0) 
    
    stream.synchronize()
    start.record(stream)
    self.scale_main_diag()
    end.record(stream)
    end.synchronize()

    ms = cp.cuda.get_elapsed_time(start, end)
    print(" scale_main_diag ",ms / 1000.0)  
    
    stream.synchronize()
    start.record(stream)
    self.reorder_sol(P)
    end.record(stream)
    end.synchronize()
    
    ms = cp.cuda.get_elapsed_time(start, end)
    print(" reorder_sol ",ms / 1000.0)    
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
  def set_system(self):
    
    np_dtype = np.dtype(self.dtype)
  
    d  = np.zeros(0,np_dtype)
    dl = np.zeros(0,np_dtype)
    du = np.zeros(0,np_dtype)
    
    N = self.N
    
    if N % 2 == 0:
      self.batch_count=int(N/2)+1
    else:
      self.batch_count=math.ceil(N/2)    
    
    s = (N-1)/2
    
    for i in range(self.batch_count):
      
      # fill up diagonal
      
      # upper batch
      m = i
      j = np.arange(0,N-m,dtype=np_dtype)
      d_ub = -2 * (s * (2 * j + 1 + m) - j * (j + m))
      if m==0:
        d_ub[0] += 1/2
      
      # lower batch
      m = N-i
      j = np.arange(0,N-m,dtype=np_dtype)
      d_lb = -2 * (s * (2 * j + 1 + m) - j * (j + m))
      
      d = np.append(d,d_ub)
      d = np.append(d,d_lb)
      
      # fill upper off diagonal
      
      # upper batch
      m = i
      j = np.arange(1,N-m+1,dtype=np_dtype)
      du_ub = np.zeros(N-m,dtype=np_dtype)
      du_ub[:-1] = np.sqrt((j[:-1]+m)*(N-j[:-1]-m))*np.sqrt(j[:-1]*(N-j[:-1]))
      
      # lower batch
      m = N-i
      j = np.arange(1,N-m+1,dtype=np_dtype)
      du_lb = np.zeros(N-m,dtype=np_dtype)
      du_lb[:-1] = np.sqrt((j[:-1]+m)*(N-j[:-1]-m))*np.sqrt(j[:-1]*(N-j[:-1]))
      
      du = np.append(du,du_ub)
      du = np.append(du,du_lb)
      
      # fill lower off diagonal
      
      # upper batch
      m = i
      dl_ub = np.zeros(N-m,dtype=np_dtype)
      dl_ub[1:] = du_ub[:-1]

      # lower batch
      m = N-i
      dl_lb = np.zeros(N-m,dtype=np_dtype)
      dl_lb[1:] = du_lb[:-1]
      
      dl = np.append(dl,dl_ub)
      dl = np.append(dl,dl_lb)

    xr = np.zeros(N*self.batch_count,np_dtype)
    xi = np.zeros(N*self.batch_count,np_dtype)
    
    # load on GPU
    self.d  = cp.asarray(d)
    self.dl = cp.asarray(dl)
    self.du = cp.asarray(du)
    self.xr = cp.asarray(xr)
    self.xi = cp.asarray(xi)
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
  def set_state(self):
    
    self.handle = cusp.create()
    
    N = self.N
    batch_stride = N
    batch_count = self.batch_count
    
    if self.single_precision:
      
      buf_bytes = cusp.sgtsv2strided_batch_buffer_size_ext(
        self.handle,
        N,
        self.dl.data.ptr,
        self.d.data.ptr,
        self.du.data.ptr,
        self.xr.data.ptr,
        batch_count,
        batch_stride
        )
    
    else:

      buf_bytes = cusp.dgtsv2strided_batch_buffer_size_ext(
        self.handle,
        N,
        self.dl.data.ptr,
        self.d.data.ptr,
        self.du.data.ptr,
        self.xr.data.ptr,
        batch_count,
        batch_stride
        )
        
    self.p_buffer = cp.empty(buf_bytes,dtype=cp.uint8)
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
  def solve_system(self):
    
    N = self.N
    batch_stride = N
    batch_count = self.batch_count

    if self.single_precision:
    
      cusp.sgtsv2strided_batch(
        self.handle,
        N,
        self.dl.data.ptr,
        self.d.data.ptr,
        self.du.data.ptr,
        self.xr.data.ptr,
        batch_count,
        batch_stride,
        self.p_buffer.data.ptr
        )

      cusp.sgtsv2strided_batch(
        self.handle,
        N,
        self.dl.data.ptr,
        self.d.data.ptr,
        self.du.data.ptr,
        self.xi.data.ptr,
        batch_count,
        batch_stride,
        self.p_buffer.data.ptr
        )

    else:

      cusp.dgtsv2strided_batch(
        self.handle,
        N,
        self.dl.data.ptr,
        self.d.data.ptr,
        self.du.data.ptr,
        self.xr.data.ptr,
        batch_count,
        batch_stride,
        self.p_buffer.data.ptr
        )

      cusp.dgtsv2strided_batch(
        self.handle,
        N,
        self.dl.data.ptr,
        self.d.data.ptr,
        self.du.data.ptr,
        self.xi.data.ptr,
        batch_count,
        batch_stride,
        self.p_buffer.data.ptr
        )
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
  def fill_rhs(self,W):
    
    threads = (16,16)
    blocks  = (math.ceil(self.N / threads[0]),
               math.ceil(self.batch_count / threads[1]))


    self.extract_diag(blocks,threads,(W,self.N,self.batch_count,self.xr,self.xi))
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
  def scale_main_diag(self):

    x = self.xi[:self.N]
    
    q = cp.mean(x,dtype=x.dtype)
    
    cp.subtract(x, q, out=x)
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
  def reorder_sol(self,P):
    
    threads = (16,16)
    blocks  = (math.ceil(self.N / threads[0]),
               math.ceil(self.batch_count / threads[1]))

    self.reorder_diag(blocks,threads,(self.xr,self.xi,self.N,self.batch_count,P))
# -------------------------------------------------------------------------------------- #