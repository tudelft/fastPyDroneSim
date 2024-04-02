import numba as nb
import numpy as np
from numba import cuda
np.random.seed = 42

@cuda.jit("void(float32[:, :, ::1], float32[:, ::1], float32[:, ::1])", fastmath=True)
def sgemv_kernel(A, x, b):
    i1 = cuda.grid(1)
    for i in range(b.shape[1]):
        b[i1, i] = A[i1, i, 0] * x[i1, 0]
        for j in range(1,x.shape[1]):
            b[i1, i] = cuda.fma(A[i1, i, j], x[i1, j], b[i1, i])

def sgemv_ufunc(A, x, b, m):
    for i in range(m):
        b[i] = A[i, 0] * x[0]
        for j in range(1, m):
            b[i] += A[i, j] * x[j]

sgemv_simd = nb.jit("void(float32[:, ::1], float32[::1], float32[::1], int32)", nopython=True, fastmath=False)(sgemv_ufunc)
sgemv_cuda_device = cuda.jit("void(float32[:, ::1], float32[::1], float32[::1], int32)",
                             nopython=True, fastmath=True, device=True, inline=False)(sgemv_ufunc)

@cuda.jit("void(float32[:, :, ::1], float32[:, ::1], float32[:, ::1], int32)",
          fastmath=True)
def sgemv_call_jit(A, x, b, m):
    i1 = cuda.grid(1)
    sgemv_cuda_device(A[i1], x[i1], b[i1], m)

def sgemv_simd_vec(A, x, b, m):
    sgemv_simd(A, x, b, m)


def sgemv_naive(A, x, b, m):
    for i in range(m):
        b[i] = A[i, 0] * x[0]
        for j in range(1, m):
            b[i] += A[i, j] * x[j]

def sgemv_np(A, x, b):
    b[:] = A @ x

sgemv_naive_guvec = nb.guvectorize(
    [(nb.float32[:,::1], nb.float32[::1], nb.float32[::1], nb.int32)],
    '(M,M),(M),(M),()',
    target='parallel', nopython=True)(sgemv_naive)

sgemv_simd_guvec = nb.guvectorize(
    [(nb.float32[:,::1], nb.float32[::1], nb.float32[::1], nb.int32)],
    '(M,M),(M),(M),()',
    target='parallel', nopython=True)(sgemv_simd_vec)

sgemv_naive_cuda = nb.guvectorize(
    [(nb.float32[:,:], nb.float32[:], nb.float32[:], nb.int32)],
    '(M,M),(M),(M),()',
    target='cuda', nopython=True)(sgemv_naive)

sgemv_np_guvec = nb.guvectorize(
    [(nb.float32[:,:], nb.float32[:], nb.float32[:])],
    '(M,M),(M),(M)',
    target='parallel', nopython=True)(sgemv_np)

M = 101
N = 2048*4

A = np.random.random((N, M, M)).astype(np.float32)
x = np.random.random((N, M)).astype(np.float32)
b = np.zeros_like(x).astype(np.float32)

stream = cuda.stream()
cA = cuda.to_device(A)
cx = cuda.to_device(x)
cb = cuda.to_device(b)
stream.synchronize()

#sgemv_naive_guvec(A,x,b,M) --> 11.4ms
#sgemv_simd_guvec(A,x,b,M) --> 11ms
#sgemv_naive_cuda(A,x,b,M) --> 34ms, vectorizes the wrong thing
#sgemv_kernel[64, 128](cA, cx, cb) --> first call takes loooong, then 200us
#    first call takes long because i didnt pass the function signature initially. Now fixed


# M = 101, N = 1000
# sgemv_naive            : N/A
# sgemv_naive_guvec      :    446us
# sgemv_np               :     22us
# sgemv_np_jit           :    543us
# sgemv_np_contiguous    :     22us
# sgemv_np_contiguous_jit:     27us
# A @ x.T                : 894ms
# A @ x.T                : 894ms
