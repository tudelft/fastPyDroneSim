import numba as nb
import numpy as np
np.random.seed = 42

def sgemv_naive(A, x, b, n):
    for i in range(n):
        b[0] = A[i, 0] * x[0]
        for j in range(1,n):
            b[0] += A[i, j] * x[j]

def sgemv_np(A, x, b):
    b[:] = A @ x

def sgemv_np_contiguous(A, x, b):
    b[:] = np.ascontiguousarray(A) @ np.ascontiguousarray(x)

sgemv_naive_jit = nb.jit("void(float32[:,:],float32[:],float32[:],int32)")(sgemv_naive)
sgemv_np_jit = nb.jit("void(float32[:,:],float32[:],float32[:])")(sgemv_np)
sgemv_np_contiguous_jit = nb.jit("void(float32[:,:],float32[:],float32[:])")(sgemv_np_contiguous)

M = 1001

A = np.random.random((M, M)).astype(np.float32)
x = np.random.random(M).astype(np.float32)
b = np.empty_like(x).astype(np.float32)

# sgemv(A, x, b)
# sgemv_naive(A, x, b, M)
# sgemv_naive_jit(A, x, b, M)
# A @ x

# M = 1001
# sgemv_naive            : 214000us
# sgemv_naive_jit        :    446us
# sgemv_np               :     22us
# sgemv_np_jit           :    543us
# sgemv_np_contiguous    :     22us
# sgemv_np_contiguous_jit:     27us
# A @ x                  :     23us
