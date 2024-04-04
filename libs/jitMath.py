"""
    Basic subroutines for fast quadrotor simulation

    Copyright (C) 2024 Till Blaha -- TU Delft

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __main__ import jitter

@jitter("void(f4[:, ::1], f4[::1], f4[::1])")
def sgemv(A, x, b):
    # matrix-vector multiplication
    # no loop unrolling because we only know sizes at runtime... what gives
    for i in range(b.shape[0]):
        b[i] = A[i, 0] * x[0]
        for j in range(1, x.shape[0]):
            b[i] += A[i, j] * x[j]

@jitter("void(f4[:, ::1], f4[::1], f4[::1])")
def sgemv_add(A, x, b):
    # matrix-vector multiplication
    # no loop unrolling because we only know sizes at runtime... what gives
    for i in range(b.shape[0]):
        b[i] += A[i, 0] * x[0]
        for j in range(1, x.shape[0]):
            b[i] += A[i, j] * x[j]

@jitter("void(f4[::1], f4[::1], f4[::1])")
def cross(u, v, w):
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]

@jitter("void(f4[::1], f4[::1], f4[::1])")
def cross_add(u, v, w):
    w[0] += u[1]*v[2] - u[2]*v[1]
    w[1] += u[2]*v[0] - u[0]*v[2]
    w[2] += u[0]*v[1] - u[1]*v[0]

@jitter("void(f4[::1], f4[::1], f4[::1])")
def quatRotate(q, v, t):
    # crazy algorithm due to Fabian Giesen (A faster quaternion-vector multiplication)
    # https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
    # v' = v  +  q[0] * cross(2*q[1:], v)  +  cross(q[1:], cross(2*q[1:], v))
    # v' = v  +  q[0] * 2*cross(q[1:], v)  +  cross(q[1:], 2*cross(q[1:], v))

    # 15 multiplications, 15 additions
    cross(q[1:], v, t)
    for j in range(3):
        t[j] += t[j]

    cross_add(q[1:], t, v)
    for j in range(3):
        v[j] += q[0] * t[j]

@jitter("void(f4[::1],f4[::1],f4[::1])")
def quatDot(q, O, qDot):
    Ox, Oy, Oz = O
    qw, qx, qy, qz = q
    qDot[0] = .5 * (-Ox*qx - Oy*qy - Oz*qz)
    qDot[1] = .5 * ( Ox*qw + Oz*qy - Oy*qz)
    qDot[2] = .5 * ( Oy*qw - Oz*qx + Ox*qz)
    qDot[3] = .5 * ( Oz*qw + Oy*qx - Ox*qy)

#%% quadrotor specific stuff
from math import sqrt

@jitter("void(f4[::1],f4[::1],f4[::1],f4[::1],f4[::1])")
def motorDot(w, d, itau, wmax, wDot):
    #wDot[:] = (wc - w) / tau
    for j in range(4):
        #dLim = 0. if d[j] < 0. else 1. if d[j] > 1. else d[j]
        #wDot[j] = (sqrt(dLim)*wmax[j] - w[j]) * itau[j]
        wDot[j] = (sqrt( d[j] )*wmax[j] - w[j]) * itau[j]

@jitter("void(f4[::1], f4[::1], f4[:, ::1], f4[:, ::1], f4[::1], f4[::1])")
def forcesAndMoments(omega, omegaDot, G1, G2, fm, workspace):
    for j in range(4):
        workspace[j] = omega[j]*omega[j]

    fm[0] = 0.
    fm[1] = 0.
    sgemv(G1, workspace[:4], fm[2:])
    sgemv_add(G2, omegaDot, fm[5:])

