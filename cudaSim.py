#!/usr/bin/env python3

"""
    Vectorized quadrotor simulation with websocket pose output

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

import numpy as np
import numba as nb
from numba import cuda
from math import sqrt, acos, hypot
GRAVITY = 9.80665

### data

# todo list
# 1. benchmark SIMD (and contiguous arrays)
#     --> contiguity can save around half the time, even in simple cases
# 2. benchmark np.jit called from np.cuda.jit
#     --> nb.jit doesnt work with an error no one on the internet has seen
#     --> calling nb.cuda.jit(..., device=True) from a Kernel works like a charm
# 3. adapt sim for cuda
# 4. make class to get G1/G2 from system parameters
# 5. investiage why first run is slow in CUDA
#     --> this happens when a function signature isnt passed

# sim data
blocks = 128 # 128 seems best
threads_per_block = 64 # 32 or 64 seems best --> 4096 or 8192 parallel simulations
N = blocks*threads_per_block # run N simulations in parallel. At least 8192 for decent GPU utilization (RTX A1000)
dt = 0.002 # step time is dt seconds (forward Euler)
T = 10  # run for T seconds

# drone data
omegaMaxs = np.tile(np.array(
    [4000., 4000., 4000., 4000.], dtype=np.float32), (N, 1)) \
    + np.random.random((N, 4)).astype(np.float32)
taus = np.tile(np.array(
    [0.03, 0.03, 0.03, 0.03], dtype=np.float32), (N, 1)) \
    + np.random.random((N, 4)).astype(np.float32)*1e-3
itaus = 1. / taus
G1s        = 1. / (4000.**2) * np.tile( np.array(
    [
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [-8.,-8.,-8.,-8.],
        [-300.,-300.,300.,300.],
        [-200.,+200.,-200.,+200.],
        [-50.,+50.,+50.,-50.],
    ], dtype=np.float32), (N,1,1)) + (np.random.random((N, 6, 4)).astype(np.float32) * 1e-8)
G2s        = 1e-5 * np.tile(np.array(
    [
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [-50.,+50.,+50.,-50.],
    ], dtype=np.float32), (N,1,1)) + (np.random.random((N, 6, 4)).astype(np.float32) * 1e-5)

# position setpoints
grid_size = int(np.ceil(np.sqrt(N)))
x_vals = np.linspace(-3, 3, grid_size)
y_vals = np.linspace(-3, 3, grid_size)

X, Y = np.meshgrid(x_vals, y_vals)

vectors = np.column_stack((X.ravel(), Y.ravel(), -1.5*np.ones_like(X.ravel())))

pSets = vectors[:N].astype(np.float32)

# position controller gains (attitude/rate hardcoded for now, sorry)
posP = 4*np.ones((N, 3), dtype=np.float32)
velP = 2*np.ones((N, 3), dtype=np.float32)
#velI = 0.1*np.ones((N, 3), dtype=np.float32) # not implemented yet
#velIlimit = np.ones((N, 3), dtype=np.float32)
#velEI = np.zeros((N, 3), dtype=np.float32)


### Numba stuff

def dummy_decorator(f=None, *args, **kwargs):
    def decorator(func):
        return func

    if callable(f):
        return f
    else:
        return decorator

#vectorize = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True)
#jit = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
#nb.set_num_threads(max(nb.config.NUMBA_DEFAULT_NUM_THREADS-4, 1))
jit = lambda signature: nb.cuda.jit(signature, 
                                    fastmath=True, device=True, inline=False)
kernel = lambda signature: nb.cuda.jit(signature,
                                       fastmath=True, device=False)

#vectorize = dummy_decorator
#jit = dummy_decorator


### precompute

G1pinvs = np.linalg.pinv(G1s)


### states
xs = np.empty((N, 17), dtype=np.float32)
xDots = np.zeros_like(xs, dtype=np.float32)


### functions (f4 is shorthand for float32)
@jit("void(f4[::1],f4[::1],f4[::1],f4[::1],f4[::1])")
def motorDot(w, d, itau, wmax, wDot):
    #wDot[:] = (wc - w) / tau
    for j in range(4):
        dLim = 0. if d[j] < 0. else 1. if d[j] > 1. else d[j] 
        wDot[j] = (sqrt(dLim)*wmax[j] - w[j]) * itau[j]

@jit("void(f4[:, ::1], f4[::1], f4[::1], i4)")
def sgemv(A, x, b, add):
    # matrix-vector multiplication
    # no loop unrolling because we only know sizes at runtime... what gives
    for i in range(b.shape[0]):
        if add:
            b[i] += A[i, 0] * x[0]
        else:
            b[i] = A[i, 0] * x[0]

        for j in range(1, x.shape[0]):
            b[i] += A[i, j] * x[j]

@jit("void(f4[::1], f4[::1], f4[:, ::1], f4[:, ::1], f4[::1], f4[::1])")
def forcesAndMoments(omega, omegaDot, G1, G2, fm, workspace):
    for j in range(4):
        workspace[j] = omega[j]*omega[j]

    sgemv(G1, workspace, fm, False)

    #g2Term = np.empty(3, dtype=np.float32)
    sgemv(G2, omegaDot, fm[3:], True)

    #for j in range(3):
    #    fm[3+j] = g2Term[j]


@jit("void(f4[::1], f4[::1], f4[::1], i4)")
def cross(u, v, w, add):
    if add:
        w[0] += u[1]*v[2] - u[2]*v[1]
        w[1] += u[2]*v[0] - u[0]*v[2]
        w[2] += u[0]*v[1] - u[1]*v[0]
    else:
        w[0] = u[1]*v[2] - u[2]*v[1]
        w[1] = u[2]*v[0] - u[0]*v[2]
        w[2] = u[0]*v[1] - u[1]*v[0]

@jit("void(f4[::1], f4[::1], f4[::1])")
def quatRotate(q, v, t):
    # crazy algorithm due to Fabian Giesen (A faster quaternion-vector multiplication)
    # https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
    # v' = v  +  q[0] * cross(2*q[1:], v)  +  cross(q[1:], cross(2*q[1:], v))
    # v' = v  +  q[0] * 2*cross(q[1:], v)  +  cross(q[1:], 2*cross(q[1:], v))

    cross(q[1:], v, t, False)
    for j in range(3):
        t[j] *= 2.

    cross(q[1:], t, v, True)
    for j in range(3):
        v[j] += q[0] * t[j]

@jit("void(f4[::1],f4[::1],f4[::1])")
def quatDot(q, O, qDot):
    Ox, Oy, Oz = O
    qw, qx, qy, qz = q
    qDot[0] = .5 * (-Ox*qx - Oy*qy - Oz*qz)
    qDot[1] = .5 * ( Ox*qw + Oz*qy - Oy*qz)
    qDot[2] = .5 * ( Oy*qw - Oz*qx + Ox*qz)
    qDot[3] = .5 * ( Oz*qw + Oy*qx - Ox*qy)


@kernel("void(f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, :, ::1], f4[:, :, ::1], f4[:, ::1], f4[:, ::1])")
def step(x, d, itau, wmax, G1, G2, xdot, workspace):
    i1 = cuda.grid(1)

    #pos   = x[i1, 0:3]
    #vel   = x[i1, 3:6]
    q     = x[i1, 6:10]
    Omega = x[i1, 10:13]
    omega = x[i1, 13:17]

    # motor model
    #omegaDot = np.empty(4, dtype=np.float32)
    #omegaDot = cuda.device_array(4, dtype=np.float32)
    motorDot(omega, d[i1], itau[i1], wmax[i1], xdot[i1, 13:17])

    # forces
    #fm = np.zeros(6, dtype=np.float32)
    forcesAndMoments( omega, xdot[i1, 13:17], G1[i1], G2[i1], xdot[i1, 3:6], workspace[i1, :6])

    ## kinematics
    #qDot = np.zeros(4, dtype=np.float32)
    quatDot(q, Omega, xdot[i1, 6:10])

    quatRotate(q, xdot[i1, 3:6], workspace[i1, :3])
    xdot[i1, 5] += GRAVITY

    ## step forward
    for j in range(17):
        x[i1, j] += dt * xdot[i1, j]

    # normalize quaternion
    ni = 1. / sqrt(x[i1, 6]**2 + x[i1, 7]**2 + x[i1, 8]**2 + x[i1, 9]**2)
    for j in range(6,10):
        x[i1, j] *= ni


#@vectorize([(nb.int32, nb.float32[:], nb.float32[:])],
#                   '(),(states),(n)')
#def Controller(i, x, d):
#    pos = x[:3].copy()
#    vel = x[3:6].copy()
#
#    velSp = posP[i]/velP[i] * (pSets[i] - pos)
#    velE = velSp - vel
#    #velEI[i][:] = np.clip(velEI[i] + velE, -velIlimit[i], velIlimit[i])
#
#    accSp = velP[i]*velE# + velI[i]*velEI[i]
#
#    qi = x[6:10].copy()
#    qi[0] = -qi[0]
#
#    fsp = accSp - np.array([0, 0, GRAVITY], dtype=np.float32)
#    quatRotate(qi, fsp)
#
#    fzsp = np.linalg.norm(fsp)
#    if (abs(fzsp) < 1e-5):
#        # we are supposed to be falling, don't control attitude
#        tiltE = np.array([0., 0., 0.], dtype=np.float32)
#        cosTilt = 1.
#    else:
#        fsp /= fzsp
#
#        tilt = np.array([-fsp[1], fsp[0], 0.], dtype=np.float32)
#
#        sinTilt = hypot(fsp[0], fsp[1])
#        cosTilt = -fsp[2] # dot prodict 
#        if (sinTilt < 1e-5):
#            # either we are aligned with attitude set or 180deg away
#            if (cosTilt > 0):
#                # aligned
#                tiltE = np.array([0., 0., 0.], dtype=np.float32)
#            else:
#                # 180 deg
#                tiltE = np.array([np.pi, 0., 0.], dtype=np.float32)
#        else:
#            tiltE = tilt / sinTilt * acos(cosTilt)
#
#    OmegaSp = -200./20. * tiltE
#    OmegaE = OmegaSp - x[10:13].copy()
#    OmegaDotSp = 20. * OmegaE
#
#    fspBody = cosTilt * np.array([0., 0., -fzsp], dtype=np.float32)
#    fspBody[2] = 0. if fspBody[2] > 0. else fspBody[2]
#    v = np.array([fspBody[0], fspBody[1], fspBody[2],
#                  OmegaDotSp[0], OmegaDotSp[1], OmegaDotSp[2]], dtype=np.float32)
#    #G1u = G1u * omegaMaxs[i]**2
#    #d[:] = np.linalg.pinv(G1u) @ v
#    G1pinvu = G1pinv[i] / (omegaMaxs[i]*omegaMaxs[i])[:, np.newaxis]
#    d[:] = G1pinvu @ v
#    for j in range(len(d)):
#        d[j] = 0. if d[j] < 0. else 1. if d[j] > 1. else d[j]


### websocket stuff
ws = None

import asyncio
import websockets
import json
from time import time
async def handle_client(websocket, path):
    # Send data to the client
    global ws
    ws = websocket
    while True:
        await asyncio.Future() # run forever without blokcing?

async def start_server():
    # Start WebSocket server
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # run forever

#async def mainDebug():
#    # if the dummy decorators are used, use this main.
#    # Nice for debugging, as it only runs a single sim without numba jit compiling
#
#    # havent run this function in a while, may be buggy now
#
#    x0 = np.zeros(17).astype(np.float32);
#    #x0[6:10] = np.random.random(4) - 0.5
#    x0[2] = -1.5
#    x0[6] = 1.
#    x0[7] = 0.
#    x0[6:10] /= np.linalg.norm(x0[6:10])
#    x[0, :] = x0.copy()
#
#    coro = start_server()
#    asyncio.create_task(coro)
#    await asyncio.sleep(3)
#
#    d = np.zeros((N, 4), dtype=np.float32)
#
#    i = 0
#    while True:
#        Controller(0, x[0], d[0])
#        Dot(0, x[0], d[0], xDot[0])
#        step(xDot[0], dt, x[0])
#
#        if ws is not None and not i % int(0.1/dt):
#            data = {"id": 0, "pos": list(x[0, :3]), "quat": list(x[0, 6:10])}
#            await ws.send(json.dumps(data))
#
#        i += 1
#        print(i)
#        await asyncio.sleep(0.1)

ts = 0

async def main():
#def main():
    x0 = np.random.random((N, 17)).astype(np.float32) - 0.5
    #x0[:, :3] = 0.
    #x0[:, 3:6] = 0.
    #x0[:, 6] = 1.
    #x0[:, 7:10] = 0.
    x0[:, 6:10] /= np.linalg.norm(x0[:, 6:10], axis=1)[:, np.newaxis]
    #x0[:, 10:13] = 0.
    #x0[:, 13:17] = 0.

    xs[:] = x0.copy()
    workspaces = np.empty((N, 6), dtype=np.float32)

    coro = start_server()
    asyncio.create_task(coro)
    print("initialized websocket")
    await asyncio.sleep(2)


    ds = np.random.random((N, 4)).astype(np.float32)
    d_ds = cuda.to_device(ds)
    d_xs = cuda.to_device(xs)
    d_itaus = cuda.to_device(itaus)
    d_omegaMaxs = cuda.to_device(omegaMaxs)
    d_G1s = cuda.to_device(G1s)
    d_G2s = cuda.to_device(G2s)
    d_xDots = cuda.to_device(xDots)
    d_workspaces = cuda.to_device(workspaces)

    global ts
    ts = time()
    for i in range(int(T / dt)):
        #tStart = time()

        #Controller(idxs, x, d)
        #cuda.synchronize()

        step[128,64](d_xs, d_ds, d_itaus, d_omegaMaxs, d_G1s, d_G2s, d_xDots, d_workspaces)
        #step[128,64](d_xs, d_ds, d_itaus, d_omegaMaxs, d_G1s, d_G2s, d_xDots, d_workspaces)

        if ws is not None and not i % int(0.1/dt):
            xs[:] = d_xs.copy_to_host()
            j = 0
            for xD in xs[:200].astype(np.float64):
                data = {"id": j, "pos": list(xD[:3]), "quat": list(xD[6:10])}
                #data = {"id": i, "pos": [1,2,3.], "quat": [1., 0., 0., 0.]}
                j += 1;
                await ws.send(json.dumps(data))

        #dtReal = time() - tStart
        #e = dt - dtReal
        #sleep(max(e, 0))


#asyncio.run(mainDebug())
asyncio.run(main())
#main()
runtime = time() - ts

print(f"Achieved {N*T / dt / runtime / 1e6:.2f}M ticks per second.")