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
dt = 0.002 # step time is dt seconds (forward Euler)
T = 10  # run for T seconds
log_interval = 5   # log stat every couple of iterations. 0 == off
# may run out of graphics RAM if N*iters too large or log-interval too low

# on an RTX A1000 mobile workstation GPU, this works best
if log_interval > 0:
    blocks = 128 # 128 seems best. Must be multiple of 32 (?)
    threads_per_block = 64 # depends on global memorty usage. Must be multiple of 64
else:
    blocks = 256
    threads_per_block = 256

# pre calc
N = blocks*threads_per_block # run N simulations in parallel. At least 8192 for decent GPU utilization (RTX A1000)
Nlog = 0 if log_interval == 0 else N / log_interval
Nviz = 512
iters = int(T / dt)


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
#        [0.,0.,0.,0.],
#        [0.,0.,0.,0.],
        [-8.,-8.,-8.,-8.],
        [-300.,-300.,300.,300.],
        [-200.,+200.,-200.,+200.],
        [-50.,+50.,+50.,-50.],
    ], dtype=np.float32), (N,1,1)) + (np.random.random((N, 4, 4)).astype(np.float32) * 1e-8)
G2s        = 0* 1e-5 * np.tile(np.array(
    [
#        [0.,0.,0.,0.],
#        [0.,0.,0.,0.],
#        [0.,0.,0.,0.],
#        [0.,0.,0.,0.],
#        [0.,0.,0.,0.],
        [-50.,+50.,+50.,-50.],
    ], dtype=np.float32), (N,1,1)) + (np.random.random((N, 1, 4)).astype(np.float32) * 1e-5)

# position setpoints
grid_size = int(np.ceil(np.sqrt(N)))
x_vals = np.linspace(-7, 7, grid_size)
y_vals = np.linspace(-7, 7, grid_size)

X, Y = np.meshgrid(x_vals, y_vals)

vectors = np.column_stack((X.ravel(), Y.ravel(), -1.5*np.ones_like(X.ravel())))

pSets = vectors[:N].astype(np.float32)

# position controller gains (attitude/rate hardcoded for now, sorry)
posPs = 2*np.ones((N, 3), dtype=np.float32)
velPs = 2*np.ones((N, 3), dtype=np.float32)
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
                                    fastmath=False, device=True, inline=False)
kernel = lambda signature: nb.cuda.jit(signature,
                                       fastmath=False, device=False)

#jit = dummy_decorator
#kernel = dummy_decorator


### precompute control allocation (FIXME: should be a weighted pseudoinverse!!)
G1pinvs = np.linalg.pinv(G1s) / (omegaMaxs*omegaMaxs)[:, :, np.newaxis]
#G1pinvs = np.linalg.pinv(G1s) / 4000 / 4000


### states
xs = np.empty((N, 17), dtype=np.float32)
xDots = np.zeros_like(xs, dtype=np.float32)
if log_interval > 0:
    xs_log = np.empty((int(iters / log_interval), N, 17), dtype=np.float32)
    xs_log[:] = np.nan
else:
    xs_log = np.empty((1, N, 17), dtype=np.float32)
    xs_log[:] = np.nan


### functions (f4 is shorthand for float32)
@jit("void(f4[::1],f4[::1],f4[::1],f4[::1],f4[::1])")
def motorDot(w, d, itau, wmax, wDot):
    #wDot[:] = (wc - w) / tau
    for j in range(4):
        #dLim = 0. if d[j] < 0. else 1. if d[j] > 1. else d[j]
        #wDot[j] = (sqrt(dLim)*wmax[j] - w[j]) * itau[j]
        wDot[j] = (sqrt( d[j] )*wmax[j] - w[j]) * itau[j]

@jit("void(f4[:, ::1], f4[::1], f4[::1])")
def sgemv(A, x, b):
    # matrix-vector multiplication
    # no loop unrolling because we only know sizes at runtime... what gives
    for i in range(b.shape[0]):
        b[i] = A[i, 0] * x[0]
        for j in range(1, x.shape[0]):
            b[i] += A[i, j] * x[j]

@jit("void(f4[:, ::1], f4[::1], f4[::1])")
def sgemv_add(A, x, b):
    # matrix-vector multiplication
    # no loop unrolling because we only know sizes at runtime... what gives
    for i in range(b.shape[0]):
        b[i] += A[i, 0] * x[0]

        for j in range(1, x.shape[0]):
            b[i] += A[i, j] * x[j]

@jit("void(f4[::1], f4[::1], f4[:, ::1], f4[:, ::1], f4[::1], f4[::1])")
def forcesAndMoments(omega, omegaDot, G1, G2, fm, workspace):
    for j in range(4):
        workspace[j] = omega[j]*omega[j]

    fm[0] = 0.
    fm[1] = 0.
    sgemv(G1, workspace[:4], fm[2:])
    sgemv_add(G2, omegaDot, fm[5:])

@jit("void(f4[::1], f4[::1], f4[::1])")
def cross(u, v, w):
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]

@jit("void(f4[::1], f4[::1], f4[::1])")
def cross_add(u, v, w):
    w[0] += u[1]*v[2] - u[2]*v[1]
    w[1] += u[2]*v[0] - u[0]*v[2]
    w[2] += u[0]*v[1] - u[1]*v[0]

@jit("void(f4[::1], f4[::1], f4[::1])")
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

@jit("void(f4[::1],f4[::1],f4[::1])")
def quatDot(q, O, qDot):
    Ox, Oy, Oz = O
    qw, qx, qy, qz = q
    qDot[0] = .5 * (-Ox*qx - Oy*qy - Oz*qz)
    qDot[1] = .5 * ( Ox*qw + Oz*qy - Oy*qz)
    qDot[2] = .5 * ( Oy*qw - Oz*qx + Ox*qz)
    qDot[3] = .5 * ( Oz*qw + Oy*qx - Ox*qy)


@kernel("void(f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, :, ::1], f4[:, :, ::1], f4, i4, f4[:, :, ::1])")
def step(x, d, itau, wmax, G1, G2, dt, current_iter, x_log):
#def step(x, d, itau, wmax, G1, G2, iters, x_log):
    i1 = cuda.grid(1)

    x_local = cuda.local.array(17, dtype=nb.float32)
    for j in range(6,17):
        x_local[j] = x[i1, j]

    #pos   = x_local[0:3]
    #vel   = x_local[3:6]
    q     = x_local[6:10]
    Omega = x_local[10:13]
    omega = x_local[13:17]

    xdot_local = cuda.local.array(17, dtype=nb.float32)
    #posDot = xdot_local[0:3]
    velDot = xdot_local[3:6]
    qDot = xdot_local[6:10]
    OmegaDot = xdot_local[10:13]
    omegaDot = xdot_local[13:17]

    # workspace needed for a few jit functions
    work = cuda.local.array(4, dtype=nb.float32)

    #%% motor forces
    # motor model
    motorDot(omega, d[i1], itau[i1], wmax[i1], omegaDot)

    # forces and moments
    fm = cuda.local.array(6, dtype=nb.float32)
    forcesAndMoments( omega, omegaDot, G1[i1], G2[i1], fm, work[:4])

    for j in range(3):
        velDot[j] = fm[j] # still needs to be rotated, see next section
        OmegaDot[j] = fm[j+3]

    #%% kinematics
    quatRotate(q, velDot, work[:3])
    velDot[2] += GRAVITY

    quatDot(q, Omega, qDot)

    #%% step forward
    for j in range(0,3): # position
        x[i1, j] += dt * x[i1, j+3]

    for j in range(3,6): # velocity
        x[i1, j] += dt * xdot_local[j]

    # quaternion needs to be normalzied after stepping. do that efficiently
    # without sqrt
    qnorm2 = 0.
    for j in range(4):
        q[j] += dt * qDot[j]
        #q[j] *= q[j]
        qnorm2 += q[j]*q[j]

    # q now contains the squares of each elements. Normalize these!
    iqnorm = 1. / sqrt(qnorm2)
    for j in range(4):
        x[i1, j+6] = q[j] * iqnorm

    for j in range(10,17): # Omega and omega
        x[i1, j] += dt * xdot_local[j]

    #%% save state
    if log_interval > 0:
        log_idx = current_iter // log_interval
        for j in range(17):
            x_log[log_idx, i1, j] = x[i1, j]


@kernel("void(f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, :, ::1])")
def controller(x, d, posPs, velPs, pSets, G1pinv):
    i1 = cuda.grid(1)

    x_local = cuda.local.array(17, dtype=nb.float32)
    for j in range(17):
        x_local[j] = x[i1, j]

    pos   = x_local[0:3]
    vel   = x_local[3:6]
    qi    = x_local[6:10]
    qi[0] = -qi[0]
    Omega = x_local[10:13]

    # this could be faster if allocating local memory
    posP = posPs[i1]
    velP = velPs[i1]
    pSet = pSets[i1]

    #%% position control
    # calculate specific force setpoint forceSp to achieve stable position
    forceSp = cuda.local.array(3, dtype=nb.float32)

    # more better would be having an integrator, but I don't feel like making
    # another state for that
    for j in range(3):
        # xy are controlled separately, which isn't nice, but easy to code
        #                       |---  velocity set point ---|
        forceSp[j] = velP[j] * ( ( posP[j] * (pSet[j] - pos[j]) ) - vel[j] )

    #forceSp[0] = 0.
    #forceSp[1] = 0.
    #forceSp[2] = 0.

    forceSp[2] -= GRAVITY

    workspace = cuda.local.array(3, dtype=nb.float32)
    quatRotate(qi, forceSp, workspace)

    #%% attitude control
    # Calculate rate derivative setpoint to steer towards commanded tilt.
    # Conveniently forget about yaw
    # tilt error is tilt_error_angle * cross(actual_tilt, commanded_tilt)
    # but the singularities make computations a bit lengthy
    tiltErr = cuda.local.array(2, dtype=nb.float32) # roll pitch only

    cosTilt = 1.
    for j in range(2):
        tiltErr[j] = 0.

    fzsp = sqrt(forceSp[0]**2 + forceSp[1]**2 + forceSp[2]**2)
    if (fzsp < 1e-5):
        # we are supposed to be falling, don't control attitude
        # ie leave tiltErr=0. and cosTilt=1.
        pass
    else:
        ifzsp = 1. / fzsp
        for j in range(3):
            forceSp[j] *= ifzsp

        sinTilt = hypot(forceSp[0], forceSp[1]) # cross product magnitude
        cosTilt = -forceSp[2] # dot product 
        if (sinTilt < 1e-5):
            # either we are aligned with attitude set or 180deg away
            if (cosTilt > 0):
                # aligned, tiltErr stays 0
                pass
            else:
                # 180 deg, lets use roll (index 0)
                tiltErr[0] = -np.pi
        else:
            tiltAngleOverSinTilt = acos(cosTilt) / sinTilt
            tiltErr[0] = +forceSp[1] * tiltAngleOverSinTilt
            tiltErr[1] = -forceSp[0] * tiltAngleOverSinTilt

    # control yaw to point foward at all times
    yawRateSp = 0.
    if (cosTilt > 0.):
        # more of less upright, lets control yaw
        # error angle is 90deg - angle(body_x, (0 1 0))
        acosAngle = 2*qi[1]*qi[2] - 2*qi[3]*qi[0]
        acosAngle = -1. if acosAngle < -1. else +1. if acosAngle > +1. else acosAngle
        yawE = 0.5*np.pi - acos(acosAngle)

        if (1 - 2*qi[2]*qi[2] - 2* qi[3]*qi[3]) < 0.:
            # dot product of global x and body x is positive! We are looking
            # backwards and the angle is relative to pi
            if yawE >= 0.:
                yawE = np.pi - yawE
            else:
                yawE = -np.pi - yawE

        yawRateSp = -1 * cosTilt * yawE

    # rate derivative setpoints
    OmegaDotSp = cuda.local.array(3, dtype=nb.float32) # rotation setpoint roll pitch yaw
    for j in range(2):
        # hardcoded gains, oops
        #                     |--- Rate setpoint ----------|
        OmegaDotSp[j] = 20. * (10. * tiltErr[j] - Omega[j])

    OmegaDotSp[2] = 20. * (yawRateSp - Omega[2])

    #%% NDI allocation
    # pseudocontrols:
    v = cuda.local.array(4, dtype=nb.float32)

    if cosTilt > 0:
        v[0] = -cosTilt*fzsp # z-force
    else:
        v[0] = 0.
    v[1] = OmegaDotSp[0]
    v[2] = OmegaDotSp[1]
    v[3] = OmegaDotSp[2]

    #d[:] = G1pinv[i1] @ v
    sgemv(G1pinv[i1], v, d[i1])
    for j in range(4):
        d[i1, j] = 0. if d[i1, j] < 0. else 1. if d[i1, j] > 1. else d[i1, j]
        #d[i1, j] = 0.


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

ds = np.random.random((N, 4)).astype(np.float32)

from ipdb import set_trace

#np.random.seed(42)
x0 = np.random.random((N, 17)).astype(np.float32) - 0.5

async def main():
#def main():
    #x0[:, :3] = 0.
    #x0[:, 3:6] = 0.
    #x0[:, 6] = 1.
    #x0[:, 7:10] = 0.
    x0[:, 6:10] /= np.linalg.norm(x0[:, 6:10], axis=1)[:, np.newaxis]
    #x0[:, 10:13] = 0.
    #x0[:, 13:17] = 0.

    xs[:] = x0.copy()

    coro = start_server()
    asyncio.create_task(coro)
    print("initialized websocket")
    await asyncio.sleep(1)

    tsAll = time()

    d_ds = cuda.to_device(ds)
    d_xs = cuda.to_device(xs)
    d_xs_log = cuda.to_device(xs_log)
    d_itaus = cuda.to_device(itaus)
    d_omegaMaxs = cuda.to_device(omegaMaxs)
    d_G1s = cuda.to_device(G1s)
    d_G2s = cuda.to_device(G2s)

    d_posPs = cuda.to_device(posPs)
    d_velPs = cuda.to_device(velPs)
    d_pSets = cuda.to_device(pSets)
    d_G1pinvs = cuda.to_device(G1pinvs)
    cuda.synchronize()

    ts = time()
    iters = int(T / dt)
    for i in range(iters):
        #tStart = time()

        step[blocks,threads_per_block](d_xs, d_ds, d_itaus, d_omegaMaxs, d_G1s, d_G2s, dt, i, d_xs_log)
        controller[blocks,threads_per_block](d_xs, d_ds, d_posPs, d_velPs, d_pSets, d_G1pinvs)

        if ws is not None and not i % int(0.05/dt):
            # visualize every 0.1 seconds
            xs[:] = d_xs.copy_to_host()
            j = 0
            for j, xD in enumerate(xs[::int(N/Nviz+0.5)].astype(np.float64)):
                # plot a maximum of 256
                if not np.isnan(xD).any():
                    data = {"id": j, "pos": list(xD[:3]), "quat": list(xD[6:10])}
                    await ws.send(json.dumps(data))

        #dtReal = time() - tStart
        #e = dt - dtReal
        #await asyncio.sleep(max(e, 0)/4)

    cuda.synchronize()
    runtime = time() - ts

    if log_interval > 0:
        xs_log[:] = d_xs_log.copy_to_host()

    runtimeOverhead = time() - tsAll
    return runtime, runtimeOverhead

#asyncio.run(mainDebug())
runtime, runtimeOverhead = asyncio.run(main())
#main()

print(f"Achieved {N*iters / runtime / 1e6:.2f}M ticks per second (sim only) across {runtime:.6f} seconds.")
print(f"Retrieved {Nlog*iters / runtimeOverhead / 1e6:.2f}M ticks per second across {runtimeOverhead:.3f} seconds.")